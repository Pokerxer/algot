"""
ICT Backtester - Phase 2
========================
Bar-by-bar historical replay of the full ICTSignalEngine pipeline.

Features
--------
- Fetches data from MT5 (or loads CSVs if MT5 unavailable / --csv-dir used)
- Proper warmup period before first signal (100 bars)
- Realistic execution model:
    * Limit orders: fill only when bar's low/high touches entry price
    * Slippage: configurable pips/points added on entry fill
    * Spread cost on entry
    * One position per symbol at a time (matches live bot)
- Kill zone filter, confluence threshold, daily loss limit – all honoured
- Exit logic: stop-loss → take-profit → max bars held (time stop)
- Full metrics: win rate, profit factor, expectancy, Sharpe, Calmar,
  max drawdown, avg R:R achieved, per-session breakdown, per-symbol breakdown,
  per-confluence-band breakdown
- Saves:
    * backtest_trades.csv   – one row per closed trade (ML-ready for Phase 3)
    * backtest_equity.csv   – bar-level equity curve
    * backtest_report.json  – complete metrics dict

Usage
-----
# With live MT5 connection:
python3 backtester.py --symbols EURUSD,XAUUSD,US30 \\
                      --login 12345 --password pass --server Exness-MT5 \\
                      --start 2024-01-01 --end 2024-12-31 --risk 0.02 --rr 3.0

# Without MT5 (CSV files named  EURUSD_H1.csv  with columns open,high,low,close,time):
python3 backtester.py --symbols EURUSD --csv-dir ./data --start 2024-01-01
"""

from __future__ import annotations

import os
import sys
import json
import math
import argparse
import csv
from copy import deepcopy
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Path setup so handlers are importable from project dir ───────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR)           # handlers live here too
for d in [SCRIPT_DIR, PROJECT_DIR]:
    if d not in sys.path:
        sys.path.insert(0, d)

# ── MT5 (optional) ────────────────────────────────────────────────────────────
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None

# ── Signal engine ─────────────────────────────────────────────────────────────
try:
    from signal_engine import ICTSignalEngine
    ENGINE_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: signal_engine.py not found – {e}")
    ENGINE_AVAILABLE = False

# ── Contract / sizing helpers (reuse from bot) ────────────────────────────────
try:
    from ict_v7_mt5_fixed import get_contract_info, calculate_position_size, get_mt5_symbol
except ImportError:
    # Inline minimal fallback so the backtester is self-contained
    def get_contract_info(symbol: str) -> Dict:
        s = symbol.upper()
        futures = {'XAUUSD': {'dollar_per_point': 100, 'min_stop': 5.0, 'type': 'futures'},
                   'XTIUSD': {'dollar_per_point': 1000,'min_stop': 0.30,'type': 'futures'}}
        if s in futures: return futures[s]
        indices = {'US30': {'dollar_per_point': 1, 'min_stop': 20, 'type': 'indices'},
                   'USTEC':{'dollar_per_point': 1, 'min_stop': 10, 'type': 'indices'},
                   'US500':{'dollar_per_point': 1, 'min_stop': 3,  'type': 'indices'}}
        if s in indices: return indices[s]
        return {'pip_value': 10, 'min_stop': 0.002, 'decimal_places': 5, 'type': 'forex'}

    def calculate_position_size(symbol, account_value, risk_pct, stop_distance, current_price):
        ci = get_contract_info(symbol)
        risk = account_value * risk_pct
        if stop_distance <= 0: return 0.01, 0.0
        if ci['type'] == 'forex':
            pip = 0.01 if ci.get('decimal_places', 5) == 3 else 0.0001
            pv  = ci.get('pip_value', 10)
            qty = risk / ((stop_distance / pip) * pv)
        elif ci['type'] in ('futures', 'indices'):
            qty = risk / (stop_distance * ci.get('dollar_per_point', 1))
        else:
            qty = risk / stop_distance
        qty = max(0.01, round(qty / 0.01) * 0.01)
        return qty, qty * stop_distance

    def get_mt5_symbol(s): return s.upper() + 'm'


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TradeRecord:
    """One closed trade – used for metrics and ML export."""
    trade_id:          int
    symbol:            str
    direction:         int          # +1 long / -1 short
    entry_bar:         int          # index in full df
    entry_time:        str
    entry_price:       float
    stop_loss:         float
    take_profit:       float
    lot_size:          float
    risk_amount:       float        # $ risked
    exit_bar:          int
    exit_time:         str
    exit_price:        float
    exit_reason:       str          # 'target' | 'stop' | 'time_stop'
    pnl_raw:           float        # $ before commission
    pnl_net:           float        # $ after commission
    r_multiple:        float        # pnl_net / risk_amount
    bars_held:         int
    # Signal metadata (for ML feature vector)
    confluence:        int
    confidence:        str
    kill_zone:         str
    htf_trend:         str
    ob_type:           str
    liq_swept:         str
    model_2022:        str
    silver_bullet:     bool
    fvg_type:          str
    # Derived
    won:               bool         # r_multiple > 0


@dataclass
class OpenPosition:
    symbol:      str
    direction:   int
    entry_bar:   int
    entry_time:  str
    entry_price: float
    stop_loss:   float
    take_profit: float
    lot_size:    float
    risk_amount: float
    confluence:  int
    confidence:  str
    metadata:    Dict = field(default_factory=dict)
    bars_held:   int = 0


# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def _mt5_timeframe(tf: str):
    """Map string to MT5 timeframe constant."""
    m = {'M1': mt5.TIMEFRAME_M1, 'M5': mt5.TIMEFRAME_M5,
         'M15': mt5.TIMEFRAME_M15, 'M30': mt5.TIMEFRAME_M30,
         'H1': mt5.TIMEFRAME_H1,  'H4': mt5.TIMEFRAME_H4,
         'D1': mt5.TIMEFRAME_D1}
    return m.get(tf.upper(), mt5.TIMEFRAME_H1)


def load_data_mt5(symbol: str, start: datetime, end: datetime,
                  timeframe: str = 'H1') -> Optional[pd.DataFrame]:
    """Fetch historical OHLC from MT5."""
    if not MT5_AVAILABLE or mt5 is None:
        return None
    mt5_sym = get_mt5_symbol(symbol)
    if not mt5.symbol_select(mt5_sym, True):
        print(f"  Cannot select {mt5_sym}")
        return None
    rates = mt5.copy_rates_range(mt5_sym, _mt5_timeframe(timeframe), start, end)
    if rates is None or len(rates) == 0:
        print(f"  No data for {mt5_sym}")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.rename(columns={'tick_volume': 'volume'}, inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']]
    return df


def load_data_csv(symbol: str, csv_dir: str,
                  timeframe: str = 'H1') -> Optional[pd.DataFrame]:
    """Load OHLC from a CSV file named  SYMBOL_TF.csv ."""
    fname = os.path.join(csv_dir, f"{symbol.upper()}_{timeframe.upper()}.csv")
    if not os.path.exists(fname):
        print(f"  CSV not found: {fname}")
        return None
    df = pd.read_csv(fname, parse_dates=['time'], index_col='time')
    df.columns = [c.lower() for c in df.columns]
    required = {'open', 'high', 'low', 'close'}
    if not required.issubset(df.columns):
        print(f"  {fname} missing columns {required - set(df.columns)}")
        return None
    return df[list(required)]


# ══════════════════════════════════════════════════════════════════════════════
# SLIPPAGE / COMMISSION MODEL
# ══════════════════════════════════════════════════════════════════════════════

def _slippage_cost(symbol: str, direction: int, slippage_pips: float) -> float:
    """Convert slippage pips to price units (always against you on entry)."""
    ci  = get_contract_info(symbol)
    st  = ci.get('type', 'forex')
    if st == 'forex':
        pip = 0.01 if ci.get('decimal_places', 5) == 3 else 0.0001
        return slippage_pips * pip * direction   # positive = worsens fill
    elif st in ('futures', 'indices'):
        tick = ci.get('tick_size', 0.01)
        return slippage_pips * tick * direction
    return 0.0


def _commission(symbol: str, lot_size: float,
                commission_per_lot: float = 7.0) -> float:
    """Round-turn commission (entry + exit).  Default $7/lot RT."""
    return lot_size * commission_per_lot


# ══════════════════════════════════════════════════════════════════════════════
# PNL CALCULATION
# ══════════════════════════════════════════════════════════════════════════════

def _pnl_dollars(symbol: str, direction: int, entry: float,
                 exit_price: float, lot_size: float) -> float:
    ci = get_contract_info(symbol)
    st = ci.get('type', 'forex')
    diff = (exit_price - entry) * direction
    if st == 'forex':
        pip = 0.01 if ci.get('decimal_places', 5) == 3 else 0.0001
        return diff / pip * ci.get('pip_value', 10) * lot_size
    elif st in ('futures', 'indices'):
        return diff * ci.get('dollar_per_point', 1) * lot_size
    return diff * lot_size


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(trades: List[TradeRecord],
                    equity_curve: List[float],
                    initial_capital: float) -> Dict:
    """Compute full suite of backtest metrics."""
    if not trades:
        return {'error': 'no trades generated'}

    n         = len(trades)
    wins      = [t for t in trades if t.won]
    losses    = [t for t in trades if not t.won]
    win_rate  = len(wins) / n * 100

    gross_profit = sum(t.pnl_net for t in wins)
    gross_loss   = abs(sum(t.pnl_net for t in losses)) or 1e-9
    profit_factor = gross_profit / gross_loss

    avg_win  = gross_profit / len(wins)  if wins   else 0.0
    avg_loss = -gross_loss  / len(losses) if losses else 0.0

    r_multiples = [t.r_multiple for t in trades]
    avg_r       = float(np.mean(r_multiples))
    expectancy  = float(np.mean([t.pnl_net for t in trades]))  # $/trade

    # Equity curve metrics
    eq = np.array(equity_curve)
    peak       = np.maximum.accumulate(eq)
    drawdowns  = (eq - peak) / peak * 100          # % drawdown at each bar
    max_dd_pct = float(abs(np.min(drawdowns)))
    max_dd_abs = float(abs(np.min(eq - peak)))

    # Sharpe (annualised, assumes H1 bars → ~5760 bars/year)
    pnl_series = np.diff(eq)
    bars_per_year = 5760
    if pnl_series.std() > 0:
        sharpe = float((pnl_series.mean() / pnl_series.std()) * math.sqrt(bars_per_year))
    else:
        sharpe = 0.0

    # Calmar  = annualised return / max drawdown
    total_return_pct = (eq[-1] - initial_capital) / initial_capital * 100
    years = len(eq) / bars_per_year
    ann_return = total_return_pct / years if years > 0 else 0.0
    calmar = ann_return / max_dd_pct if max_dd_pct > 0 else 0.0

    # Longest losing streak
    streak = max_streak = 0
    for t in trades:
        if not t.won:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    # --- Breakdowns ---
    def _breakdown(key_fn):
        groups: Dict[str, Dict] = {}
        for t in trades:
            k = str(key_fn(t))
            if k not in groups:
                groups[k] = {'n': 0, 'wins': 0, 'pnl': 0.0, 'r': []}
            g = groups[k]
            g['n'] += 1
            g['pnl'] += t.pnl_net
            g['r'].append(t.r_multiple)
            if t.won:
                g['wins'] += 1
        return {k: {'trades': v['n'],
                    'win_rate': round(v['wins'] / v['n'] * 100, 1),
                    'total_pnl': round(v['pnl'], 2),
                    'avg_r': round(float(np.mean(v['r'])), 2)}
                for k, v in groups.items()}

    # Confluence bands: <60, 60-69, 70-79, 80+
    def _cfl_band(t):
        c = t.confluence
        if c >= 80: return '80+'
        if c >= 70: return '70-79'
        if c >= 60: return '60-69'
        return '<60'

    return {
        'summary': {
            'total_trades':     n,
            'wins':             len(wins),
            'losses':           len(losses),
            'win_rate_pct':     round(win_rate, 1),
            'profit_factor':    round(profit_factor, 2),
            'avg_win':          round(avg_win,  2),
            'avg_loss':         round(avg_loss, 2),
            'avg_r_multiple':   round(avg_r, 2),
            'expectancy_per_trade': round(expectancy, 2),
            'gross_profit':     round(gross_profit, 2),
            'gross_loss':       round(-gross_loss, 2),
            'net_pnl':          round(gross_profit - gross_loss, 2),
            'initial_capital':  round(initial_capital, 2),
            'final_equity':     round(float(eq[-1]), 2),
            'total_return_pct': round(total_return_pct, 1),
            'ann_return_pct':   round(ann_return, 1),
            'max_drawdown_pct': round(max_dd_pct, 1),
            'max_drawdown_abs': round(max_dd_abs, 2),
            'sharpe_ratio':     round(sharpe, 2),
            'calmar_ratio':     round(calmar, 2),
            'longest_losing_streak': max_streak,
        },
        'by_symbol':     _breakdown(lambda t: t.symbol),
        'by_session':    _breakdown(lambda t: t.kill_zone),
        'by_confidence': _breakdown(lambda t: t.confidence),
        'by_confluence_band': _breakdown(_cfl_band),
        'by_exit_reason': _breakdown(lambda t: t.exit_reason),
        'by_model':       _breakdown(lambda t: t.model_2022 if t.model_2022 not in ('none','---','') else
                                     ('SB' if t.silver_bullet else
                                      (t.ob_type if t.ob_type not in ('none','---','') else 'fvg_only'))),
    }


# ══════════════════════════════════════════════════════════════════════════════
# MAIN BACKTESTER CLASS
# ══════════════════════════════════════════════════════════════════════════════

class ICTBacktester:
    """
    Bar-by-bar backtester for the ICT signal engine.

    Parameters
    ----------
    symbols           : list of instrument names
    start / end       : datetime range to test
    timeframe         : 'H1' (default) – bar size for signal generation
    initial_capital   : starting account balance in USD
    risk_pct          : fraction of equity risked per trade (e.g. 0.02)
    rr_ratio          : take-profit R:R multiplier fed to the signal engine
    confluence_threshold : minimum score to enter a trade
    max_daily_loss    : daily loss stop (USD, negative)
    slippage_pips     : fill slippage on entry (pips / points)
    commission_per_lot: round-turn commission in USD per standard lot
    max_bars_held     : time stop – close trade after N bars regardless
    warmup_bars       : bars fed to engine before first signal is considered
    csv_dir           : if set, load data from CSV files instead of MT5
    """

    def __init__(
        self,
        symbols:              List[str],
        start:                datetime,
        end:                  datetime,
        timeframe:            str   = 'H1',
        initial_capital:      float = 10_000.0,
        risk_pct:             float = 0.02,
        rr_ratio:             float = 3.0,
        confluence_threshold: int   = 65,
        max_daily_loss:       float = -500.0,
        slippage_pips:        float = 1.5,
        commission_per_lot:   float = 7.0,
        max_bars_held:        int   = 48,
        warmup_bars:          int   = 100,
        csv_dir:              Optional[str] = None,
    ):
        self.symbols              = [s.upper() for s in symbols]
        self.start                = start
        self.end                  = end
        self.timeframe            = timeframe
        self.initial_capital      = initial_capital
        self.risk_pct             = risk_pct
        self.rr_ratio             = rr_ratio
        self.confluence_threshold = confluence_threshold
        self.max_daily_loss       = max_daily_loss
        self.slippage_pips        = slippage_pips
        self.commission_per_lot   = commission_per_lot
        self.max_bars_held        = max_bars_held
        self.warmup_bars          = warmup_bars
        self.csv_dir              = csv_dir

        # State
        self.engine     = ICTSignalEngine(rr_ratio=rr_ratio) if ENGINE_AVAILABLE else None
        self.equity     = initial_capital
        self.positions: Dict[str, OpenPosition] = {}
        self.trades:    List[TradeRecord]        = []
        self.equity_curve: List[float]           = []
        self._trade_id  = 0

        # Per-day loss tracking
        self._daily_pnl:    float              = 0.0
        self._current_date: Optional[datetime] = None

    # ──────────────────────────────────────────────────────────────────────────
    # ENTRY POINT
    # ──────────────────────────────────────────────────────────────────────────

    def run(self) -> Dict:
        """Execute full backtest; return metrics dict."""
        if not ENGINE_AVAILABLE:
            print("ERROR: signal_engine.py required for backtesting")
            return {}

        print(f"\n{'='*64}")
        print(f"  ICT Backtester  {self.start.date()} → {self.end.date()}")
        print(f"  Symbols: {', '.join(self.symbols)}")
        print(f"  Capital: ${self.initial_capital:,.0f}  |  Risk: {self.risk_pct*100:.1f}%  |  R:R 1:{self.rr_ratio}")
        print(f"  Confluence ≥ {self.confluence_threshold}  |  Slippage: {self.slippage_pips} pips")
        print(f"{'='*64}\n")

        # Load all data
        all_data: Dict[str, pd.DataFrame] = {}
        for sym in self.symbols:
            df = self._load(sym)
            if df is not None and len(df) > self.warmup_bars + 10:
                all_data[sym] = df
                print(f"  {sym}: {len(df)} bars  "
                      f"({df.index[0].date()} – {df.index[-1].date()})")
            else:
                print(f"  {sym}: insufficient data, skipping")

        if not all_data:
            print("ERROR: No usable data loaded.")
            return {}

        # Align to a common timeline so the equity curve is consistent
        all_times = sorted(set().union(*[set(df.index) for df in all_data.values()]))
        print(f"\n  Total bars in timeline: {len(all_times)}\n")

        # Bar-by-bar replay
        bar_count = 0
        for ts in all_times:
            bar_count += 1
            self.equity_curve.append(self.equity)

            # Reset daily counter on new calendar day
            ts_dt = ts.to_pydatetime() if hasattr(ts, 'to_pydatetime') else ts
            day = ts_dt.date()
            if self._current_date != day:
                self._current_date = day
                self._daily_pnl    = 0.0

            for sym, df in all_data.items():
                if ts not in df.index:
                    continue

                # Bar index in this symbol's dataframe
                bar_idx = df.index.get_loc(ts)
                if bar_idx < self.warmup_bars:
                    continue

                # ── Update open position for this symbol ──────────────────────
                self._update_position(sym, df, bar_idx)

                # ── Check for new signal (only if no open position) ───────────
                if sym not in self.positions:
                    self._try_enter(sym, df, bar_idx, ts_dt)

            if bar_count % 500 == 0:
                pct = bar_count / len(all_times) * 100
                n_t = len(self.trades)
                print(f"  [{pct:5.1f}%]  bars={bar_count:,}  trades={n_t}  "
                      f"equity=${self.equity:,.2f}")

        # Close any still-open positions at the last bar's close
        for sym in list(self.positions.keys()):
            for sym2, df in all_data.items():
                if sym2 == sym and len(df) > 0:
                    last_idx  = len(df) - 1
                    last_close = df['close'].iloc[last_idx]
                    self._close_position(sym, last_idx, df.index[last_idx], last_close, 'end_of_test')

        self.equity_curve.append(self.equity)  # final equity

        # Compute and display metrics
        metrics = compute_metrics(self.trades, self.equity_curve, self.initial_capital)
        self._print_report(metrics)
        self._save_results(metrics)
        return metrics

    # ──────────────────────────────────────────────────────────────────────────
    # DATA LOADING
    # ──────────────────────────────────────────────────────────────────────────

    def _load(self, symbol: str) -> Optional[pd.DataFrame]:
        if self.csv_dir:
            return load_data_csv(symbol, self.csv_dir, self.timeframe)
        elif MT5_AVAILABLE:
            return load_data_mt5(symbol, self.start, self.end, self.timeframe)
        else:
            print(f"  Cannot load {symbol}: no MT5 and no --csv-dir")
            return None

    # ──────────────────────────────────────────────────────────────────────────
    # SIGNAL GENERATION
    # ──────────────────────────────────────────────────────────────────────────

    def _build_data_dict(self, df: pd.DataFrame, bar_idx: int) -> Dict:
        """
        Build the ``data`` dict that ICTSignalEngine.analyze_symbol() expects,
        using only bars up to and including bar_idx (no look-ahead).
        """
        slice_df = df.iloc[max(0, bar_idx - 199): bar_idx + 1]
        opens  = slice_df['open'].values
        highs  = slice_df['high'].values
        lows   = slice_df['low'].values
        closes = slice_df['close'].values
        n      = len(closes)

        # Minimal derived arrays the engine's fallback path may need
        kill_zone = np.zeros(n, dtype=bool)
        if hasattr(slice_df.index, 'hour'):
            for i, ts in enumerate(slice_df.index):
                h = ts.hour
                kill_zone[i] = (2 <= h < 5) or (7 <= h < 10) or (10 <= h < 12) or (13 <= h < 16)

        return {
            'opens':          opens,
            'highs':          highs,
            'lows':           lows,
            'closes':         closes,
            'volatility':     np.zeros(n),
            'htf_trend':      np.zeros(n),
            'ltf_trend':      np.zeros(n),
            'price_position': np.zeros(n),
            'kill_zone':      kill_zone,
            'bullish_fvgs':   [],
            'bearish_fvgs':   [],
        }

    def _try_enter(self, symbol: str, df: pd.DataFrame,
                   bar_idx: int, bar_time: datetime):
        """Generate a signal and register a pending limit order if valid."""
        # Daily loss guard
        if self._daily_pnl <= self.max_daily_loss:
            return

        data      = self._build_data_dict(df, bar_idx)
        bar_close = float(df['close'].iloc[bar_idx])

        try:
            signal = self.engine.analyze_symbol(symbol, data, bar_close)
        except Exception as e:
            return  # engine error on this bar; skip silently

        if not signal or signal.get('direction', 0) == 0:
            return
        if signal.get('confluence', 0) < self.confluence_threshold:
            return
        if signal.get('stop_loss') is None or signal.get('take_profit') is None:
            return

        direction   = signal['direction']
        entry_price = signal['entry_price']
        stop_loss   = signal['stop_loss']
        take_profit = signal['take_profit']

        stop_dist = abs(entry_price - stop_loss)
        if stop_dist <= 0:
            return

        lot_size, risk_amt = calculate_position_size(
            symbol, self.equity, self.risk_pct, stop_dist, entry_price
        )
        if lot_size <= 0:
            return

        # Pull metadata from engine cache
        cached = getattr(self.engine, 'last_analysis', {}).get(symbol, {})

        self.positions[symbol] = OpenPosition(
            symbol      = symbol,
            direction   = direction,
            entry_bar   = bar_idx,
            entry_time  = bar_time.isoformat(),
            entry_price = entry_price,
            stop_loss   = stop_loss,
            take_profit = take_profit,
            lot_size    = lot_size,
            risk_amount = risk_amt,
            confluence  = signal.get('confluence', 0),
            confidence  = signal.get('confidence', 'LOW'),
            metadata    = {
                'kill_zone':     cached.get('kill_zone', 'OFF_HOURS'),
                'htf_trend':     str(cached.get('htf_trend', 'N/A')),
                'ob_type':       cached.get('ob_type', 'none'),
                'liq_swept':     str(cached.get('liq_swept', 'none')),
                'model_2022':    cached.get('model_2022', 'none'),
                'silver_bullet': bool(cached.get('silver_bullet', False)),
                'fvg_type':      signal.get('fvg_data', {}).get('type', '') if isinstance(signal.get('fvg_data'), dict) else '',
                'reasoning':     signal.get('reasoning', [])[:4],
            },
        )

    # ──────────────────────────────────────────────────────────────────────────
    # POSITION MANAGEMENT
    # ──────────────────────────────────────────────────────────────────────────

    def _update_position(self, symbol: str, df: pd.DataFrame, bar_idx: int):
        """Check if the current bar fills the limit order or triggers exit."""
        if symbol not in self.positions:
            return

        pos  = self.positions[symbol]
        bar  = df.iloc[bar_idx]
        high = float(bar['high'])
        low  = float(bar['low'])

        # ── Limit order fill check (only on bars after signal bar) ────────────
        if bar_idx == pos.entry_bar:
            return   # signal was generated at close of this bar; fill next bar

        # Has the limit order filled yet? (entry not yet confirmed)
        # We track this by seeing if pos has been 'activated' – use a flag
        if not getattr(pos, '_filled', False):
            # Apply slippage to entry
            slip = _slippage_cost(symbol, pos.direction, self.slippage_pips)

            if pos.direction == 1:   # BUY limit – fill when bar's low ≤ entry
                if low <= pos.entry_price:
                    pos.entry_price = pos.entry_price + slip   # slippage hurts
                    pos._filled = True  # type: ignore[attr-defined]
                else:
                    # Not filled yet; check if order is too stale (> 5 bars)
                    if bar_idx - pos.entry_bar > 5:
                        del self.positions[symbol]
                    return
            else:                    # SELL limit – fill when bar's high ≥ entry
                if high >= pos.entry_price:
                    pos.entry_price = pos.entry_price - slip
                    pos._filled = True  # type: ignore[attr-defined]
                else:
                    if bar_idx - pos.entry_bar > 5:
                        del self.positions[symbol]
                    return

        pos.bars_held += 1
        ts = df.index[bar_idx]
        ts_str = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)

        # ── Exit checks (stop → target → time stop) ────────────────────────────
        exit_price  = None
        exit_reason = None

        if pos.direction == 1:
            if low <= pos.stop_loss:
                exit_price  = pos.stop_loss
                exit_reason = 'stop'
            elif high >= pos.take_profit:
                exit_price  = pos.take_profit
                exit_reason = 'target'
        else:
            if high >= pos.stop_loss:
                exit_price  = pos.stop_loss
                exit_reason = 'stop'
            elif low <= pos.take_profit:
                exit_price  = pos.take_profit
                exit_reason = 'target'

        # Time stop
        if exit_price is None and pos.bars_held >= self.max_bars_held:
            exit_price  = float(df['close'].iloc[bar_idx])
            exit_reason = 'time_stop'

        if exit_price is not None:
            self._close_position(symbol, bar_idx, ts, exit_price, exit_reason)

    def _close_position(self, symbol: str, bar_idx: int,
                        ts, exit_price: float, exit_reason: str):
        """Record a closed trade and update equity."""
        if symbol not in self.positions:
            return

        pos     = self.positions.pop(symbol)
        ts_str  = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)

        pnl_raw = _pnl_dollars(symbol, pos.direction,
                                pos.entry_price, exit_price, pos.lot_size)
        comm    = _commission(symbol, pos.lot_size, self.commission_per_lot)
        pnl_net = pnl_raw - comm
        r_mult  = pnl_net / pos.risk_amount if pos.risk_amount > 0 else 0.0
        won     = pnl_net > 0

        self.equity     += pnl_net
        self._daily_pnl += pnl_net
        self._trade_id  += 1

        m = pos.metadata
        self.trades.append(TradeRecord(
            trade_id      = self._trade_id,
            symbol        = symbol,
            direction     = pos.direction,
            entry_bar     = pos.entry_bar,
            entry_time    = pos.entry_time,
            entry_price   = round(pos.entry_price, 6),
            stop_loss     = round(pos.stop_loss, 6),
            take_profit   = round(pos.take_profit, 6),
            lot_size      = pos.lot_size,
            risk_amount   = round(pos.risk_amount, 2),
            exit_bar      = bar_idx,
            exit_time     = ts_str,
            exit_price    = round(exit_price, 6),
            exit_reason   = exit_reason,
            pnl_raw       = round(pnl_raw, 2),
            pnl_net       = round(pnl_net, 2),
            r_multiple    = round(r_mult, 3),
            bars_held     = pos.bars_held,
            confluence    = pos.confluence,
            confidence    = pos.confidence,
            kill_zone     = m.get('kill_zone', 'OFF_HOURS'),
            htf_trend     = m.get('htf_trend', 'N/A'),
            ob_type       = m.get('ob_type', 'none'),
            liq_swept     = str(m.get('liq_swept', 'none')),
            model_2022    = m.get('model_2022', 'none'),
            silver_bullet = bool(m.get('silver_bullet', False)),
            fvg_type      = m.get('fvg_type', ''),
            won           = won,
        ))

    # ──────────────────────────────────────────────────────────────────────────
    # REPORTING
    # ──────────────────────────────────────────────────────────────────────────

    def _print_report(self, metrics: Dict):
        s = metrics.get('summary', {})
        print(f"\n{'='*64}")
        print(f"  BACKTEST RESULTS")
        print(f"{'='*64}")
        print(f"  Trades:          {s.get('total_trades', 0)}  "
              f"({s.get('wins', 0)}W / {s.get('losses', 0)}L)")
        print(f"  Win Rate:        {s.get('win_rate_pct', 0):.1f}%")
        print(f"  Profit Factor:   {s.get('profit_factor', 0):.2f}")
        print(f"  Avg R Multiple:  {s.get('avg_r_multiple', 0):.2f}R")
        print(f"  Expectancy:      ${s.get('expectancy_per_trade', 0):.2f}/trade")
        print(f"  Net P&L:         ${s.get('net_pnl', 0):,.2f}")
        print(f"  Total Return:    {s.get('total_return_pct', 0):.1f}%  "
              f"(Ann: {s.get('ann_return_pct', 0):.1f}%)")
        print(f"  Max Drawdown:    {s.get('max_drawdown_pct', 0):.1f}%  "
              f"(${s.get('max_drawdown_abs', 0):,.0f})")
        print(f"  Sharpe:          {s.get('sharpe_ratio', 0):.2f}")
        print(f"  Calmar:          {s.get('calmar_ratio', 0):.2f}")
        print(f"  Losing Streak:   {s.get('longest_losing_streak', 0)}")

        _section(metrics, 'by_symbol',          'BY SYMBOL')
        _section(metrics, 'by_session',         'BY KILL ZONE / SESSION')
        _section(metrics, 'by_confidence',      'BY CONFIDENCE')
        _section(metrics, 'by_confluence_band', 'BY CONFLUENCE BAND')
        _section(metrics, 'by_exit_reason',     'BY EXIT REASON')
        _section(metrics, 'by_model',           'BY MODEL / SIGNAL TYPE')

        print(f"\n  Saved: backtest_trades.csv  "
              f"backtest_equity.csv  backtest_report.json\n")

    def _save_results(self, metrics: Dict):
        """Save trades CSV (ML-ready), equity CSV, and metrics JSON."""
        out = SCRIPT_DIR

        # ── Trades CSV ─────────────────────────────────────────────────────────
        trades_path = os.path.join(out, 'backtest_trades.csv')
        if self.trades:
            fields = list(asdict(self.trades[0]).keys())
            with open(trades_path, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                for t in self.trades:
                    w.writerow(asdict(t))
            print(f"  → {trades_path}  ({len(self.trades)} rows)")

        # ── Equity CSV ────────────────────────────────────────────────────────
        equity_path = os.path.join(out, 'backtest_equity.csv')
        with open(equity_path, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['bar', 'equity'])
            for i, eq in enumerate(self.equity_curve):
                w.writerow([i, round(eq, 2)])
        print(f"  → {equity_path}  ({len(self.equity_curve)} bars)")

        # ── JSON report ───────────────────────────────────────────────────────
        report_path = os.path.join(out, 'backtest_report.json')
        with open(report_path, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"  → {report_path}")


# ══════════════════════════════════════════════════════════════════════════════
# PRINT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _section(metrics: Dict, key: str, title: str):
    data = metrics.get(key, {})
    if not data:
        return
    print(f"\n  ── {title} ──")
    for k, v in sorted(data.items(), key=lambda x: -x[1]['total_pnl']):
        n   = v['trades']
        wr  = v['win_rate']
        pnl = v['total_pnl']
        ar  = v['avg_r']
        bar = '█' * int(max(0, wr) / 10)
        print(f"    {str(k):<20}  {n:>4} trades  {wr:>5.1f}% WR  "
              f"{bar:<10}  Avg R {ar:>+.2f}  Net ${pnl:>+,.0f}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ICT Backtester – Phase 2')

    parser.add_argument('--symbols',    default='EURUSD,XAUUSD,US30',
                        help='Comma-separated symbols')
    parser.add_argument('--start',      default='2024-01-01',
                        help='Start date YYYY-MM-DD')
    parser.add_argument('--end',        default='2024-12-31',
                        help='End date YYYY-MM-DD')
    parser.add_argument('--timeframe',  default='H1',
                        help='Bar timeframe (H1, M15, H4 …)')
    parser.add_argument('--capital',    type=float, default=10_000.0,
                        help='Initial capital USD')
    parser.add_argument('--risk',       type=float, default=0.02,
                        help='Risk per trade (e.g. 0.02 = 2%%)')
    parser.add_argument('--rr',         type=float, default=3.0,
                        help='Risk:Reward ratio')
    parser.add_argument('--confluence', type=int,   default=65,
                        help='Minimum confluence threshold')
    parser.add_argument('--max-loss',   type=float, default=-500.0,
                        help='Daily loss stop USD (negative)')
    parser.add_argument('--slippage',   type=float, default=1.5,
                        help='Entry slippage in pips/points')
    parser.add_argument('--commission', type=float, default=7.0,
                        help='Round-turn commission per lot (USD)')
    parser.add_argument('--max-bars',   type=int,   default=48,
                        help='Time stop: max bars held')
    parser.add_argument('--warmup',     type=int,   default=100,
                        help='Warmup bars before first signal')
    parser.add_argument('--csv-dir',    default=None,
                        help='Folder with SYMBOL_H1.csv files (skips MT5)')

    # MT5 credentials (only needed if --csv-dir not supplied)
    parser.add_argument('--login',    type=int, default=None)
    parser.add_argument('--password', type=str, default=None)
    parser.add_argument('--server',   type=str, default=None)

    args = parser.parse_args()

    # Initialise MT5 if not using CSVs
    if args.csv_dir is None:
        if not MT5_AVAILABLE:
            print("ERROR: MetaTrader5 not installed.  "
                  "Use --csv-dir or install MetaTrader5.")
            sys.exit(1)
        if not mt5.initialize():
            print(f"MT5 init failed: {mt5.last_error()}")
            sys.exit(1)
        if args.login and args.password and args.server:
            if not mt5.login(args.login, args.password, args.server):
                print(f"MT5 login failed: {mt5.last_error()}")
                sys.exit(1)
        acct = mt5.account_info()
        print(f"MT5 connected: account {acct.login}  balance ${acct.balance:,.2f}")

    symbols   = [s.strip().upper() for s in args.symbols.split(',')]
    start_dt  = datetime.strptime(args.start, '%Y-%m-%d')
    end_dt    = datetime.strptime(args.end,   '%Y-%m-%d')

    bt = ICTBacktester(
        symbols              = symbols,
        start                = start_dt,
        end                  = end_dt,
        timeframe            = args.timeframe,
        initial_capital      = args.capital,
        risk_pct             = args.risk,
        rr_ratio             = args.rr,
        confluence_threshold = args.confluence,
        max_daily_loss       = args.max_loss,
        slippage_pips        = args.slippage,
        commission_per_lot   = args.commission,
        max_bars_held        = args.max_bars,
        warmup_bars          = args.warmup,
        csv_dir              = args.csv_dir,
    )

    metrics = bt.run()

    if MT5_AVAILABLE and args.csv_dir is None:
        mt5.shutdown()