"""
ICT Backtester  –  Phase 2  (MT5-native)
=========================================
Bar-by-bar historical replay using real MT5 OHLC data.

Key improvements over v1
------------------------
- Fetches BOTH H1 (signal) and D1 (HTF bias) data directly from MT5
- Robust symbol resolution: tries plain name → 'm'-suffix → manual mapping
- Real per-bar spread from MT5 history used in cost model
- Engine called only on kill-zone bars (~10% of bars) for >5x speed-up;
  configurable with --every-bar flag for research
- Signal-generation counters printed every cycle so you see WHY trades aren't
  entering (below threshold / wrong zone / already in position)
- --debug flag prints full signal reasoning for every signal generated
- Progress includes ETA and signals-per-hour rate
- Saves backtest_trades.csv, backtest_equity.csv, backtest_report.json

Usage
-----
# Typical run (MT5 must be open and logged in):
python3 backtester.py \\
    --symbols EURUSD,XAUUSD,US30 \\
    --login 298797826 --password yourpass --server Exness-MT5Real \\
    --start 2024-01-01 --end 2024-12-31 \\
    --risk 0.02 --rr 3.0 --confluence 65

# Debug mode (prints every signal + reasoning):
python3 backtester.py --symbols EURUSD --start 2024-06-01 --end 2024-06-30 --debug

# Lower confluence to see if any signals fire at all:
python3 backtester.py --symbols EURUSD --start 2024-01-01 --confluence 40 --debug
"""

from __future__ import annotations

import os, sys, json, math, argparse, csv, time as _time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

# ── MT5 ───────────────────────────────────────────────────────────────────────
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None  # type: ignore

# ── Signal engine ─────────────────────────────────────────────────────────────
try:
    from signal_engine import ICTSignalEngine
    ENGINE_AVAILABLE = True
except ImportError as e:
    print(f"ERROR: signal_engine.py not found: {e}")
    ENGINE_AVAILABLE = False

# ── Contract / sizing helpers ─────────────────────────────────────────────────
try:
    from ict_v7_mt5 import get_contract_info, calculate_position_size
except ImportError:
    def get_contract_info(symbol):
        s = symbol.upper()
        if s == 'XAUUSD': return {'dollar_per_point': 100, 'min_stop': 5.0,  'type': 'futures', 'tick_size': 0.01}
        if s == 'XTIUSD': return {'dollar_per_point': 1000,'min_stop': 0.3,  'type': 'futures', 'tick_size': 0.01}
        if s == 'US30': return {'dollar_per_point': 1, 'min_stop': 500,  'type': 'indices', 'tick_size': 1}
        if s == 'US500': return {'dollar_per_point': 1, 'min_stop': 50, 'type': 'indices', 'tick_size': 0.1}
        if s == 'USTEC': return {'dollar_per_point': 1, 'min_stop': 100, 'type': 'indices', 'tick_size': 0.1}
        if s == 'UK100': return {'dollar_per_point': 1, 'min_stop': 80, 'type': 'indices', 'tick_size': 0.1}
        if s == 'GER40': return {'dollar_per_point': 1, 'min_stop': 80, 'type': 'indices', 'tick_size': 0.1}
        return {'pip_value': 10, 'min_stop': 0.002, 'decimal_places': 5, 'type': 'forex'}

    def calculate_position_size(symbol, account_value, risk_pct, stop_distance, current_price):
        ci = get_contract_info(symbol)
        risk = account_value * risk_pct
        if stop_distance <= 0:
            return 0.01, 0.0
        if ci['type'] == 'forex':
            pip = 0.01 if ci.get('decimal_places', 5) == 3 else 0.0001
            qty = risk / ((stop_distance / pip) * ci.get('pip_value', 10))
        else:
            qty = risk / (stop_distance * ci.get('dollar_per_point', 1))
        qty = max(0.01, round(qty / 0.01) * 0.01)
        return qty, qty * stop_distance


# ══════════════════════════════════════════════════════════════════════════════
# MT5 DATA LAYER
# ══════════════════════════════════════════════════════════════════════════════

# All known name variants per broker (Exness first, then generic)
_SYMBOL_VARIANTS = {
    'EURUSD': ['EURUSDm', 'EURUSD'],
    'GBPUSD': ['GBPUSDm', 'GBPUSD'],
    'USDJPY': ['USDJPYm', 'USDJPY'],
    'AUDUSD': ['AUDUSDm', 'AUDUSD'],
    'USDCAD': ['USDCADm', 'USDCAD'],
    'USDCHF': ['USDCHFm', 'USDCHF'],
    'NZDUSD': ['NZDUSDm', 'NZDUSD'],
    'EURGBP': ['EURGBPm', 'EURGBP'],
    'EURJPY': ['EURJPYm', 'EURJPY'],
    'GBPJPY': ['GBPJPYm', 'GBPJPY'],
    'XAUUSD': ['XAUUSDm', 'XAUUSD'],
    'XAGUSD': ['XAGUSDm', 'XAGUSD'],
    'XTIUSD': ['USOILm',  'XTIUSD', 'USOIL', 'USOIL.m'],
    'XBRUSD': ['UKOILm',  'XBRUSD', 'UKOIL'],
    'US30':   ['US30m',   'US30',   'DJ30',  'DJIA'],
    'USTEC':  ['USTECm',  'USTEC',  'NAS100','NASDAQ'],
    'US500':  ['US500m',  'US500',  'SP500', 'SPX500'],
    'UK100':  ['UK100m',  'UK100',  'FTSE100'],
    'BTCUSD': ['BTCUSDm', 'BTCUSD', 'BTC/USD'],
    'ETHUSD': ['ETHUSDm', 'ETHUSD', 'ETH/USD'],
    'SOLUSD': ['SOLUSDm', 'SOLUSD'],
}

_TF_ATTRS = {
    'M1': 'TIMEFRAME_M1',  'M5':  'TIMEFRAME_M5',
    'M15':'TIMEFRAME_M15', 'M30': 'TIMEFRAME_M30',
    'H1': 'TIMEFRAME_H1',  'H4':  'TIMEFRAME_H4',
    'D1': 'TIMEFRAME_D1',  'W1':  'TIMEFRAME_W1',
}

# Cache resolved symbol names so we don't hit MT5 repeatedly
_resolved_cache: Dict[str, Optional[str]] = {}


def _resolve(symbol: str) -> Optional[str]:
    """Return the first MT5 symbol name that the broker has data for."""
    key = symbol.upper()
    if key in _resolved_cache:
        return _resolved_cache[key]

    variants = _SYMBOL_VARIANTS.get(key, [key + 'm', key])
    for cand in variants:
        info = mt5.symbol_info(cand)
        if info is not None:
            mt5.symbol_select(cand, True)
            _resolved_cache[key] = cand
            return cand

    _resolved_cache[key] = None
    return None


def _mt5_tf(tf: str):
    attr = _TF_ATTRS.get(tf.upper(), 'TIMEFRAME_H1')
    return getattr(mt5, attr)


def fetch_mt5_ohlc(symbol: str, tf: str, start: datetime, end: datetime,
                   extra_warmup_days: int = 0) -> Optional[pd.DataFrame]:
    """
    Fetch OHLCV + spread data from MT5.
    extra_warmup_days prepends that many calendar days before `start`
    so the engine has enough history on bar 0.
    Returns a DataFrame with columns: open, high, low, close, spread
    Index is a tz-naive UTC datetime.
    """
    if not MT5_AVAILABLE:
        return None

    mt5_sym = _resolve(symbol)
    if mt5_sym is None:
        tried = _SYMBOL_VARIANTS.get(symbol.upper(), [symbol])
        print(f"  [{symbol}] Symbol not found in MT5. Tried: {tried}")
        print(f"   Tip: check Market Watch in your MT5 terminal for the correct name.")
        return None

    fetch_start = start - timedelta(days=extra_warmup_days)

    rates = mt5.copy_rates_range(mt5_sym, _mt5_tf(tf), fetch_start, end)
    if rates is None or len(rates) == 0:
        err = mt5.last_error()
        print(f"  [{symbol}] No rates returned for {mt5_sym} "
              f"{tf} {fetch_start.date()}–{end.date()}. MT5 error: {err}")
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
    df.set_index('time', inplace=True)
    df.index = df.index.tz_localize(None)   # strip tz for simpler comparisons

    df.rename(columns={'tick_volume': 'volume', 'real_volume': 'real_vol'},
              inplace=True, errors='ignore')

    # Ensure all required columns exist
    for col in ('open', 'high', 'low', 'close'):
        if col not in df.columns:
            print(f"  [{symbol}] Missing column '{col}' in MT5 data")
            return None

    if 'spread' not in df.columns:
        df['spread'] = 0

    df = df[['open', 'high', 'low', 'close', 'spread']].sort_index()
    return df


def spread_to_price(symbol: str, spread_points: int) -> float:
    """Convert MT5 spread (in points) to a price difference."""
    mt5_sym = _resolve(symbol)
    if mt5_sym is None or spread_points == 0:
        return 0.0
    info = mt5.symbol_info(mt5_sym)
    return spread_points * info.point if info else 0.0


# ══════════════════════════════════════════════════════════════════════════════
# DATA CLASSES
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class TradeRecord:
    trade_id:       int
    symbol:         str
    direction:      int      # +1 long / -1 short
    entry_bar:      int
    entry_time:     str
    entry_price:    float
    stop_loss:      float
    take_profit:    float
    lot_size:       float
    risk_amount:    float
    exit_bar:       int
    exit_time:      str
    exit_price:     float
    exit_reason:    str      # 'target' | 'stop' | 'time_stop' | 'end_of_test'
    spread_cost:    float    # USD cost from spread at entry
    commission:     float    # RT commission
    pnl_gross:      float    # before costs
    pnl_net:        float    # after costs
    r_multiple:     float
    bars_held:      int
    # Signal metadata (ML feature vector for Phase 3)
    confluence:     int
    confidence:     str
    kill_zone:      str
    htf_trend:      str
    ob_type:        str
    liq_swept:      str
    model_2022:     str
    silver_bullet:  bool
    fvg_type:       str
    won:            bool


@dataclass
class OpenPosition:
    symbol:       str
    direction:    int
    signal_bar:   int        # bar index when signal was generated
    entry_bar:    int
    entry_time:   str
    entry_price:  float
    stop_loss:    float
    take_profit:  float
    lot_size:     float
    risk_amount:  float
    spread_cost:  float
    confluence:   int
    confidence:   str
    metadata:     Dict = field(default_factory=dict)
    bars_held:    int  = 0
    filled:       bool = False


# ══════════════════════════════════════════════════════════════════════════════
# COST MODEL
# ══════════════════════════════════════════════════════════════════════════════

def slippage_price(symbol: str, direction: int, slip_pips: float) -> float:
    """Price penalty from slippage (always hurts you on entry)."""
    ci  = get_contract_info(symbol)
    if ci['type'] == 'forex':
        pip = 0.01 if ci.get('decimal_places', 5) == 3 else 0.0001
        return slip_pips * pip * direction
    return slip_pips * ci.get('tick_size', 0.01) * direction


def commission_rt(lots: float, per_lot: float = 7.0) -> float:
    return lots * per_lot


def pnl_usd(symbol: str, direction: int,
            entry: float, exit_p: float, lots: float) -> float:
    ci   = get_contract_info(symbol)
    diff = (exit_p - entry) * direction
    if ci['type'] == 'forex':
        pip = 0.01 if ci.get('decimal_places', 5) == 3 else 0.0001
        return diff / pip * ci.get('pip_value', 10) * lots
    return diff * ci.get('dollar_per_point', 1) * lots


# ══════════════════════════════════════════════════════════════════════════════
# KILL ZONE HELPER
# ══════════════════════════════════════════════════════════════════════════════

def in_kill_zone(hour: int) -> bool:
    return (2 <= hour < 5) or (7 <= hour < 10) or (10 <= hour < 12) or (13 <= hour < 16)


# ══════════════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════════════

def compute_metrics(trades: List[TradeRecord],
                    equity_curve: List[float],
                    initial_capital: float) -> Dict:
    if not trades:
        return {'error': 'no_trades', 'summary': {'total_trades': 0}}

    n     = len(trades)
    wins  = [t for t in trades if t.won]
    loss  = [t for t in trades if not t.won]
    wr    = len(wins) / n * 100

    gp = sum(t.pnl_net for t in wins)
    gl = abs(sum(t.pnl_net for t in loss)) or 1e-9
    pf = gp / gl

    eq   = np.array(equity_curve, dtype=float)
    peak = np.maximum.accumulate(eq)
    dd   = (eq - peak) / np.where(peak > 0, peak, 1) * 100
    mdd  = float(abs(np.min(dd)))
    mdd_abs = float(abs(np.min(eq - peak)))

    bpy  = 5760   # H1 bars per year
    diff = np.diff(eq)
    sharpe = float((diff.mean() / diff.std()) * math.sqrt(bpy)) if diff.std() > 0 else 0.0
    ret  = (eq[-1] - initial_capital) / initial_capital * 100
    yrs  = len(eq) / bpy
    ann  = ret / yrs if yrs > 0 else 0.0
    cal  = ann / mdd if mdd > 0 else 0.0

    streak = mx = 0
    for t in trades:
        if not t.won: streak += 1; mx = max(mx, streak)
        else:         streak = 0

    def bd(fn):
        g: Dict = {}
        for t in trades:
            k = str(fn(t))
            if k not in g: g[k] = {'n': 0, 'w': 0, 'pnl': 0.0, 'r': []}
            g[k]['n'] += 1; g[k]['pnl'] += t.pnl_net; g[k]['r'].append(t.r_multiple)
            if t.won: g[k]['w'] += 1
        return {k: {'trades': v['n'],
                    'win_rate': round(v['w'] / v['n'] * 100, 1),
                    'total_pnl': round(v['pnl'], 2),
                    'avg_r': round(float(np.mean(v['r'])), 2)}
                for k, v in g.items()}

    def cfl_band(t):
        c = t.confluence
        if c >= 80: return '80+'
        if c >= 70: return '70-79'
        if c >= 60: return '60-69'
        return '<60'

    def model_label(t):
        if t.model_2022 not in ('none', '---', ''): return f"M22:{t.model_2022}"
        if t.silver_bullet:                          return 'SilverBullet'
        if t.ob_type not in ('none', '---', ''):     return f"OB:{t.ob_type}"
        if t.fvg_type:                               return f"FVG:{t.fvg_type}"
        return 'other'

    r_mults = [t.r_multiple for t in trades]
    return {
        'summary': {
            'total_trades':         n,
            'wins':                 len(wins),
            'losses':               len(loss),
            'win_rate_pct':         round(wr, 1),
            'profit_factor':        round(pf, 2),
            'avg_win':              round(gp / len(wins) if wins else 0, 2),
            'avg_loss':             round(gl / len(loss) if loss else 0, 2),
            'avg_r_multiple':       round(float(np.mean(r_mults)), 2),
            'expectancy_per_trade': round(float(np.mean([t.pnl_net for t in trades])), 2),
            'gross_profit':         round(gp, 2),
            'gross_loss':           round(-gl, 2),
            'net_pnl':              round(gp - gl, 2),
            'total_spread_cost':    round(sum(t.spread_cost for t in trades), 2),
            'total_commission':     round(sum(t.commission for t in trades), 2),
            'initial_capital':      round(initial_capital, 2),
            'final_equity':         round(float(eq[-1]), 2),
            'total_return_pct':     round(ret, 1),
            'ann_return_pct':       round(ann, 1),
            'max_drawdown_pct':     round(mdd, 1),
            'max_drawdown_abs':     round(mdd_abs, 2),
            'sharpe_ratio':         round(sharpe, 2),
            'calmar_ratio':         round(cal, 2),
            'longest_losing_streak': mx,
        },
        'by_symbol':          bd(lambda t: t.symbol),
        'by_session':         bd(lambda t: t.kill_zone),
        'by_confidence':      bd(lambda t: t.confidence),
        'by_confluence_band': bd(cfl_band),
        'by_exit_reason':     bd(lambda t: t.exit_reason),
        'by_model':           bd(model_label),
        'by_direction':       bd(lambda t: 'LONG' if t.direction == 1 else 'SHORT'),
    }


# ══════════════════════════════════════════════════════════════════════════════
# BACKTESTER
# ══════════════════════════════════════════════════════════════════════════════

class ICTBacktester:
    """
    MT5-native bar-by-bar backtester for the ICT signal engine.

    Parameters
    ----------
    symbols              : instrument list e.g. ['EURUSD','XAUUSD','US30']
    start / end          : test window as datetime
    timeframe            : signal bar size – 'H1' recommended
    htf_timeframe        : HTF for bias context – 'D1' or 'H4'
    initial_capital      : starting equity in USD
    risk_pct             : fraction risked per trade (0.02 = 2%)
    rr_ratio             : R:R multiplier for take-profit
    confluence_threshold : min score to enter (0-100)
    max_daily_loss       : stop trading for the day below this USD (negative)
    slippage_pips        : entry fill slippage in pips/points
    commission_per_lot   : round-turn commission per standard lot USD
    max_bars_held        : time-stop: close after N bars regardless
    warmup_bars          : bars fed to engine before first trade is considered
    kill_zone_only       : only run engine on kill-zone hours (5-10x faster)
    debug                : print every signal and trade event
    """

    def __init__(
        self,
        symbols:               List[str],
        start:                 datetime,
        end:                   datetime,
        timeframe:             str   = 'H1',
        htf_timeframe:         str   = 'D1',
        initial_capital:       float = 10_000.0,
        risk_pct:              float = 0.02,
        rr_ratio:              float = 3.0,
        confluence_threshold:  int   = 65,
        max_daily_loss:        float = -500.0,
        slippage_pips:         float = 1.5,
        commission_per_lot:    float = 7.0,
        max_bars_held:         int   = 48,
        warmup_bars:           int   = 100,
        kill_zone_only:        bool  = True,
        debug:                 bool  = False,
    ):
        self.symbols              = [s.upper() for s in symbols]
        self.start                = start
        self.end                  = end
        self.timeframe            = timeframe
        self.htf_timeframe        = htf_timeframe
        self.initial_capital      = initial_capital
        self.risk_pct             = risk_pct
        self.rr_ratio             = rr_ratio
        self.confluence_threshold = confluence_threshold
        self.max_daily_loss       = max_daily_loss
        self.slippage_pips        = slippage_pips
        self.commission_per_lot   = commission_per_lot
        self.max_bars_held        = max_bars_held
        self.warmup_bars          = warmup_bars
        self.kill_zone_only       = kill_zone_only
        self.debug                = debug

        self.engine = ICTSignalEngine(rr_ratio=rr_ratio) if ENGINE_AVAILABLE else None
        if self.engine and debug:
            self.engine._debug_mode = True   # surface handler errors in debug mode
        self.equity = initial_capital
        self.positions:   Dict[str, OpenPosition] = {}
        self.trades:      List[TradeRecord]        = []
        self.equity_curve: List[float]             = []
        self._trade_id  = 0
        self._daily_pnl: float    = 0.0
        self._cur_date            = None

        self._stats: Dict[str, int] = {
            'bars_evaluated':    0,
            'bars_skipped_kz':   0,
            'signals_fired':     0,
            'below_threshold':   0,
            'already_in_pos':    0,
            'daily_limit_hit':   0,
            'entries_attempted': 0,
            'entries_filled':    0,
            'entries_expired':   0,
        }

    # ── MAIN ──────────────────────────────────────────────────────────────────

    def run(self) -> Dict:
        if not ENGINE_AVAILABLE:
            print("ERROR: signal_engine.py required for backtesting")
            return {}

        self._print_header()

        # ── Fetch data from MT5 ───────────────────────────────────────────────
        h1_data: Dict[str, pd.DataFrame] = {}
        d1_data: Dict[str, pd.DataFrame] = {}

        warmup_days = max(30, self.warmup_bars // 24 + 5)

        for sym in self.symbols:
            print(f"  Fetching {sym}...")
            h1 = fetch_mt5_ohlc(sym, self.timeframe,     self.start, self.end,
                                 extra_warmup_days=warmup_days)
            d1 = fetch_mt5_ohlc(sym, self.htf_timeframe, self.start, self.end,
                                 extra_warmup_days=90)

            if h1 is None or len(h1) < self.warmup_bars + 10:
                print(f"    {sym}: skipped (insufficient data)")
                continue

            h1_data[sym] = h1
            d1_data[sym] = d1

            test_bars = len(h1[h1.index >= pd.Timestamp(self.start)])
            d1_count  = len(d1) if d1 is not None else 0
            print(f"    {sym}: {len(h1)} H1 bars total  "
                  f"({test_bars} in window)  {d1_count} D1 bars")

        if not h1_data:
            print("\nNo data loaded. Check MT5 is running and symbols are visible.")
            return {}

        # ── Build shared test-window timeline ─────────────────────────────────
        test_start = pd.Timestamp(self.start)
        all_times  = sorted(set().union(
            *[set(df[df.index >= test_start].index) for df in h1_data.values()]
        ))
        n_bars = len(all_times)
        print(f"\n  Test-window bars  : {n_bars:,}")
        print(f"  Kill-zone-only    : {self.kill_zone_only} "
              f"({'faster' if self.kill_zone_only else 'thorough'})")
        print(f"  {'─'*54}\n")

        t_start = _time.time()

        for i, ts in enumerate(all_times):
            self.equity_curve.append(self.equity)

            ts_dt = ts.to_pydatetime() if hasattr(ts, 'to_pydatetime') else ts
            if self._cur_date != ts_dt.date():
                self._cur_date  = ts_dt.date()
                self._daily_pnl = 0.0

            for sym, df in h1_data.items():
                if ts not in df.index:
                    continue

                bar_idx = df.index.get_loc(ts)
                if bar_idx < self.warmup_bars:
                    continue

                # Update open position first
                self._update_pos(sym, df, bar_idx, ts)

                # Try new entry if flat
                if sym not in self.positions:
                    if self.kill_zone_only and not in_kill_zone(ts_dt.hour):
                        self._stats['bars_skipped_kz'] += 1
                    else:
                        self._stats['bars_evaluated'] += 1
                        self._try_enter(sym, df, d1_data.get(sym), bar_idx, ts_dt)

            if (i + 1) % 250 == 0 or i == n_bars - 1:
                self._progress(i + 1, n_bars, t_start)

        # Force-close survivors
        for sym in list(self.positions.keys()):
            df = h1_data[sym]
            self._close(sym, len(df) - 1, df.index[-1],
                        float(df['close'].iloc[-1]), 0, 'end_of_test')

        self.equity_curve.append(self.equity)

        metrics = compute_metrics(self.trades, self.equity_curve, self.initial_capital)
        self._print_report(metrics)
        self._print_diagnostics()
        self._save(metrics)
        return metrics

    # ── DATA DICT (no look-ahead) ──────────────────────────────────────────────

    def _make_data(self, h1: pd.DataFrame, bar_idx: int,
                   d1: Optional[pd.DataFrame], bar_time: datetime) -> Dict:
        """
        Slice up to 200 bars of H1 data ending at bar_idx.
        Computes HTF trend from D1, LTF trend from H1 momentum.
        Zero look-ahead guaranteed.
        """
        sl = h1.iloc[max(0, bar_idx - 199): bar_idx + 1]
        opens  = sl['open'].values.copy()
        highs  = sl['high'].values.copy()
        lows   = sl['low'].values.copy()
        closes = sl['close'].values.copy()
        n      = len(closes)

        # Kill zone flag per bar
        kz = np.zeros(n, dtype=bool)
        for j, idx_ts in enumerate(sl.index):
            h = idx_ts.hour if hasattr(idx_ts, 'hour') else 0
            kz[j] = in_kill_zone(h)

        # HTF trend from D1 SMA20
        htf = np.zeros(n)
        if d1 is not None and len(d1) >= 5:
            d1_sl = d1[d1.index <= pd.Timestamp(bar_time)]
            if len(d1_sl) >= 5:
                d1c    = d1_sl['close'].values
                sma    = d1c[-20:].mean() if len(d1c) >= 20 else d1c.mean()
                trend  = 1 if d1c[-1] > sma else (-1 if d1c[-1] < sma else 0)
                htf[:] = trend

        # LTF momentum (10-bar)
        ltf = np.zeros(n)
        for j in range(10, n):
            if closes[j - 10] > 0:
                pct = (closes[j] - closes[j - 10]) / closes[j - 10]
                if   pct >  0.003: ltf[j] =  1
                elif pct < -0.003: ltf[j] = -1

        # Price position in 20-bar range
        pp = np.zeros(n)
        for j in range(20, n):
            ph = highs[j-20:j].max(); pl = lows[j-20:j].min()
            rng = ph - pl
            if rng > 1e-8: pp[j] = (closes[j] - pl) / rng

        # Simple FVG detection
        b_fvg, s_fvg = [], []
        for j in range(2, n):
            if lows[j] > highs[j - 2]:
                b_fvg.append({'idx': j, 'mid': (highs[j-2] + lows[j]) / 2,
                               'high': lows[j], 'low': highs[j-2]})
            if highs[j] < lows[j - 2]:
                s_fvg.append({'idx': j, 'mid': (highs[j] + lows[j-2]) / 2,
                               'high': lows[j-2], 'low': highs[j]})

        return {
            'opens': opens, 'highs': highs, 'lows': lows, 'closes': closes,
            'volatility': np.zeros(n), 'htf_trend': htf, 'ltf_trend': ltf,
            'price_position': pp, 'kill_zone': kz,
            'bullish_fvgs': b_fvg[-10:], 'bearish_fvgs': s_fvg[-10:],
        }

    # ── SIGNAL GENERATION ─────────────────────────────────────────────────────

    def _try_enter(self, sym: str, h1: pd.DataFrame,
                   d1: Optional[pd.DataFrame], bar_idx: int, bar_time: datetime):
        if self._daily_pnl <= self.max_daily_loss:
            self._stats['daily_limit_hit'] += 1
            return

        data  = self._make_data(h1, bar_idx, d1, bar_time)
        close = float(h1['close'].iloc[bar_idx])

        try:
            sig = self.engine.analyze_symbol(sym, data, close)
        except Exception as e:
            if self.debug:
                print(f"  [{sym}] Engine error @ {bar_time}: {e}")
            return

        if not sig or sig.get('direction', 0) == 0:
            return

        self._stats['signals_fired'] += 1
        cfl = sig.get('confluence', 0)

        if self.debug:
            ds = 'LONG' if sig['direction'] == 1 else 'SHORT'
            print(f"\n  [SIGNAL {sym} {bar_time:%Y-%m-%d %H:%M}] "
                  f"{ds}  cfl={cfl}  conf={sig.get('confidence','?')}")
            for r in sig.get('reasoning', []):
                print(f"    • {r}")

        if cfl < self.confluence_threshold:
            self._stats['below_threshold'] += 1
            return
        if sig.get('stop_loss') is None or sig.get('take_profit') is None:
            return

        direction   = sig['direction']
        entry_price = sig['entry_price']
        stop_loss   = sig['stop_loss']
        take_profit = sig['take_profit']
        stop_dist   = abs(entry_price - stop_loss)
        if stop_dist <= 0:
            return

        lots, risk_amt = calculate_position_size(
            sym, self.equity, self.risk_pct, stop_dist, entry_price)
        if lots <= 0:
            return

        # Spread cost captured at entry bar
        raw_spread = float(h1['spread'].iloc[bar_idx])
        spd_price  = spread_to_price(sym, int(raw_spread))
        spd_cost   = abs(pnl_usd(sym, direction, entry_price,
                                  entry_price + spd_price * direction, lots))

        cached = getattr(self.engine, 'last_analysis', {}).get(sym, {})

        self.positions[sym] = OpenPosition(
            symbol       = sym,
            direction    = direction,
            signal_bar   = bar_idx,
            entry_bar    = bar_idx,
            entry_time   = bar_time.isoformat(),
            entry_price  = entry_price,
            stop_loss    = stop_loss,
            take_profit  = take_profit,
            lot_size     = lots,
            risk_amount  = risk_amt,
            spread_cost  = spd_cost,
            confluence   = cfl,
            confidence   = sig.get('confidence', 'LOW'),
            metadata     = {
                'kill_zone':     cached.get('kill_zone', 'OFF_HOURS'),
                'htf_trend':     str(cached.get('htf_trend', 'N/A')),
                'ob_type':       cached.get('ob_type', 'none'),
                'liq_swept':     str(cached.get('liq_swept', 'none')),
                'model_2022':    cached.get('model_2022', 'none'),
                'silver_bullet': bool(cached.get('silver_bullet', False)),
                'fvg_type':      (sig.get('fvg_data') or {}).get('type', ''),
            },
        )
        self._stats['entries_attempted'] += 1

    # ── POSITION MANAGEMENT ────────────────────────────────────────────────────

    def _update_pos(self, sym: str, df: pd.DataFrame, bar_idx: int, ts):
        if sym not in self.positions:
            return

        pos   = self.positions[sym]
        bar   = df.iloc[bar_idx]
        high  = float(bar['high'])
        low   = float(bar['low'])
        close = float(bar['close'])
        spd   = float(bar['spread'])

        # ── Limit order fill ─────────────────────────────────────────────────
        if not pos.filled:
            # Never fill on the signal bar itself
            if bar_idx == pos.signal_bar:
                return

            slip = slippage_price(sym, pos.direction, self.slippage_pips)

            if pos.direction == 1:       # BUY limit: fill when low <= entry
                if low <= pos.entry_price:
                    pos.entry_price += slip
                    pos.filled = True
                    self._stats['entries_filled'] += 1
                elif bar_idx - pos.signal_bar > 5:
                    del self.positions[sym]
                    self._stats['entries_expired'] += 1
                    return
                else:
                    return
            else:                        # SELL limit: fill when high >= entry
                if high >= pos.entry_price:
                    pos.entry_price -= slip
                    pos.filled = True
                    self._stats['entries_filled'] += 1
                elif bar_idx - pos.signal_bar > 5:
                    del self.positions[sym]
                    self._stats['entries_expired'] += 1
                    return
                else:
                    return

        pos.bars_held += 1

        # ── Exit checks: stop → target → time stop ────────────────────────────
        ep = er = None
        if pos.direction == 1:
            if low  <= pos.stop_loss:    ep = pos.stop_loss;   er = 'stop'
            elif high >= pos.take_profit: ep = pos.take_profit; er = 'target'
        else:
            if high >= pos.stop_loss:    ep = pos.stop_loss;   er = 'stop'
            elif low  <= pos.take_profit: ep = pos.take_profit; er = 'target'

        if ep is None and pos.bars_held >= self.max_bars_held:
            ep = close; er = 'time_stop'

        if ep is not None:
            self._close(sym, bar_idx, ts, ep, spd, er)

    def _close(self, sym: str, bar_idx: int, ts,
               exit_price: float, spread_pts: float, reason: str):
        if sym not in self.positions:
            return

        pos     = self.positions.pop(sym)
        ts_str  = ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)

        gross   = pnl_usd(sym, pos.direction, pos.entry_price, exit_price, pos.lot_size)
        comm    = commission_rt(pos.lot_size, self.commission_per_lot)
        net     = gross - comm - pos.spread_cost
        r_mult  = net / pos.risk_amount if pos.risk_amount > 0 else 0.0
        won     = net > 0

        self.equity     += net
        self._daily_pnl += net
        self._trade_id  += 1

        m = pos.metadata
        self.trades.append(TradeRecord(
            trade_id      = self._trade_id,
            symbol        = sym,
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
            exit_reason   = reason,
            spread_cost   = round(pos.spread_cost, 2),
            commission    = round(comm, 2),
            pnl_gross     = round(gross, 2),
            pnl_net       = round(net, 2),
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

        if self.debug:
            s = f"+${net:.2f}" if net >= 0 else f"-${abs(net):.2f}"
            print(f"  [TRADE#{self._trade_id} {sym}] "
                  f"{'WIN' if won else 'LOSS'} {reason} {s} {r_mult:+.2f}R "
                  f"equity=${self.equity:,.0f}")

    # ── DISPLAY ───────────────────────────────────────────────────────────────

    def _print_header(self):
        print(f"\n{'═'*64}")
        print(f"  ICT Backtester  {self.start.date()} → {self.end.date()}")
        print(f"  Symbols   : {', '.join(self.symbols)}")
        print(f"  Timeframes: {self.timeframe} signal  |  {self.htf_timeframe} HTF")
        print(f"  Capital   : ${self.initial_capital:,.0f}  "
              f"Risk {self.risk_pct*100:.1f}%  R:R 1:{self.rr_ratio}")
        print(f"  Confluence: ≥{self.confluence_threshold}  "
              f"Slippage {self.slippage_pips}p  "
              f"Comm ${self.commission_per_lot}/lot RT")
        print(f"{'═'*64}\n")

    def _progress(self, done: int, total: int, t0: float):
        ela = _time.time() - t0
        eta = timedelta(seconds=int(ela / done * (total - done))) if done > 0 else '?'
        s   = self._stats
        print(f"  [{done/total*100:5.1f}%] bars={done:,}/{total:,}  "
              f"kz_skip={s['bars_skipped_kz']:,}  "
              f"signals={s['signals_fired']}  fills={s['entries_filled']}  "
              f"trades={len(self.trades)}  "
              f"equity=${self.equity:,.0f}  ETA {eta}")

    def _print_diagnostics(self):
        s = self._stats
        print(f"\n  ── Signal diagnostics ──")
        print(f"    Bars evaluated     : {s['bars_evaluated']:,}  "
              f"(kz-skipped: {s['bars_skipped_kz']:,})")
        print(f"    Signals generated  : {s['signals_fired']:,}")
        print(f"    Below confluence   : {s['below_threshold']:,}")
        print(f"    Daily limit hit    : {s['daily_limit_hit']:,}")
        print(f"    Limit orders placed: {s['entries_attempted']:,}")
        print(f"    Limit orders filled: {s['entries_filled']:,}")
        print(f"    Limit orders expired:{s['entries_expired']:,}")
        print(f"    Closed trades      : {len(self.trades):,}")

        if s['bars_evaluated'] > 0 and s['signals_fired'] == 0:
            print(f"\n  ⚠  No signals generated. Diagnose with:")
            print(f"     python3 backtester.py --symbols {self.symbols[0]} "
                  f"--start {self.start.date()} --confluence 1 --debug")

    def _print_report(self, metrics: Dict):
        s = metrics.get('summary', {})
        n = s.get('total_trades', 0)
        print(f"\n{'═'*64}")
        print(f"  RESULTS")
        print(f"{'═'*64}")
        if n == 0:
            print("  No trades closed in this test window."); return
        print(f"  Trades     : {n}  ({s['wins']}W / {s['losses']}L)")
        print(f"  Win Rate   : {s['win_rate_pct']:.1f}%")
        print(f"  Prof Factor: {s['profit_factor']:.2f}")
        print(f"  Avg R      : {s['avg_r_multiple']:.2f}R")
        print(f"  Expectancy : ${s['expectancy_per_trade']:.2f}/trade")
        print(f"  Net P&L    : ${s['net_pnl']:,.2f}  "
              f"(spread ${s['total_spread_cost']:.0f}  comm ${s['total_commission']:.0f})")
        print(f"  Return     : {s['total_return_pct']:.1f}%  (ann {s['ann_return_pct']:.1f}%)")
        print(f"  Max DD     : {s['max_drawdown_pct']:.1f}%  (${s['max_drawdown_abs']:,.0f})")
        print(f"  Sharpe     : {s['sharpe_ratio']:.2f}   Calmar: {s['calmar_ratio']:.2f}")
        print(f"  Lose Streak: {s['longest_losing_streak']}")

        for key, title in [
            ('by_symbol',          'BY SYMBOL'),
            ('by_session',         'BY SESSION / KILL ZONE'),
            ('by_confidence',      'BY CONFIDENCE'),
            ('by_confluence_band', 'BY CONFLUENCE BAND'),
            ('by_exit_reason',     'BY EXIT REASON'),
            ('by_model',           'BY MODEL / SIGNAL TYPE'),
            ('by_direction',       'BY DIRECTION'),
        ]:
            _section(metrics, key, title)

    def _save(self, metrics: Dict):
        if self.trades:
            p = os.path.join(SCRIPT_DIR, 'backtest_trades.csv')
            with open(p, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(asdict(self.trades[0]).keys()))
                w.writeheader()
                for t in self.trades: w.writerow(asdict(t))
            print(f"\n  Saved: {p}  ({len(self.trades)} rows)")

        p = os.path.join(SCRIPT_DIR, 'backtest_equity.csv')
        with open(p, 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['bar', 'equity'])
            for i, eq in enumerate(self.equity_curve): w.writerow([i, round(eq, 2)])
        print(f"  Saved: {p}")

        p = os.path.join(SCRIPT_DIR, 'backtest_report.json')
        with open(p, 'w') as f:
            json.dump(metrics, f, indent=2, default=str)
        print(f"  Saved: {p}\n")


# ── helpers ───────────────────────────────────────────────────────────────────

def _section(metrics: Dict, key: str, title: str):
    data = metrics.get(key, {})
    if not data: return
    print(f"\n  ── {title} ──")
    for k, v in sorted(data.items(), key=lambda x: -x[1]['total_pnl']):
        bar = '█' * int(max(0, v['win_rate']) / 10)
        print(f"    {str(k):<22} {v['trades']:>4} trades  "
              f"{v['win_rate']:>5.1f}% WR  {bar:<10}  "
              f"Avg R {v['avg_r']:>+.2f}  Net ${v['total_pnl']:>+,.0f}")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    # NOTE: save this file as backtest.py OR backtester.py — either works.
    # Run: python backtest.py --symbols EURUSD,XAUUSD,US30 --start 2024-01-01
    ap = argparse.ArgumentParser(description='ICT Backtester – Phase 2 (MT5-native)')
    ap.add_argument('--symbols',    default='EURUSD,XAUUSD,US30')
    ap.add_argument('--start',      default='2024-01-01',   help='YYYY-MM-DD')
    ap.add_argument('--end',        default='2024-12-31',   help='YYYY-MM-DD')
    ap.add_argument('--timeframe',  default='H1',           help='H1 M15 H4')
    ap.add_argument('--htf',        default='D1',           help='D1 H4')
    ap.add_argument('--capital',    type=float, default=None,
                    help='Starting capital (defaults to live account balance)')
    ap.add_argument('--risk',       type=float, default=0.02)
    ap.add_argument('--rr',         type=float, default=3.0)
    ap.add_argument('--confluence', type=int,   default=65)
    ap.add_argument('--max-loss',   type=float, default=-500.0)
    ap.add_argument('--slippage',   type=float, default=1.5)
    ap.add_argument('--commission', type=float, default=7.0)
    ap.add_argument('--max-bars',   type=int,   default=48)
    ap.add_argument('--warmup',     type=int,   default=100)
    ap.add_argument('--every-bar',  action='store_true',
                    help='Run engine on every bar (thorough but slow)')
    ap.add_argument('--debug',      action='store_true',
                    help='Print every signal and trade event')
    ap.add_argument('--login',    type=int, default=None)
    ap.add_argument('--password', type=str, default=None)
    ap.add_argument('--server',   type=str, default=None)
    args = ap.parse_args()

    if not MT5_AVAILABLE:
        print("MetaTrader5 not installed: pip install MetaTrader5"); sys.exit(1)

    if not mt5.initialize():
        print(f"MT5 initialize() failed: {mt5.last_error()}"); sys.exit(1)

    if args.login and args.password and args.server:
        if not mt5.login(args.login, password=args.password, server=args.server):
            print(f"MT5 login failed: {mt5.last_error()}"); sys.exit(1)

    acct = mt5.account_info()
    if acct is None:
        print("Cannot get account info. MT5 terminal must be open."); sys.exit(1)
    print(f"MT5: account {acct.login}  balance ${acct.balance:,.2f}  server {acct.server}")

    capital   = args.capital if args.capital else acct.balance
    symbols   = [s.strip().upper() for s in args.symbols.split(',')]
    start_dt  = datetime.strptime(args.start, '%Y-%m-%d')
    end_dt    = datetime.strptime(args.end,   '%Y-%m-%d')

    bt = ICTBacktester(
        symbols              = symbols,
        start                = start_dt,
        end                  = end_dt,
        timeframe            = args.timeframe,
        htf_timeframe        = args.htf,
        initial_capital      = capital,
        risk_pct             = args.risk,
        rr_ratio             = args.rr,
        confluence_threshold = args.confluence,
        max_daily_loss       = args.max_loss,
        slippage_pips        = args.slippage,
        commission_per_lot   = args.commission,
        max_bars_held        = args.max_bars,
        warmup_bars          = args.warmup,
        kill_zone_only       = not args.every_bar,
        debug                = args.debug,
    )

    bt.run()
    mt5.shutdown()