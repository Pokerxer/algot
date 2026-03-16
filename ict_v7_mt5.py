"""
ICT V7 Trading Bot - MT5 Version  (R:R & Position-Sizing FIXED)
===============================================================
Fixes applied vs original:
  1. V7SignalGenerator now accepts and stores rr_ratio so _combine_signals
     uses the actual configured ratio instead of hard-coded 2.0.
  2. _enter_trade no longer silently recalculates target_price; it relies on
     the TP already stored in the signal, keeping logs and MT5 orders in sync.
  3. calculate_position_size (futures / metals) formula corrected:
       WRONG: qty = risk / (stop_distance / tick_size * multiplier)
       RIGHT: qty = risk / (stop_distance * multiplier)
     The old code was off by exactly 1/tick_size (×100 error for XAUUSD).
  4. actual_risk for futures now uses the same corrected formula.
  5. Indices position sizing uses a configurable point_value_per_lot instead
     of the hardcoded 1, with sensible per-symbol defaults.
  6. Forex pip validation bounds aligned: min 10 pips, max 150 pips
     (was 15 / 100, which conflicted with the 80-pip soft ceiling).

Usage:
    python3 ict_v7_mt5_fixed.py --symbols "BTCUSD,ETHUSD,XAUUSD,XTIUSD" \
                                 --login 12345 --password "yourpass"
"""

import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import sys
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from fvg_handler import FVGHandler, FairValueGap, FVGStatus
from gap_handler import GapHandler, Gap, GapType, GapDirection

try:
    from mtf_coordinator import MTFCoordinator, TimeframePurpose, TimeframeRelation
    from market_structure_handler import (
        MarketStructureHandler, MarketStructureAnalysis,
        StructureBreakType, TrendState, PriceZone
    )
    MTF_AVAILABLE = True
except ImportError as e:
    MTF_AVAILABLE = False
    print(f"WARNING: MTF modules not available: {e}")

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("WARNING: MetaTrader5 not installed. Install with: pip install MetaTrader5")

try:
    import telegram_notify as tn
    if tn and hasattr(tn, 'init_bot'):
        try:
            tn.init_bot()
            print("Telegram bot initialized")
        except Exception as e:
            print(f"Telegram bot init failed: {e}")
except (ImportError, NameError, Exception) as e:
    tn = None
    print(f"WARNING: telegram_notify not available: {e}")


FOREX_SYMBOLS = {
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD', 'AUDUSD', 'USDCHF',
    'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY', 'XAUUSD', 'XAGUSD'
}

MT5_SYMBOLS = {
    'BTCUSD': 'BTCUSDm', 'ETHUSD': 'ETHUSDm', 'SOLUSD': 'SOLUSDm',
    'XAUUSD': 'XAUUSDm', 'XAGUSD': 'XAGUSDm',
    'XTIUSD': 'USOILm',  'XBRUSD': 'UKOILm', 'XNGUSD': 'XNGUSDm',
    'EURUSD': 'EURUSDm', 'GBPUSD': 'GBPUSDm', 'USDJPY': 'USDJPYm',
    'USDCAD': 'USDCADm', 'AUDUSD': 'AUDUSDm', 'USDCHF': 'USDCHFm',
    'NZDUSD': 'NZDUSDm',
    'EURGBP': 'EURGBPm', 'EURJPY': 'EURJPYm', 'GBPJPY': 'GBPJPYm',
    'EURAUD': 'EURAUDm', 'EURCAD': 'EURCADm', 'GBPAUD': 'GBPAUDm',
    'AUDJPY': 'AUDJPYm', 'CADJPY': 'CADJPYm', 'CHFJPY': 'CHFJPYm',
    'US30':  'US30m',  'USTEC': 'USTECm', 'US500': 'US500m',
    'UK100': 'UK100m', 'AUS200': 'AUS200m',
}

EXNESS_SYMBOLS = MT5_SYMBOLS.copy()


def get_mt5_symbol(symbol: str) -> str:
    s = symbol.upper()
    if s in MT5_SYMBOLS:
        return MT5_SYMBOLS[s]
    if s in EXNESS_SYMBOLS:
        return EXNESS_SYMBOLS[s]
    return s + 'm'


CONVERT_THRESHOLD_GBP = 500


def is_trading_paused() -> bool:
    if tn and hasattr(tn, 'is_trading_paused'):
        return tn.is_trading_paused()
    return False


def is_valid_trading_session(is_forex: bool = False) -> Tuple[bool, str]:
    return True, "24/7"


def get_quote_currency(symbol: str) -> str:
    if len(symbol) >= 6:
        return symbol[-3:]
    return ''


def init_mt5(login: int = None, password: str = None, server: str = None) -> bool:
    if not MT5_AVAILABLE:
        print("ERROR: MetaTrader5 package not available")
        return False

    if not mt5.initialize():
        print(f"MT5 initialize() failed: {mt5.last_error()}")
        return False

    if login and password and server:
        authorized = mt5.login(login=login, password=password, server=server)
        if not authorized:
            print(f"MT5 login failed: {mt5.last_error()}")
            return False
        print(f"Connected to MT5 account {login}")

    account_info = mt5.account_info()
    if account_info is None:
        print("Failed to get account info")
        return False

    print(f"MT5 Account: {account_info.login} | Balance: ${account_info.balance:.2f}")
    return True


def get_mt5_symbol_info(symbol: str) -> Optional[Dict]:
    if not MT5_AVAILABLE:
        return None
    symbol = symbol.upper()
    info = mt5.symbol_info(symbol)
    if info is None:
        return None
    return {
        'symbol': info.name,
        'bid': info.bid,
        'ask': info.ask,
        'last': info.last,
        'point': info.point,
        'digits': info.digits,
        'trade_contract_size': info.trade_contract_size,
        'volume_min': info.volume_min,
        'volume_max': info.volume_max,
        'volume_step': info.volume_step,
        'trade_tick_size': info.trade_tick_size,
        'trade_tick_value': info.trade_tick_value,
    }


def get_contract_info(symbol: str) -> Dict:
    """Return comprehensive contract information for position sizing.

    For futures/metals the key field is ``dollar_per_point`` which represents
    the USD profit/loss for a 1-unit price move per 1 standard lot.  This is
    the *only* multiplier needed for position sizing – no tick_size arithmetic
    should be applied on top of it.
    """
    symbol = symbol.upper()

    # ── Futures / Metals / Oil ────────────────────────────────────────────────
    # dollar_per_point  = P&L in USD when price moves 1.0 unit, 1 standard lot
    #   XAUUSD : 1 lot = 100 oz  →  $100 per $1 price move
    #   XAGUSD : 1 lot = 5000 oz →  $50  per $0.001 move  = $50 000 per $1
    #            (Exness mini contract is typically 50 oz → $50/unit/lot,
    #             adjust dollar_per_point to your broker's contract spec)
    #   XTIUSD : 1 lot = 1000 bbl→  $1000 per $1 price move
    futures_info = {
        'XAUUSD': {'dollar_per_point': 100,   'min_stop': 5.0,   'tick_size': 0.01,  'type': 'futures'},
        'XAGUSD': {'dollar_per_point': 50,    'min_stop': 0.10,  'tick_size': 0.001, 'type': 'futures'},
        'XTIUSD': {'dollar_per_point': 1000,  'min_stop': 0.30,  'tick_size': 0.01,  'type': 'futures'},
        'XBRUSD': {'dollar_per_point': 1000,  'min_stop': 0.30,  'tick_size': 0.01,  'type': 'futures'},
        'XNGUSD': {'dollar_per_point': 10000, 'min_stop': 0.02,  'tick_size': 0.001, 'type': 'futures'},
    }

    # ── Forex ─────────────────────────────────────────────────────────────────
    # pip_value_per_lot = USD P&L per 1 pip move, 1 standard lot (100 000 units)
    forex_info = {
        'EURUSD': {'pip_value': 10,   'min_stop': 0.0020, 'decimal_places': 5, 'type': 'forex'},
        'GBPUSD': {'pip_value': 10,   'min_stop': 0.0020, 'decimal_places': 5, 'type': 'forex'},
        'USDJPY': {'pip_value': 9.1,  'min_stop': 0.20,   'decimal_places': 3, 'type': 'forex'},
        'AUDUSD': {'pip_value': 10,   'min_stop': 0.0020, 'decimal_places': 5, 'type': 'forex'},
        'USDCAD': {'pip_value': 7.5,  'min_stop': 0.0020, 'decimal_places': 5, 'type': 'forex'},
        'USDCHF': {'pip_value': 10.8, 'min_stop': 0.0020, 'decimal_places': 5, 'type': 'forex'},
        'NZDUSD': {'pip_value': 10,   'min_stop': 0.0020, 'decimal_places': 5, 'type': 'forex'},
        'GBPJPY': {'pip_value': 9.1,  'min_stop': 0.20,   'decimal_places': 3, 'type': 'forex'},
        'EURJPY': {'pip_value': 9.1,  'min_stop': 0.20,   'decimal_places': 3, 'type': 'forex'},
        'EURGBP': {'pip_value': 10,   'min_stop': 0.0020, 'decimal_places': 5, 'type': 'forex'},
    }

    # ── Crypto ────────────────────────────────────────────────────────────────
    # 1 lot = 1 coin; dollar_per_point = current price (dynamic), but for sizing
    # purposes we treat it as: qty = risk / (stop_pct * account_value)
    crypto_info = {
        'BTCUSD': {'min_stop_pct': 0.015, 'tick_size': 0.01, 'type': 'crypto'},
        'ETHUSD': {'min_stop_pct': 0.015, 'tick_size': 0.01, 'type': 'crypto'},
        'SOLUSD': {'min_stop_pct': 0.020, 'tick_size': 0.01, 'type': 'crypto'},
        'LTCUSD': {'min_stop_pct': 0.020, 'tick_size': 0.01, 'type': 'crypto'},
    }

    # ── Indices ───────────────────────────────────────────────────────────────
    # dollar_per_point = USD P&L per 1 index point per 1 standard lot
    # (broker-dependent – values below are typical for Exness standard lots)
    indices_info = {
        'US30':   {'dollar_per_point': 1,   'min_stop': 20,  'tick_size': 1,   'type': 'indices'},
        'US500':  {'dollar_per_point': 1,   'min_stop': 3,   'tick_size': 0.1, 'type': 'indices'},
        'USTEC':  {'dollar_per_point': 1,   'min_stop': 10,  'tick_size': 0.1, 'type': 'indices'},
        'GER40':  {'dollar_per_point': 1,   'min_stop': 15,  'tick_size': 0.1, 'type': 'indices'},
        'UK100':  {'dollar_per_point': 1,   'min_stop': 8,   'tick_size': 0.1, 'type': 'indices'},
        'FRA40':  {'dollar_per_point': 1,   'min_stop': 8,   'tick_size': 0.1, 'type': 'indices'},
        'JPN225': {'dollar_per_point': 1,   'min_stop': 80,  'tick_size': 1,   'type': 'indices'},
        'AUS200': {'dollar_per_point': 1,   'min_stop': 8,   'tick_size': 0.1, 'type': 'indices'},
    }

    if symbol in futures_info:
        return futures_info[symbol]
    if symbol in forex_info:
        return forex_info[symbol]
    if symbol in crypto_info:
        return crypto_info[symbol]
    if symbol in indices_info:
        return indices_info[symbol]

    # Default to generic forex
    return {'pip_value': 10, 'min_stop': 0.0020, 'decimal_places': 5, 'type': 'forex'}


def fetch_mt5_rates(symbol: str, timeframe: str = "H1", num_bars: int = 500) -> Optional[pd.DataFrame]:
    if not MT5_AVAILABLE:
        return None

    timeframe_map = {
        "M1":  mt5.TIMEFRAME_M1,  "M5":  mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
        "H1":  mt5.TIMEFRAME_H1,  "H4":  mt5.TIMEFRAME_H4,
        "D1":  mt5.TIMEFRAME_D1,  "W1":  mt5.TIMEFRAME_W1,
    }
    mt5_timeframe = timeframe_map.get(timeframe, mt5.TIMEFRAME_H1)

    if not mt5.symbol_select(symbol, True):
        print(f"Could not select symbol: {symbol}")
        return None

    rates = mt5.copy_rates_from_pos(symbol, mt5_timeframe, 0, num_bars)
    if rates is None or len(rates) == 0:
        print(f"No rates returned for {symbol}")
        return None

    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.columns = ['open', 'low', 'high', 'close', 'tick_volume', 'spread', 'real_volume']
    return df


def prepare_data_mt5(symbol: str, lookback: int = 200) -> Optional[Dict]:
    symbol = symbol.upper()
    mt5_symbol = get_mt5_symbol(symbol)

    df = fetch_mt5_rates(mt5_symbol, "H1", num_bars=lookback + 50)
    if df is None or len(df) < 50:
        print(f"MT5 failed for {mt5_symbol}")
        return None

    print(f"Using MT5 data for {mt5_symbol}: {len(df)} rows")

    highs  = df['high'].values
    lows   = df['low'].values
    closes = df['close'].values
    opens  = df['open'].values

    bullish_fvgs = []
    bearish_fvgs = []
    for i in range(3, len(df)):
        if lows[i] > highs[i-2]:
            bullish_fvgs.append({'idx': i, 'mid': (highs[i-2] + lows[i]) / 2, 'high': lows[i]})
        if highs[i] < lows[i-2]:
            bearish_fvgs.append({'idx': i, 'mid': (highs[i] + lows[i-2]) / 2, 'low': highs[i]})

    df_daily = fetch_mt5_rates(mt5_symbol, "D1", num_bars=60)
    if df_daily is None or len(df_daily) < 5:
        htf_trend = np.zeros(len(df))
    else:
        daily_highs = df_daily['high'].values
        daily_lows  = df_daily['low'].values
        htf = []
        for i in range(1, len(df_daily)):
            if (daily_highs[i] > np.max(daily_highs[max(0,i-5):i]) and
                    daily_lows[i] > np.min(daily_lows[max(0,i-5):i])):
                htf.append(1)
            elif (daily_highs[i] < np.max(daily_highs[max(0,i-5):i]) and
                      daily_lows[i] < np.min(daily_lows[max(0,i-5):i])):
                htf.append(-1)
            else:
                htf.append(0)

        htf_trend = np.zeros(len(df))
        for i in range(len(df)):
            bar_time = df.index[i]
            for j in range(len(df_daily) - 1, -1, -1):
                if df_daily.index[j] <= bar_time:
                    htf_trend[i] = htf[j] if j < len(htf) else 0
                    break

    trend = np.zeros(len(df))
    for i in range(20, len(df)):
        momentum   = closes[i] - closes[i-10]
        pct_change = momentum / closes[i-10] if closes[i-10] > 0 else 0
        ema_fast   = np.mean(closes[max(0,i-5):i+1])
        ema_slow   = np.mean(closes[max(0,i-13):i+1])
        ema_bullish = ema_fast > ema_slow
        ema_bearish = ema_fast < ema_slow
        if pct_change > 0.005 or (pct_change > 0.001 and ema_bullish):
            trend[i] = 1
        elif pct_change < -0.005 or (pct_change < -0.001 and ema_bearish):
            trend[i] = -1

    price_position = np.zeros(len(df))
    for i in range(20, len(df)):
        ph  = np.max(highs[i-20:i])
        pl  = np.min(lows[i-20:i])
        rng = ph - pl
        if rng < 0.001:
            rng = 0.001
        price_position[i] = (closes[i] - pl) / rng

    hours     = df.index.hour.values
    kill_zone = np.zeros(len(df), dtype=bool)
    for i in range(len(hours)):
        h = hours[i]
        kill_zone[i] = (1 <= h < 5) or (7 <= h < 12) or (13 <= h < 16)

    volatility = np.zeros(len(df))
    for i in range(14, len(df)):
        trs = []
        for j in range(max(0, i-14), i+1):
            tr = (max(highs[j] - lows[j],
                      abs(highs[j] - closes[j-1]) if j > 0 else 0,
                      abs(lows[j]  - closes[j-1]) if j > 0 else 0))
            trs.append(tr)
        volatility[i] = np.mean(trs) if trs else 0

    return {
        'opens':        opens,
        'highs':        highs,
        'lows':         lows,
        'closes':       closes,
        'volatility':   volatility,
        'htf_trend':    htf_trend,
        'ltf_trend':    trend,
        'price_position': price_position,
        'kill_zone':    kill_zone,
        'bullish_fvgs': bullish_fvgs,
        'bearish_fvgs': bearish_fvgs,
    }


def get_signal(data: Dict, idx: int) -> Dict:
    return {'confluence': 0, 'direction': 0}


# ═══════════════════════════════════════════════════════════════════════════════
# POSITION SIZING  (fully corrected)
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_position_size(
    symbol: str,
    account_value: float,
    risk_pct: float,
    stop_distance: float,
    current_price: float,
) -> Tuple[float, float]:
    """
    Calculate lot size so that a loss equal to ``stop_distance`` costs exactly
    ``risk_pct`` of ``account_value``.

    Returns
    -------
    qty         : float  – lot size (rounded to 0.01)
    actual_risk : float  – exact dollar risk at that lot size
    """
    contract_info = get_contract_info(symbol)
    symbol_type   = contract_info['type']
    risk_amount   = account_value * risk_pct

    if stop_distance <= 0:
        return 0.0, 0.0

    qty = 0.0

    # ── Forex ─────────────────────────────────────────────────────────────────
    if symbol_type == 'forex':
        decimal_places = contract_info.get('decimal_places', 5)
        pip_size       = 0.01 if decimal_places == 3 else 0.0001
        pip_value      = contract_info.get('pip_value', 10)  # $/pip/lot

        stop_pips = stop_distance / pip_size
        if stop_pips <= 0:
            return 0.0, 0.0

        # qty (lots) = $ risk / (pips × $/pip/lot)
        qty         = risk_amount / (stop_pips * pip_value)
        actual_risk = qty * stop_pips * pip_value

    # ── Futures / Metals / Oil ────────────────────────────────────────────────
    elif symbol_type == 'futures':
        # dollar_per_point = USD profit when price moves 1 unit, 1 lot
        # e.g. XAUUSD: $100/unit/lot  →  for a $5 stop and $600 risk:
        #   qty = 600 / (5 × 100) = 1.2 lots  ← CORRECT
        # The old code incorrectly divided stop_distance by tick_size first,
        # which multiplied the denominator by 100 and gave 0.012 lots instead.
        dollar_per_point = contract_info.get('dollar_per_point', 100)

        # qty = risk / (stop_distance × $/unit/lot)
        qty         = risk_amount / (stop_distance * dollar_per_point)
        actual_risk = qty * stop_distance * dollar_per_point

    # ── Indices ───────────────────────────────────────────────────────────────
    elif symbol_type == 'indices':
        # dollar_per_point is the USD value of a 1-point move per standard lot.
        # Typical Exness: US30 = $1/point/lot, USTEC = $1/point/lot, etc.
        # Adjust get_contract_info() if your broker differs.
        dollar_per_point = contract_info.get('dollar_per_point', 1)

        # qty = risk / (stop_points × $/point/lot)
        qty         = risk_amount / (stop_distance * dollar_per_point)
        actual_risk = qty * stop_distance * dollar_per_point

    # ── Crypto ────────────────────────────────────────────────────────────────
    elif symbol_type == 'crypto':
        # For crypto 1 lot = 1 coin; P&L = price_move × qty × 1
        if current_price <= 0:
            return 0.0, 0.0
        qty         = risk_amount / stop_distance
        actual_risk = qty * stop_distance

    else:
        qty         = risk_amount / stop_distance
        actual_risk = qty * stop_distance

    # Enforce MT5 minimum / rounding
    qty = max(qty, 0.01)
    qty = round(qty / 0.01) * 0.01

    return qty, actual_risk


# ═══════════════════════════════════════════════════════════════════════════════
# MT5 ORDER HELPERS  (unchanged from original)
# ═══════════════════════════════════════════════════════════════════════════════

def place_mt5_order(
    symbol: str, order_type: str, volume: float,
    price: float, stop_loss: float = None,
    take_profit: float = None, magic: int = 123456,
) -> Optional[Dict]:
    if not MT5_AVAILABLE:
        return None

    symbol     = symbol.upper()
    mt5_symbol = get_mt5_symbol(symbol)

    symbol_info = mt5.symbol_info(mt5_symbol)
    if symbol_info is None:
        print(f"Symbol {mt5_symbol} not found")
        return None

    if not symbol_info.visible:
        if not mt5.symbol_select(mt5_symbol, True):
            print(f"Could not select symbol: {mt5_symbol}")
            return None

    point = symbol_info.point

    requested_price = price
    use_limit       = True

    if order_type.upper() == "BUY":
        order_type_enum = mt5.ORDER_TYPE_BUY_LIMIT
        market_price    = symbol_info.ask
    else:
        order_type_enum = mt5.ORDER_TYPE_SELL_LIMIT
        market_price    = symbol_info.bid

    if order_type.upper() == "BUY" and requested_price <= market_price:
        order_type_enum = mt5.ORDER_TYPE_BUY
        use_limit       = False
    elif order_type.upper() == "SELL" and requested_price >= market_price:
        order_type_enum = mt5.ORDER_TYPE_SELL
        use_limit       = False

    if use_limit:
        request = {
            "action":      mt5.TRADE_ACTION_PENDING,
            "symbol":      mt5_symbol,
            "volume":      volume,
            "type":        order_type_enum,
            "price":       requested_price,
            "magic":       magic,
            "comment":     "ICT V7",
            "type_time":   mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
    else:
        request = {
            "action":      mt5.TRADE_ACTION_DEAL,
            "symbol":      mt5_symbol,
            "volume":      volume,
            "type":        order_type_enum,
            "price":       market_price,
            "deviation":   20,
            "magic":       magic,
            "comment":     "ICT V7",
            "type_time":   mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

    if stop_loss:
        stop_loss = round(stop_loss / point) * point
        if order_type.upper() == "BUY":
            if stop_loss >= symbol_info.ask:
                stop_loss = symbol_info.ask - point
        else:
            if stop_loss <= symbol_info.bid:
                stop_loss = symbol_info.bid + point
        request["sl"] = stop_loss

    if take_profit:
        take_profit = round(take_profit / point) * point
        if order_type.upper() == "BUY":
            if take_profit <= symbol_info.ask:
                take_profit = symbol_info.ask + point
        else:
            if take_profit >= symbol_info.bid:
                take_profit = symbol_info.bid - point
        request["tp"] = take_profit

    volume = round(volume / symbol_info.volume_step) * symbol_info.volume_step
    volume = max(symbol_info.volume_min, min(volume, symbol_info.volume_max))
    request["volume"] = volume

    print(f"  Order: {order_type} {mt5_symbol} vol={volume} @ {price} "
          f"SL={request.get('sl')} TP={request.get('tp')}")

    result = mt5.order_send(request)
    if result is None:
        print(f"Order send failed: {mt5.last_error()}")
        return None
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order failed: {result.comment}")
        return None

    return {
        'order_id': result.order,
        'volume':   result.volume,
        'price':    result.price,
        'retcode':  result.retcode,
    }


def get_mt5_positions() -> List[Dict]:
    if not MT5_AVAILABLE:
        return []
    positions = mt5.positions_get()
    if positions is None:
        return []
    result = []
    for pos in positions:
        result.append({
            'ticket':        pos.ticket,
            'symbol':        pos.symbol,
            'volume':        pos.volume,
            'price_open':    pos.price_open,
            'price_current': pos.price_current,
            'profit':        pos.profit,
            'type':          pos.type,
            'sl':            pos.sl,
            'tp':            pos.tp,
            'time':          pos.time,
        })
    return result


def close_mt5_position(ticket: int, volume: float = None) -> bool:
    if not MT5_AVAILABLE:
        return False
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        return False
    pos        = positions[0]
    order_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
    close_volume = volume if volume else pos.volume
    request = {
        "action":      mt5.TRADE_ACTION_DEAL,
        "symbol":      pos.symbol,
        "volume":      close_volume,
        "type":        order_type,
        "position":    ticket,
        "price":       mt5.symbol_info(pos.symbol).bid if pos.type == 0 else mt5.symbol_info(pos.symbol).ask,
        "deviation":   20,
        "magic":       123456,
        "comment":     "ICT V7 Close",
        "type_time":   mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL GENERATOR  (rr_ratio now injected correctly)
# ═══════════════════════════════════════════════════════════════════════════════

class V7SignalGenerator:
    """Enhanced signal generator combining V5 ICT with FVG, Gap, MTF and Market Structure."""

    # ── FIX 1: accept rr_ratio in __init__ so _combine_signals uses the real value
    def __init__(self, rr_ratio: float = 2.0):
        self.rr_ratio = rr_ratio          # ← stored; was never set before

        self.fvg_handler = FVGHandler(
            sensitivity=0.0001,
            min_gap_size=0.0,
            track_body_respect=True,
            detect_volume_imbalances=True,
            detect_suspension_blocks=True,
        )
        self.gap_handler = GapHandler(
            large_gap_pips_forex=40.0,
            large_gap_points_indices=50.0,
            keep_gaps_days=3,
        )

        if MTF_AVAILABLE:
            self.mtf_coordinator = MTFCoordinator()
            self.ms_handler      = MarketStructureHandler(swing_lookback=5, min_displacement_pct=0.1)
        else:
            self.mtf_coordinator = None
            self.ms_handler      = None

        self.last_analysis = {}

    def analyze_symbol(self, symbol: str, data: Dict, current_price: float) -> Dict:
        idx       = len(data['closes']) - 1
        v5_signal = get_signal(data, idx)

        df = pd.DataFrame({
            'open':  data['opens'],
            'high':  data['highs'],
            'low':   data['lows'],
            'close': data['closes'],
        })

        fvgs         = self.fvg_handler.detect_all_fvgs(df)
        fvg_analysis = self.fvg_handler.analyze_fvgs(df)
        gap_analysis = self.gap_handler.analyze(df, current_price)

        ms_analysis = None
        if self.ms_handler is not None:
            try:
                ms_analysis = self.ms_handler.analyze(df)
            except Exception as e:
                print(f"MS analysis error: {e}")

        combined_signal = self._combine_signals(
            symbol, v5_signal, fvg_analysis, gap_analysis,
            current_price, data, idx, ms_analysis,
        )

        self.last_analysis[symbol] = {
            'timestamp':      datetime.now().isoformat(),
            'v5_confluence':  v5_signal['confluence'] if v5_signal else 0,
            'fvg_count':      len(fvgs),
            'active_fvgs':    len(fvg_analysis.active_fvgs),
            'high_prob_fvgs': len(fvg_analysis.high_prob_fvgs),
            'gap_levels':     len(gap_analysis.all_levels),
        }
        return combined_signal

    def _combine_signals(
        self, symbol: str, v5_signal: Optional[Dict],
        fvg_analysis, gap_analysis, current_price: float,
        data: Dict, idx: int, ms_analysis=None,
    ) -> Dict:

        signal = {
            'symbol':      symbol,
            'direction':   0,
            'confluence':  0,
            'entry_price': current_price,
            'stop_loss':   None,
            'take_profit': None,
            'confidence':  'LOW',
            'reasoning':   [],
            'v5_signal':   v5_signal,
            'fvg_data':    None,
            'gap_data':    None,
        }

        htf    = data['htf_trend'][idx]
        ltf    = data['ltf_trend'][idx]
        kz     = data['kill_zone'][idx]
        pp     = data['price_position'][idx]
        closes = data['closes'][idx]

        near_bull_fvg = next(
            (f for f in reversed(data['bullish_fvgs']) if f['idx'] < idx and f['mid'] < closes), None)
        near_bear_fvg = next(
            (f for f in reversed(data['bearish_fvgs']) if f['idx'] < idx and f['mid'] > closes), None)

        base_confluence = 0
        if kz:
            base_confluence += 15
            signal['reasoning'].append("Kill Zone: +15")

        if htf == 1 and ltf >= 0:
            base_confluence += 25
            signal['direction'] = 1
            signal['reasoning'].append("HTF+LTF Bullish: +25")
        elif htf == -1 and ltf <= 0:
            base_confluence += 25
            signal['direction'] = -1
            signal['reasoning'].append("HTF+LTF Bearish: +25")
        elif htf == 0 and ltf == 1:
            base_confluence += 15
            signal['direction'] = 1
            signal['reasoning'].append("LTF Bullish (HTF flat): +15")
        elif htf == 0 and ltf == -1:
            base_confluence += 15
            signal['direction'] = -1
            signal['reasoning'].append("LTF Bearish (HTF flat): +15")

        if pp < 0.25:
            base_confluence += 20
            signal['reasoning'].append("Price near lows: +20")
        elif pp > 0.75:
            base_confluence += 20
            signal['reasoning'].append("Price near highs: +20")

        if near_bull_fvg and ltf >= 0:
            base_confluence += 15
            signal['reasoning'].append("V5 Bull FVG: +15")
        if near_bear_fvg and ltf <= 0:
            base_confluence += 15
            signal['reasoning'].append("V5 Bear FVG: +15")

        if ms_analysis is not None:
            try:
                if hasattr(ms_analysis, 'recent_breaks'):
                    for brk in ms_analysis.recent_breaks[-3:]:
                        if brk.break_type == StructureBreakType.MSS:
                            if signal['direction'] == 1 and brk.direction == 'bullish':
                                base_confluence += 20
                                signal['reasoning'].append("Bullish MSS: +20")
                            elif signal['direction'] == -1 and brk.direction == 'bearish':
                                base_confluence += 20
                                signal['reasoning'].append("Bearish MSS: +20")
                        elif brk.break_type == StructureBreakType.BOS:
                            if signal['direction'] == 1 and brk.direction == 'bullish':
                                base_confluence += 15
                                signal['reasoning'].append("Bullish BOS: +15")
                            elif signal['direction'] == -1 and brk.direction == 'bearish':
                                base_confluence += 15
                                signal['reasoning'].append("Bearish BOS: +15")

                if hasattr(ms_analysis, 'trend_state'):
                    if ms_analysis.trend_state == TrendState.BULLISH and signal['direction'] == 1:
                        base_confluence += 10
                        signal['reasoning'].append("MS Bullish: +10")
                    elif ms_analysis.trend_state == TrendState.BEARISH and signal['direction'] == -1:
                        base_confluence += 10
                        signal['reasoning'].append("MS Bearish: +10")

                if hasattr(ms_analysis, 'current_zone'):
                    if ms_analysis.current_zone == PriceZone.DISCOUNT and signal['direction'] == 1:
                        base_confluence += 10
                        signal['reasoning'].append("Discount Zone: +10")
                    elif ms_analysis.current_zone == PriceZone.PREMIUM and signal['direction'] == -1:
                        base_confluence += 10
                        signal['reasoning'].append("Premium Zone: +10")
            except Exception:
                pass

        signal['confluence'] = base_confluence

        # ── FVG confluence ────────────────────────────────────────────────────
        fvg_confluence = 0
        if fvg_analysis.best_bisi_fvg and signal['direction'] == 1:
            fvg = fvg_analysis.best_bisi_fvg
            distance = abs(current_price - fvg.consequent_encroachment)
            if distance < fvg.size * 2:
                fvg_confluence += 20
                signal['fvg_data'] = {'type': 'BISI', 'ce': fvg.consequent_encroachment, 'distance': distance}
                signal['reasoning'].append(f"FVG BISI at {fvg.consequent_encroachment:.4f}")
                if fvg.is_high_probability:
                    fvg_confluence += 15
                    signal['reasoning'].append("High Probability FVG")

        elif fvg_analysis.best_sibi_fvg and signal['direction'] == -1:
            fvg = fvg_analysis.best_sibi_fvg
            distance = abs(current_price - fvg.consequent_encroachment)
            if distance < fvg.size * 2:
                fvg_confluence += 20
                signal['fvg_data'] = {'type': 'SIBI', 'ce': fvg.consequent_encroachment, 'distance': distance}
                signal['reasoning'].append(f"FVG SIBI at {fvg.consequent_encroachment:.4f}")
                if fvg.is_high_probability:
                    fvg_confluence += 15
                    signal['reasoning'].append("High Probability FVG")

        # ── Gap confluence ────────────────────────────────────────────────────
        gap_confluence = 0
        if gap_analysis.current_gap:
            gap = gap_analysis.current_gap
            if gap_analysis.in_gap_zone:
                gap_confluence += 10
                signal['reasoning'].append(f"In {gap.gap_type.value} gap zone")
                if gap.quadrants:
                    ce_distance = abs(current_price - gap.quadrants.ce)
                    if ce_distance < (gap.quadrants.range_size * 0.1):
                        gap_confluence += 15
                        signal['reasoning'].append("At Gap CE (50%)")
                        signal['gap_data'] = {
                            'type':      gap.gap_type.value,
                            'ce':        gap.quadrants.ce,
                            'direction': gap.direction.value,
                        }

        if gap_analysis.nearest_level:
            level_price, level_name = gap_analysis.nearest_level
            distance_pct = abs(current_price - level_price) / current_price * 100
            if distance_pct < 0.5:
                gap_confluence += 10
                signal['reasoning'].append(f"Near {level_name}")

        total_confluence    = base_confluence + fvg_confluence + gap_confluence
        signal['confluence'] = min(total_confluence, 100)

        if total_confluence >= 80:
            signal['confidence'] = 'HIGH'
        elif total_confluence >= 60:
            signal['confidence'] = 'MEDIUM'
        elif total_confluence >= 50:
            signal['confidence'] = 'LOW'
        else:
            signal['direction'] = 0

        # ── Entry / Stop / Take-Profit ────────────────────────────────────────
        if signal['direction'] != 0:
            entry = current_price
            if signal['fvg_data']:
                fvg_ce = signal['fvg_data']['ce']
                if signal['direction'] == 1 and current_price > fvg_ce:
                    entry = fvg_ce
                elif signal['direction'] == -1 and current_price < fvg_ce:
                    entry = fvg_ce

            signal['entry_price'] = entry

            contract_info = get_contract_info(symbol)
            is_forex      = contract_info['type'] == 'forex'

            highs_arr  = data['highs']
            lows_arr   = data['lows']
            closes_arr = data['closes']

            # ATR (14-period)
            atr = np.mean([
                max(
                    highs_arr[i] - lows_arr[i],
                    abs(highs_arr[i] - closes_arr[i-1]) if i > 0 else 0,
                    abs(lows_arr[i]  - closes_arr[i-1]) if i > 0 else 0,
                )
                for i in range(max(0, idx - 14), idx + 1)
            ])

            atr_multiplier     = 2.0
            min_atr_multiplier = 1.5

            if is_forex:
                decimal_places = contract_info.get('decimal_places', 5)
                pip_size       = 0.01 if decimal_places == 3 else 0.0001

                atr_pips          = atr / pip_size
                # Soft bounds in pips (guidance only – final check below)
                min_stop_pips     = max(15,  atr_pips * min_atr_multiplier)
                max_stop_pips     = max(80,  atr_pips * atr_multiplier)
                atr_stop_distance = atr * atr_multiplier

                min_stop_distance = min_stop_pips * pip_size
                max_stop_distance = max_stop_pips * pip_size
                stop_distance     = max(min_stop_distance, min(atr_stop_distance, max_stop_distance))

                risk_pips = stop_distance / pip_size

                # ── FIX 6: aligned validation bounds (10–150 pips) ────────────
                # Old bounds were 15–100 which rejected legitimate volatile moves.
                if risk_pips < 10 or risk_pips > 150:
                    signal['direction']   = 0
                    signal['stop_loss']   = entry
                    signal['take_profit'] = entry
                    return signal

            else:
                # Non-forex: ATR-based stop with instrument minimum floor
                min_stop      = contract_info.get('min_stop', 20)
                stop_distance = max(atr * atr_multiplier, min_stop)

            # ── FIX 1 (cont.): use self.rr_ratio – set correctly via __init__ ─
            rr = self.rr_ratio   # was: getattr(self, 'rr_ratio', 2.0)  → always 2.0

            if signal['direction'] == 1:
                signal['stop_loss']   = entry - stop_distance
                signal['take_profit'] = entry + stop_distance * rr
            else:
                signal['stop_loss']   = entry + stop_distance
                signal['take_profit'] = entry - stop_distance * rr

        return signal


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE TRADER
# ═══════════════════════════════════════════════════════════════════════════════

class V7MT5LiveTrader:

    def __init__(
        self, symbols: List[str], risk_pct: float = 0.02,
        poll_interval: int = 30, rr_ratio: float = 3.0,
        confluence_threshold: int = 60, max_daily_loss: float = -2000,
    ):
        self.symbols              = symbols
        self.risk_pct             = risk_pct
        self.poll_interval        = poll_interval
        self.rr_ratio             = rr_ratio
        self.confluence_threshold = confluence_threshold
        self.max_daily_loss       = max_daily_loss

        # ── FIX 1 (cont.): pass rr_ratio into signal generator ────────────────
        self.signal_generator = V7SignalGenerator(rr_ratio=rr_ratio)

        self.mode             = 'paper'
        self.positions        = {}
        self.historical_data  = {}
        self.last_poll_time   = {}
        self.last_signal_time = {}
        self.last_signals     = {}

        self.daily_pnl    = 0.0
        self.trade_count  = 0
        self.account_value = 100_000

        self.update_account()
        self.currency_pnl = {}
        self.running      = False
        self.magic        = 123456

        self._sync_positions()

    def _sync_positions(self):
        print("\nSyncing positions with MT5...")
        try:
            positions = get_mt5_positions()
            for pos in positions:
                mt5_symbol = pos['symbol'].upper()
                symbol     = mt5_symbol.replace('M', '')
                if (symbol in [s.upper() for s in self.symbols] or
                        mt5_symbol in [get_mt5_symbol(s).upper() for s in self.symbols]):
                    direction  = 1 if pos['type'] == 0 else -1
                    symbol_key = symbol if symbol in [s.upper() for s in self.symbols] else mt5_symbol
                    self.positions[symbol_key] = {
                        'ticket':        pos['ticket'],
                        'entry':         pos['price_open'],
                        'direction':     direction,
                        'volume':        pos['volume'],
                        'stop':          pos['sl'],
                        'target':        pos['tp'],
                        'bars_held':     0,
                        'current_price': pos['price_current'],
                        'pnl':           pos['profit'],
                    }
                    print(f"  Found open position: {mt5_symbol} x {pos['volume']} "
                          f"@ {pos['price_open']:.4f}  P&L: ${pos['profit']:.2f}")

            if not self.positions:
                print("  No open positions found")
            else:
                print(f"  Total open positions: {len(self.positions)}")
        except Exception as e:
            print(f"  Error syncing positions: {e}")

    def get_current_price(self, symbol: str) -> Optional[float]:
        if not MT5_AVAILABLE:
            return None
        info = mt5.symbol_info(get_mt5_symbol(symbol))
        return info.bid if info else None

    def _check_position_exit(self, symbol: str, current_price: float):
        mt5_sym = get_mt5_symbol(symbol)
        pos_key = symbol if symbol in self.positions else (mt5_sym if mt5_sym in self.positions else None)
        if pos_key is None:
            return

        try:
            pos           = self.positions[pos_key]
            pos['bars_held']     = pos.get('bars_held', 0) + 1
            pos['current_price'] = current_price

            contract_info = get_contract_info(symbol)
            symbol_type   = contract_info['type']
            direction     = pos['direction']
            entry         = pos['entry']
            stop          = pos.get('stop', 0)
            target        = pos.get('target', 0)
            volume        = pos['volume']
            price_diff    = (current_price - entry) if direction == 1 else (entry - current_price)

            if symbol_type == 'forex':
                pnl = price_diff * volume * 100_000
            elif symbol_type in ('futures', 'indices'):
                dollar_per_point = contract_info.get('dollar_per_point', 1)
                pnl              = price_diff * volume * dollar_per_point
            else:
                pnl = price_diff * volume

            exit_reason = None
            if direction == 1:
                if stop   and current_price <= stop:   exit_reason = 'stop'
                elif target and current_price >= target: exit_reason = 'target'
            else:
                if stop   and current_price >= stop:   exit_reason = 'stop'
                elif target and current_price <= target: exit_reason = 'target'

            if exit_reason:
                self._handle_position_closed(pos_key, current_price, pnl, exit_reason)
            else:
                pos['pnl'] = pnl
                print(f"  [{symbol}] Unrealized P&L: ${pnl:.2f}")

        except Exception as e:
            print(f"[{symbol}] Error checking position: {e}")

    def _handle_position_closed(self, symbol: str, exit_price: float, pnl: float, exit_reason: str):
        if symbol not in self.positions:
            return
        pos           = self.positions[symbol]
        direction_str = 'LONG' if pos['direction'] == 1 else 'SHORT'
        pnl_str       = f"+${pnl:.2f}" if pnl > 0 else f"-${abs(pnl):.2f}"
        print(f"[{symbol}] V7 EXIT ({exit_reason}): {direction_str} @ {exit_price:.4f} | P&L: {pnl_str}")
        print(f"  Daily P&L: ${self.daily_pnl:.2f}")
        if tn:
            try:
                tn.send_trade_exit(symbol, pos['direction'], pnl, exit_reason,
                                   pos['entry'], exit_price, pos.get('bars_held', 0))
            except Exception:
                pass
        del self.positions[symbol]

    def _enter_trade(self, symbol: str, signal: Dict, current_price: float):
        if is_trading_paused():
            print(f"[{symbol}] Signal found but trading is PAUSED")
            return

        if self.daily_pnl <= self.max_daily_loss:
            print(f"[{symbol}] Daily loss limit reached (${self.daily_pnl:.2f}), skipping trade")
            return

        is_forex = symbol.upper() in FOREX_SYMBOLS
        in_session, session_name = is_valid_trading_session(is_forex=is_forex)
        if not in_session:
            print(f"[{symbol}] Outside trading hours ({session_name}), skipping")
            return

        mt5_sym = get_mt5_symbol(symbol)
        if symbol in self.positions or mt5_sym in self.positions:
            print(f"[{symbol}] Already has open position, skipping entry")
            return

        try:
            entry_price = signal['entry_price']
            stop_price  = signal['stop_loss']

            if stop_price is None:
                print(f"[{symbol}] No stop_loss in signal, skipping")
                return

            stop_distance = abs(entry_price - stop_price)
            if stop_distance <= 0:
                return

            # ── FIX 2: use the TP the signal generator already computed ────────
            # The original code recalculated target_price independently here,
            # creating a mismatch between what was logged in the signal and what
            # went into the MT5 order.  Now both always reflect the same value.
            target_price = signal['take_profit']
            if target_price is None:
                # Fallback – should never happen if signal is valid
                target_price = (entry_price + stop_distance * self.rr_ratio
                                if signal['direction'] == 1
                                else entry_price - stop_distance * self.rr_ratio)

            qty, risk_amount = calculate_position_size(
                symbol, self.account_value, self.risk_pct,
                stop_distance, entry_price,
            )
            if qty <= 0:
                print(f"[{symbol}] Failed: qty={qty}, stop_dist={stop_distance:.5f}")
                return

            # Sanity-check: confirm the effective R:R matches configuration
            reward_distance = abs(target_price - entry_price)
            effective_rr    = reward_distance / stop_distance if stop_distance > 0 else 0
            print(f"[{symbol}] Lot calc: Risk=${risk_amount:.2f}  "
                  f"Stop={stop_distance:.5f}  Lots={qty:.2f}  "
                  f"R:R=1:{effective_rr:.2f}")

            direction_str = 'LONG' if signal['direction'] == 1 else 'SHORT'

            fvg_info = signal.get('fvg_data') or {}
            gap_info = signal.get('gap_data') or {}
            pd_zone  = (f"FVG {fvg_info.get('type', '')}" if fvg_info
                        else f"Gap {gap_info.get('type', '')}" if gap_info
                        else None)

            if self.mode == 'shadow':
                print(f"[{symbol}] V7 SHADOW SIGNAL: {direction_str} @ {current_price:.4f}")
                print(f"  Stop: {stop_price:.4f} | Target: {target_price:.4f} "
                      f"(R:R 1:{effective_rr:.2f})")
                print(f"  Confluence: {signal['confluence']}/100 | {pd_zone or 'No PD'}")
                print(f"  Risk: ${risk_amount:.2f} | Qty: {qty:.2f}")
                self._log_shadow_trade(symbol, signal, current_price,
                                       stop_price, target_price, qty, risk_amount)
                if tn:
                    try:
                        tn.send_signal_alert(
                            symbol=symbol, direction=signal['direction'],
                            confluence=signal['confluence'],
                            pd_zone=pd_zone or '', current_price=current_price,
                        )
                    except Exception:
                        pass
                return

            order_type = "BUY" if signal['direction'] == 1 else "SELL"
            result = place_mt5_order(
                symbol, order_type, qty, entry_price,
                stop_loss=stop_price, take_profit=target_price,
                magic=self.magic,
            )

            if result:
                self.positions[symbol] = {
                    'ticket':        result['order_id'],
                    'entry':         entry_price,
                    'stop':          stop_price,
                    'target':        target_price,
                    'direction':     signal['direction'],
                    'volume':        qty,
                    'confluence':    signal['confluence'],
                    'confidence':    signal['confidence'],
                    'entry_time':    datetime.now(),
                    'reasoning':     signal['reasoning'],
                    'bars_held':     0,
                    'current_price': entry_price,
                }
                self.trade_count += 1
                print(f"[{symbol}] V7 ENTRY: {direction_str} x {qty:.2f} @ {entry_price:.4f}")
                print(f"  Confidence: {signal['confidence']} | Confluence: {signal['confluence']}/100")
                print(f"  Stop: {stop_price:.4f} | Target: {target_price:.4f} | R:R 1:{effective_rr:.2f}")
                if signal['reasoning']:
                    print(f"  Reasoning: {' | '.join(signal['reasoning'][:3])}")
                if tn:
                    try:
                        tn.send_trade_entry(
                            symbol, signal['direction'], qty,
                            entry_price, signal['confluence'], target_price, stop_price,
                            pd_zone=pd_zone, risk_amount=risk_amount,
                        )
                    except RuntimeError as e:
                        if "event loop" in str(e).lower():
                            print(f"[{symbol}] Telegram notification skipped (asyncio issue)")
                        else:
                            print(f"[{symbol}] Telegram notification error: {e}")
                    except Exception as e:
                        print(f"[{symbol}] Telegram notification error: {e}")
            else:
                print(f"[{symbol}] Failed to place order")

        except Exception as e:
            print(f"[{symbol}] V7 Error entering trade: {e}")

    def _log_shadow_trade(
        self, symbol: str, signal: Dict, current_price: float,
        stop_price: float, target_price: float, qty: float, risk_amount: float,
    ):
        try:
            trade = {
                'timestamp':   datetime.now().isoformat(),
                'symbol':      symbol,
                'direction':   'LONG' if signal['direction'] == 1 else 'SHORT',
                'entry':       current_price,
                'stop':        stop_price,
                'target':      target_price,
                'rr_ratio':    self.rr_ratio,
                'qty':         qty,
                'risk_amount': risk_amount,
                'confluence':  signal['confluence'],
                'confidence':  signal['confidence'],
                'fvg_data':    signal.get('fvg_data'),
                'gap_data':    signal.get('gap_data'),
                'reasoning':   signal.get('reasoning', [])[:3],
            }
            with open('v7_shadow_trades.json', 'a') as f:
                f.write(json.dumps(trade) + '\n')
        except Exception as e:
            print(f"Error logging shadow trade: {e}")

    def _refresh_data(self):
        for symbol in self.symbols:
            try:
                data = prepare_data_mt5(symbol, lookback=200)
                if data and len(data.get('closes', [])) >= 50:
                    self.historical_data[symbol] = data
            except Exception as e:
                print(f"[{symbol}] Error refreshing data: {e}")

    def poll_symbols(self):
        current_time = time.time()
        for symbol in self.symbols:
            if current_time - self.last_poll_time.get(symbol, 0) < self.poll_interval:
                continue
            self.last_poll_time[symbol] = current_time
            try:
                data = prepare_data_mt5(symbol, lookback=200)
                if data is None or len(data.get('closes', [])) < 50:
                    continue
                self.historical_data[symbol] = data

                idx          = len(data['closes']) - 1
                live_price   = self.get_current_price(symbol)
                current_price = live_price if live_price else data['closes'][idx]

                try:
                    signal = self.signal_generator.analyze_symbol(symbol, data, current_price)
                except Exception as e:
                    print(f"[{symbol}] Signal error: {e}")
                    continue

                if signal and signal.get('direction', 0) != 0:
                    fvg_type = signal.get('fvg_data', {}).get('type', '') if isinstance(signal.get('fvg_data'), dict) else ''
                    gap_type = signal.get('gap_data', {}).get('type', '') if isinstance(signal.get('gap_data'), dict) else ''
                    self.last_signals[symbol] = {
                        'direction':  signal.get('direction', 0),
                        'confluence': signal.get('confluence', 0),
                        'confidence': signal.get('confidence', 'LOW'),
                        'pd_zone':    fvg_type or gap_type,
                        'entry':      signal.get('entry_price', current_price),
                        'timestamp':  datetime.now().isoformat(),
                    }

                if tn and signal:
                    try:
                        htf = data.get('htf_trend', np.zeros(len(data['closes'])))[idx]
                        ltf = data.get('ltf_trend', np.zeros(len(data['closes'])))[idx]
                        tn.update_market_data(symbol, {
                            'price':      current_price,
                            'htf_trend':  htf,
                            'ltf_trend':  ltf,
                            'confluence': signal.get('confluence', 0),
                            'confidence': signal.get('confidence', 'LOW'),
                        })
                    except Exception:
                        pass

                mt5_sym = get_mt5_symbol(symbol)
                if symbol in self.positions or mt5_sym in self.positions:
                    self._check_position_exit(symbol, current_price)
                else:
                    current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
                    if self.last_signal_time.get(symbol, datetime.min) >= current_hour:
                        continue
                    if (signal and signal.get('direction', 0) != 0 and
                            signal.get('confluence', 0) >= self.confluence_threshold):
                        self.last_signal_time[symbol] = current_hour
                        self._enter_trade(symbol, signal, current_price)

            except Exception as e:
                print(f"[{symbol}] Error polling: {e}")

    def update_account(self):
        if not MT5_AVAILABLE:
            return
        try:
            account_info = mt5.account_info()
            if account_info:
                self.account_value = account_info.balance
                print(f"Account balance: ${self.account_value:,.2f}")
        except Exception as e:
            print(f"Error updating account: {e}")

    def check_positions(self):
        if not MT5_AVAILABLE:
            return
        try:
            mt5_positions = get_mt5_positions()
            mt5_tickets   = {p['ticket']: p['symbol'].upper() for p in mt5_positions}
            for symbol in list(self.positions.keys()):
                pos = self.positions[symbol]
                if pos.get('ticket') and pos['ticket'] not in mt5_tickets:
                    print(f"[{symbol}] Position closed externally")
                    self._handle_position_closed(
                        symbol, pos.get('current_price', pos['entry']),
                        pos.get('pnl', 0), 'external',
                    )
        except Exception as e:
            print(f"Error checking positions: {e}")

    def start(self):
        self.running = True
        print(f"\nV7 MT5 Trader started for {self.symbols}")

    def stop(self):
        self.running = False
        print("V7 MT5 Trader stopped")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def run_v7_trading(
    symbols: List[str], interval: int = 30, risk_pct: float = 0.02,
    login: int = None, password: str = None, server: str = None,
    mode: str = 'paper', rr_ratio: float = 3.0,
    confluence_threshold: int = 60, max_daily_loss: float = -2000,
):
    if not MT5_AVAILABLE:
        print("ERROR: MetaTrader5 not installed. Run: pip install MetaTrader5")
        return

    if not init_mt5(login, password, server):
        print("Failed to initialize MT5")
        return

    print(f"\nICT V7 - MT5 Trading")
    print(f"Mode: {mode.upper()}")
    print(f"Symbols: {symbols}")
    print(f"Risk: {risk_pct*100}% | R:R 1:{rr_ratio}")
    print(f"Confluence: {confluence_threshold}+ | Max Loss: ${max_daily_loss}")
    print("-" * 50)

    if tn:
        try:
            tn.send_startup(symbols=symbols, risk_pct=risk_pct,
                            interval=interval, mode=f"V7 {mode.upper()}")
        except Exception as e:
            print(f"Telegram startup notification failed: {e}")

    trader = V7MT5LiveTrader(
        symbols, risk_pct, poll_interval=interval,
        rr_ratio=rr_ratio, confluence_threshold=confluence_threshold,
        max_daily_loss=max_daily_loss,
    )
    print(f"Trader account value: ${trader.account_value:,.2f}")
    trader.mode = mode

    if tn and hasattr(tn, 'set_live_trader'):
        tn.set_live_trader(trader)
        print("Live trader registered with Telegram")

    if tn and hasattr(tn, 'start_polling_background'):
        try:
            tn.start_polling_background()
            print("Telegram command polling started")
        except Exception as e:
            print(f"Failed to start Telegram polling: {e}")

    trader.start()
    trader._refresh_data()

    print("\nTrading started. Press Ctrl+C to stop.\n")

    iteration = 0
    try:
        while trader.running:
            iteration += 1
            time.sleep(1)

            if iteration % interval == 0:
                trader.poll_symbols()
            if iteration % 300 == 0:
                trader._refresh_data()
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Data refreshed")
            if iteration % 60 == 0:
                trader.update_account()
                trader.check_positions()
            if iteration % 3600 == 0 and trader.positions:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Active positions: {len(trader.positions)}")
            if trader.daily_pnl <= trader.max_daily_loss and iteration % 60 == 0:
                print(f"[WARNING] Daily loss limit reached: ${trader.daily_pnl:.2f}")

    except KeyboardInterrupt:
        print("\n\nShutdown...")
    finally:
        trader.stop()
        trader.update_account()
        if MT5_AVAILABLE:
            mt5.shutdown()
        print(f"\nTrades: {trader.trade_count} | "
              f"Daily P&L: ${trader.daily_pnl:.2f} | "
              f"Final: ${trader.account_value:,.2f}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='ICT V7 - MT5 Trading with FVG + Gap')
    parser.add_argument("--symbols",
                        default="EURUSD,GBPUSD,USDJPY,USDCAD,AUDUSD,XAUUSD,XTIUSD,US30,USTEC,US500")
    parser.add_argument("--interval",    type=int,   default=30)
    parser.add_argument("--risk",        type=float, default=0.03)
    parser.add_argument("--login",       type=int,   default=None)
    parser.add_argument("--password",    type=str,   default=None)
    parser.add_argument("--server",      type=str,   default=None)
    parser.add_argument("--mode",        type=str,   default="paper",
                        choices=["shadow", "paper", "live"])
    parser.add_argument("--rr",          type=float, default=3.0)
    parser.add_argument("--confluence",  type=int,   default=60)
    parser.add_argument("--max-loss",    type=float, default=-2000)
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(',')]

    print("=" * 60)
    print("ICT V7 Trading Bot - MetaTrader 5  (R:R FIXED)")
    print("=" * 60)
    print(f"Mode:       {args.mode.upper()}")
    print(f"Symbols:    {', '.join(symbols)}")
    print(f"Risk:       {args.risk*100}%  |  R:R 1:{args.rr}")
    print(f"Confluence: {args.confluence}+  |  Max Loss: ${args.max_loss}")
    print(f"MT5 Login:  {args.login or 'Demo'}")
    print("=" * 60)

    run_v7_trading(
        symbols, args.interval, args.risk,
        args.login, args.password, args.server, args.mode,
        rr_ratio=args.rr, confluence_threshold=args.confluence,
        max_daily_loss=args.max_loss,
    )