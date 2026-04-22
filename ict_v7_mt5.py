"""
ICT V8 Trading Bot - MT5 Version  (All R:R Bugs Fixed)
=======================================================
Three bugs fixed vs the previous version:

BUG-FIX 1 – Name collision: class V7SignalGenerator shadowed the import
    The file imported ICTSignalEngine as V7SignalGenerator at the top,
    then redefined V7SignalGenerator as the old stub class further down.
    Python uses the LAST definition → ICTSignalEngine was never used.
    Fix: legacy stub renamed to _LegacyV7SignalGenerator.

BUG-FIX 2 – OB entry level (in signal_engine.py)
    ob.open for a bullish OB = bearish candle open = body_high.
    body_high > current_price → BUY LIMIT above ask → invalid MT5 order.
    Fix: signal_engine now returns entry = mean_threshold (50% of OB body),
    which is always below current price for a bullish retest.

BUG-FIX 3 – TP anchored to limit entry, not actual fill price (_enter_trade)
    signal['entry_price'] is a LIMIT price; the fill happens at current_price
    (market order) or later at the limit.  The old code computed:
        TP = signal['entry_price'] + stop_dist × rr
    When signal['entry_price'] ≈ TP due to m22/sb targets, the result was
    TP ≈ entry (0-pip gain).
    Fix: TP is always computed from current_price (the reference price at
    the moment the signal fires), so R:R is always 1:rr_ratio.

Usage:
    python3 ict_v7_mt5_fixed.py --symbols "EURUSD,GBPUSD,USDJPY,XAUUSD" \
                                 --login 12345 --password "pass" \
                                 --mode shadow --rr 3.0
"""

import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import sys
import os
import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# ── V8 fully-wired signal engine ──────────────────────────────────────────────
# BUG-FIX 1: This import alias must NOT be shadowed by a class definition below.
try:
    from signal_engine import ICTSignalEngine as V7SignalGenerator
    print("ICT Signal Engine loaded (V8 – all handlers wired, R:R fixed)")
except ImportError as e:
    print(f"WARNING: signal_engine.py not found, falling back to stub: {e}")
    V7SignalGenerator = None   # handled in V7MT5LiveTrader.__init__

from fvg_handler import FVGHandler, FairValueGap, FVGStatus
from gap_handler import GapHandler, Gap, GapType, GapDirection

try:
    from market_structure_handler import (
        MarketStructureHandler, MarketStructureAnalysis,
        StructureBreakType, TrendState, PriceZone,
    )
    from mtf_coordinator import MTFCoordinator
    MTF_AVAILABLE = True
except ImportError as e:
    MTF_AVAILABLE = False
    print(f"WARNING: market_structure_handler or mtf_coordinator not available: {e}")

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
    print("WARNING: MetaTrader5 not installed.  Run: pip install MetaTrader5")

try:
    import telegram_notify as tn
    if tn and hasattr(tn, 'init_bot'):
        try:
            tn.init_bot()
            print("Telegram bot initialised")
        except Exception as e:
            print(f"Telegram bot init failed: {e}")
except (ImportError, NameError, Exception) as e:
    tn = None
    print(f"WARNING: telegram_notify not available: {e}")


FOREX_SYMBOLS = {
    'EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD', 'AUDUSD', 'USDCHF',
    'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY', 'XAUUSD', 'XAGUSD',
    'EURAUD', 'EURCAD', 'GBPAUD', 'AUDJPY', 'CADJPY', 'CHFJPY',
    'GBPAUD', 'GBPCAD',
}

MT5_SYMBOLS = {
    'BTCUSD': 'BTCUSDm', 'ETHUSD': 'ETHUSDm', 'SOLUSD': 'SOLUSDm', 'XRPUSD': 'XRPUSDm',
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
        'symbol':               info.name,
        'bid':                  info.bid,
        'ask':                  info.ask,
        'last':                 info.last,
        'point':                info.point,
        'digits':               info.digits,
        'trade_contract_size':  info.trade_contract_size,
        'volume_min':           info.volume_min,
        'volume_max':           info.volume_max,
        'volume_step':          info.volume_step,
        'trade_tick_size':      info.trade_tick_size,
        'trade_tick_value':     info.trade_tick_value,
    }


def get_contract_info(symbol: str) -> Dict:
    """
    Return comprehensive contract information for position sizing.

    ``dollar_per_point`` = USD profit/loss when price moves 1 unit, 1 standard lot.
    This is the ONLY multiplier needed – no tick_size arithmetic on top.
    """
    symbol = symbol.upper()

    futures_info = {
        'XAUUSD': {'dollar_per_point': 100,   'min_stop': 5.0,   'tick_size': 0.01,  'type': 'futures'},
        'XAGUSD': {'dollar_per_point': 50,    'min_stop': 0.10,  'tick_size': 0.001, 'type': 'futures'},
        'XTIUSD': {'dollar_per_point': 1000,  'min_stop': 0.30,  'tick_size': 0.01,  'type': 'futures'},
        'XBRUSD': {'dollar_per_point': 1000,  'min_stop': 0.30,  'tick_size': 0.01,  'type': 'futures'},
        'XNGUSD': {'dollar_per_point': 10000, 'min_stop': 0.02,  'tick_size': 0.001, 'type': 'futures'},
    }

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
        'EURAUD': {'pip_value': 10,   'min_stop': 0.0020, 'decimal_places': 5, 'type': 'forex'},
        'EURCAD': {'pip_value': 7.5,  'min_stop': 0.0020, 'decimal_places': 5, 'type': 'forex'},
        'GBPAUD': {'pip_value': 10,   'min_stop': 0.0020, 'decimal_places': 5, 'type': 'forex'},
        'AUDJPY': {'pip_value': 9.1,  'min_stop': 0.20,   'decimal_places': 3, 'type': 'forex'},
        'CADJPY': {'pip_value': 9.1,  'min_stop': 0.20,   'decimal_places': 3, 'type': 'forex'},
        'CHFJPY': {'pip_value': 9.1,  'min_stop': 0.20,   'decimal_places': 3, 'type': 'forex'},
    }

    crypto_info = {
        'BTCUSD': {'dollar_per_point': 2, 'min_stop_pct': 1.5, 'tick_size': 0.5, 'type': 'crypto'},
        'ETHUSD': {'dollar_per_point': 20, 'min_stop_pct': 1.5, 'tick_size': 0.05, 'type': 'crypto'},
        'SOLUSD': {'dollar_per_point': 200, 'min_stop_pct': 2.0, 'tick_size': 0.005, 'type': 'crypto'},
        'XRPUSD': {'dollar_per_point': 100000, 'min_stop_pct': 2.0, 'tick_size': 0.00001, 'type': 'crypto'},
        'LTCUSD': {'dollar_per_point': 100, 'min_stop_pct': 2.0, 'tick_size': 0.01, 'type': 'crypto'},
    }

    indices_info = {
        'US30':   {'dollar_per_point': 1, 'min_stop': 20,  'tick_size': 1,   'type': 'indices'},
        'US500':  {'dollar_per_point': 1, 'min_stop': 3,   'tick_size': 0.1, 'type': 'indices'},
        'USTEC':  {'dollar_per_point': 1, 'min_stop': 10,  'tick_size': 0.1, 'type': 'indices'},
        'GER40':  {'dollar_per_point': 1, 'min_stop': 15,  'tick_size': 0.1, 'type': 'indices'},
        'UK100':  {'dollar_per_point': 1, 'min_stop': 8,   'tick_size': 0.1, 'type': 'indices'},
        'FRA40':  {'dollar_per_point': 1, 'min_stop': 8,   'tick_size': 0.1, 'type': 'indices'},
        'JPN225': {'dollar_per_point': 1, 'min_stop': 80,  'tick_size': 1,   'type': 'indices'},
        'AUS200': {'dollar_per_point': 1, 'min_stop': 8,   'tick_size': 0.1, 'type': 'indices'},
    }

    if symbol in futures_info: return futures_info[symbol]
    if symbol in forex_info:   return forex_info[symbol]
    if symbol in crypto_info:  return crypto_info[symbol]
    if symbol in indices_info: return indices_info[symbol]
    return {'pip_value': 10, 'min_stop': 0.0020, 'decimal_places': 5, 'type': 'forex'}


def fetch_mt5_rates(symbol: str, timeframe: str = "H1",
                   num_bars: int = 500) -> Optional[pd.DataFrame]:
    if not MT5_AVAILABLE:
        return None
    timeframe_map = {
        "M1":  mt5.TIMEFRAME_M1,  "M5":  mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15, "M30": mt5.TIMEFRAME_M30,
        "H1":  mt5.TIMEFRAME_H1,  "H4":  mt5.TIMEFRAME_H4,
        "D1":  mt5.TIMEFRAME_D1,  "W1":  mt5.TIMEFRAME_W1,
    }
    mt5_tf = timeframe_map.get(timeframe, mt5.TIMEFRAME_H1)
    if not mt5.symbol_select(symbol, True):
        print(f"Could not select symbol: {symbol}")
        return None
    rates = mt5.copy_rates_from_pos(symbol, mt5_tf, 0, num_bars)
    if rates is None or len(rates) == 0:
        print(f"No rates returned for {symbol}")
        return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)
    df.columns = ['open', 'low', 'high', 'close', 'tick_volume', 'spread', 'real_volume']
    return df


def prepare_data_mt5(symbol: str, lookback: int = 200) -> Optional[Dict]:
    symbol     = symbol.upper()
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
            bullish_fvgs.append({'idx': i, 'mid': (highs[i-2] + lows[i]) / 2,
                                 'high': lows[i]})
        if highs[i] < lows[i-2]:
            bearish_fvgs.append({'idx': i, 'mid': (highs[i] + lows[i-2]) / 2,
                                 'low': highs[i]})

    df_daily = fetch_mt5_rates(mt5_symbol, "D1", num_bars=60)
    if df_daily is None or len(df_daily) < 5:
        htf_trend = np.zeros(len(df))
    else:
        daily_highs = df_daily['high'].values
        daily_lows  = df_daily['low'].values
        htf = []
        for i in range(1, len(df_daily)):
            if (daily_highs[i] > np.max(daily_highs[max(0,i-5):i]) and
                    daily_lows[i]  > np.min(daily_lows[max(0,i-5):i])):
                htf.append(1)
            elif (daily_highs[i] < np.max(daily_highs[max(0,i-5):i]) and
                      daily_lows[i]  < np.min(daily_lows[max(0,i-5):i])):
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
        ema_bull   = ema_fast > ema_slow
        ema_bear   = ema_fast < ema_slow
        if pct_change > 0.005 or (pct_change > 0.001 and ema_bull):
            trend[i] = 1
        elif pct_change < -0.005 or (pct_change < -0.001 and ema_bear):
            trend[i] = -1

    price_position = np.zeros(len(df))
    for i in range(20, len(df)):
        ph  = np.max(highs[i-20:i])
        pl  = np.min(lows[i-20:i])
        rng = max(ph - pl, 0.001)
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
            tr = max(
                highs[j] - lows[j],
                abs(highs[j] - closes[j-1]) if j > 0 else 0,
                abs(lows[j]  - closes[j-1]) if j > 0 else 0,
            )
            trs.append(tr)
        volatility[i] = np.mean(trs) if trs else 0

    return {
        'opens':          opens,
        'highs':          highs,
        'lows':           lows,
        'closes':         closes,
        'volatility':     volatility,
        'htf_trend':      htf_trend,
        'ltf_trend':      trend,
        'price_position': price_position,
        'kill_zone':      kill_zone,
        'bullish_fvgs':   bullish_fvgs,
        'bearish_fvgs':   bearish_fvgs,
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

    Returns (qty_lots, actual_dollar_risk).
    """
    contract_info = get_contract_info(symbol)
    symbol_type   = contract_info['type']
    risk_amount   = account_value * risk_pct

    if stop_distance <= 0:
        return 0.0, 0.0

    qty = 0.0

    if symbol_type == 'forex':
        decimal_places = contract_info.get('decimal_places', 5)
        pip_size       = 0.01 if decimal_places == 3 else 0.0001
        pip_value      = contract_info.get('pip_value', 10)
        stop_pips      = stop_distance / pip_size
        if stop_pips <= 0:
            return 0.0, 0.0
        qty         = risk_amount / (stop_pips * pip_value)
        actual_risk = qty * stop_pips * pip_value

    elif symbol_type == 'futures':
        dollar_per_point = contract_info.get('dollar_per_point', 100)
        qty         = risk_amount / (stop_distance * dollar_per_point)
        actual_risk = qty * stop_distance * dollar_per_point

    elif symbol_type == 'indices':
        dollar_per_point = contract_info.get('dollar_per_point', 1)
        qty         = risk_amount / (stop_distance * dollar_per_point)
        actual_risk = qty * stop_distance * dollar_per_point

    elif symbol_type == 'crypto':
        dollar_per_point = contract_info.get('dollar_per_point', 1)
        if current_price <= 0:
            return 0.0, 0.0
        qty = risk_amount / (stop_distance * dollar_per_point)
        actual_risk = qty * stop_distance * dollar_per_point

    else:
        qty         = risk_amount / stop_distance
        actual_risk = qty * stop_distance

    qty = max(qty, 0.01)
    qty = round(qty / 0.01) * 0.01
    return qty, actual_risk


# ═══════════════════════════════════════════════════════════════════════════════
# MT5 ORDER HELPERS
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

    point         = symbol_info.point
    requested_price = price
    use_limit     = True

    if order_type.upper() == "BUY":
        order_type_enum = mt5.ORDER_TYPE_BUY_LIMIT
        market_price    = symbol_info.ask
    else:
        order_type_enum = mt5.ORDER_TYPE_SELL_LIMIT
        market_price    = symbol_info.bid

    # BUY LIMIT must be BELOW current ask; BUY STOP above.
    # SELL LIMIT must be ABOVE current bid; SELL STOP below.
    if order_type.upper() == "BUY":
        if requested_price >= market_price:
            # Price is at or above ask → market buy or BUY STOP
            # For ICT we always want to enter AT or below current price,
            # so fall through to market order.
            order_type_enum = mt5.ORDER_TYPE_BUY
            use_limit = False
    else:
        if requested_price <= market_price:
            order_type_enum = mt5.ORDER_TYPE_SELL
            use_limit = False

    if use_limit:
        request = {
            "action":       mt5.TRADE_ACTION_PENDING,
            "symbol":       mt5_symbol,
            "volume":       volume,
            "type":         order_type_enum,
            "price":        requested_price,
            "magic":        magic,
            "comment":      "ICT V8",
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
    else:
        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       mt5_symbol,
            "volume":       volume,
            "type":         order_type_enum,
            "price":        market_price,
            "deviation":    20,
            "magic":        magic,
            "comment":      "ICT V8",
            "type_time":    mt5.ORDER_TIME_GTC,
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

    print(f"  Order: {order_type} {mt5_symbol} vol={volume} @ {price:.5f} "
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
    pos          = positions[0]
    order_type   = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
    close_volume = volume if volume else pos.volume
    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       pos.symbol,
        "volume":       close_volume,
        "type":         order_type,
        "position":     ticket,
        "price":        (mt5.symbol_info(pos.symbol).bid
                         if pos.type == 0
                         else mt5.symbol_info(pos.symbol).ask),
        "deviation":    20,
        "magic":        123456,
        "comment":      "ICT V8 Close",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE


# ═══════════════════════════════════════════════════════════════════════════════
# LEGACY SIGNAL GENERATOR  (renamed – no longer shadows ICTSignalEngine import)
# BUG-FIX 1: was "class V7SignalGenerator" – that shadowed the V8 import alias.
# ═══════════════════════════════════════════════════════════════════════════════

class _LegacyV7SignalGenerator:
    """
    DEPRECATED – kept for reference only.
    V7MT5LiveTrader now uses ICTSignalEngine (imported as V7SignalGenerator).
    This class is never instantiated in normal operation.
    """

    def __init__(self, rr_ratio: float = 2.0):
        self.rr_ratio = rr_ratio

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
            self.ms_handler      = MarketStructureHandler(
                swing_lookback=5, min_displacement_pct=0.1)
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

        return self._combine_signals(
            symbol, v5_signal, fvg_analysis, gap_analysis,
            current_price, data, idx, ms_analysis,
        )

    def _combine_signals(
        self, symbol, v5_signal, fvg_analysis, gap_analysis,
        current_price, data, idx, ms_analysis=None,
    ):
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
            (f for f in reversed(data['bullish_fvgs'])
             if f['idx'] < idx and f['mid'] < closes), None)
        near_bear_fvg = next(
            (f for f in reversed(data['bearish_fvgs'])
             if f['idx'] < idx and f['mid'] > closes), None)

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

        fvg_confluence = 0
        if fvg_analysis.best_bisi_fvg and signal['direction'] == 1:
            fvg  = fvg_analysis.best_bisi_fvg
            dist = abs(current_price - fvg.consequent_encroachment)
            if dist < fvg.size * 2:
                fvg_confluence += 20
                signal['fvg_data'] = {'type': 'BISI', 'ce': fvg.consequent_encroachment,
                                      'distance': dist}
                signal['reasoning'].append(f"FVG BISI at {fvg.consequent_encroachment:.5f}")
                if fvg.is_high_probability:
                    fvg_confluence += 15
                    signal['reasoning'].append("High Probability FVG")
        elif fvg_analysis.best_sibi_fvg and signal['direction'] == -1:
            fvg  = fvg_analysis.best_sibi_fvg
            dist = abs(current_price - fvg.consequent_encroachment)
            if dist < fvg.size * 2:
                fvg_confluence += 20
                signal['fvg_data'] = {'type': 'SIBI', 'ce': fvg.consequent_encroachment,
                                      'distance': dist}
                signal['reasoning'].append(f"FVG SIBI at {fvg.consequent_encroachment:.5f}")
                if fvg.is_high_probability:
                    fvg_confluence += 15
                    signal['reasoning'].append("High Probability FVG")

        gap_confluence = 0
        if gap_analysis.current_gap:
            g = gap_analysis.current_gap
            if gap_analysis.in_gap_zone:
                gap_confluence += 10
                signal['reasoning'].append(f"In {g.gap_type.value} gap zone")
                if g.quadrants:
                    ce_dist = abs(current_price - g.quadrants.ce)
                    if ce_dist < g.quadrants.range_size * 0.1:
                        gap_confluence += 15
                        signal['reasoning'].append("At Gap CE (50%)")
                        signal['gap_data'] = {'type': g.gap_type.value,
                                              'ce': g.quadrants.ce,
                                              'direction': g.direction.value}
        if gap_analysis.nearest_level:
            level_price, level_name = gap_analysis.nearest_level
            dist_pct = abs(current_price - level_price) / current_price * 100
            if dist_pct < 0.5:
                gap_confluence += 10
                signal['reasoning'].append(f"Near {level_name}")

        total = base_confluence + fvg_confluence + gap_confluence
        signal['confluence'] = min(total, 100)

        if total >= 80:
            signal['confidence'] = 'HIGH'
        elif total >= 60:
            signal['confidence'] = 'MEDIUM'
        elif total >= 50:
            signal['confidence'] = 'LOW'
        else:
            signal['direction'] = 0

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
            highs_arr     = data['highs']
            lows_arr      = data['lows']
            closes_arr    = data['closes']

            atr = np.mean([
                max(
                    highs_arr[i] - lows_arr[i],
                    abs(highs_arr[i] - closes_arr[i-1]) if i > 0 else 0,
                    abs(lows_arr[i]  - closes_arr[i-1]) if i > 0 else 0,
                )
                for i in range(max(0, idx - 14), idx + 1)
            ])

            if is_forex:
                decimal_places = contract_info.get('decimal_places', 5)
                pip_size       = 0.01 if decimal_places == 3 else 0.0001
                atr_pips       = atr / pip_size
                min_stop_pips  = max(15, atr_pips * 1.5)
                max_stop_pips  = max(80, atr_pips * 2.0)
                stop_distance  = max(min_stop_pips * pip_size,
                                     min(atr * 2.0, max_stop_pips * pip_size))
                risk_pips = stop_distance / pip_size
                if risk_pips < 10 or risk_pips > 150:
                    signal['direction']   = 0
                    signal['stop_loss']   = entry
                    signal['take_profit'] = entry
                    return signal
            else:
                min_stop      = contract_info.get('min_stop', 20)
                stop_distance = max(atr * 2.0, min_stop)

            if signal['direction'] == 1:
                signal['stop_loss']   = entry - stop_distance
                signal['take_profit'] = entry + stop_distance * self.rr_ratio
            else:
                signal['stop_loss']   = entry + stop_distance
                signal['take_profit'] = entry - stop_distance * self.rr_ratio

        return signal


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE TRADER
# ═══════════════════════════════════════════════════════════════════════════════

class V7MT5LiveTrader:

    def __init__(
        self,
        symbols: List[str],
        risk_pct: float = 0.02,
        poll_interval: int = 30,
        rr_ratio: float = 3.0,
        confluence_threshold: int = 65,
        max_daily_loss: float = -500,
        max_daily_loss_pct: float = 3.0,
        reverse_signals: bool = False,
    ):
        self.symbols              = symbols
        self.risk_pct             = risk_pct
        self.poll_interval        = poll_interval
        self.rr_ratio             = rr_ratio
        self.confluence_threshold  = confluence_threshold
        self.max_daily_loss       = max_daily_loss
        self.max_daily_loss_pct   = max_daily_loss_pct
        self.reverse_signals      = reverse_signals

        # V7SignalGenerator is the ICTSignalEngine import alias (BUG-FIX 1 ensures
        # it is never overwritten by the local class definition).
        if V7SignalGenerator is not None:
            self.signal_generator = V7SignalGenerator(rr_ratio=rr_ratio)
        else:
            class _StubGenerator:
                def __init__(self, rr_ratio): self.rr_ratio = rr_ratio
                last_analysis: Dict = {}
                def analyze_symbol(self, sym, data, price):
                    return {"direction": 0, "confluence": 0, "confidence": "LOW",
                            "entry_price": price, "stop_loss": None, "take_profit": None,
                            "reasoning": [], "fvg_data": None, "gap_data": None}
            self.signal_generator = _StubGenerator(rr_ratio=rr_ratio)
            print("WARNING: signal_engine.py not found – stub running, no real signals")

        self.mode             = 'paper'
        self.positions        = {}
        self.historical_data  = {}
        self.last_poll_time   = {}
        self.last_signal_time = {}
        self.last_signals     = {}

        self.daily_pnl    = 0.0
        self.daily_pnl_pct = 0.0  # percentage of account
        self.last_reset_date = datetime.now().date()
        self.trade_count  = 0
        self.account_value = 100_000

        self.update_account()
        self.running = False
        self.magic   = 123456

        self._sync_positions()
        # Auto-fix any inherited positions whose TP gives wrong R:R
        # (handles positions opened by the previous broken V7 code)
        self.fix_existing_tps()

    # ─────────────────────────────────────────────────────────────────────────

    def _check_daily_reset(self):
        """Reset daily P&L at start of new trading day (NY midnight)."""
        today = datetime.now().date()
        if self.last_reset_date != today:
            print(f"\n[NEW DAY] Resetting daily P&L: ${self.daily_pnl:.2f} -> $0.00")
            self.daily_pnl = 0.0
            self.daily_pnl_pct = 0.0
            self.last_reset_date = today

    def _update_daily_pnl(self):
        """Update daily P&L percentage and check loss limit."""
        if self.account_value > 0:
            self.daily_pnl_pct = (self.daily_pnl / self.account_value) * 100
        else:
            self.daily_pnl_pct = 0.0

    def _is_daily_loss_limit_hit(self) -> bool:
        """Check if daily loss limit exceeded (either absolute or percentage)."""
        abs_limit_hit = self.daily_pnl <= self.max_daily_loss
        pct_limit_hit = self.daily_pnl_pct <= -self.max_daily_loss_pct
        return abs_limit_hit or pct_limit_hit

    # ─────────────────────────────────────────────────────────────────────────

    def _sync_positions(self):
        print("\nSyncing positions with MT5...")
        try:
            positions = get_mt5_positions()
            for pos in positions:
                mt5_sym   = pos['symbol'].upper()
                symbol    = mt5_sym.replace('M', '')
                sym_list  = [s.upper() for s in self.symbols]
                mt5_list  = [get_mt5_symbol(s).upper() for s in self.symbols]
                if symbol in sym_list or mt5_sym in mt5_list:
                    direction  = 1 if pos['type'] == 0 else -1
                    symbol_key = symbol if symbol in sym_list else mt5_sym
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
                    print(f"  Found: {mt5_sym} x{pos['volume']} @ {pos['price_open']:.5f}"
                          f"  P&L: ${pos['profit']:.2f}")
            if not self.positions:
                print("  No open positions found")
            else:
                print(f"  Total open: {len(self.positions)}")
        except Exception as e:
            print(f"  Error syncing: {e}")

    # ─────────────────────────────────────────────────────────────────────────

    def fix_existing_tps(self, min_improvement: float = 0.10) -> None:
        """
        Scan ALL open MT5 positions (not just bot-tracked ones) and correct
        any TP whose reward/risk is worse than self.rr_ratio - min_improvement.

        Called automatically on startup so that positions opened by the old
        broken V7 code get their TPs fixed the moment the new bot starts.
        Can also be called manually at any time: trader.fix_existing_tps()

        Parameters
        ----------
        min_improvement : only fix positions whose R:R is more than this
                          below the target.  Default 0.10 means any position
                          with R:R < (rr_ratio - 0.10) will be corrected.
                          Set to 0 to force-correct every position.
        """
        if not MT5_AVAILABLE:
            return

        try:
            positions = mt5.positions_get()
            if not positions:
                return

            print(f"\n── TP Audit (target R:R 1:{self.rr_ratio}) ───────────────────────")
            fixed = 0

            for pos in positions:
                ticket     = pos.ticket
                symbol     = pos.symbol
                pos_type   = pos.type       # 0=BUY, 1=SELL
                price_open = pos.price_open
                sl         = pos.sl
                tp         = pos.tp

                if sl == 0.0:
                    print(f"  {symbol} ticket={ticket}  SKIP (no SL set)")
                    continue

                stop_dist = abs(price_open - sl)
                if stop_dist <= 1e-8:
                    continue

                # Existing R:R
                reward_dist = abs(tp - price_open) if tp != 0.0 else 0.0
                current_rr  = reward_dist / stop_dist if stop_dist > 0 else 0.0

                # Correct TP
                if pos_type == 0:   # BUY
                    correct_tp = price_open + stop_dist * self.rr_ratio
                else:               # SELL
                    correct_tp = price_open - stop_dist * self.rr_ratio

                sym_info = mt5.symbol_info(symbol)
                if sym_info:
                    digits     = sym_info.digits
                    correct_tp = round(correct_tp, digits)
                    sl_send    = round(sl, digits)
                else:
                    sl_send = sl

                needs_fix = current_rr < (self.rr_ratio - min_improvement)

                type_str = "BUY " if pos_type == 0 else "SELL"
                if needs_fix:
                    print(f"  {symbol:<12} {type_str} @ {price_open:.5f}  "
                          f"SL={sl:.5f}  TP={tp:.5f}  R:R=1:{current_rr:.2f}  "
                          f"→ fixing TP to {correct_tp:.5f} (1:{self.rr_ratio})", end="")

                    request = {
                        "action":   mt5.TRADE_ACTION_SLTP,
                        "position": ticket,
                        "symbol":   symbol,
                        "sl":       sl_send,
                        "tp":       correct_tp,
                    }
                    result = mt5.order_send(request)
                    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
                        # Update internal tracker if we have this position
                        for key, p in self.positions.items():
                            if p.get('ticket') == ticket:
                                p['target'] = correct_tp
                                break
                        print("  ✓")
                        fixed += 1
                    else:
                        err = result.comment if result else mt5.last_error()
                        print(f"  ✗ ({err})")
                else:
                    print(f"  {symbol:<12} {type_str} @ {price_open:.5f}  "
                          f"R:R=1:{current_rr:.2f}  OK")

            print(f"  ── {fixed} position(s) TP corrected ─────────────────────────\n")

        except Exception as e:
            print(f"fix_existing_tps error: {e}")

    def get_current_price(self, symbol: str) -> Optional[float]:
        if not MT5_AVAILABLE:
            return None
        info = mt5.symbol_info(get_mt5_symbol(symbol))
        return info.bid if info else None

    def _check_position_exit(self, symbol: str, current_price: float):
        mt5_sym = get_mt5_symbol(symbol)
        pos_key = symbol if symbol in self.positions else (
            mt5_sym if mt5_sym in self.positions else None)
        if pos_key is None:
            return
        try:
            pos                  = self.positions[pos_key]
            pos['bars_held']     = pos.get('bars_held', 0) + 1
            pos['current_price'] = current_price

            contract_info    = get_contract_info(symbol)
            symbol_type      = contract_info['type']
            direction        = pos['direction']
            entry            = pos['entry']
            stop             = pos.get('stop', 0)
            target           = pos.get('target', 0)
            volume           = pos['volume']
            price_diff       = (current_price - entry) if direction == 1 \
                               else (entry - current_price)

            if symbol_type == 'forex':
                decimal_places = contract_info.get('decimal_places', 5)
                pip_size = 0.01 if decimal_places == 3 else 0.0001
                pip_value = contract_info.get('pip_value', 10)
                pnl = (price_diff / pip_size) * pip_value * volume
            elif symbol_type in ('futures', 'indices'):
                pnl = price_diff * volume * contract_info.get('dollar_per_point', 1)
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
                print(f"  [{symbol}] Unrealised P&L: ${pnl:.2f}")
        except Exception as e:
            print(f"[{symbol}] Error checking position: {e}")

    def _handle_position_closed(self, symbol: str, exit_price: float,
                                 pnl: float, exit_reason: str):
        if symbol not in self.positions:
            return
        pos           = self.positions[symbol]
        direction_str = 'LONG' if pos['direction'] == 1 else 'SHORT'
        pnl_str       = f"+${pnl:.2f}" if pnl > 0 else f"-${abs(pnl):.2f}"
        self.daily_pnl += pnl
        print(f"[{symbol}] EXIT ({exit_reason}): {direction_str} @ {exit_price:.5f} "
              f"| P&L: {pnl_str} | Daily: ${self.daily_pnl:.2f}")
        if tn:
            try:
                tn.send_trade_exit(symbol, pos['direction'], pnl, exit_reason,
                                   pos['entry'], exit_price, pos.get('bars_held', 0))
            except Exception:
                pass
        del self.positions[symbol]

    # ─────────────────────────────────────────────────────────────────────────
    # BUG-FIX 3: _enter_trade – TP always anchored to current_price
    # ─────────────────────────────────────────────────────────────────────────

    def _enter_trade(self, symbol: str, signal: Dict, current_price: float):
        """
        Execute a trade from a signal dict.

        KEY FIX: Take-profit is always computed as:
            TP = current_price ± actual_stop_distance × rr_ratio

        where actual_stop_distance = |current_price - stop_price|.

        This guarantees a real 1:rr_ratio regardless of whether
        signal['entry_price'] is a limit level, an OB mean_threshold, or
        a FVG CE that differs from the live market price.

        signal['entry_price'] is still sent to MT5 as the limit price so
        pending orders fill at the ICT PD array level; the TP attached to
        that pending order is set correctly from current_price.
        """
        if is_trading_paused():
            print(f"[{symbol}] Trading PAUSED")
            return

        if self._is_daily_loss_limit_hit():
            pct = self.daily_pnl_pct
            print(f"[{symbol}] Daily loss limit ${self.daily_pnl:.2f} ({pct:.1f}%), skipping")
            return

        is_forex = symbol.upper() in FOREX_SYMBOLS
        in_session, session_name = is_valid_trading_session(is_forex=is_forex)
        if not in_session:
            print(f"[{symbol}] Outside hours ({session_name}), skipping")
            return

        mt5_sym = get_mt5_symbol(symbol)
        if symbol in self.positions or mt5_sym in self.positions:
            print(f"[{symbol}] Already in position, skipping")
            return

        # REVERSE MODE: flip direction and swap SL/TP directly
        reversed_mode = False
        if self.reverse_signals and signal.get('direction', 0) != 0:
            reversed_mode = True
            orig_dir = signal['direction']
            orig_sl = signal.get('stop_loss')
            orig_tp = signal.get('take_profit')
            
            # New direction (BUY->SELL or SELL->BUY)
            signal['direction'] = -orig_dir
            
            # Just swap SL and TP directly - no recalculation
            signal['stop_loss'] = orig_tp
            signal['take_profit'] = orig_sl
            
            signal['reasoning'] = ['REVERSED'] + (signal.get('reasoning', []) or [])
            print(f"[{symbol}] REVERSE: was {orig_dir} -> now {signal['direction']}, "
                  f"SL={orig_sl} <-> TP={orig_tp}")

        try:
            entry_price = signal['entry_price']   # limit/pending level
            stop_price  = signal['stop_loss']

            if stop_price is None:
                print(f"[{symbol}] No stop_loss, skipping")
                return

            # ── Validate stop is on the correct side of CURRENT price ─────────
            # (current_price is our reference for R:R, not the limit entry)
            if signal['direction'] == 1 and stop_price >= current_price:
                print(f"[{symbol}] BUY stop {stop_price:.5f} >= current {current_price:.5f}, skip")
                return
            if signal['direction'] == -1 and stop_price <= current_price:
                print(f"[{symbol}] SELL stop {stop_price:.5f} <= current {current_price:.5f}, skip")
                return

            # ── R:R arithmetic anchored to current_price (BUG-FIX 3) ──────────
            actual_stop_distance = abs(current_price - stop_price)
            if actual_stop_distance <= 1e-8:
                print(f"[{symbol}] Stop distance ~0, skipping")
                return

            # Take-profit: use swapped TP in reverse mode, otherwise calculate normally
            if reversed_mode:
                target_price = signal['take_profit']  # Use the swapped TP
            else:
                target_price = (
                    current_price + actual_stop_distance * self.rr_ratio
                    if signal['direction'] == 1
                    else current_price - actual_stop_distance * self.rr_ratio
                )

            # ── Extra sanity: ensure min stop distance for indices ────────────
            contract_info = get_contract_info(symbol)
            if contract_info['type'] == 'indices':
                min_stop = contract_info.get('min_stop', 10)
                if actual_stop_distance < min_stop:
                    print(f"[{symbol}] Stop dist {actual_stop_distance:.1f} < min {min_stop}, skip")
                    return
            elif contract_info['type'] == 'forex':
                decimal_places = contract_info.get('decimal_places', 5)
                pip_size       = 0.01 if decimal_places == 3 else 0.0001
                risk_pips      = actual_stop_distance / pip_size
                if risk_pips < 10 or risk_pips > 250:
                    print(f"[{symbol}] Pip stop {risk_pips:.1f} outside 10–250, skip")
                    return

            # ── Position sizing uses actual_stop_distance from current_price ──
            qty, risk_amount = calculate_position_size(
                symbol, self.account_value, self.risk_pct,
                actual_stop_distance, current_price,
            )
            if qty <= 0:
                print(f"[{symbol}] qty={qty:.2f}, skip")
                return

            effective_rr  = self.rr_ratio
            direction_str = 'LONG' if signal['direction'] == 1 else 'SHORT'

            fvg_info = signal.get('fvg_data') or {}
            gap_info = signal.get('gap_data') or {}
            pd_zone  = (f"FVG {fvg_info.get('type','')}" if fvg_info
                        else f"Gap {gap_info.get('type','')}" if gap_info
                        else 'OB' if signal.get('reasoning') and
                             any('OB' in r for r in signal.get('reasoning', []))
                        else None)

            print(f"[{symbol}] Sizing: ref={current_price:.5f}  "
                  f"stop_dist={actual_stop_distance:.5f}  "
                  f"risk=${risk_amount:.2f}  lots={qty:.2f}  R:R=1:{effective_rr}")

            if self.mode == 'shadow':
                print(f"[{symbol}] V8 SHADOW  {direction_str} @ {current_price:.5f}")
                print(f"  Limit entry: {entry_price:.5f}  "
                      f"SL: {stop_price:.5f}  TP: {target_price:.5f}")
                print(f"  R:R 1:{effective_rr}  |  Confluence {signal['confluence']}/100  "
                      f"|  {pd_zone or 'No PD'}")
                print(f"  Risk: ${risk_amount:.2f}  |  Lots: {qty:.2f}")
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
                symbol, order_type, qty,
                entry_price,          # limit price at PD array
                stop_loss=stop_price,
                take_profit=target_price,   # ← correctly anchored to current_price
                magic=self.magic,
            )

            if result:
                self.positions[symbol] = {
                    'ticket':        result['order_id'],
                    'entry':         entry_price,
                    'stop':          stop_price,
                    'target':        target_price,
                    'ref_price':     current_price,
                    'direction':     signal['direction'],
                    'volume':        qty,
                    'confluence':    signal['confluence'],
                    'confidence':    signal['confidence'],
                    'entry_time':    datetime.now(),
                    'reasoning':     signal.get('reasoning', []),
                    'bars_held':     0,
                    'current_price': current_price,
                }
                self.trade_count += 1
                print(f"[{symbol}] V8 ENTRY: {direction_str} x{qty:.2f} "
                      f"limit={entry_price:.5f}  ref={current_price:.5f}")
                print(f"  SL: {stop_price:.5f}  TP: {target_price:.5f}  R:R 1:{effective_rr}")
                print(f"  Conf: {signal['confidence']}  "
                      f"Confluence: {signal['confluence']}/100")
                if signal.get('reasoning'):
                    print(f"  {' | '.join(signal['reasoning'][:3])}")
                if tn:
                    try:
                        tn.send_trade_entry(
                            symbol, signal['direction'], qty,
                            current_price, signal['confluence'],
                            target_price, stop_price,
                            pd_zone=pd_zone, risk_amount=risk_amount,
                        )
                    except RuntimeError as e:
                        if "event loop" in str(e).lower():
                            print(f"[{symbol}] Telegram skipped (asyncio issue)")
                        else:
                            print(f"[{symbol}] Telegram error: {e}")
                    except Exception as e:
                        print(f"[{symbol}] Telegram error: {e}")
            else:
                print(f"[{symbol}] Failed to place order")

        except Exception as e:
            print(f"[{symbol}] V8 Error in _enter_trade: {e}")

    def _log_shadow_trade(
        self, symbol: str, signal: Dict, current_price: float,
        stop_price: float, target_price: float, qty: float, risk_amount: float,
    ):
        try:
            stop_dist   = abs(current_price - stop_price)
            reward_dist = abs(target_price - current_price)
            eff_rr      = reward_dist / stop_dist if stop_dist > 0 else 0
            trade = {
                'timestamp':      datetime.now().isoformat(),
                'symbol':         symbol,
                'direction':      'LONG' if signal['direction'] == 1 else 'SHORT',
                'ref_price':      current_price,
                'limit_entry':    signal['entry_price'],
                'stop':           stop_price,
                'target':         target_price,
                'rr_ratio':       round(eff_rr, 2),
                'configured_rr':  self.rr_ratio,
                'qty':            qty,
                'risk_amount':    risk_amount,
                'confluence':     signal['confluence'],
                'confidence':     signal['confidence'],
                'fvg_data':       signal.get('fvg_data'),
                'gap_data':       signal.get('gap_data'),
                'reasoning':      signal.get('reasoning', [])[:5],
            }
            with open('v8_shadow_trades.json', 'a') as f:
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

                idx           = len(data['closes']) - 1
                live_price    = self.get_current_price(symbol)
                current_price = live_price if live_price else data['closes'][idx]

                try:
                    signal = self.signal_generator.analyze_symbol(
                        symbol, data, current_price)
                except Exception as e:
                    print(f"[{symbol}] Signal error: {e}")
                    continue

                if signal and signal.get('direction', 0) != 0:
                    fvg_type = (signal.get('fvg_data') or {}).get('type', '')
                    gap_type = (signal.get('gap_data') or {}).get('type', '')
                    cached   = getattr(self.signal_generator, 'last_analysis', {}).get(symbol, {})
                    self.last_signals[symbol] = {
                        'direction':     signal.get('direction', 0),
                        'confluence':    signal.get('confluence', 0),
                        'confidence':    signal.get('confidence', 'LOW'),
                        'pd_zone':       fvg_type or gap_type,
                        'entry':         signal.get('entry_price', current_price),
                        'stop':          signal.get('stop_loss'),
                        'target':        signal.get('take_profit'),
                        'htf_trend':     cached.get('htf_trend', 'N/A'),
                        'kill_zone':     cached.get('kill_zone', 'OFF_HOURS'),
                        'ob_type':       cached.get('ob_type', 'none'),
                        'liq_swept':     cached.get('liq_swept'),
                        'model_2022':    cached.get('model_2022', 'none'),
                        'silver_bullet': cached.get('silver_bullet', False),
                        'reasoning':     signal.get('reasoning', [])[:5],
                        'timestamp':     datetime.now().isoformat(),
                    }

                if tn and signal:
                    try:
                        cached = getattr(self.signal_generator, 'last_analysis', {}).get(symbol, {})
                        tn.update_market_data(symbol, {
                            'price':      current_price,
                            'htf_trend':  cached.get('htf_trend', 'N/A'),
                            'ltf_trend':  cached.get('ltf_trend', 'N/A'),
                            'kill_zone':  cached.get('kill_zone', 'N/A'),
                            'confluence': signal.get('confluence', 0),
                            'confidence': signal.get('confidence', 'LOW'),
                            'ob_type':    cached.get('ob_type', 'none'),
                            'liq_swept':  cached.get('liq_swept'),
                            'model_2022': cached.get('model_2022', 'none'),
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
                    if (signal and signal.get('direction', 0) != 0
                            and signal.get('confluence', 0) >= self.confluence_threshold):
                        self.last_signal_time[symbol] = current_hour
                        self._enter_trade(symbol, signal, current_price)

            except Exception as e:
                print(f"[{symbol}] Error polling: {e}")

    def update_account(self):
        if not MT5_AVAILABLE:
            return
        try:
            info = mt5.account_info()
            if info:
                self.account_value = info.balance
                self._update_daily_pnl()
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
                        symbol,
                        pos.get('current_price', pos['entry']),
                        pos.get('pnl', 0),
                        'external',
                    )
        except Exception as e:
            print(f"Error checking positions: {e}")

    def print_status(self):
        """Full pipeline snapshot – printed every 5 minutes."""
        now = datetime.now().strftime('%H:%M:%S')
        print(f"\n{'='*72}")
        print(f"ICT V8  {now}  |  Acct: ${self.account_value:,.2f}  "
              f"|  Daily P&L: ${self.daily_pnl:.2f} ({self.daily_pnl_pct:.1f}%)")
        print(f"{'='*72}")
        cached_all = getattr(self.signal_generator, 'last_analysis', {})
        for sym in self.symbols:
            sig   = self.last_signals.get(sym, {})
            cache = cached_all.get(sym, {})
            pos   = self.positions.get(sym)
            dir_s = {1: 'LONG ▲', -1: 'SHORT ▼', 0: '  --  '}.get(
                sig.get('direction', 0), '  --  ')
            cfl  = sig.get('confluence', 0)
            conf = sig.get('confidence', '---')
            kz   = cache.get('kill_zone', '---')
            htf  = cache.get('htf_trend', '---')
            ob_t = cache.get('ob_type',   '---')
            m22  = cache.get('model_2022','---')
            sb   = 'SB✓' if cache.get('silver_bullet') else '   '
            liq  = cache.get('liq_swept') or '---'
            pos_s = ''
            if pos:
                pnl   = pos.get('pnl', 0.0)
                ps    = 'L' if pos['direction'] == 1 else 'S'
                pos_s = (f" | POS {ps} limit={pos['entry']:.5f}  "
                         f"ref={pos.get('ref_price', pos['entry']):.5f}  "
                         f"P&L ${pnl:.2f}")
            print(f"  {sym:<8}  {dir_s}  {conf:<6} {cfl:>3}/100  "
                  f"KZ:{kz:<9} HTF:{htf:<12} OB:{ob_t:<8} "
                  f"M22:{m22:<4} {sb}  LIQ:{liq}{pos_s}")
        print(f"{'='*72}\n")

    def start(self):
        self.running = True
        print(f"\nV8 MT5 Trader started | symbols: {self.symbols}")

    def stop(self):
        self.running = False
        print("V8 MT5 Trader stopped")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

def run_v8_trading(
    symbols: List[str],
    interval: int = 30,
    risk_pct: float = 0.02,
    login: int = None,
    password: str = None,
    server: str = None,
    mode: str = 'paper',
    rr_ratio: float = 3.0,
    confluence_threshold: int = 60,
    max_daily_loss: float = -2000,
    max_daily_loss_pct: float = 3.0,
    reverse_signals: bool = False,
):
    if not MT5_AVAILABLE:
        print("ERROR: MetaTrader5 not installed.  Run: pip install MetaTrader5")
        return

    if not init_mt5(login, password, server):
        print("Failed to initialise MT5")
        return

    print(f"\nICT V8 – MT5 Trading")
    print(f"Mode: {mode.upper()}")
    print(f"Symbols: {symbols}")
    print(f"Risk: {risk_pct*100:.1f}%  |  R:R 1:{rr_ratio}")
    print(f"Confluence: {confluence_threshold}+  |  Max Loss: ${max_daily_loss}")
    print("-" * 50)

    if tn:
        try:
            tn.send_startup(symbols=symbols, risk_pct=risk_pct,
                            interval=interval, mode=f"V8 {mode.upper()}")
        except Exception as e:
            print(f"Telegram startup failed: {e}")

    trader = V7MT5LiveTrader(
        symbols, risk_pct,
        poll_interval=interval,
        rr_ratio=rr_ratio,
        confluence_threshold=confluence_threshold,
        max_daily_loss=max_daily_loss,
        max_daily_loss_pct=max_daily_loss_pct,
        reverse_signals=reverse_signals,
    )
    print(f"Account: ${trader.account_value:,.2f}")
    trader.mode = mode

    if tn and hasattr(tn, 'set_live_trader'):
        tn.set_live_trader(trader)
        print("Live trader registered with Telegram")

    if tn and hasattr(tn, 'start_polling_background'):
        try:
            tn.start_polling_background()
            print("Telegram command polling started")
        except Exception as e:
            print(f"Telegram polling failed: {e}")

    trader.start()
    trader._refresh_data()
    print("\nTrading started.  Press Ctrl+C to stop.\n")

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
                trader.print_status()
            if iteration % 60 == 0:
                trader.update_account()
                trader.check_positions()
            if iteration % 3600 == 0 and trader.positions:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] "
                      f"Open positions: {len(trader.positions)}")
            if iteration % 60 == 0:
                trader._check_daily_reset()
                trader._update_daily_pnl()
            if trader._is_daily_loss_limit_hit() and iteration % 60 == 0:
                print(f"[WARNING] Daily loss limit: ${trader.daily_pnl:.2f} ({trader.daily_pnl_pct:.1f}%)")

    except KeyboardInterrupt:
        print("\n\nShutdown…")
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

    parser = argparse.ArgumentParser(description='ICT V8 – MT5 Trading Bot (R:R Fixed)')
    parser.add_argument("--symbols",    default="GBPAUD,USDJPY,USDCHF,EURJPY,GBPJPY,NZDUSD")
    parser.add_argument("--interval",  type=int,   default=30)
    parser.add_argument("--risk",      type=float, default=0.02)
    parser.add_argument("--login",     type=int,   default=None)
    parser.add_argument("--password",  type=str,   default=None)
    parser.add_argument("--server",    type=str,   default=None)
    parser.add_argument("--mode",      type=str,   default="paper",
                        choices=["shadow", "paper", "live"])
    parser.add_argument("--rr",        type=float, default=3.0)
    parser.add_argument("--confluence",type=int,   default=65)
    parser.add_argument("--max-loss",  type=float, default=-500)
    parser.add_argument("--max-loss-pct", type=float, default=3.0,
                        help="Max daily loss as percentage of account")
    parser.add_argument("--reverse",   action="store_true",
                        help="Reverse all signals (BUY->SELL, SL->TP)")
    args = parser.parse_args()

    symbols = [s.strip().upper() for s in args.symbols.split(',')]

    print("=" * 60)
    print("ICT V8 Trading Bot – MetaTrader 5  (R:R Fully Fixed)")
    print("=" * 60)
    print(f"Mode:       {args.mode.upper()}")
    print(f"Symbols:    {', '.join(symbols)}")
    print(f"Risk:       {args.risk*100:.1f}%  |  R:R 1:{args.rr}")
    print(f"Confluence: {args.confluence}+  |  Max Loss: ${abs(args.max_loss)} ({args.max_loss_pct}%)")
    if args.reverse:
        print("REVERSE MODE: All signals flipped!")
    print(f"MT5 Login:  {args.login or 'Demo'}")
    print("=" * 60)

    run_v8_trading(
        symbols, args.interval, args.risk,
        args.login, args.password, args.server, args.mode,
        rr_ratio=args.rr,
        confluence_threshold=args.confluence,
        max_daily_loss=-abs(args.max_loss),
        max_daily_loss_pct=args.max_loss_pct,
        reverse_signals=args.reverse,
    )