"""
ICT V7 Trading Bot - MT5 Version
=================================
Combines V6 FVG + Gap Analysis with MetaTrader 5 connectivity.
Full Telegram integration for commands and notifications.

Usage:
    python3 ict_v7_mt5.py --symbols "BTCUSD,ETHUSD,XAUUSD,XTIUSD" --login 12345 --password "yourpass"
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


FOREX_SYMBOLS = {'EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD', 'AUDUSD', 'USDCHF', 'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY', 'XAUUSD', 'XAGUSD'}

MT5_SYMBOLS = {
    # Crypto
    'BTCUSD': 'BTCUSDm', 'ETHUSD': 'ETHUSDm', 'SOLUSD': 'SOLUSDm',
    # Metals (Gold & Silver)
    'XAUUSD': 'XAUUSDm', 'XAGUSD': 'XAGUSDm',
    # Oil
    'XTIUSD': 'USOILm', 'XBRUSD': 'UKOILm', 'XNGUSD': 'XNGUSDm',
    # Major Forex
    'EURUSD': 'EURUSDm', 'GBPUSD': 'GBPUSDm', 'USDJPY': 'USDJPYm',
    'USDCAD': 'USDCADm', 'AUDUSD': 'AUDUSDm', 'USDCHF': 'USDCHFm', 'NZDUSD': 'NZDUSDm',
    # Forex Crosses
    'EURGBP': 'EURGBPm', 'EURJPY': 'EURJPYm', 'GBPJPY': 'GBPJPYm',
    'EURAUD': 'EURAUDm', 'EURCAD': 'EURCADm', 'GBPAUD': 'GBPAUDm',
    'AUDJPY': 'AUDJPYm', 'CADJPY': 'CADJPYm', 'CHFJPY': 'CHFJPYm',
    # Indices
    'US30': 'US30m', 'USTEC': 'USTECm', 'US500': 'US500m', 'UK100': 'UK100m', 'AUS200': 'AUS200m',
}

# Exness-specific mapping (fallback)
EXNESS_SYMBOLS = MT5_SYMBOLS.copy()

def get_mt5_symbol(symbol: str) -> str:
    """Convert symbol to MT5 format with suffix"""
    s = symbol.upper()
    if s in MT5_SYMBOLS:
        return MT5_SYMBOLS[s]
    # Try Exness-specific
    if s in EXNESS_SYMBOLS:
        return EXNESS_SYMBOLS[s]
    return s + 'm'
CONVERT_THRESHOLD_GBP = 500


def is_trading_paused() -> bool:
    if tn and hasattr(tn, 'is_trading_paused'):
        return tn.is_trading_paused()
    return False


def is_valid_trading_session(is_forex: bool = False) -> Tuple[bool, str]:
    import pytz
    from datetime import time as dt_time
    
    # Skip session check - allow all hours
    return True, "24/7"
    
    et_tz = pytz.timezone('US/Eastern')
    now_et = datetime.now(et_tz)
    current_time = now_et.time()
    
    if is_forex:
        # Extended forex session - 24 hours during week
        # Current hour
        hour = now_et.weekday()
        if hour < 5:  # Mon-Fri
            return True, "FOREX"
        return False, "CLOSED"
    
    london_open = dt_time(2, 0)
    london_close = dt_time(17, 0)
    
    ny_open = dt_time(9, 30)
    ny_close = dt_time(16, 15)
    
    if london_open <= current_time <= london_close:
        return True, "LONDON"
    
    if ny_open <= current_time <= ny_close:
        return True, "NY"
    
    return False, "CLOSED"


def get_quote_currency(symbol: str) -> str:
    if len(symbol) >= 6:
        return symbol[-3:]
    return ''


def init_mt5(login: int = None, password: str = None, server: str = None) -> bool:
    """Initialize MT5 connection."""
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
    """Get symbol info from MT5."""
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
    """Get comprehensive contract information for position sizing."""
    symbol = symbol.upper()
    
    futures_info = {
        'XTIUSD': {'multiplier': 1000, 'min_stop': 0.50, 'tick_size': 0.01},  # Crude Oil
        'XBRUSD': {'multiplier': 1000, 'min_stop': 0.50, 'tick_size': 0.01},  # Brent
        'XNGUSD': {'multiplier': 10000, 'min_stop': 0.03, 'tick_size': 0.001}, # Natural Gas
        'XAUUSD': {'multiplier': 100, 'min_stop': 10, 'tick_size': 0.01},     # Gold
        'XAGUSD': {'multiplier': 5000, 'min_stop': 0.50, 'tick_size': 0.001}, # Silver
    }
    
    forex_info = {
        'EURUSD': {'pip_value': 10, 'min_stop': 0.0050, 'decimal_places': 5},
        'GBPUSD': {'pip_value': 10, 'min_stop': 0.0050, 'decimal_places': 5},
        'USDJPY': {'pip_value': 9.1, 'min_stop': 0.50, 'decimal_places': 3},
        'AUDUSD': {'pip_value': 10, 'min_stop': 0.0050, 'decimal_places': 5},
        'USDCAD': {'pip_value': 7.5, 'min_stop': 0.0050, 'decimal_places': 5},
        'USDCHF': {'pip_value': 10.8, 'min_stop': 0.0050, 'decimal_places': 5},
        'NZDUSD': {'pip_value': 10, 'min_stop': 0.0050, 'decimal_places': 5},
        'GBPJPY': {'pip_value': 9.1, 'min_stop': 0.50, 'decimal_places': 3},
        'EURJPY': {'pip_value': 9.1, 'min_stop': 0.50, 'decimal_places': 3},
        'EURGBP': {'pip_value': 10, 'min_stop': 0.0050, 'decimal_places': 5},
    }
    
    crypto_info = {
        'BTCUSD': {'multiplier': 1, 'min_stop_pct': 0.015, 'tick_size': 0.01},
        'ETHUSD': {'multiplier': 1, 'min_stop_pct': 0.015, 'tick_size': 0.01},
        'SOLUSD': {'multiplier': 1, 'min_stop_pct': 0.02, 'tick_size': 0.01},
        'LTCUSD': {'multiplier': 1, 'min_stop_pct': 0.02, 'tick_size': 0.01},
    }
    
    indices_info = {
        'US30': {'multiplier': 1, 'min_stop': 30, 'tick_size': 1},    # Dow Jones
        'US100': {'multiplier': 1, 'min_stop': 15, 'tick_size': 0.1}, # Nasdaq 100
        'US500': {'multiplier': 1, 'min_stop': 5, 'tick_size': 0.1},  # S&P 500
        'GER40': {'multiplier': 1, 'min_stop': 20, 'tick_size': 0.1},  # DAX
        'UK100': {'multiplier': 1, 'min_stop': 10, 'tick_size': 0.1},  # FTSE 100
        'FRA40': {'multiplier': 1, 'min_stop': 10, 'tick_size': 0.1},  # CAC 40
        'JPN225': {'multiplier': 1, 'min_stop': 100, 'tick_size': 1},   # Nikkei 225
        'AUS200': {'multiplier': 1, 'min_stop': 10, 'tick_size': 0.1}, # ASX 200
    }
    
    if symbol in futures_info:
        info = futures_info[symbol].copy()
        info['type'] = 'futures'
        return info
    elif symbol in forex_info:
        info = forex_info[symbol].copy()
        info['type'] = 'forex'
        return info
    elif symbol in crypto_info:
        info = crypto_info[symbol].copy()
        info['type'] = 'crypto'
        return info
    elif symbol in indices_info:
        info = indices_info[symbol].copy()
        info['type'] = 'indices'
        return info
    else:
        return {'type': 'forex', 'pip_value': 10, 'min_stop': 0.0050, 'decimal_places': 5}


def fetch_mt5_rates(symbol: str, timeframe: str = "H1", num_bars: int = 500) -> Optional[pd.DataFrame]:
    """Fetch historical rates from MT5."""
    if not MT5_AVAILABLE:
        return None
    
    timeframe_map = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
        "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1,
    }
    
    mt5_timeframe = timeframe_map.get(timeframe, mt5.TIMEFRAME_H1)
    
    # Select symbol first
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
    """Prepare data from MT5."""
    symbol = symbol.upper()
    mt5_symbol = get_mt5_symbol(symbol)
    
    df = fetch_mt5_rates(mt5_symbol, "H1", num_bars=lookback + 50)
    if df is None or len(df) < 50:
        print(f"MT5 failed for {mt5_symbol}")
        return None
    
    print(f"Using MT5 data for {mt5_symbol}: {len(df)} rows")
    
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    opens = df['open'].values
    
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
        daily_lows = df_daily['low'].values
        htf = []
        for i in range(1, len(df_daily)):
            if daily_highs[i] > np.max(daily_highs[max(0,i-5):i]) and daily_lows[i] > np.min(daily_lows[max(0,i-5):i]):
                htf.append(1)
            elif daily_highs[i] < np.max(daily_highs[max(0,i-5):i]) and daily_lows[i] < np.min(daily_lows[max(0,i-5):i]):
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
        momentum = closes[i] - closes[i-10]
        pct_change = momentum / closes[i-10] if closes[i-10] > 0 else 0
        
        ema_fast = np.mean(closes[max(0,i-5):i+1])
        ema_slow = np.mean(closes[max(0,i-13):i+1])
        ema_bullish = ema_fast > ema_slow
        ema_bearish = ema_fast < ema_slow
        
        if pct_change > 0.005 or (pct_change > 0.001 and ema_bullish):
            trend[i] = 1
        elif pct_change < -0.005 or (pct_change < -0.001 and ema_bearish):
            trend[i] = -1
    
    price_position = np.zeros(len(df))
    for i in range(20, len(df)):
        ph = np.max(highs[i-20:i])
        pl = np.min(lows[i-20:i])
        rng = ph - pl
        if rng < 0.001:
            rng = 0.001
        price_position[i] = (closes[i] - pl) / rng
    
    hours = df.index.hour.values
    kill_zone = np.zeros(len(df), dtype=bool)
    for i in range(len(hours)):
        h = hours[i]
        kill_zone[i] = (1 <= h < 5) or (7 <= h < 12) or (13.5 <= h < 16)
    
    volatility = np.zeros(len(df))
    for i in range(14, len(df)):
        trs = []
        for j in range(max(0, i-14), i+1):
            tr = max(highs[j] - lows[j], abs(highs[j] - closes[j-1]), abs(lows[j] - closes[j-1])) if j > 0 else highs[j] - lows[j]
            trs.append(tr)
        volatility[i] = np.mean(trs) if trs else 0
    
    return {
        'opens': opens,
        'highs': highs,
        'lows': lows,
        'closes': closes,
        'volatility': volatility,
        'htf_trend': htf_trend,
        'ltf_trend': trend,
        'price_position': price_position,
        'kill_zone': kill_zone,
        'bullish_fvgs': bullish_fvgs,
        'bearish_fvgs': bearish_fvgs,
    }


def get_signal(data: Dict, idx: int) -> Dict:
    """Get V5 signal at index."""
    return {
        'confluence': 0,
        'direction': 0,
    }


def calculate_position_size(symbol: str, account_value: float, risk_pct: float, 
                           stop_distance: float, current_price: float) -> Tuple[float, float]:
    """Calculate position size for MT5."""
    contract_info = get_contract_info(symbol)
    symbol_type = contract_info['type']
    
    if symbol_type == 'crypto':
        risk_amount = 2000
    elif symbol.upper() in ['XAUUSD', 'GOLD']:
        risk_amount = 2000
    elif symbol_type == 'futures':
        risk_amount = 1000
    elif symbol_type == 'indices':
        risk_amount = account_value * risk_pct  # Use percentage for indices
    else:
        risk_amount = account_value * risk_pct
    
    if symbol_type in ['forex', 'crypto']:
        if symbol_type == 'forex':
            decimal_places = contract_info.get('decimal_places', 5)
            pip_size = 0.0001 if decimal_places == 5 else 0.01
            stop_pips = stop_distance / pip_size
            if stop_pips > 0:
                risk_amount = min(risk_amount, stop_pips * contract_info.get('pip_value', 10))
        else:
            risk_amount = min(risk_amount, stop_distance * contract_info.get('multiplier', 1))
    
    if stop_distance <= 0:
        return 0, 0
    
    if symbol_type == 'futures':
        qty = risk_amount / (stop_distance * contract_info.get('multiplier', 1))
    elif symbol_type == 'indices':
        # For indices, qty = risk_amount / stop_distance (1 point = $1 typically)
        qty = risk_amount / stop_distance
    else:
        qty = risk_amount / stop_distance
    
    qty = max(qty, 0.01)  # Min 0.01 lots
    qty = round(qty / 0.01) * 0.01  # Round to 0.01
    
    return qty, risk_amount


def place_mt5_order(symbol: str, order_type: str, volume: float, 
                    price: float, stop_loss: float = None, 
                    take_profit: float = None, magic: int = 123456) -> Optional[Dict]:
    """Place an order on MT5."""
    if not MT5_AVAILABLE:
        return None
    
    symbol = symbol.upper()
    mt5_symbol = get_mt5_symbol(symbol)  # Convert to MT5 format
    
    symbol_info = mt5.symbol_info(mt5_symbol)
    if symbol_info is None:
        print(f"Symbol {mt5_symbol} not found")
        return None
    
    if not symbol_info.visible:
        if not mt5.symbol_select(mt5_symbol, True):
            print(f"Could not select symbol: {mt5_symbol}")
            return None
    
    point = symbol_info.point
    
    if order_type.upper() == "BUY":
        order_type_enum = mt5.ORDER_TYPE_BUY
        price = symbol_info.ask
    else:
        order_type_enum = mt5.ORDER_TYPE_SELL
        price = symbol_info.bid
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": mt5_symbol,
        "volume": volume,
        "type": order_type_enum,
        "price": price,
        "deviation": 20,
        "magic": magic,
        "comment": "ICT V7",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    if stop_loss:
        # Round to symbol's digit precision
        stop_loss = round(stop_loss / point) * point
        # Ensure SL is at least 1 point away from current price
        if order_type.upper() == "BUY":
            min_sl = symbol_info.bid + point
            max_tp = symbol_info.bid - point
        else:
            min_sl = symbol_info.ask - point
            max_tp = symbol_info.ask + point
        
        if order_type.upper() == "BUY" and stop_loss >= symbol_info.bid:
            stop_loss = min_sl
        elif order_type.upper() == "SELL" and stop_loss <= symbol_info.ask:
            stop_loss = min_sl
            
        request["sl"] = stop_loss
        
    if take_profit:
        take_profit = round(take_profit / point) * point
        if order_type.upper() == "BUY" and take_profit <= symbol_info.ask:
            take_profit = max_tp
        elif order_type.upper() == "SELL" and take_profit >= symbol_info.bid:
            take_profit = max_tp
        request["tp"] = take_profit
    
    # Ensure volume is valid
    volume = round(volume / symbol_info.volume_step) * symbol_info.volume_step
    volume = max(symbol_info.volume_min, min(volume, symbol_info.volume_max))
    request["volume"] = volume
    
    # Debug output
    print(f"  Order: {order_type} {symbol_info.name} vol={volume} @ {price} SL={request.get('sl')} TP={request.get('tp')}")
    
    result = mt5.order_send(request)
    
    if result is None:
        print(f"Order send failed: {mt5.last_error()}")
        return None
    
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"Order failed: {result.comment}")
        return None
    
    return {
        'order_id': result.order,
        'volume': result.volume,
        'price': result.price,
        'retcode': result.retcode,
    }


def get_mt5_positions() -> List[Dict]:
    """Get all open positions from MT5."""
    if not MT5_AVAILABLE:
        return []
    
    positions = mt5.positions_get()
    if positions is None:
        return []
    
    result = []
    for pos in positions:
        result.append({
            'ticket': pos.ticket,
            'symbol': pos.symbol,
            'volume': pos.volume,
            'price_open': pos.price_open,
            'price_current': pos.price_current,
            'profit': pos.profit,
            'type': pos.type,
            'sl': pos.sl,
            'tp': pos.tp,
            'time': pos.time,
        })
    
    return result


def close_mt5_position(ticket: int, volume: float = None) -> bool:
    """Close a position by ticket."""
    if not MT5_AVAILABLE:
        return False
    
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        return False
    
    pos = positions[0]
    
    order_type = mt5.ORDER_TYPE_SELL if pos.type == 0 else mt5.ORDER_TYPE_BUY
    
    close_volume = volume if volume else pos.volume
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": pos.symbol,
        "volume": close_volume,
        "type": order_type,
        "position": ticket,
        "price": mt5.symbol_info(pos.symbol).bid if pos.type == 0 else mt5.symbol_info(pos.symbol).ask,
        "deviation": 20,
        "magic": 123456,
        "comment": "ICT V7 Close",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    return result is not None and result.retcode == mt5.TRADE_RETCODE_DONE


class V7SignalGenerator:
    """Enhanced signal generator combining V5 ICT with FVG, Gap, MTF and Market Structure"""
    
    def __init__(self):
        self.fvg_handler = FVGHandler(
            sensitivity=0.0001,
            min_gap_size=0.0,
            track_body_respect=True,
            detect_volume_imbalances=True,
            detect_suspension_blocks=True
        )
        self.gap_handler = GapHandler(
            large_gap_pips_forex=40.0,
            large_gap_points_indices=50.0,
            keep_gaps_days=3
        )
        
        if MTF_AVAILABLE:
            self.mtf_coordinator = MTFCoordinator()
            self.ms_handler = MarketStructureHandler(
                swing_lookback=5,
                min_displacement_pct=0.1
            )
        else:
            self.mtf_coordinator = None
            self.ms_handler = None
            
        self.last_analysis = {}
    
    def analyze_symbol(self, symbol: str, data: Dict, current_price: float) -> Dict:
        idx = len(data['closes']) - 1
        v5_signal = get_signal(data, idx)
        
        df = pd.DataFrame({
            'open': data['opens'],
            'high': data['highs'],
            'low': data['lows'],
            'close': data['closes']
        })
        
        fvgs = self.fvg_handler.detect_all_fvgs(df)
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
            current_price, data, idx, ms_analysis
        )
        
        self.last_analysis[symbol] = {
            'timestamp': datetime.now().isoformat(),
            'v5_confluence': v5_signal['confluence'] if v5_signal else 0,
            'fvg_count': len(fvgs),
            'active_fvgs': len(fvg_analysis.active_fvgs),
            'high_prob_fvgs': len(fvg_analysis.high_prob_fvgs),
            'gap_levels': len(gap_analysis.all_levels)
        }
        
        return combined_signal
    
    def _combine_signals(self, symbol: str, v5_signal: Optional[Dict], 
                        fvg_analysis, gap_analysis, current_price: float,
                        data: Dict, idx: int, ms_analysis=None) -> Dict:
        
        signal = {
            'symbol': symbol,
            'direction': 0,
            'confluence': 0,
            'entry_price': current_price,
            'stop_loss': None,
            'take_profit': None,
            'confidence': 'LOW',
            'reasoning': [],
            'v5_signal': v5_signal,
            'fvg_data': None,
            'gap_data': None
        }
        
        htf = data['htf_trend'][idx]
        ltf = data['ltf_trend'][idx]
        kz = data['kill_zone'][idx]
        pp = data['price_position'][idx]
        closes = data['closes'][idx]
        
        near_bull_fvg = next((f for f in reversed(data['bullish_fvgs']) if f['idx'] < idx and f['mid'] < closes), None)
        near_bear_fvg = next((f for f in reversed(data['bearish_fvgs']) if f['idx'] < idx and f['mid'] > closes), None)
        
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
            except Exception as e:
                pass
        
        signal['confluence'] = base_confluence
        
        fvg_confluence = 0
        if fvg_analysis.best_bisi_fvg and signal['direction'] == 1:
            fvg = fvg_analysis.best_bisi_fvg
            distance = abs(current_price - fvg.consequent_encroachment)
            if distance < fvg.size * 2:
                fvg_confluence += 20
                signal['fvg_data'] = {
                    'type': 'BISI',
                    'ce': fvg.consequent_encroachment,
                    'distance': distance
                }
                signal['reasoning'].append(f"FVG BISI at {fvg.consequent_encroachment:.4f}")
                
                if fvg.is_high_probability:
                    fvg_confluence += 15
                    signal['reasoning'].append("High Probability FVG")
        
        elif fvg_analysis.best_sibi_fvg and signal['direction'] == -1:
            fvg = fvg_analysis.best_sibi_fvg
            distance = abs(current_price - fvg.consequent_encroachment)
            if distance < fvg.size * 2:
                fvg_confluence += 20
                signal['fvg_data'] = {
                    'type': 'SIBI',
                    'ce': fvg.consequent_encroachment,
                    'distance': distance
                }
                signal['reasoning'].append(f"FVG SIBI at {fvg.consequent_encroachment:.4f}")
                
                if fvg.is_high_probability:
                    fvg_confluence += 15
                    signal['reasoning'].append("High Probability FVG")
        
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
                            'type': gap.gap_type.value,
                            'ce': gap.quadrants.ce,
                            'direction': gap.direction.value
                        }
        
        if gap_analysis.nearest_level:
            level_price, level_name = gap_analysis.nearest_level
            distance_pct = abs(current_price - level_price) / current_price * 100
            if distance_pct < 0.5:
                gap_confluence += 10
                signal['reasoning'].append(f"Near {level_name}")
        
        total_confluence = base_confluence + fvg_confluence + gap_confluence
        signal['confluence'] = min(total_confluence, 100)
        
        if total_confluence >= 80:
            signal['confidence'] = 'HIGH'
        elif total_confluence >= 60:
            signal['confidence'] = 'MEDIUM'
        elif total_confluence >= 50:
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
            is_forex = contract_info['type'] == 'forex'
            
            highs = data['highs']
            lows = data['lows']
            closes_arr = data['closes']
            
            atr = np.mean([max(
                highs[i] - lows[i],
                abs(highs[i] - closes_arr[i-1]) if i > 0 else 0,
                abs(lows[i] - closes_arr[i-1]) if i > 0 else 0
            ) for i in range(max(0, idx-14), idx+1)])
            
            atr_multiplier = 2.0
            min_atr_multiplier = 1.5
            
            if is_forex:
                decimal_places = contract_info.get('decimal_places', 5)
                pip_size = 0.01 if decimal_places == 3 else 0.0001
                atr_pips = atr / pip_size
                min_stop_pips = max(25, atr_pips * min_atr_multiplier)
                max_stop_pips = max(80, atr_pips * atr_multiplier)
                atr_stop_distance = atr * atr_multiplier
                min_stop_distance = min_stop_pips * pip_size
                max_stop_distance = max_stop_pips * pip_size
                stop_distance = max(min_stop_distance, min(atr_stop_distance, max_stop_distance))
                
                # Validate forex pips
                risk_pips = stop_distance / pip_size
                if risk_pips < 15 or risk_pips > 100:
                    signal['direction'] = 0
                    signal['stop_loss'] = entry
                    signal['take_profit'] = entry
                    return signal
            else:
                # Indices, metals, crypto - use ATR-based stops
                min_stop = contract_info.get('min_stop', 20)
                stop_distance = max(atr * atr_multiplier, min_stop)
            
            rr = getattr(self, 'rr_ratio', 2.0)
            
            if signal['direction'] == 1:
                signal['stop_loss'] = entry - stop_distance
                signal['take_profit'] = entry + (stop_distance * rr)
            else:
                signal['stop_loss'] = entry + stop_distance
                signal['take_profit'] = entry - (stop_distance * rr)
        
        return signal


class V7MT5LiveTrader:
    """V7 Live Trader with MT5 integration and Telegram"""
    
    def __init__(self, symbols: List[str], risk_pct: float = 0.02, 
                 poll_interval: int = 30, rr_ratio: float = 3.0, 
                 confluence_threshold: int = 60, max_daily_loss: float = -2000):
        self.symbols = symbols
        self.risk_pct = risk_pct
        self.poll_interval = poll_interval
        self.signal_generator = V7SignalGenerator()
        self.mode = 'paper'
        self.rr_ratio = rr_ratio
        self.confluence_threshold = confluence_threshold
        self.max_daily_loss = max_daily_loss
        
        self.positions = {}
        self.historical_data = {}
        self.last_poll_time = {}
        self.last_signal_time = {}
        self.last_signals = {}
        
        self.daily_pnl = 0.0
        self.trade_count = 0
        self.account_value = 100000
        
        self.currency_pnl = {}
        self.running = False
        self.magic = 123456
        
        self._sync_positions()
    
    def _sync_positions(self):
        """Sync positions with MT5 on startup."""
        print("\nSyncing positions with MT5...")
        try:
            positions = get_mt5_positions()
            for pos in positions:
                symbol = pos['symbol'].upper()
                if symbol in [s.upper() for s in self.symbols]:
                    direction = 1 if pos['type'] == 0 else -1
                    self.positions[symbol] = {
                        'ticket': pos['ticket'],
                        'entry': pos['price_open'],
                        'direction': direction,
                        'volume': pos['volume'],
                        'stop': pos['sl'],
                        'target': pos['tp'],
                        'bars_held': 0,
                        'current_price': pos['price_current']
                    }
                    print(f"  Found open position: {symbol} x {pos['volume']} @ {pos['price_open']:.4f}")
            
            if not self.positions:
                print("  No open positions found")
        except Exception as e:
            print(f"  Error syncing positions: {e}")
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price from MT5."""
        if not MT5_AVAILABLE:
            return None
        
        mt5_symbol = get_mt5_symbol(symbol)
        info = mt5.symbol_info(mt5_symbol)
        if info is None:
            return None
        
        return info.bid
    
    def _check_position_exit(self, symbol: str, current_price: float):
        """Check if position hit stop/target."""
        if symbol not in self.positions:
            return
        
        try:
            pos = self.positions[symbol]
            pos['bars_held'] = pos.get('bars_held', 0) + 1
            pos['current_price'] = current_price
            
            direction = pos['direction']
            entry = pos['entry']
            stop = pos.get('stop', 0)
            target = pos.get('target', 0)
            
            exit_reason = None
            pnl = 0
            
            if direction == 1:
                if stop and current_price <= stop:
                    exit_reason = 'stop'
                    pnl = (stop - entry) * pos['volume']
                elif target and current_price >= target:
                    exit_reason = 'target'
                    pnl = (target - entry) * pos['volume']
                else:
                    pnl = (current_price - entry) * pos['volume']
            else:
                if stop and current_price >= stop:
                    exit_reason = 'stop'
                    pnl = (entry - stop) * pos['volume']
                elif target and current_price <= target:
                    exit_reason = 'target'
                    pnl = (entry - target) * pos['volume']
                else:
                    pnl = (entry - current_price) * pos['volume']
            
            if exit_reason:
                self._handle_position_closed(symbol, current_price, pnl, exit_reason)
            else:
                self.daily_pnl = self.daily_pnl - (pos.get('pnl', 0) or 0) + pnl
                pos['pnl'] = pnl
                
        except Exception as e:
            print(f"[{symbol}] Error checking position: {e}")
    
    def _handle_position_closed(self, symbol: str, exit_price: float, pnl: float, exit_reason: str):
        """Handle position closure."""
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        direction_str = 'LONG' if pos['direction'] == 1 else 'SHORT'
        pnl_str = f"+${pnl:.2f}" if pnl > 0 else f"-${abs(pnl):.2f}"
        
        print(f"[{symbol}] V7 EXIT ({exit_reason}): {direction_str} @ {exit_price:.4f} | P&L: {pnl_str}")
        print(f"  Daily P&L: ${self.daily_pnl:.2f}")
        
        if tn:
            try:
                tn.send_trade_exit(
                    symbol, pos['direction'], pnl, exit_reason,
                    pos['entry'], exit_price, pos.get('bars_held', 0)
                )
            except:
                pass
        
        del self.positions[symbol]
    
    def _enter_trade(self, symbol: str, signal: Dict, current_price: float):
        """Enter trade using signal."""
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
        
        if symbol in self.positions:
            print(f"[{symbol}] Already has open position, skipping entry")
            return
        
        try:
            entry_price = signal['entry_price']
            stop_price = signal['stop_loss']
            
            stop_distance = abs(entry_price - stop_price)
            if stop_distance <= 0:
                return
            
            if signal['direction'] == 1:
                target_price = entry_price + (stop_distance * self.rr_ratio)
            else:
                target_price = entry_price - (stop_distance * self.rr_ratio)
            
            qty, risk_amount = calculate_position_size(
                symbol, self.account_value, self.risk_pct, 
                stop_distance, entry_price
            )
            if qty <= 0:
                return
            
            direction_str = 'LONG' if signal['direction'] == 1 else 'SHORT'
            
            fvg_info = signal.get('fvg_data', {})
            gap_info = signal.get('gap_data', {})
            pd_zone = None
            if fvg_info:
                pd_zone = f"FVG {fvg_info.get('type', '')}"
            elif gap_info:
                pd_zone = f"Gap {gap_info.get('type', '')}"
            
            if self.mode == 'shadow':
                print(f"[{symbol}] V7 SHADOW SIGNAL: {direction_str} @ {current_price:.4f}")
                print(f"  Stop: {stop_price:.4f} | Target: {target_price:.4f} (R:R 1:{self.rr_ratio})")
                print(f"  Confluence: {signal['confluence']}/100 | {pd_zone or 'No PD'}")
                print(f"  Risk: ${risk_amount:.2f} | Qty: {qty}")
                
                self._log_shadow_trade(symbol, signal, current_price, stop_price, target_price, qty, risk_amount)
                
                if tn:
                    try:
                        tn.send_signal_alert(
                            symbol=symbol,
                            direction=signal['direction'],
                            confluence=signal['confluence'],
                            pd_zone=pd_zone or '',
                            current_price=current_price
                        )
                    except:
                        pass
                return
            
            order_type = "BUY" if signal['direction'] == 1 else "SELL"
            
            result = place_mt5_order(
                symbol, order_type, qty, entry_price,
                stop_loss=stop_price, take_profit=target_price,
                magic=self.magic
            )
            
            if result:
                self.positions[symbol] = {
                    'ticket': result['order_id'],
                    'entry': entry_price,
                    'stop': stop_price,
                    'target': target_price,
                    'direction': signal['direction'],
                    'volume': qty,
                    'confluence': signal['confluence'],
                    'confidence': signal['confidence'],
                    'entry_time': datetime.now(),
                    'reasoning': signal['reasoning'],
                    'bars_held': 0,
                    'current_price': entry_price
                }
                
                self.trade_count += 1
                
                print(f"[{symbol}] V7 ENTRY: {direction_str} x {qty} @ {entry_price:.4f}")
                print(f"  Confidence: {signal['confidence']} | Confluence: {signal['confluence']}/100")
                print(f"  Stop: {stop_price:.4f} | Target: {target_price:.4f}")
                if signal['reasoning']:
                    print(f"  Reasoning: {' | '.join(signal['reasoning'][:3])}")
                
                if tn:
                    try:
                        tn.send_trade_entry(
                            symbol, signal['direction'], qty, 
                            entry_price, signal['confluence'], target_price, stop_price,
                            pd_zone=pd_zone, risk_amount=risk_amount
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
    
    def _log_shadow_trade(self, symbol: str, signal: Dict, current_price: float, 
                          stop_price: float, target_price: float, qty: float, risk_amount: float):
        """Log shadow trade to JSON file."""
        try:
            trade = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'direction': 'LONG' if signal['direction'] == 1 else 'SHORT',
                'entry': current_price,
                'stop': stop_price,
                'target': target_price,
                'qty': qty,
                'risk_amount': risk_amount,
                'confluence': signal['confluence'],
                'confidence': signal['confidence'],
                'fvg_data': signal.get('fvg_data'),
                'gap_data': signal.get('gap_data'),
                'reasoning': signal.get('reasoning', [])[:3]
            }
            
            with open('v7_shadow_trades.json', 'a') as f:
                f.write(json.dumps(trade) + '\n')
        except Exception as e:
            print(f"Error logging shadow trade: {e}")
    
    def _refresh_data(self):
        """Refresh data for all symbols."""
        for symbol in self.symbols:
            try:
                data = prepare_data_mt5(symbol, lookback=200)
                if data and len(data.get('closes', [])) >= 50:
                    self.historical_data[symbol] = data
            except Exception as e:
                print(f"[{symbol}] Error refreshing data: {e}")
    
    def poll_symbols(self):
        """Poll symbols for signals."""
        current_time = time.time()
        
        for symbol in self.symbols:
            last_poll = self.last_poll_time.get(symbol, 0)
            if current_time - last_poll < self.poll_interval:
                continue
            
            self.last_poll_time[symbol] = current_time
            
            try:
                if symbol not in self.historical_data:
                    data = prepare_data_mt5(symbol, lookback=200)
                    if data is None or len(data.get('closes', [])) < 50:
                        continue
                    self.historical_data[symbol] = data
                else:
                    data = prepare_data_mt5(symbol, lookback=200)
                    if data and len(data.get('closes', [])) >= 50:
                        self.historical_data[symbol] = data
                
                idx = len(data['closes']) - 1
                current_price = data['closes'][idx]
                
                try:
                    signal = self.signal_generator.analyze_symbol(symbol, data, current_price)
                except Exception as e:
                    print(f"[{symbol}] Signal error: {e}")
                    continue
                
                if signal and signal.get('direction', 0) != 0:
                    fvg_type = signal.get('fvg_data', {}).get('type', '') if isinstance(signal.get('fvg_data'), dict) else ''
                    gap_type = signal.get('gap_data', {}).get('type', '') if isinstance(signal.get('gap_data'), dict) else ''
                    self.last_signals[symbol] = {
                        'direction': signal.get('direction', 0),
                        'confluence': signal.get('confluence', 0),
                        'confidence': signal.get('confidence', 'LOW'),
                        'pd_zone': fvg_type or gap_type,
                        'entry': signal.get('entry_price', current_price),
                        'timestamp': datetime.now().isoformat()
                    }
                
                if tn and signal:
                    try:
                        htf = data.get('htf_trend', np.zeros(len(data['closes'])))[idx]
                        ltf = data.get('ltf_trend', np.zeros(len(data['closes'])))[idx]
                        tn.update_market_data(symbol, {
                            'price': current_price,
                            'htf_trend': htf,
                            'ltf_trend': ltf,
                            'confluence': signal.get('confluence', 0),
                            'confidence': signal.get('confidence', 'LOW')
                        })
                    except:
                        pass
                
                if symbol in self.positions:
                    self._check_position_exit(symbol, current_price)
                else:
                    current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
                    last_signal = self.last_signal_time.get(symbol)
                    
                    if last_signal and last_signal >= current_hour:
                        continue
                    
                    if signal and signal.get('direction', 0) != 0 and signal.get('confluence', 0) >= self.confluence_threshold:
                        self.last_signal_time[symbol] = current_hour
                        self._enter_trade(symbol, signal, current_price)
                        
            except Exception as e:
                print(f"[{symbol}] Error polling: {e}")
    
    def update_account(self):
        """Update account info from MT5."""
        if not MT5_AVAILABLE:
            return
        
        try:
            account_info = mt5.account_info()
            if account_info:
                self.account_value = account_info.balance
        except Exception as e:
            print(f"Error updating account: {e}")
    
    def check_positions(self):
        """Check if any positions were closed externally."""
        if not MT5_AVAILABLE:
            return
        
        try:
            mt5_positions = get_mt5_positions()
            mt5_tickets = {p['ticket']: p['symbol'].upper() for p in mt5_positions}
            
            for symbol in list(self.positions.keys()):
                pos = self.positions[symbol]
                if pos.get('ticket') and pos['ticket'] not in mt5_tickets:
                    print(f"[{symbol}] Position closed externally")
                    self._handle_position_closed(symbol, pos.get('current_price', pos['entry']), pos.get('pnl', 0), 'external')
        except Exception as e:
            print(f"Error checking positions: {e}")
    
    def start(self):
        """Start the trading loop."""
        self.running = True
        print(f"\nV7 MT5 Trader started for {self.symbols}")
    
    def stop(self):
        """Stop the trading loop."""
        self.running = False
        print("V7 MT5 Trader stopped")


def run_v7_trading(symbols: List[str], interval: int = 30, risk_pct: float = 0.02,
                  login: int = None, password: str = None, server: str = None,
                  mode: str = 'paper', rr_ratio: float = 3.0, 
                  confluence_threshold: int = 60, max_daily_loss: float = -2000):
    """Run V7 trading with MT5 and Telegram integration."""
    
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
            tn.send_startup(
                symbols=symbols,
                risk_pct=risk_pct,
                interval=interval,
                mode=f"V7 {mode.upper()}"
            )
        except Exception as e:
            print(f"Telegram startup notification failed: {e}")
    
    trader = V7MT5LiveTrader(
        symbols, risk_pct, poll_interval=interval,
        rr_ratio=rr_ratio, confluence_threshold=confluence_threshold,
        max_daily_loss=max_daily_loss
    )
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
            
            if iteration % 60 == 0:
                trader.check_positions()
            
            if iteration % 3600 == 0 and trader.positions:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Active positions: {len(trader.positions)}")
            
            if trader.daily_pnl <= trader.max_daily_loss and iteration % 60 == 0:
                print(f"[WARNING] Daily loss limit reached: ${trader.daily_pnl:.2f}")
            
    except KeyboardInterrupt:
        print("\n\nShutdown...")
    finally:
        trader.stop()
        if MT5_AVAILABLE:
            mt5.shutdown()
        print(f"\nTrades: {trader.trade_count} | Daily P&L: ${trader.daily_pnl:.2f} | Final: ${trader.account_value:,.2f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ICT V7 - MT5 Trading with FVG + Gap')
    parser.add_argument("--symbols", default="EURUSD,GBPUSD,USDJPY,USDCAD,AUDUSD,XAUUSD,XTIUSD,US30,USTEC,US500",
                        help="Comma-separated symbols")
    parser.add_argument("--interval", type=int, default=30,
                        help="Poll interval in seconds")
    parser.add_argument("--risk", type=float, default=0.02,
                        help="Risk per trade (e.g., 0.02 for 2%%)")
    parser.add_argument("--login", type=int, default=None,
                        help="MT5 login (account number)")
    parser.add_argument("--password", type=str, default=None,
                        help="MT5 password")
    parser.add_argument("--server", type=str, default=None,
                        help="MT5 server")
    parser.add_argument("--mode", type=str, default="paper",
                        choices=["shadow", "paper", "live"],
                        help="Trading mode")
    parser.add_argument("--rr", type=float, default=2.0,
                        help="Risk:Reward ratio (e.g., 2.0 for 1:2, 4.0 for 1:4)")
    parser.add_argument("--confluence", type=int, default=60,
                        help="Minimum confluence threshold (0-100)")
    parser.add_argument("--max-loss", type=float, default=-2000,
                        help="Max daily loss before stopping (negative value)")
    
    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(',')]
    
    print("="*60)
    print("ICT V7 Trading Bot - MetaTrader 5")
    print("="*60)
    print(f"Mode: {args.mode.upper()}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Risk: {args.risk*100}% | R:R 1:{args.rr}")
    print(f"Confluence: {args.confluence}+ | Max Loss: ${args.max_loss}")
    print(f"MT5 Login: {args.login or 'Demo'}")
    print("="*60)
    
    run_v7_trading(
        symbols, args.interval, args.risk,
        args.login, args.password, args.server, args.mode,
        rr_ratio=args.rr, confluence_threshold=args.confluence,
        max_daily_loss=args.max_loss
    )
