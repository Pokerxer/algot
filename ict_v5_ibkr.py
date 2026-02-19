"""
ICT V5 - Interactive Brokers Trading System
============================================
Trade Forex, Indices, Futures, and Crypto via IBKR.

Usage:
    python3 ict_v5_ibkr.py --symbols "SPX,NDX,GC,ES,NQ,BTCUSD" --interval 30

IBKR Symbols:
    Indices: SPX, NDX, DJI, RUT
    Futures: ES (S&P), NQ (Nasdaq), GC (Gold), SI (Silver), CL (Oil), NG (Natural Gas)
    Forex: EURUSD, GBPUSD, USDJPY, AUDUSD
    Crypto: BTCUSD, ETHUSD
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import time
import os
import random
import pickle
from datetime import datetime, timedelta
from collections import deque

try:
    import telegram_notify as tn
except ImportError:
    tn = None

np.random.seed(42)
random.seed(42)

Q_TABLE_FILE = "v5_q_table.pkl"

IBKR_AVAILABLE = False

# Global IB connection for data fetching (reused across calls)
_data_ib = None
_data_ib_connected = False


def get_data_ib_connection():
    """Get or create a shared IB connection for data fetching."""
    global _data_ib, _data_ib_connected
    
    try:
        from ib_insync import IB
    except ImportError:
        return None
    
    if _data_ib is None:
        _data_ib = IB()
    
    if not _data_ib_connected or not _data_ib.isConnected():
        try:
            if _data_ib.isConnected():
                _data_ib.disconnect()
            _data_ib.connect('127.0.0.1', 7497, clientId=99)
            _data_ib_connected = True
        except Exception as e:
            print(f"Could not connect data IB: {e}")
            _data_ib_connected = False
            return None
    
    return _data_ib


def disconnect_data_ib():
    """Disconnect the shared data IB connection."""
    global _data_ib, _data_ib_connected
    if _data_ib is not None and _data_ib_connected:
        try:
            _data_ib.disconnect()
        except:
            pass
        _data_ib_connected = False


# Cache for historical data (fetch once, update incrementally)
_data_cache = {}

def fetch_ibkr_data(symbol, days=30, interval="1h", ib=None, use_cache=True):
    """Fetch historical data from IBKR with caching for live trading.
    
    Args:
        symbol: Symbol to fetch
        days: Number of days of history
        interval: Bar interval ("1h" or "1d")
        ib: Optional existing IB connection to reuse
        use_cache: If True, returns cached data and only fetches new bars
    """
    try:
        from ib_insync import util
    except ImportError:
        print("ERROR: ib_insync not installed. Run: pip install ib_insync")
        return None
    
    # Use provided connection or get shared one
    if ib is None:
        ib = get_data_ib_connection()
        if ib is None:
            return None
    
    cache_key = f"{symbol}_{interval}"
    
    # If caching enabled and we have cached data, try to update incrementally
    if use_cache and cache_key in _data_cache:
        cached_df = _data_cache[cache_key]
        last_date = cached_df.index[-1]
        
        # Only fetch last 2 days to get new bars (much faster)
        try:
            contract = get_ibkr_contract(symbol)
            bars = ib.reqHistoricalData(
                contract,
                endDateTime='',  # Up to now
                durationStr="2 D",  # Only last 2 days
                barSizeSetting="1 hour" if interval == "1h" else "1 day",
                whatToShow='MIDPOINT',
                useRTH=False,
                formatDate=2
            )
            
            if bars:
                new_df = util.df(bars)
                new_df.set_index('date', inplace=True)
                new_df.index = pd.to_datetime(new_df.index)
                
                # Merge: keep old data, append only new bars
                new_bars = new_df[new_df.index > last_date]
                if not new_bars.empty:
                    updated_df = pd.concat([cached_df, new_bars])
                    # Keep only last 'days' worth of data
                    cutoff = updated_df.index[-1] - pd.Timedelta(days=days)
                    updated_df = updated_df[updated_df.index > cutoff]
                    _data_cache[cache_key] = updated_df
                    return updated_df
                else:
                    return cached_df
            else:
                return cached_df
                
        except Exception as e:
            print(f"Error updating {symbol}: {e}, using cache")
            return cached_df
    
    # Full fetch (first time or cache disabled)
    try:
        contract = get_ibkr_contract(symbol)
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=f"{days} D",
            barSizeSetting="1 hour" if interval == "1h" else "1 day",
            whatToShow='MIDPOINT',
            useRTH=False,
            formatDate=2
        )
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None
    
    if not bars:
        return None
    
    df = util.df(bars)
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index)
    
    # Cache the result
    if use_cache:
        _data_cache[cache_key] = df
    
    return df


def get_live_price(symbol, ib):
    """Get current live price using market data subscription."""
    try:
        contract = get_ibkr_contract(symbol)
        ticker = ib.reqMktData(contract, '', False, False)
        
        # Wait a bit for price
        ib.sleep(0.5)
        
        price = ticker.last if ticker.last else ticker.close
        return price
    except Exception as e:
        print(f"Error getting live price for {symbol}: {e}")
        return None


def prepare_data_ibkr(symbol, lookback=200, ib=None):
    """Prepare data using IBKR as primary source, Yahoo as fallback.
    
    Args:
        symbol: Symbol to fetch
        lookback: Number of bars to look back
        ib: Optional existing IB connection to reuse (avoids reconnecting)
    """
    df = fetch_ibkr_data(symbol, days=30, interval="1h", ib=ib)
    
    if df is None or len(df) < 50:
        print(f"IBKR failed for {symbol}, using Yahoo...")
        return prepare_data(symbol, lookback)
    
    print(f"Using IBKR data for {symbol}: {len(df)} rows")
    
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
    
    df_daily = fetch_ibkr_data(symbol, days=60, interval="1d")
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
        
        df_daily_index = pd.DatetimeIndex(df_daily.index).tz_localize(None)
        df_index = pd.DatetimeIndex(df.index).tz_localize(None)
        htf_trend = np.zeros(len(df))
        for i in range(len(df)):
            bar_time = df_index[i]
            for j in range(len(df_daily) - 1, -1, -1):
                if df_daily_index[j] <= bar_time:
                    htf_trend[i] = htf[j] if j < len(htf) else 0
                    break
    
    trend = np.zeros(len(df))
    for i in range(20, len(df)):
        rh = np.max(highs[max(0,i-20):i])
        rl = np.min(lows[max(0,i-20):i])
        if rh > highs[i-5] and rl > lows[i-5]:
            trend[i] = 1
        elif rh < highs[i-5] and rl < lows[i-5]:
            trend[i] = -1
    
    price_position = np.zeros(len(df))
    for i in range(20, len(df)):
        ph = np.max(highs[i-20:i])
        pl = np.min(lows[i-20:i])
        rng = ph - pl
        if rng < 0.001:
            rng = 0.001
        price_position[i] = (closes[i] - pl) / rng
    
    hours = pd.to_datetime(df.index).hour.values
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
        'df': df, 'highs': highs, 'lows': lows, 'closes': closes, 'opens': opens,
        'bullish_fvgs': bullish_fvgs, 'bearish_fvgs': bearish_fvgs,
        'htf_trend': htf_trend, 'ltf_trend': trend, 'price_position': price_position,
        'kill_zone': kill_zone, 'volatility': volatility, 'hours': hours
    }


class Actions:
    ENTRY_PASS = 0
    ENTRY_NOW = 1
    ENTRY_PULLBACK = 2
    ENTRY_LIMIT = 3
    EXIT_HOLD = 4
    EXIT_NOW = 5
    MOVE_STOP_BE = 6
    TRAIL_STOP = 7


class QLearningAgent:
    def __init__(self, state_size, action_size, n_bins=10):
        self.state_size = state_size
        self.action_size = action_size
        self.n_bins = n_bins
        self.q_table = {}
        self.epsilon = 0.05
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.alpha = 0.1
        self.gamma = 0.95
        self.train_steps = 0

    def discretize_state(self, state):
        key = []
        for val in state:
            if np.isnan(val) or np.isinf(val):
                key.append(0)
            else:
                bin_idx = int(np.clip(val * self.n_bins, 0, self.n_bins - 1))
                key.append(bin_idx)
        return tuple(key)

    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        disc_state = self.discretize_state(state)
        if disc_state not in self.q_table:
            self.q_table[disc_state] = np.zeros(self.action_size)
        return int(np.argmax(self.q_table[disc_state]))

    def update_q(self, state, action, reward, next_state):
        disc_state = self.discretize_state(state)
        disc_next = self.discretize_state(next_state)
        if disc_state not in self.q_table:
            self.q_table[disc_state] = np.zeros(self.action_size)
        if disc_next not in self.q_table:
            self.q_table[disc_next] = np.zeros(self.action_size)
        current_q = self.q_table[disc_state][action]
        max_next_q = np.max(self.q_table[disc_next])
        self.q_table[disc_state][action] = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump({
                'q_table': self.q_table,
                'epsilon': self.epsilon,
                'alpha': self.alpha,
                'gamma': self.gamma
            }, f)
        print(f"Q-table saved: {len(self.q_table)} states to {filename}")

    def load(self, filename):
        if os.path.exists(filename):
            with open(filename, 'rb') as f:
                data = pickle.load(f)
                self.q_table = data['q_table']
                self.epsilon = data.get('epsilon', 0.01)
                self.alpha = data.get('alpha', 0.1)
                self.gamma = data.get('gamma', 0.95)
            print(f"Q-table loaded: {len(self.q_table)} states from {filename}")
            return True
        return False


def get_ibkr_contract(symbol):
    """Convert symbol string to IBKR contract."""
    from ib_insync import Stock, Forex, CFD, Future, Crypto
    
    symbol = symbol.upper()
    
    # Crypto - IBKR uses base symbol (BTC, not BTCUSD)
    crypto_map = {
        'BTCUSD': 'BTC',
        'ETHUSD': 'ETH',
        'LTCUSD': 'LTC',
        'SOLUSD': 'SOL',
        'LINKUSD': 'LINK',
        'UNIUSD': 'UNI',
    }
    
    if symbol in crypto_map:
        return Crypto(crypto_map[symbol], exchange='PAXOS', currency='USD')
    
    # Forex - use combined symbol format (e.g., 'EURUSD')
    forex_map = {
        'EURUSD': 'EURUSD',
        'GBPUSD': 'GBPUSD',
        'USDJPY': 'USDJPY',
        'AUDUSD': 'AUDUSD',
        'USDCAD': 'USDCAD',
        'USDCHF': 'USDCHF',
        'NZDUSD': 'NZDUSD',
        'GBPJPY': 'GBPJPY',
        'EURJPY': 'EURJPY',
    }
    
    if symbol in forex_map:
        return Forex(forex_map[symbol])  # Single string like 'EURUSD'
    
    # Indices - use Stock for underlying
    if symbol == 'SPX':
        return Stock('SPX', 'SMART', 'USD')
    elif symbol == 'NDX':
        return Stock('NDX', 'SMART', 'USD')
    elif symbol == 'DJI':
        return Stock('DJI', 'SMART', 'USD')
    elif symbol == 'RUT':
        return Stock('RUT', 'SMART', 'USD')
    elif symbol == 'VIX':
        return Stock('VIX', 'SMART', 'USD')
    
    # Futures - use proper exchanges with expiry (current: Feb 2026)
    # Multiplier values: GC=100, SI=5000, CL=1000, NG=10000, ES=50, NQ=20
    futures_map = {
        'ES': ('ES', 'CME', 'USD', '202603', '50'),      # E-mini S&P 500
        'NQ': ('NQ', 'CME', 'USD', '202603', '20'),      # E-mini Nasdaq-100
        'MNQ': ('MNQ', 'CME', 'USD', '202603', '2'),     # Micro Nasdaq
        'GC': ('GC', 'COMEX', 'USD', '202604', '100'),   # Gold (Comex)
        'SI': ('SI', 'COMEX', 'USD', '202603', '5000'),  # Silver (Comex)
        'CL': ('CL', 'NYMEX', 'USD', '202603', '1000'),  # Crude Oil (NYMEX)
        'NG': ('NG', 'NYMEX', 'USD', '202603', '10000'), # Natural Gas (NYMEX)
        'YM': ('YM', 'CBOT', 'USD', '202603', '5'),      # E-mini Dow (CBOT)
        'RTY': ('RTY', 'CME', 'USD', '202603', '50'),    # E-mini Russell 2000 (CME)
    }
    
    if symbol in futures_map:
        fut = futures_map[symbol]
        return Future(fut[0], exchange=fut[1], currency=fut[2], lastTradeDateOrContractMonth=fut[3], multiplier=fut[4])
    
    # Default to stock
    return Stock(symbol, 'SMART', 'USD')


def prepare_data(symbol, lookback=200):
    """Fetch and prepare data for a symbol."""
    yahoo_symbol = symbol
    
    # Map some symbols for Yahoo
    yahoo_map = {
        'SPX': '^GSPC',
        'NDX': '^IXIC',
        'DJI': '^DJI',
        'BTCUSD': 'BTC-USD',
        'ETHUSD': 'ETH-USD',
        'EURUSD': 'EURUSD=X',
        'GBPUSD': 'GBPUSD=X',
        'GC': 'GC=F',  # Gold futures
        'SI': 'SI=F',  # Silver futures
        'CL': 'CL=F',  # Oil futures
        'NG': 'NG=F',  # Natural gas futures
        'ES': 'ES=F',  # S&P futures
        'NQ': 'NQ=F',  # Nasdaq futures
        'YM': 'YM=F',  # Dow futures
    }
    
    if symbol in yahoo_map:
        yahoo_symbol = yahoo_map[symbol]
    
    try:
        # Use 1h data for trading
        df = yf.Ticker(yahoo_symbol).history(period="10d", interval="1h")
        df = df.dropna()
        df = df[~df.index.duplicated(keep='first')]
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None
    
    if len(df) < 50:
        print(f"Not enough data for {symbol}: {len(df)} rows")
        return None
    
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    opens = df['Open'].values
    
    # Calculate FVG (Fair Value Gap)
    bullish_fvgs = []
    bearish_fvgs = []
    for i in range(3, len(df)):
        if lows[i] > highs[i-2]:
            bullish_fvgs.append({'idx': i, 'mid': (highs[i-2] + lows[i]) / 2, 'high': lows[i]})
        if highs[i] < lows[i-2]:
            bearish_fvgs.append({'idx': i, 'mid': (highs[i] + lows[i-2]) / 2, 'low': highs[i]})
    
    # Daily data for HTF trend
    try:
        df_daily = yf.Ticker(yahoo_symbol).history(period="30d", interval="1d")
        if len(df_daily) < 5:
            htf_trend = np.zeros(len(df))
        else:
            daily_highs = df_daily['High'].values
            daily_lows = df_daily['Low'].values
            htf = []
            for i in range(1, len(df_daily)):
                if daily_highs[i] > np.max(daily_highs[max(0,i-5):i]) and daily_lows[i] > np.min(daily_lows[max(0,i-5):i]):
                    htf.append(1)
                elif daily_highs[i] < np.max(daily_highs[max(0,i-5):i]) and daily_lows[i] < np.min(daily_lows[max(0,i-5):i]):
                    htf.append(-1)
                else:
                    htf.append(0)
            
            df_daily_index = pd.DatetimeIndex(df_daily.index).tz_localize(None)
            df_index = pd.DatetimeIndex(df.index).tz_localize(None)
            htf_trend = np.zeros(len(df))
            for i in range(len(df)):
                bar_time = df_index[i]
                for j in range(len(df_daily) - 1, -1, -1):
                    if df_daily_index[j] <= bar_time:
                        htf_trend[i] = htf[j] if j < len(htf) else 0
                        break
    except:
        htf_trend = np.zeros(len(df))
    
    # LTF Trend
    trend = np.zeros(len(df))
    for i in range(20, len(df)):
        rh = np.max(highs[max(0,i-20):i])
        rl = np.min(lows[max(0,i-20):i])
        if rh > highs[i-5] and rl > lows[i-5]:
            trend[i] = 1
        elif rh < highs[i-5] and rl < lows[i-5]:
            trend[i] = -1
    
    # Price Position
    price_position = np.zeros(len(df))
    for i in range(20, len(df)):
        ph = np.max(highs[i-20:i])
        pl = np.min(lows[i-20:i])
        rng = ph - pl
        if rng < 0.001:
            rng = 0.001
        price_position[i] = (closes[i] - pl) / rng
    
    # Kill Zone (NYC + London)
    hours = pd.to_datetime(df.index).hour.values
    kill_zone = np.zeros(len(df), dtype=bool)
    for i in range(len(hours)):
        h = hours[i]
        kill_zone[i] = (1 <= h < 5) or (7 <= h < 12) or (13.5 <= h < 16)
    
    # Volatility (ATR-like)
    volatility = np.zeros(len(df))
    for i in range(14, len(df)):
        trs = []
        for j in range(max(0, i-14), i+1):
            tr = max(highs[j] - lows[j], abs(highs[j] - closes[j-1]), abs(lows[j] - closes[j-1])) if j > 0 else highs[j] - lows[j]
            trs.append(tr)
        volatility[i] = np.mean(trs) if trs else 0
    
    return {
        'df': df, 'highs': highs, 'lows': lows, 'closes': closes, 'opens': opens,
        'bullish_fvgs': bullish_fvgs, 'bearish_fvgs': bearish_fvgs,
        'htf_trend': htf_trend, 'ltf_trend': trend, 'price_position': price_position,
        'kill_zone': kill_zone, 'volatility': volatility, 'hours': hours
    }


def build_state(data, idx, position=None):
    """Build state vector for RL agent."""
    closes = data['closes']
    highs = data['highs']
    lows = data['lows']
    current_price = closes[idx]
    htf = data['htf_trend'][idx]
    ltf = data['ltf_trend'][idx]
    pp = data['price_position'][idx]
    kz = data['kill_zone'][idx]
    vol = data['volatility'][idx]
    hours = data['hours'][idx]
    
    price_scale = current_price if current_price > 100 else 1
    
    # Find nearest FVG
    near_bull_fvg = next((f for f in reversed(data['bullish_fvgs']) if f['idx'] < idx and f['mid'] < current_price < f['high']), None)
    near_bear_fvg = next((f for f in reversed(data['bearish_fvgs']) if f['idx'] < idx and f['low'] < current_price < f['mid']), None)
    
    fvg_dist = 0
    if near_bull_fvg:
        fvg_dist = (current_price - near_bull_fvg['mid']) / price_scale
    elif near_bear_fvg:
        fvg_dist = (near_bear_fvg['mid'] - current_price) / price_scale
    
    # Liquidity sweep
    recent_low = np.min(lows[max(0, idx-20):idx]) if idx > 20 else lows[0]
    recent_high = np.max(highs[max(0, idx-20):idx]) if idx > 20 else highs[0]
    liquidity_sweep = 0
    if lows[idx] < recent_low:
        liquidity_sweep = 1
    elif highs[idx] > recent_high:
        liquidity_sweep = -1
    
    # Confluence
    confluence = 0
    if kz:
        confluence += 0.15
    if htf == 1 and ltf >= 0:
        confluence += 0.25
    elif htf == -1 and ltf <= 0:
        confluence += 0.25
    if pp < 0.25 or pp > 0.75:
        confluence += 0.2
    if near_bull_fvg and ltf >= 0:
        confluence += 0.15
    if near_bear_fvg and ltf <= 0:
        confluence += 0.15
    
    # Regime
    regime = 0
    if htf == 1 and ltf == 1:
        regime = 1
    elif htf == -1 and ltf == -1:
        regime = -1
    
    state = np.array([
        htf, ltf, pp, float(kz), min(fvg_dist * 10, 1), confluence, regime / 2.0,
        vol / price_scale, liquidity_sweep, 0, 0, 0, 0, 0,
        np.sin(2 * np.pi * hours / 24), np.cos(2 * np.pi * hours / 24),
        (idx % 24) / 24.0, near_bull_fvg is not None, near_bear_fvg is not None
    ], dtype=np.float32)
    
    if position is not None:
        state[10] = 1 if position['direction'] == 1 else -1
        state[11] = position.get('pnl_r', 0) / 5
        state[12] = min(position.get('bars_held', 0) / 20, 1)
    
    return state


def get_signal(data, idx):
    """Get trading signal based on ICT strategy."""
    closes = data['closes'][idx]
    htf = data['htf_trend'][idx]
    ltf = data['ltf_trend'][idx]
    kz = data['kill_zone'][idx]
    pp = data['price_position'][idx]
    
    near_bull_fvg = next((f for f in reversed(data['bullish_fvgs']) if f['idx'] < idx and f['mid'] < closes), None)
    near_bear_fvg = next((f for f in reversed(data['bearish_fvgs']) if f['idx'] < idx and f['mid'] > closes), None)
    
    confluence = 0
    if kz:
        confluence += 15
    if htf == 1 and ltf >= 0:
        confluence += 25
    elif htf == -1 and ltf <= 0:
        confluence += 25
    if pp < 0.25:
        confluence += 20
    elif pp > 0.75:
        confluence += 20
    if near_bull_fvg and ltf >= 0:
        confluence += 15
    if near_bear_fvg and ltf <= 0:
        confluence += 15
    
    if confluence >= 60:
        if htf == 1 and ltf >= 0:
            return {'direction': 1, 'confluence': confluence, 'entry': closes}
        elif htf == -1 and ltf <= 0:
            return {'direction': -1, 'confluence': confluence, 'entry': closes}
    
    return None


def run_backtest(symbols, days=180, use_ibkr=True):
    """Run backtest on historical data."""
    print(f"\n{'='*50}")
    print(f"ICT V5 Backtest - {days} days")
    print(f"Data source: {'IBKR' if use_ibkr else 'Yahoo'}")
    print(f"{'='*50}\n")
    
    results = []
    
    for symbol in symbols:
        print(f"Testing {symbol}...")
        
        if use_ibkr:
            data = prepare_data_ibkr(symbol)
        else:
            data = prepare_data(symbol)
        
        if data is None or len(data.get('closes', [])) < 50:
            print(f"  No data for {symbol}")
            continue
        
        balance = 10000
        position = None
        trades = 0
        wins = 0
        losses = 0
        
        closes = data['closes']
        highs = data['highs']
        lows = data['lows']
        htf_trend = data['htf_trend']
        ltf_trend = data['ltf_trend']
        kill_zone = data['kill_zone']
        price_position = data['price_position']
        bullish_fvgs = data['bullish_fvgs']
        bearish_fvgs = data['bearish_fvgs']
        
        for idx in range(50, len(closes) - 1):
            htf = htf_trend[idx]
            ltf = ltf_trend[idx]
            kz = kill_zone[idx]
            pp = price_position[idx]
            current_price = closes[idx]
            
            near_bull_fvg = next((f for f in reversed(bullish_fvgs) if f['idx'] < idx and f['mid'] < current_price < f['high']), None)
            near_bear_fvg = next((f for f in reversed(bearish_fvgs) if f['idx'] < idx and f['low'] < current_price < f['mid']), None)
            
            confluence = 0
            if kz:
                confluence += 15
            if htf == 1 and ltf >= 0:
                confluence += 25
            elif htf == -1 and ltf <= 0:
                confluence += 25
            if pp < 0.25:
                confluence += 20
            elif pp > 0.75:
                confluence += 20
            if near_bull_fvg and ltf >= 0:
                confluence += 15
            if near_bear_fvg and ltf <= 0:
                confluence += 15
            
            if position is None and confluence >= 60:
                if htf == 1 and ltf >= 0:
                    direction = 1
                    stop = lows[idx]
                    target = current_price + (current_price - stop) * 2
                elif htf == -1 and ltf <= 0:
                    direction = -1
                    stop = highs[idx]
                    target = current_price - (stop - current_price) * 2
                else:
                    continue
                
                position = {
                    'entry': current_price,
                    'stop': stop,
                    'target': target,
                    'direction': direction
                }
            
            elif position:
                next_close = closes[idx + 1]
                next_low = lows[idx + 1]
                next_high = highs[idx + 1]
                
                if position['direction'] == 1:
                    if next_low <= position['stop']:
                        balance -= 100
                        losses += 1
                        trades += 1
                        position = None
                    elif next_high >= position['target']:
                        balance += 200
                        wins += 1
                        trades += 1
                        position = None
                else:
                    if next_high >= position['stop']:
                        balance -= 100
                        losses += 1
                        trades += 1
                        position = None
                    elif next_low <= position['target']:
                        balance += 200
                        wins += 1
                        trades += 1
                        position = None
        
        win_rate = wins / trades * 100 if trades > 0 else 0
        result = {
            'symbol': symbol,
            'balance': balance,
            'trades': trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate
        }
        results.append(result)
        
        print(f"  {symbol}: ${balance:,.0f} | {trades} trades | {win_rate:.0f}% win rate")
    
    total_return = sum(r['balance'] for r in results) - (10000 * len(results))
    total_pct = total_return / (10000 * len(results)) * 100 if results else 0
    
    print(f"\n{'='*50}")
    print(f"Total Return: ${total_return:,.0f} ({total_pct:.1f}%)")
    print(f"{'='*50}\n")
    
    return results


def get_contract_info(symbol):
    """Get comprehensive contract information for position sizing.
    
    Returns dict with:
    - type: 'futures', 'forex', 'crypto', or 'stock'
    - multiplier: $ value per point (for futures/crypto/stock)
    - pip_value: $ value per pip (for forex)
    - min_stop: minimum recommended stop distance
    - tick_size: minimum price increment
    """
    symbol = symbol.upper()
    
    # Futures - with minimum stops and tick sizes
    futures_info = {
        'ES': {'multiplier': 50, 'min_stop': 10, 'tick_size': 0.25},      # E-mini S&P
        'NQ': {'multiplier': 20, 'min_stop': 25, 'tick_size': 0.25},      # E-mini Nasdaq
        'MNQ': {'multiplier': 2, 'min_stop': 25, 'tick_size': 0.25},      # Micro Nasdaq
        'MES': {'multiplier': 5, 'min_stop': 10, 'tick_size': 0.25},      # Micro E-mini S&P
        'GC': {'multiplier': 100, 'min_stop': 10, 'tick_size': 0.10},     # Gold
        'SI': {'multiplier': 5000, 'min_stop': 0.50, 'tick_size': 0.005}, # Silver
        'CL': {'multiplier': 1000, 'min_stop': 0.50, 'tick_size': 0.01},  # Crude Oil
        'NG': {'multiplier': 10000, 'min_stop': 0.03, 'tick_size': 0.001}, # Natural Gas
        'YM': {'multiplier': 5, 'min_stop': 50, 'tick_size': 1},          # E-mini Dow
        'RTY': {'multiplier': 50, 'min_stop': 15, 'tick_size': 0.10},     # E-mini Russell
    }
    
    # Forex pip values (per standard lot of 100k units)
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
    }
    
    # Crypto - with minimum stops as percentage of price
    crypto_info = {
        'BTCUSD': {'multiplier': 1, 'min_stop_pct': 0.015, 'tick_size': 0.01},
        'ETHUSD': {'multiplier': 1, 'min_stop_pct': 0.015, 'tick_size': 0.01},
        'SOLUSD': {'multiplier': 1, 'min_stop_pct': 0.02, 'tick_size': 0.01},
        'LTCUSD': {'multiplier': 1, 'min_stop_pct': 0.02, 'tick_size': 0.01},
        'LINKUSD': {'multiplier': 1, 'min_stop_pct': 0.025, 'tick_size': 0.01},
        'UNIUSD': {'multiplier': 1, 'min_stop_pct': 0.03, 'tick_size': 0.01},
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
    else:
        # Default stock settings
        return {'type': 'stock', 'multiplier': 1, 'min_stop': 0.02, 'tick_size': 0.01}


def get_contract_multiplier(symbol):
    """Get the contract multiplier for position sizing (backward compatibility)."""
    info = get_contract_info(symbol)
    if info['type'] == 'futures':
        return {'type': 'futures', 'multiplier': info['multiplier']}
    elif info['type'] == 'forex':
        return {'type': 'forex', 'pip_value': info['pip_value']}
    elif info['type'] == 'crypto':
        return {'type': 'crypto', 'multiplier': info['multiplier']}
    else:
        return {'type': 'stock', 'multiplier': 1}


def calculate_position_size(symbol, account_value, risk_pct, stop_distance, current_price):
    """
    Calculate proper position size with minimum stop enforcement.
    
    Risk per trade varies by asset type:
    - Crypto: $2,000 per trade
    - S&P (ES): $1,000 per trade  
    - Gold (GC): $2,000 per trade
    - Other futures: $1,000 per trade
    
    Returns: (quantity, actual_risk_per_unit)
    """
    contract_info = get_contract_info(symbol)
    symbol_type = contract_info['type']
    
    # Adjust risk amount based on symbol
    if symbol_type == 'crypto':
        risk_amount = 2000  # $2,000 for crypto
    elif symbol.upper() == 'ES':
        risk_amount = 1000  # $1,000 for S&P
    elif symbol.upper() == 'GC':
        risk_amount = 2000  # $2,000 for Gold
    else:
        risk_amount = account_value * risk_pct  # Default: 2% of account
    
    if symbol_type == 'futures':
        multiplier = contract_info['multiplier']
        min_stop = contract_info['min_stop']
        tick_size = contract_info['tick_size']
        
        # Enforce minimum stop distance to prevent oversized positions
        effective_stop = max(stop_distance, min_stop)
        
        # Round to tick size
        effective_stop = round(effective_stop / tick_size) * tick_size
        
        # Calculate risk per contract
        risk_per_contract = effective_stop * multiplier
        
        # Calculate contracts
        raw_qty = risk_amount / risk_per_contract if risk_per_contract > 0 else 0
        qty = max(1, int(raw_qty))
        
        # For high-multiplier contracts, check if 1 contract exceeds risk limit
        if qty == 1 and risk_per_contract > risk_amount * 3.0:
            # Skip this trade only if even 1 contract is way too risky (>3x)
            return 0, risk_per_contract
        
        return qty, risk_per_contract
    
    elif symbol_type == 'forex':
        pip_value = contract_info['pip_value']
        min_stop = contract_info['min_stop']
        decimal_places = contract_info.get('decimal_places', 5)
        
        # Enforce minimum stop
        effective_stop = max(stop_distance, min_stop)
        
        # Convert to pips
        if decimal_places == 3:  # JPY pairs
            pips = effective_stop * 100
        else:  # Standard pairs
            pips = effective_stop * 10000
        
        # Calculate lot size
        risk_per_lot = pips * pip_value
        lots = risk_amount / risk_per_lot if risk_per_lot > 0 else 0.01
        
        # Convert to units (1 lot = 100,000 units)
        qty = max(1000, int(lots * 100000))
        
        # Cap at reasonable maximum
        max_units = 5000000  # 50 lots max
        qty = min(qty, max_units)
        
        risk_per_unit = risk_per_lot / 100000
        return qty, risk_per_unit
    
    elif symbol_type == 'crypto':
        multiplier = contract_info['multiplier']
        min_stop_pct = contract_info.get('min_stop_pct', 0.02)
        tick_size = contract_info.get('tick_size', 0.01)
        
        # Calculate minimum stop in price terms
        min_stop = current_price * min_stop_pct
        
        # Enforce minimum stop
        effective_stop = max(stop_distance, min_stop)
        
        # Round to tick size
        effective_stop = round(effective_stop / tick_size) * tick_size
        
        # Calculate position size
        risk_per_unit = effective_stop * multiplier
        qty = risk_amount / risk_per_unit if risk_per_unit > 0 else 0
        
        # Round appropriately
        if current_price > 10000:  # BTC
            qty = round(qty, 6)
        elif current_price > 1000:  # ETH
            qty = round(qty, 5)
        else:  # Other crypto
            qty = round(qty, 2)
        
        # Enforce minimum and maximum
        qty = max(0.001, qty)
        
        # Cap position value at 10x account (leverage limit)
        max_qty = (account_value * 10) / current_price
        qty = min(qty, max_qty)
        
        return qty, risk_per_unit
    
    else:  # Stock/Other
        min_stop = contract_info.get('min_stop', 0.02)
        effective_stop = max(stop_distance, min_stop)
        
        risk_per_share = effective_stop
        qty = int(risk_amount / risk_per_share) if risk_per_share > 0 else 0
        
        # Enforce min/max
        qty = max(1, qty)
        max_shares = int(account_value / current_price) if current_price > 0 else 0
        qty = min(qty, max_shares)
        
        return qty, risk_per_share


def sync_positions_with_ibkr(ib, symbols):
    """
    Sync local position tracking with actual IBKR positions.
    Call this on startup to recover from crashes.
    """
    positions = {}
    
    try:
        ibkr_positions = ib.positions()
        
        for pos in ibkr_positions:
            # Get symbol from contract
            symbol = pos.contract.symbol
            
            # Map IBKR symbols back to our format
            if pos.contract.secType == 'CASH':
                # Forex - combine pair
                symbol = f"{pos.contract.symbol}{pos.contract.currency}"
            elif pos.contract.secType == 'CRYPTO':
                symbol = f"{pos.contract.symbol}USD"
            
            if symbol in symbols or symbol.upper() in [s.upper() for s in symbols]:
                qty = abs(pos.position)
                direction = 1 if pos.position > 0 else -1
                avg_cost = pos.avgCost
                
                # Estimate stop/target (we don't know original, use defaults)
                contract_info = get_contract_multiplier(symbol)
                if contract_info['type'] == 'futures':
                    # Use 1% of price as estimated stop distance
                    stop_dist = avg_cost * 0.01
                else:
                    stop_dist = avg_cost * 0.02
                
                positions[symbol.upper()] = {
                    'entry': avg_cost,
                    'stop': avg_cost - stop_dist if direction == 1 else avg_cost + stop_dist,
                    'target': avg_cost + stop_dist * 2 if direction == 1 else avg_cost - stop_dist * 2,
                    'direction': direction,
                    'qty': qty,
                    'confluence': 0,  # Unknown
                    'bars_held': 0,
                    'synced_from_ibkr': True
                }
                print(f"[SYNC] Found existing position: {symbol} {direction} x {qty} @ {avg_cost:.2f}")
        
        return positions
    
    except Exception as e:
        print(f"Error syncing positions: {e}")
        return {}


def wait_for_fill(ib, trade, timeout=10):
    """
    Wait for order fill and return fill details.
    Returns: (filled: bool, fill_price: float, filled_qty: int)
    """
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        ib.sleep(0.5)  # Use ib.sleep to process events
        
        if trade.isDone():
            if trade.orderStatus.status == 'Filled':
                fill_price = trade.orderStatus.avgFillPrice
                filled_qty = trade.orderStatus.filled
                return True, fill_price, int(filled_qty)
            else:
                # Order rejected or cancelled
                print(f"Order failed: {trade.orderStatus.status}")
                return False, 0, 0
    
    print(f"Order timeout after {timeout}s")
    return False, 0, 0


def place_bracket_order(ib, contract, direction, qty, stop_price, target_price):
    """
    Place a bracket order with stop-loss and take-profit.
    Returns: (parent_trade, sl_trade, tp_trade) or None on failure
    """
    try:
        from ib_insync import MarketOrder, StopOrder, LimitOrder
        
        # Create bracket order
        action = 'BUY' if direction == 1 else 'SELL'
        close_action = 'SELL' if direction == 1 else 'BUY'
        
        # Parent order (market entry)
        parent = MarketOrder(action, qty)
        parent.transmit = False  # Don't transmit yet
        
        # Stop loss order
        stop_order = StopOrder(close_action, qty, stop_price)
        stop_order.parentId = 0  # Will be set after parent is placed
        stop_order.transmit = False
        
        # Take profit order  
        tp_order = LimitOrder(close_action, qty, target_price)
        tp_order.parentId = 0
        tp_order.transmit = True  # Transmit all orders
        
        # Place parent first
        parent_trade = ib.placeOrder(contract, parent)
        ib.sleep(0.5)
        
        # Get parent order ID and attach children
        parent_id = parent_trade.order.orderId
        stop_order.parentId = parent_id
        tp_order.parentId = parent_id
        
        # Set OCA (One-Cancels-All) group for SL and TP
        oca_group = f"OCA_{parent_id}_{int(time.time())}"
        stop_order.ocaGroup = oca_group
        stop_order.ocaType = 1  # Cancel on fill
        tp_order.ocaGroup = oca_group
        tp_order.ocaType = 1
        
        # Place child orders
        sl_trade = ib.placeOrder(contract, stop_order)
        tp_trade = ib.placeOrder(contract, tp_order)
        
        return parent_trade, sl_trade, tp_trade
    
    except Exception as e:
        print(f"Error placing bracket order: {e}")
        return None


class LiveTrader:
    """Live trading with streaming 5-second bars."""
    
    def __init__(self, ib, symbols, risk_pct=0.02):
        self.ib = ib
        self.symbols = symbols
        self.risk_pct = risk_pct
        self.positions = {}
        self.active_orders = {}
        self.bar_handlers = {}
        self.historical_data = {}
        self.account_value = 100000
        self.trade_count = 0
        
        # Initialize historical data for indicator calculations
        self._init_historical_data()
        
    def _init_historical_data(self):
        """Fetch historical data once for indicator calculations."""
        print("\nInitializing historical data for indicators...")
        for symbol in self.symbols:
            try:
                data = prepare_data_ibkr(symbol, ib=self.ib, use_cache=True)
                if data and len(data.get('closes', [])) >= 50:
                    self.historical_data[symbol] = data
                    print(f"  {symbol}: {len(data['closes'])} bars loaded")
                else:
                    print(f"  {symbol}: Failed to load data")
            except Exception as e:
                print(f"  {symbol}: Error loading - {e}")
    
    def _on_realtime_bar(self, symbol, bar):
        """Handle real-time 5-second bar updates."""
        # Only process completed 5-second bars
        if not self.historical_data.get(symbol):
            return
            
        data = self.historical_data[symbol]
        current_price = bar.close
        
        # Update last price for Telegram
        if tn:
            try:
                tn.update_market_data(symbol, {
                    'price': current_price,
                    'last_update': datetime.now().isoformat()
                })
            except:
                pass
        
        # Check if we have an open position
        if symbol in self.positions:
            self._check_position_exit(symbol, current_price)
        else:
            # No position - check for entry signal on every 5-second bar
            idx = len(data['closes']) - 1
            signal = get_signal(data, idx)
            
            if signal:
                self._enter_trade(symbol, signal, current_price)
    
    def _enter_trade(self, symbol, signal, current_price):
        """Enter a new trade with bracket order."""
        try:
            data = self.historical_data[symbol]
            idx = len(data['closes']) - 1
            
            # Calculate stop and target
            if signal['direction'] == 1:
                stop = data['lows'][idx]
                target = current_price + (current_price - stop) * 2
            else:
                stop = data['highs'][idx]
                target = current_price - (stop - current_price) * 2
            
            stop_distance = abs(current_price - stop)
            if stop_distance <= 0:
                return
            
            # Calculate position size
            qty, _ = calculate_position_size(symbol, self.account_value, self.risk_pct, stop_distance, current_price)
            if qty <= 0:
                return
            
            # Place bracket order
            contract = get_ibkr_contract(symbol)
            bracket = place_bracket_order(self.ib, contract, signal['direction'], qty, stop, target)
            
            if bracket:
                parent_trade, sl_trade, tp_trade = bracket
                filled, fill_price, filled_qty = wait_for_fill(self.ib, parent_trade, timeout=10)
                
                if filled:
                    self.positions[symbol] = {
                        'entry': fill_price,
                        'stop': stop,
                        'target': target,
                        'direction': signal['direction'],
                        'qty': filled_qty,
                        'confluence': signal['confluence'],
                        'entry_time': datetime.now()
                    }
                    
                    self.active_orders[symbol] = {
                        'sl_order_id': sl_trade.order.orderId,
                        'tp_order_id': tp_trade.order.orderId
                    }
                    
                    print(f"[{symbol}] ENTRY: {'LONG' if signal['direction'] == 1 else 'SHORT'} x {filled_qty} @ {fill_price:.4f}")
                    self.trade_count += 1
                    
                    if tn:
                        try:
                            tn.send_trade_entry(symbol, signal['direction'], filled_qty, fill_price, signal['confluence'], target, stop)
                        except:
                            pass
                            
        except Exception as e:
            print(f"[{symbol}] Error entering trade: {e}")
    
    def _check_position_exit(self, symbol, current_price):
        """Check if position hit stop/target (bracket orders handle this, but we track it)."""
        try:
            # Query IBKR to see if position still exists
            ibkr_pos = self.ib.positions()
            has_position = False
            
            for p in ibkr_pos:
                p_symbol = p.contract.symbol
                if p.contract.secType == 'CASH':
                    p_symbol = f"{p.contract.symbol}{p.contract.currency}"
                elif p.contract.secType == 'CRYPTO':
                    p_symbol = f"{p.contract.symbol}USD"
                
                if p_symbol.upper() == symbol.upper() and abs(p.position) > 0:
                    has_position = True
                    break
            
            if not has_position and symbol in self.positions:
                # Position was closed by bracket order
                pos = self.positions[symbol]
                
                # Try to get fill price from IBKR
                try:
                    fills = self.ib.fills()
                    for fill in fills:
                        fill_symbol = fill.contract.symbol
                        if fill.contract.secType == 'CASH':
                            fill_symbol = f"{fill.contract.symbol}{fill.contract.currency}"
                        elif fill.contract.secType == 'CRYPTO':
                            fill_symbol = f"{fill.contract.symbol}USD"
                        
                        if fill_symbol.upper() == symbol.upper():
                            exit_price = fill.execution.price
                            if pos['direction'] == 1:
                                pnl = (exit_price - pos['entry']) * pos['qty']
                            else:
                                pnl = (pos['entry'] - exit_price) * pos['qty']
                            
                            # Apply contract multiplier for PnL
                            contract_info = get_contract_multiplier(symbol)
                            if contract_info['type'] == 'futures':
                                pnl *= contract_info['multiplier']
                            
                            print(f"[{symbol}] EXIT: @ {exit_price:.4f} P&L: ${pnl:.2f}")
                            
                            if tn:
                                try:
                                    bars_held = (datetime.now() - pos['entry_time']).seconds // 300  # Approximate
                                    tn.send_trade_exit(symbol, pos['direction'], pnl, 'bracket_order', pos['entry'], exit_price, bars_held)
                                except:
                                    pass
                            
                            del self.positions[symbol]
                            if symbol in self.active_orders:
                                del self.active_orders[symbol]
                            break
                except Exception as e:
                    print(f"[{symbol}] Error processing fill: {e}")
                    
        except Exception as e:
            print(f"[{symbol}] Error checking position: {e}")
    
    def start(self):
        """Start streaming real-time bars."""
        print(f"\nStarting real-time streaming for {len(self.symbols)} symbols...")
        print("Using 5-second bars for signal detection")
        print("-" * 50)
        
        for symbol in self.symbols:
            try:
                contract = get_ibkr_contract(symbol)
                
                # Subscribe to real-time bars (5-second intervals)
                bars = self.ib.reqRealTimeBars(contract, 5, 'MIDPOINT', False)
                
                # Create callback for this symbol
                def make_callback(sym):
                    def callback(bar):
                        self._on_realtime_bar(sym, bar)
                    return callback
                
                bars.updateEvent += make_callback(symbol)
                self.bar_handlers[symbol] = bars
                
                print(f"  {symbol}: Subscribed to 5-second bars")
                
            except Exception as e:
                print(f"  {symbol}: Failed to subscribe - {e}")
        
        print(f"\nStreaming started - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("Press Ctrl+C to stop\n")
    
    def stop(self):
        """Stop streaming."""
        print("\nStopping real-time streaming...")
        for symbol, bars in self.bar_handlers.items():
            try:
                self.ib.cancelRealTimeBars(bars)
                print(f"  {symbol}: Unsubscribed")
            except:
                pass


def run_ibkr_trading(symbols, interval=30, risk_pct=0.02, port=7497):
    """Run live trading via IBKR with streaming 5-second bars.
    
    Features:
    - Real-time 5-second bar streaming (not polling)
    - Incremental historical data loading
    - Bracket orders with automatic SL/TP
    - Position sync on startup
    - Event-driven architecture (no sleep loops)
    """
    try:
        from ib_insync import IB, MarketOrder, StopOrder, LimitOrder
    except ImportError:
        print("ERROR: ib_insync not installed. Run: pip install ib_insync")
        return
    
    # Connect to IBKR
    ib = IB()
    
    try:
        ib.connect('127.0.0.1', port, clientId=1)
        print(f"Connected to IBKR (port {port})")
    except Exception as e:
        print(f"ERROR connecting to IBKR: {e}")
        print("Make sure IBKR Gateway or TWS is running!")
        return
    
    # Get account info
    account_value = 100000
    try:
        account = ib.accountValues()
        for av in account:
            if av.tag == 'NetLiquidation' and av.currency == 'USD':
                account_value = float(av.value)
                print(f"Account value: ${account_value:,.2f}")
                break
    except Exception as e:
        print(f"Warning: Could not get account info: {e}")
    
    # Sync existing positions
    print("\nSyncing positions with IBKR...")
    positions = sync_positions_with_ibkr(ib, symbols)
    if positions:
        print(f"Found {len(positions)} existing position(s)")
    else:
        print("No existing positions found")
    
    print(f"\nICT V5 - IBKR Trading (STREAMING)")
    print(f"Symbols: {symbols}")
    print(f"Risk per trade: {risk_pct*100}%")
    print(f"Mode: {'Paper Trading' if port == 7497 else 'Live Trading'}")
    print("-" * 50)
    
    # Send startup notification
    if tn:
        try:
            message = f"""
🚀 <b>V5 Trading Bot Started</b>
━━━━━━━━━━━━━━━━━━━━
<b>Symbols:</b> {', '.join(symbols)}
<b>Risk:</b> {risk_pct*100}%
<b>Mode:</b> {'Paper Trading' if port == 7497 else 'Live Trading'}
<b>Account:</b> ${account_value:,.0f}
<b>Data:</b> Real-time 5-second bars
━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            tn.send_message(message)
        except Exception as e:
            print(f"Error sending startup notification: {e}")
    
    # Create and start trader
    trader = LiveTrader(ib, symbols, risk_pct)
    trader.account_value = account_value
    trader.positions = positions
    
    try:
        trader.start()
        
        # Run forever (or until interrupted)
        while True:
            ib.sleep(1)
            
            # Refresh account value periodically
            try:
                for av in ib.accountValues():
                    if av.tag == 'NetLiquidation' and av.currency == 'USD':
                        trader.account_value = float(av.value)
                        break
            except:
                pass
                
    except KeyboardInterrupt:
        print("\n\nShutdown requested...")
    finally:
        trader.stop()
        ib.disconnect()
        print(f"\nTotal trades executed: {trader.trade_count}")
        print(f"Final account value: ${trader.account_value:,.2f}")


if __name__ == "__main__":
    import asyncio
    asyncio.set_event_loop(asyncio.new_event_loop())
    
    import argparse
    
    parser = argparse.ArgumentParser(description='ICT V5 - IBKR Trading')
    parser.add_argument("--symbols", default="SPX,NDX,GC,ES,NQ,EURUSD,BTCUSD", 
                        help="Comma-separated symbols")
    parser.add_argument("--interval", type=int, default=30, 
                        help="Check interval in seconds")
    parser.add_argument("--risk", type=float, default=0.02, 
                        help="Risk per trade (0.02 = 2%%)")
    parser.add_argument("--port", type=int, default=7497, 
                        help="IBKR port (7497=paper, 7496=live)")
    parser.add_argument("--backtest", action="store_true",
                        help="Run backtest instead of live trading")
    parser.add_argument("--days", type=int, default=180,
                        help="Backtest days")
    parser.add_argument("--yahoo", action="store_true",
                        help="Use Yahoo Finance instead of IBKR for data")
    
    args = parser.parse_args()
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    if args.backtest:
        run_backtest(symbols, days=args.days, use_ibkr=not args.yahoo)
    else:
        run_ibkr_trading(
            symbols=symbols,
            interval=args.interval,
            risk_pct=args.risk,
            port=args.port
        )
