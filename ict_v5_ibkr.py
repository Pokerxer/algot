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


def fetch_ibkr_data(symbol, days=30, interval="1h"):
    """Fetch historical data from IBKR."""
    try:
        from ib_insync import IB, util
    except ImportError:
        print("ERROR: ib_insync not installed. Run: pip install ib_insync")
        return None
    
    contract = get_ibkr_contract(symbol)
    
    ib = IB()
    try:
        ib.connect('127.0.0.1', 7497, clientId=99)
    except Exception as e:
        print(f"Could not connect to IBKR: {e}")
        return None
    
    duration = f"{days} D"
    bar_size = "1 hour" if interval == "1h" else "1 day"
    
    try:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',  # Empty for all
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow='MIDPOINT',  # Use MIDPOINT for all
            useRTH=False,
            formatDate=2
        )
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        ib.disconnect()
        return None
    
    ib.disconnect()
    
    if not bars:
        return None
    
    df = util.df(bars)
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index)
    
    return df


def prepare_data_ibkr(symbol, lookback=200):
    """Prepare data using IBKR as primary source, Yahoo as fallback."""
    df = fetch_ibkr_data(symbol, days=30, interval="1h")
    
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


def run_ibkr_trading(symbols, interval=30, risk_pct=0.02, port=7497, train=True):
    """Run live trading via IBKR."""
    try:
        from ib_insync import IB, MarketOrder
    except ImportError:
        print("ERROR: ib_insync not installed. Run: pip install ib_insync")
        return
    
    # Connect to IBKR
    ib = IB()
    
    try:
        # Paper trading port 7497, live 7496
        ib.connect('127.0.0.1', port, clientId=1)
        print(f"Connected to IBKR (port {port})")
    except Exception as e:
        print(f"ERROR connecting to IBKR: {e}")
        print("Make sure IBKR Gateway or TWS is running!")
        return
    
    # Get account info
    try:
        account = ib.accountValues()
        for av in account:
            if av.tag == 'CashBalance' and av.currency == 'USD':
                print(f"Cash balance: ${float(av.value):,.2f}")
                break
    except Exception as e:
        print(f"Warning: Could not get account info: {e}")
    
    # Setup agents
    agents = {}
    positions = {}
    
    if train:
        combined_agent = QLearningAgent(20, 8)
        if combined_agent.load(Q_TABLE_FILE):
            print(f"Loaded Q-table - continuing to learn")
        else:
            print(f"No Q-table found - starting fresh")
        for symbol in symbols:
            agents[symbol] = combined_agent
    else:
        for symbol in symbols:
            agents[symbol] = QLearningAgent(20, 8)
    
    print(f"\nICT V5 - IBKR Trading")
    print(f"Symbols: {symbols}")
    print(f"Risk per trade: {risk_pct*100}%")
    print(f"Check interval: {interval}s")
    print("-" * 50)
    
    # Send startup notification to Telegram
    if tn:
        try:
            message = f"""
🚀 <b>V5 Trading Bot Started</b>
━━━━━━━━━━━━━━━━━━━━
<b>Symbols:</b> {', '.join(symbols)}
<b>Risk:</b> {risk_pct*100}%
<b>Interval:</b> {interval}s
<b>Mode:</b> {'Paper Trading' if port == 7497 else 'Live Trading'}
━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            tn.send_message(message)
        except Exception as e:
            print(f"Error sending startup notification: {e}")
    
    trade_count = 0
    
    while True:
        try:
            for symbol in symbols:
                data = prepare_data_ibkr(symbol)
                if data is None or len(data.get('closes', [])) < 50:
                    continue
                
                idx = len(data['closes']) - 1
                current_price = data['closes'][idx]
                
                # Get signal
                signal = get_signal(data, idx)
                
                # Send Telegram notification for signal
                if signal and tn:
                    try:
                        htf = data['htf_trend'][idx]
                        ltf = data['ltf_trend'][idx]
                        kz = data['kill_zone'][idx]
                        pp = data['price_position'][idx]
                        tp = current_price + (current_price - data['lows'][idx]) * 2 if signal['direction'] == 1 else current_price - (data['highs'][idx] - current_price) * 2
                        sl = data['lows'][idx] if signal['direction'] == 1 else data['highs'][idx]
                        tn.send_signal(symbol, signal['direction'], signal['confluence'], current_price, tp, sl, htf, ltf, kz, pp)
                    except Exception as e:
                        print(f"Error sending signal notification: {e}")
                
                # Check existing position
                if symbol in positions:
                    pos = positions[symbol]
                    
                    # Check stop/target
                    if pos['direction'] == 1:  # Long
                        if data['lows'][idx] <= pos['stop']:
                            # Stop hit
                            try:
                                contract = get_ibkr_contract(symbol)
                                ib.placeOrder(contract, MarketOrder('SELL', pos['qty']))
                                print(f"[{symbol}] STOP HIT: {pos['stop']:.2f}")
                                
                                # Calculate PnL and send notification
                                if tn:
                                    try:
                                        pnl = (pos['stop'] - pos['entry']) * pos['qty']
                                        tn.send_trade_exit(symbol, pos['direction'], pnl, 'stop_loss', pos['entry'], pos['stop'], pos.get('bars_held', 0))
                                    except Exception as e:
                                        print(f"Error sending exit notification: {e}")
                            except Exception as e:
                                print(f"Error closing position: {e}")
                            del positions[symbol]
                        elif data['highs'][idx] >= pos['target']:
                            # Target hit
                            try:
                                contract = get_ibkr_contract(symbol)
                                ib.placeOrder(contract, MarketOrder('SELL', pos['qty']))
                                print(f"[{symbol}] TARGET HIT: {pos['target']:.2f}")
                                
                                # Calculate PnL and send notification
                                if tn:
                                    try:
                                        pnl = (pos['target'] - pos['entry']) * pos['qty']
                                        tn.send_trade_exit(symbol, pos['direction'], pnl, 'take_profit', pos['entry'], pos['target'], pos.get('bars_held', 0))
                                    except Exception as e:
                                        print(f"Error sending exit notification: {e}")
                            except Exception as e:
                                print(f"Error closing position: {e}")
                            del positions[symbol]
                    else:
                        # Short position - check stop/target
                        if pos['direction'] == -1:
                            if data['highs'][idx] >= pos['stop']:
                                # Stop hit (short)
                                try:
                                    contract = get_ibkr_contract(symbol)
                                    ib.placeOrder(contract, MarketOrder('BUY', pos['qty']))
                                    print(f"[{symbol}] STOP HIT (SHORT): {pos['stop']:.2f}")
                                    
                                    # Calculate PnL and send notification
                                    if tn:
                                        try:
                                            pnl = (pos['entry'] - pos['stop']) * pos['qty']
                                            tn.send_trade_exit(symbol, pos['direction'], pnl, 'stop_loss', pos['entry'], pos['stop'], pos.get('bars_held', 0))
                                        except Exception as e:
                                            print(f"Error sending exit notification: {e}")
                                except Exception as e:
                                    print(f"Error closing position: {e}")
                                del positions[symbol]
                            elif data['lows'][idx] <= pos['target']:
                                # Target hit (short)
                                try:
                                    contract = get_ibkr_contract(symbol)
                                    ib.placeOrder(contract, MarketOrder('BUY', pos['qty']))
                                    print(f"[{symbol}] TARGET HIT (SHORT): {pos['target']:.2f}")
                                    
                                    # Calculate PnL and send notification
                                    if tn:
                                        try:
                                            pnl = (pos['entry'] - pos['target']) * pos['qty']
                                            tn.send_trade_exit(symbol, pos['direction'], pnl, 'take_profit', pos['entry'], pos['target'], pos.get('bars_held', 0))
                                        except Exception as e:
                                            print(f"Error sending exit notification: {e}")
                                except Exception as e:
                                    print(f"Error closing position: {e}")
                                del positions[symbol]
                else:
                    # No position - check for entry
                    if signal:
                        # Calculate position size
                        account_value = 100000
                        try:
                            account = ib.accountValues()
                            for av in account:
                                if av.tag == 'CashBalance' and av.currency == 'USD':
                                    account_value = float(av.value)
                                    break
                        except:
                            pass
                        risk_amount = account_value * risk_pct
                        stop_dist = current_price - data['lows'][idx] if signal['direction'] == 1 else data['highs'][idx] - current_price
                        if stop_dist > 0:
                            qty = int(risk_amount / stop_dist)
                            if qty > 0:
                                try:
                                    contract = get_ibkr_contract(symbol)
                                    if signal['direction'] == 1:
                                        order = MarketOrder('BUY', qty)
                                    else:
                                        order = MarketOrder('SELL', qty)
                                    ib.placeOrder(contract, order)
                                    
                                    positions[symbol] = {
                                        'entry': current_price,
                                        'stop': data['lows'][idx] if signal['direction'] == 1 else data['highs'][idx],
                                        'target': current_price + (current_price - data['lows'][idx]) * 2 if signal['direction'] == 1 else current_price - (data['highs'][idx] - current_price) * 2,
                                        'direction': signal['direction'],
                                        'qty': qty,
                                        'confluence': signal['confluence'],
                                        'bars_held': 0
                                    }
                                    print(f"[{symbol}] ENTRY: {signal['direction']} @ {current_price:.2f} (conf: {signal['confluence']})")
                                    trade_count += 1
                                    
                                    # Send Telegram notification for trade entry
                                    if tn:
                                        try:
                                            tp = positions[symbol]['target']
                                            sl = positions[symbol]['stop']
                                            tn.send_trade_entry(symbol, signal['direction'], qty, current_price, signal['confluence'], tp, sl)
                                        except Exception as e:
                                            print(f"Error sending entry notification: {e}")
                                except Exception as e:
                                    print(f"Error placing order: {e}")
                
                # Update position hold time
                if symbol in positions:
                    positions[symbol]['bars_held'] += 1
                
                # RL Learning (if enabled)
                if train and agents.get(symbol):
                    state = build_state(data, idx, positions.get(symbol))
                    action = agents[symbol].act(state, training=True)
                    
                    # Simple reward: based on price movement
                    if positions.get(symbol):
                        pos = positions[symbol]
                        pnl = (current_price - pos['entry']) * pos['direction'] / pos['entry']
                        reward = pnl * 10
                    else:
                        reward = 0
                    
                    # Get next state (simplified)
                    next_state = state  # In real implementation, would fetch next bar
                    agents[symbol].update_q(state, action, reward, next_state)
            
            # Save Q-table periodically
            if trade_count > 0 and trade_count % 10 == 0:
                for symbol, agent in agents.items():
                    agent.save(Q_TABLE_FILE)
            
            time.sleep(interval)
            
        except KeyboardInterrupt:
            print("\nStopping...")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(5)
    
    # Cleanup
    ib.disconnect()


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
    parser.add_argument("--no-train", action="store_true", 
                        help="Disable RL learning")
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
            port=args.port,
            train=not args.no_train
        )
