"""
ICT V4 IBKR - RL-Enhanced Live Trading System
===============================================
Trade 24/7 with kill zone bonus using Interactive Brokers.
Focus on crypto: BTC, ETH, SOL.

Changes from V4 (Alpaca):
- Uses IBKR instead of Alpaca
- Real-time data from IBKR
- Supports crypto and futures

Usage:
    python3 ict_v4_ibkr.py --symbols "BTCUSD,ETHUSD,SOLUSD" --interval 30
"""

import asyncio
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

# Set up event loop for ib_insync
try:
    asyncio.get_event_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

np.random.seed(42)
random.seed(42)

Q_TABLE_FILE = "v4_ibkr_q_table.pkl"

# Try to import IBKR
try:
    from ib_insync import IB, Crypto, Forex, Future, Stock, util
    IBKR_AVAILABLE = True
except ImportError:
    IBKR_AVAILABLE = False
    print("WARNING: ib_insync not installed. Run: pip install ib_insync")


def get_ibkr_contract(symbol):
    """Convert symbol string to IBKR contract."""
    from ib_insync import Stock, Forex, Future, Crypto
    
    symbol = symbol.upper()
    
    # Crypto - IBKR uses base symbol (BTC, not BTCUSD)
    crypto_map = {
        'BTCUSD': 'BTC',
        'ETHUSD': 'ETH',
        'SOLUSD': 'SOL',
        'LTCUSD': 'LTC',
    }
    
    if symbol in crypto_map:
        return Crypto(crypto_map[symbol], exchange='PAXOS', currency='USD')
    
    # Forex
    forex_map = {
        'EURUSD': 'EURUSD',
        'GBPUSD': 'GBPUSD',
        'USDJPY': 'USDJPY',
    }
    
    if symbol in forex_map:
        return Forex(forex_map[symbol])
    
    # Futures
    futures_map = {
        'ES': ('ES', 'CME', 'USD', '202603', '50'),
        'NQ': ('NQ', 'CME', 'USD', '202603', '20'),
        'GC': ('GC', 'COMEX', 'USD', '202604', '100'),
        'SI': ('SI', 'COMEX', 'USD', '202603', '5000'),
    }
    
    if symbol in futures_map:
        fut = futures_map[symbol]
        return Future(fut[0], exchange=fut[1], currency=fut[2], 
                     lastTradeDateOrContractMonth=fut[3], multiplier=fut[4])
    
    # Default to stock
    return Stock(symbol, 'SMART', 'USD')


def fetch_ibkr_data(symbol, days=30, interval="1h"):
    """Fetch historical data from IBKR."""
    if not IBKR_AVAILABLE:
        return None
    
    try:
        from ib_insync import IB, util
    except ImportError:
        return None
    
    contract = get_ibkr_contract(symbol)
    
    ib = IB()
    try:
        ib.connect('127.0.0.1', 7497, clientId=random.randint(1000, 50000))
    except Exception as e:
        print(f"Could not connect to IBKR: {e}")
        return None
    
    duration = f"{days} D"
    bar_size = "1 hour" if interval == "1h" else "1 day"
    
    try:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=duration,
            barSizeSetting=bar_size,
            whatToShow='MIDPOINT',
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


def prepare_data_ibkr(symbol, lookback=200):
    """Prepare data using IBKR as primary source, Yahoo as fallback."""
    # Try IBKR first
    df = fetch_ibkr_data(symbol, days=10, interval="1h")
    
    if df is None or len(df) < 50:
        print(f"IBKR failed for {symbol}, using Yahoo...")
        # Fall back to Yahoo Finance
        yahoo_map = {
            'BTCUSD': 'BTC-USD',
            'ETHUSD': 'ETH-USD',
            'SOLUSD': 'SOL-USD',
        }
        yahoo_symbol = yahoo_map.get(symbol, symbol)
        
        df = yf.Ticker(yahoo_symbol).history(period="10d", interval="1h")
        df = df.dropna()
        df = df[~df.index.duplicated(keep='first')]
        
        if len(df) < 50:
            return None
    else:
        # Rename IBKR columns to match Yahoo format
        df = df.rename(columns={
            'open': 'Open',
            'high': 'High',
            'low': 'Low',
            'close': 'Close',
            'volume': 'Volume'
        })
    
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    opens = df['Open'].values
    
    bullish_fvgs = []
    bearish_fvgs = []
    for i in range(3, len(df)):
        if lows[i] > highs[i-2]:
            bullish_fvgs.append({'idx': i, 'mid': (highs[i-2] + lows[i]) / 2, 'high': lows[i]})
        if highs[i] < lows[i-2]:
            bearish_fvgs.append({'idx': i, 'mid': (highs[i] + lows[i-2]) / 2, 'low': highs[i]})
    
    # Get daily data for HTF trend
    df_daily = fetch_ibkr_data(symbol, days=30, interval="1d")
    if df_daily is None or len(df_daily) < 5:
        # Try Yahoo for daily
        yahoo_symbol = yahoo_map.get(symbol, symbol)
        df_daily = yf.Ticker(yahoo_symbol).history(period="30d", interval="1d")
    
    if df_daily is None or len(df_daily) < 5:
        htf_trend = np.zeros(len(df))
    else:
        # Rename if from IBKR
        if 'high' in df_daily.columns:
            df_daily = df_daily.rename(columns={
                'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close'
            })
        
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


# Alias for backward compatibility
def prepare_data(symbol, lookback=200):
    """Alias for prepare_data_ibkr."""
    return prepare_data_ibkr(symbol, lookback)


def build_state(data, idx, position=None):
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
    
    near_bull_fvg = next((f for f in reversed(data['bullish_fvgs']) if f['idx'] < idx and f['mid'] < current_price < f['high']), None)
    near_bear_fvg = next((f for f in reversed(data['bearish_fvgs']) if f['idx'] < idx and f['low'] < current_price < f['mid']), None)
    
    fvg_dist = 0
    if near_bull_fvg:
        fvg_dist = (current_price - near_bull_fvg['mid']) / price_scale
    elif near_bear_fvg:
        fvg_dist = (near_bear_fvg['mid'] - current_price) / price_scale
    
    recent_low = np.min(lows[max(0, idx-20):idx]) if idx > 20 else lows[0]
    recent_high = np.max(highs[max(0, idx-20):idx]) if idx > 20 else highs[0]
    liquidity_sweep = 0
    if lows[idx] < recent_low:
        liquidity_sweep = 1
    elif highs[idx] > recent_high:
        liquidity_sweep = -1
    
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


def run_live_trading(symbols, interval=30, risk_pct=0.04, port=7497, log_file="v4_ibkr_live.log", train=True):
    """Run live trading via Interactive Brokers."""
    if not IBKR_AVAILABLE:
        print("ERROR: ib_insync not installed. Run: pip install ib_insync")
        return
    
    # Connect to IBKR
    ib = IB()
    try:
        ib.connect('127.0.0.1', port, clientId=random.randint(1000, 50000))
        print(f"Connected to IBKR (port {port})")
    except Exception as e:
        print(f"ERROR connecting to IBKR: {e}")
        print("Make sure IBKR Gateway or TWS is running!")
        return
    
    # Get account info
    try:
        account_values = ib.accountValues()
        cash = next((float(v.value) for v in account_values if v.tag == 'CashBalance' and v.currency == 'USD'), 0)
        print(f"Account Cash: ${cash:,.2f}")
    except Exception as e:
        print(f"Warning: Could not get account info: {e}")
        cash = 0
    
    agents = {}
    positions = {}
    data_cache = {}
    
    # Try to load existing Q-table
    if train:
        combined_agent = QLearningAgent(20, 8)
        if combined_agent.load(Q_TABLE_FILE):
            print(f"Loaded pre-trained Q-table - agent will continue learning")
        else:
            print(f"No pre-trained Q-table found - starting fresh")
            print(f"Agent will learn from live trading (may take time to improve)")
        # Use same agent for all symbols (transfer learning)
        for symbol in symbols:
            agents[symbol] = combined_agent
    else:
        for symbol in symbols:
            agents[symbol] = QLearningAgent(20, 8)
    
    print(f"Starting V3 Live Trading")
    print(f"Symbols: {symbols}")
    print(f"Risk per trade: {risk_pct*100}%")
    print(f"Check interval: {interval} seconds")
    print(f"Learning: {'Enabled' if train else 'Disabled'}")
    print("-" * 50)
    
    trade_count = 0
    
    while True:
        try:
            for symbol in symbols:
                data = prepare_data(symbol)
                if data is None or len(data.get('closes', [])) < 50:
                    continue
                
                idx = len(data['closes']) - 1
                
                if symbol not in agents:
                    if train:
                        agents[symbol] = combined_agent
                    else:
                        agents[symbol] = QLearningAgent(20, 8)
                
                agent = agents[symbol]
                
                current_position = positions.get(symbol)
                state = build_state(data, idx, current_position)
                signal = get_signal(data, idx)
                
                if current_position is None:
                    action = agent.act(state, training=False)
                    
                    if signal and action in [Actions.ENTRY_NOW, Actions.ENTRY_PULLBACK, Actions.ENTRY_LIMIT]:
                        entry_price = data['closes'][idx]
                        direction = signal['direction']
                        qty = 1
                        side = "buy" if direction == 1 else "sell"
                        
                        try:
                            from ib_insync import MarketOrder
                            contract = get_ibkr_contract(symbol)
                            action = "BUY" if direction == 1 else "SELL"
                            order = MarketOrder(action, qty)
                            trade = ib.placeOrder(contract, order)
                            
                            log_msg = f"{datetime.now()} | {symbol} | {side.upper()} | Entry: ${entry_price:.2f} | Confluence: {signal['confluence']}"
                            print(log_msg)
                            with open(log_file, "a") as f:
                                f.write(log_msg + "\n")
                            
                            positions[symbol] = {
                                'direction': direction,
                                'entry': entry_price,
                                'stop_loss': entry_price * (1 - risk_pct if direction == 1 else 1 + risk_pct),
                                'take_profit': entry_price * (1 + risk_pct * 2 if direction == 1 else 1 - risk_pct * 2),
                                'bars_held': 0,
                                'entry_time': datetime.now()
                            }
                        except Exception as e:
                            print(f"Order error: {e}")
                
                else:
                    current_price = data['closes'][idx]
                    direction = current_position['direction']
                    
                    pnl_pct = (current_price - current_position['entry']) / current_position['entry'] if direction == 1 else (current_position['entry'] - current_price) / current_position['entry']
                    
                    hit_stop = (direction == 1 and current_price <= current_position['stop_loss']) or (direction == -1 and current_price >= current_position['stop_loss'])
                    hit_target = (direction == 1 and current_price >= current_position['take_profit']) or (direction == -1 and current_price <= current_position['take_profit'])
                    
                    current_position['bars_held'] += 1
                    
                    if hit_stop or hit_target or current_position['bars_held'] >= 20:
                        side = "sell" if direction == 1 else "buy"
                        pnl = current_price - current_position['entry'] if direction == 1 else current_position['entry'] - current_price
                        
                        try:
                            from ib_insync import MarketOrder
                            contract = get_ibkr_contract(symbol)
                            close_action = "SELL" if direction == 1 else "BUY"
                            close_order = MarketOrder(close_action, 1)
                            ib.placeOrder(contract, close_order)
                            
                            result = "WIN" if pnl > 0 else "LOSS"
                            log_msg = f"{datetime.now()} | {symbol} | {side.upper()} | Exit: ${current_price:.2f} | PnL: ${pnl:.2f} | {result}"
                            print(log_msg)
                            with open(log_file, "a") as f:
                                f.write(log_msg + "\n")
                            
                            if train:
                                trade_count += 1
                                agent.save(Q_TABLE_FILE)
                                print(f"Q-table saved (total trades: {trade_count})")
                        except Exception as e:
                            print(f"Close error: {e}")
                        
                        del positions[symbol]
                    
                    else:
                        action = agent.act(state, training=False)
                        
                        if action == Actions.EXIT_NOW:
                            side = "sell" if direction == 1 else "buy"
                            pnl = current_price - current_position['entry'] if direction == 1 else current_position['entry'] - current_price
                            
                            try:
                                from ib_insync import MarketOrder
                                contract = get_ibkr_contract(symbol)
                                close_action = "SELL" if direction == 1 else "BUY"
                                close_order = MarketOrder(close_action, 1)
                                ib.placeOrder(contract, close_order)
                                
                                result = "WIN" if pnl > 0 else "LOSS"
                                log_msg = f"{datetime.now()} | {symbol} | RL EXIT | Exit: ${current_price:.2f} | PnL: ${pnl:.2f} | {result}"
                                print(log_msg)
                                with open(log_file, "a") as f:
                                    f.write(log_msg + "\n")
                            except Exception as e:
                                print(f"RL exit error: {e}")
                            
                            del positions[symbol]
                        
                        elif action == Actions.MOVE_STOP_BE:
                            current_position['stop_loss'] = current_position['entry']
                        
                        elif action == Actions.TRAIL_STOP:
                            if direction == 1:
                                current_position['stop_loss'] = current_price * 0.99
                            else:
                                current_position['stop_loss'] = current_price * 1.01
            
            cycle_count = getattr(run_live_trading, 'cycle_count', 0) + 1
            run_live_trading.cycle_count = cycle_count
            if cycle_count % 4 == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Monitoring {len(symbols)} symbols | Positions: {len(positions)}")
            
            time.sleep(interval)
            
        except KeyboardInterrupt:
            print("\nStopping V4 Live Trading...")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(interval)
            
        except KeyboardInterrupt:
            print("\nStopping V3 Live Trading...")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(interval)


def run_backtest(symbols, days=180, use_ibkr=True):
    """Run backtest on historical data."""
    print(f"\n{'='*50}")
    print(f"ICT V4 IBKR Backtest - {days} days")
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


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ICT V4 IBKR Live Trading")
    parser.add_argument("--symbols", default="BTCUSD,ETHUSD,SOLUSD", help="Comma-separated symbols (BTCUSD,ETHUSD,SOLUSD)")
    parser.add_argument("--interval", type=int, default=30, help="Check interval in seconds")
    parser.add_argument("--risk", type=float, default=0.04, help="Risk per trade")
    parser.add_argument("--port", type=int, default=7497, help="IBKR port (7497=paper, 7496=live)")
    parser.add_argument("--train", action="store_true", default=True, help="Enable RL learning")
    parser.add_argument("--no-train", dest="train", action="store_false", help="Disable RL learning")
    parser.add_argument("--backtest", action="store_true", help="Run backtest instead of live trading")
    parser.add_argument("--days", type=int, default=180, help="Backtest days")
    parser.add_argument("--yahoo", action="store_true", help="Use Yahoo Finance instead of IBKR for data")
    
    args = parser.parse_args()
    
    symbols = [s.strip() for s in args.symbols.split(",")]
    
    if args.backtest:
        run_backtest(symbols, days=args.days, use_ibkr=not args.yahoo)
    else:
        run_live_trading(symbols, args.interval, args.risk, port=args.port, train=args.train)
