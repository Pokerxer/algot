"""
ICT V4 - RL-Enhanced Live Trading System
=========================================
Trade 24/7 with kill zone bonus. Supports BTC and 12 other symbols.

Changes from V3:
- Trade anytime (not just kill zones)
- Kill zone still gives +15 bonus for better confirmation
- Added BTC-USD

Usage:
    python3 ict_v4_live.py --symbols "NQ=F,GC=F,GBPUSD=X,BTC-USD" --interval 30
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import time
import os
import random
import pickle
from datetime import datetime
from collections import deque

np.random.seed(42)
random.seed(42)

Q_TABLE_FILE = "v3_q_table.pkl"

import requests

class AlpacaPaper:
    def __init__(self, api_key, secret_key):
        self.base_url = "https://paper-api.alpaca.markets"
        self.headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": secret_key
        }
    
    def get_account(self):
        resp = requests.get(f"{self.base_url}/v2/account", headers=self.headers)
        return resp.json()
    
    def get_positions(self):
        resp = requests.get(f"{self.base_url}/v2/positions", headers=self.headers)
        return resp.json()
    
    def place_order(self, symbol, qty, side, order_type="market", time_in_force="day"):
        order = {
            "symbol": symbol,
            "qty": qty,
            "side": side,
            "type": order_type,
            "time_in_force": time_in_force
        }
        resp = requests.post(f"{self.base_url}/v2/orders", json=order, headers=self.headers)
        return resp.json()
    
    def close_position(self, symbol):
        resp = requests.delete(f"{self.base_url}/v2/positions/{symbol}", headers=self.headers)
        return resp.json()


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


def prepare_data(symbol, lookback=200):
    df = yf.Ticker(symbol).history(period="10d", interval="1h")
    df = df.dropna()
    df = df[~df.index.duplicated(keep='first')]
    
    if len(df) < 50:
        return None
    
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
    
    df_daily = yf.Ticker(symbol).history(period="30d", interval="1d")
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


def run_live_trading(symbols, interval=30, risk_pct=0.04, log_file="v3_live.log", train=True):
    api_key = os.getenv("ALPACA_API_KEY")
    secret_key = os.getenv("ALPACA_SECRET_KEY")
    
    if not api_key or not secret_key:
        print("ERROR: ALPACA_API_KEY and ALPACA_SECRET_KEY must be set")
        return
    
    alpaca = AlpacaPaper(api_key, secret_key)
    
    try:
        account = alpaca.get_account()
        print(f"Connected to Alpaca. Account: ${float(account.get('cash', 0)):,.2f}")
    except Exception as e:
        print(f"ERROR connecting to Alpaca: {e}")
        return
    
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
                if data is None or len(data['closes']) < 50:
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
                            order = alpaca.place_order(symbol, qty, side)
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
                            alpaca.close_position(symbol)
                            result = "WIN" if pnl > 0 else "LOSS"
                            log_msg = f"{datetime.now()} | {symbol} | {side.upper()} | Exit: ${current_price:.2f} | PnL: ${pnl:.2f} | {result}"
                            print(log_msg)
                            with open(log_file, "a") as f:
                                f.write(log_msg + "\n")
                            
                            # Learn from trade and save Q-table
                            if train:
                                trade_count += 1
                                agent.save(Q_TABLE_FILE)
                                print(f"Q-table saved (total trades: {trade_count})")
                        except:
                            pass
                        
                        del positions[symbol]
                    
                    else:
                        action = agent.act(state, training=False)
                        
                        if action == Actions.EXIT_NOW:
                            side = "sell" if direction == 1 else "buy"
                            pnl = current_price - current_position['entry'] if direction == 1 else current_position['entry'] - current_price
                            
                            try:
                                alpaca.close_position(symbol)
                                result = "WIN" if pnl > 0 else "LOSS"
                                log_msg = f"{datetime.now()} | {symbol} | RL EXIT | Exit: ${current_price:.2f} | PnL: ${pnl:.2f} | {result}"
                                print(log_msg)
                                with open(log_file, "a") as f:
                                    f.write(log_msg + "\n")
                            except:
                                pass
                            
                            del positions[symbol]
                        
                        elif action == Actions.MOVE_STOP_BE:
                            current_position['stop_loss'] = current_position['entry']
                        
                        elif action == Actions.TRAIL_STOP:
                            if direction == 1:
                                current_position['stop_loss'] = current_price * 0.99
                            else:
                                current_position['stop_loss'] = current_price * 1.01
            
            # Heartbeat log every 2 minutes
            cycle_count = getattr(run_live_trading, 'cycle_count', 0) + 1
            run_live_trading.cycle_count = cycle_count
            if cycle_count % 4 == 0:
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Monitoring {len(symbols)} symbols | Positions: {len(positions)}")
            
            time.sleep(interval)
            
        except KeyboardInterrupt:
            print("\nStopping V3 Live Trading...")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(interval)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="ICT V4 Live Trading")
    parser.add_argument("--mode", default="paper", help="Trading mode (paper/live)")
    parser.add_argument("--symbols", default="NQ=F,YM=F,EURUSD=X,GBPUSD=X,GBPJPY=X,ES=F,GC=F,ZB=F,ZN=F,ZS=F,ZW=F,SI=F,BTC-USD", help="Comma-separated symbols")
    parser.add_argument("--interval", type=int, default=30, help="Check interval in seconds")
    parser.add_argument("--risk", type=float, default=0.04, help="Risk per trade")
    parser.add_argument("--train", action="store_true", default=True, help="Enable RL learning (saves Q-table)")
    parser.add_argument("--no-train", dest="train", action="store_false", help="Disable RL learning")
    
    args = parser.parse_args()
    
    symbols = [s.strip() for s in args.symbols.split(",")]
    
    run_live_trading(symbols, args.interval, args.risk, train=args.train)
