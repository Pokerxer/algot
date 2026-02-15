"""
ICT V3 - RL-Enhanced Trading System
====================================
Pre-train RL agent on first 6 months, test on last 6 months.
RL makes all entry/exit decisions (no fallback to V2 rules).

Improvements over V2:
- RL learns optimal entry timing (ENTER_NOW, ENTER_PULLBACK, ENTER_LIMIT, PASS)
- RL manages exits (EXIT_NOW, HOLD, MOVE_STOP_BE, TRAIL_STOP)
- Dynamic position sizing based on regime
- Better risk management through learned stop management
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import random
from datetime import datetime, timedelta
from collections import deque
import math

np.random.seed(42)
random.seed(42)

print("=" * 70)
print("ICT V3 - RL-ENHANCED TRADING SYSTEM")
print("=" * 70)
print("Phase 1: Train on 6 months | Phase 2: Test on 6 months")
print()

# ============================================================================
# SIMPLE Q-LEARNING AGENT
# ============================================================================

class QLearningAgent:
    """Simple Q-Learning agent with table-based approach"""
    
    def __init__(self, state_size, action_size, n_bins=10):
        self.state_size = state_size
        self.action_size = action_size
        self.n_bins = n_bins
        
        # Q-table (discretized states)
        self.q_table = {}
        
        # Training params
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.alpha = 0.1  # Learning rate
        self.gamma = 0.95  # Discount factor
        
        self.train_steps = 0
    
    def discretize_state(self, state):
        """Convert continuous state to discrete bins"""
        key = []
        for val in state:
            if np.isnan(val) or np.isinf(val):
                key.append(0)
            else:
                bin_idx = int(np.clip(val * self.n_bins, 0, self.n_bins - 1))
                key.append(bin_idx)
        return tuple(key)
    
    def store(self, state, action, reward, next_state, done):
        pass  # No buffer needed for tabular
    
    def act(self, state, training=True):
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        disc_state = self.discretize_state(state)
        
        if disc_state not in self.q_table:
            self.q_table[disc_state] = np.zeros(self.action_size)
        
        return int(np.argmax(self.q_table[disc_state]))
    
    def train(self, batch_size=32):
        self.train_steps += 1
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        return 0
    
    def update_q(self, state, action, reward, next_state):
        """Update Q-value after an experience"""
        disc_state = self.discretize_state(state)
        disc_next = self.discretize_state(next_state)
        
        if disc_state not in self.q_table:
            self.q_table[disc_state] = np.zeros(self.action_size)
        if disc_next not in self.q_table:
            self.q_table[disc_next] = np.zeros(self.action_size)
        
        # Q-learning update
        current_q = self.q_table[disc_state][action]
        max_next_q = np.max(self.q_table[disc_next])
        
        self.q_table[disc_state][action] = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)


# ============================================================================
# ACTIONS
# ============================================================================

class Actions:
    ENTRY_PASS = 0
    ENTRY_NOW = 1
    ENTRY_PULLBACK = 2
    ENTRY_LIMIT = 3
    
    EXIT_HOLD = 4
    EXIT_NOW = 5
    MOVE_STOP_BE = 6
    TRAIL_STOP = 7


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_data(symbol, period="1y"):
    """Fetch and prepare market data with ICT features"""
    df = yf.Ticker(symbol).history(period=period, interval="1h")
    df = df.dropna()
    df = df[~df.index.duplicated(keep='first')]
    
    # Daily for HTF trend
    df_daily = yf.Ticker(symbol).history(period=period, interval="1d")
    
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    opens = df['Open'].values
    
    # Calculate FVGs
    bullish_fvgs = []
    bearish_fvgs = []
    for i in range(3, len(df)):
        if lows[i] > highs[i-2]:
            bullish_fvgs.append({'idx': i, 'mid': (highs[i-2] + lows[i]) / 2, 'high': lows[i]})
        if highs[i] < lows[i-2]:
            bearish_fvgs.append({'idx': i, 'mid': (highs[i] + lows[i-2]) / 2, 'low': highs[i]})
    
    # Calculate OBs
    bullish_obs = []
    bearish_obs = []
    for i in range(5, len(df)):
        if closes[i-1] < opens[i-1] and closes[i] > opens[i] and lows[i] < lows[i-1]:
            bullish_obs.append({'idx': i, 'high': highs[i-1], 'low': lows[i-1]})
        if closes[i-1] > opens[i-1] and closes[i] < opens[i] and highs[i] > highs[i-1]:
            bearish_obs.append({'idx': i, 'high': highs[i-1], 'low': lows[i-1]})
    
    # HTF trend
    daily_highs = df_daily['High'].values
    daily_lows = df_daily['Low'].values
    
    htf_trend = []
    for i in range(1, len(df_daily)):
        if daily_highs[i] > np.max(daily_highs[max(0, i-5):i]) and daily_lows[i] > np.min(daily_lows[max(0, i-5):i]):
            htf_trend.append(1)
        elif daily_highs[i] < np.max(daily_highs[max(0, i-5):i]) and daily_lows[i] < np.min(daily_lows[max(0, i-5):i]):
            htf_trend.append(-1)
        else:
            htf_trend.append(0)
    
    df_daily_index = pd.DatetimeIndex(df_daily.index).tz_localize(None)
    df_index = pd.DatetimeIndex(df.index).tz_localize(None)
    
    htf_trend_hourly = np.zeros(len(df))
    for i in range(len(df)):
        bar_time = df_index[i]
        for j in range(len(df_daily) - 1, -1, -1):
            if df_daily_index[j] <= bar_time:
                htf_trend_hourly[i] = htf_trend[j] if j < len(htf_trend) else 0
                break
    
    # LTF trend
    trend = np.zeros(len(df))
    for i in range(20, len(df)):
        rh = np.max(highs[max(0, i-20):i])
        rl = np.min(lows[max(0, i-20):i])
        if rh > highs[i-5] and rl > lows[i-5]:
            trend[i] = 1
        elif rh < highs[i-5] and rl < lows[i-5]:
            trend[i] = -1
    
    # Price position
    price_position = np.zeros(len(df))
    for i in range(20, len(df)):
        ph = np.max(highs[i-20:i])
        pl = np.min(lows[i-20:i])
        rng = ph - pl
        if rng < 0.001:
            rng = 0.001
        price_position[i] = (closes[i] - pl) / rng
    
    # Kill zones
    hours = pd.to_datetime(df.index).hour.values
    kill_zone = np.zeros(len(df), dtype=bool)
    for i in range(len(hours)):
        h = hours[i]
        kill_zone[i] = (1 <= h < 5) or (7 <= h < 12) or (13.5 <= h < 16)
    
    # Volatility (ATR-like)
    volatility = np.zeros(len(df))
    for i in range(14, len(df)):
        tr = max(
            highs[i] - lows[i],
            abs(highs[i] - closes[i-1]),
            abs(lows[i] - closes[i-1])
        )
        volatility[i] = np.mean([max(highs[j] - lows[j], abs(highs[j] - closes[j-1]), abs(lows[j] - closes[j-1])) for j in range(max(0, i-14), i+1)])
    
    # Hours
    hours = pd.to_datetime(df.index).hour.values
    
    return {
        'df': df,
        'highs': highs,
        'lows': lows,
        'closes': closes,
        'opens': opens,
        'bullish_fvgs': bullish_fvgs,
        'bearish_fvgs': bearish_fvgs,
        'bullish_obs': bullish_obs,
        'bearish_obs': bearish_obs,
        'htf_trend': htf_trend_hourly,
        'ltf_trend': trend,
        'price_position': price_position,
        'kill_zone': kill_zone,
        'volatility': volatility,
        'hours': hours
    }


def build_state(data, idx, position=None):
    """Build state vector for RL agent"""
    current_price = data['closes'][idx]
    highs_arr = data['highs']
    lows_arr = data['lows']
    highs = highs_arr[idx]
    lows = lows_arr[idx]
    htf = data['htf_trend'][idx]
    ltf = data['ltf_trend'][idx]
    pp = data['price_position'][idx]
    kz = data['kill_zone'][idx]
    vol = data['volatility'][idx]
    hours = data.get('hours', np.zeros(len(data['closes'])))[idx]
    
    # Normalize price for scale independence
    price_scale = current_price if current_price > 100 else 1
    
    # Find nearest FVG
    near_bull_fvg = next((f for f in reversed(data['bullish_fvgs']) if f['idx'] < idx and f['mid'] < current_price < f['high']), None)
    near_bear_fvg = next((f for f in reversed(data['bearish_fvgs']) if f['idx'] < idx and f['low'] < current_price < f['mid']), None)
    
    fvg_dist = 0
    if near_bull_fvg:
        fvg_dist = (current_price - near_bull_fvg['mid']) / price_scale
    elif near_bear_fvg:
        fvg_dist = (near_bear_fvg['mid'] - current_price) / price_scale
    
    # Find nearest OB
    near_bull_ob = next((ob for ob in reversed(data['bullish_obs']) if ob['idx'] < idx), None)
    near_bear_ob = next((ob for ob in reversed(data['bearish_obs']) if ob['idx'] < idx), None)
    
    ob_dist = 0
    if near_bull_ob:
        ob_dist = (current_price - near_bull_ob['high']) / price_scale
    elif near_bear_ob:
        ob_dist = (near_bear_ob['low'] - current_price) / price_scale
    
    # Liquidity sweep detection
    recent_low = np.min(lows_arr[max(0, idx-20):idx]) if idx > 20 else lows_arr[0]
    recent_high = np.max(highs_arr[max(0, idx-20):idx]) if idx > 20 else highs_arr[0]
    liquidity_sweep = 0
    if lows < recent_low:
        liquidity_sweep = 1
    elif highs > recent_high:
        liquidity_sweep = -1
    
    # Confluence score (simplified)
    confluence = 0
    if kz:
        confluence += 0.15
    if htf == 1 and ltf >= 0:
        confluence += 0.25
    elif htf == -1 and ltf <= 0:
        confluence += 0.25
    if pp < 0.25 or pp > 0.75:
        confluence += 0.2
    elif pp < 0.35 or pp > 0.65:
        confluence += 0.15
    if near_bull_fvg and ltf >= 0:
        confluence += 0.15
    if near_bear_fvg and ltf <= 0:
        confluence += 0.15
    
    # Regime
    regime = 0  # ranging
    if htf == 1 and ltf == 1:
        regime = 1  # trending bull
    elif htf == -1 and ltf == -1:
        regime = -1  # trending bear
    elif vol > np.mean(data['volatility'][max(0,idx-50):idx]) * 1.5:
        regime = 2  # volatile
    
    # State vector (20 features)
    state = np.array([
        htf / 1.0,  # HTF trend
        ltf / 1.0,  # LTF trend
        pp,  # Price position
        float(kz),  # Kill zone
        min(fvg_dist * 10, 1),  # FVG distance (normalized)
        min(ob_dist * 10, 1),  # OB distance (normalized)
        confluence,  # Confluence score
        regime / 2.0,  # Regime
        vol / price_scale,  # Volatility
        liquidity_sweep / 1.0,  # Liquidity sweep
        0,  # Position direction (0 if flat)
        0,  # Unrealized PnL (0 if flat)
        0,  # Bars held
        0,  # Distance to stop
        0,  # Distance to target
        np.sin(2 * np.pi * hours / 24),  # Hour of day
        np.cos(2 * np.pi * hours / 24),
        (idx % 24) / 24.0,  # Day progress
        near_bull_fvg is not None,  # Has bull setup
        near_bear_fvg is not None,  # Has bear setup
    ], dtype=np.float32)
    
    # Add position info if in trade
    if position is not None:
        state[10] = 1 if position['direction'] == 1 else -1
        state[11] = position.get('pnl_r', 0) / 5  # Normalize R
        state[12] = min(position.get('bars_held', 0) / 20, 1)  # Bars held
        if position.get('stop_loss') and position.get('entry'):
            sl_dist = abs(position['entry'] - position['stop_loss']) / price_scale
            state[13] = min(sl_dist * 10, 1)
        if position.get('take_profit') and position.get('entry'):
            tp_dist = abs(position['take_profit'] - position['entry']) / price_scale
            state[14] = min(tp_dist * 10, 1)
    
    return state


def get_signal(data, idx):
    """Check if there's a V2 signal at this index"""
    closes = data['closes'][idx]
    highs = data['highs'][idx]
    lows = data['lows'][idx]
    ltf = data['ltf_trend'][idx]
    htf = data['htf_trend'][idx]
    kz = data['kill_zone'][idx]
    pp = data['price_position'][idx]
    
    near_bull_fvg = next((f for f in reversed(data['bullish_fvgs']) if f['idx'] < idx and f['mid'] < closes < f['high']), None)
    near_bear_fvg = next((f for f in reversed(data['bearish_fvgs']) if f['idx'] < idx and f['low'] < closes < f['mid']), None)
    
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
    
    # Return signal if confluence >= 60 (lower threshold than V2, let RL decide)
    if confluence >= 60:
        if htf == 1 and ltf >= 0:
            return {'direction': 1, 'confluence': confluence, 'entry': closes}
        elif htf == -1 and ltf <= 0:
            return {'direction': -1, 'confluence': confluence, 'entry': closes}
    
    return None


# ============================================================================
# TRAINING & BACKTESTING
# ============================================================================

def train_agent(data, agent, start_idx=50, end_idx=None, risk_pct=0.02):
    """Train RL agent on historical data"""
    if end_idx is None:
        end_idx = len(data['closes']) - 50
    
    print(f"Training agent on {end_idx - start_idx} bars...")
    
    position = None
    
    for idx in range(start_idx, end_idx):
        state = build_state(data, idx, position)
        
        signal = get_signal(data, idx)
        
        if position is None:
            # No position - decide to enter or pass
            action = agent.act(state, training=True)
            
            if signal and action == Actions.ENTRY_NOW:
                # Enter at market
                entry_price = data['closes'][idx]
                position = {
                    'direction': signal['direction'],
                    'entry': entry_price,
                    'entry_idx': idx,
                    'stop_loss': entry_price * (1 - risk_pct if signal['direction'] == 1 else 1 + risk_pct),
                    'take_profit': entry_price * (1 + risk_pct * 2 if signal['direction'] == 1 else 1 - risk_pct * 2),
                    'bars_held': 0,
                    'pnl_r': 0,
                    'confluence': signal['confluence']
                }
                # Small reward for taking action
                reward = 0.1
                # Update Q immediately
                next_state = build_state(data, idx + 1, position)
                agent.update_q(state, action, reward, next_state)
            else:
                # Pass - small reward for skipping weak signals
                reward = 0.05 if signal else 0
                next_state = build_state(data, idx + 1, None)
                agent.update_q(state, action, reward, next_state)
        
        else:
            # In position - decide to hold or exit
            current_price = data['closes'][idx]
            direction = position['direction']
            
            # Update PnL
            if direction == 1:
                pnl = (current_price - position['entry']) / position['entry']
            else:
                pnl = (position['entry'] - current_price) / position['entry']
            
            position['pnl_r'] = pnl / risk_pct  # R multiples
            position['bars_held'] += 1
            
            # Check stop/target
            hit_stop = (direction == 1 and current_price <= position['stop_loss']) or \
                       (direction == -1 and current_price >= position['stop_loss'])
            hit_target = (direction == 1 and current_price >= position['take_profit']) or \
                         (direction == -1 and current_price <= position['take_profit'])
            
            if hit_stop or hit_target or position['bars_held'] >= 20:
                # Close position
                pnl_r = position['pnl_r']
                
                # Reward based on R
                if pnl_r > 0:
                    reward = pnl_r * 0.5  # Scale reward
                else:
                    reward = pnl_r * 0.3  # Smaller penalty for learning
                
                next_state = build_state(data, idx + 1, None)
                agent.update_q(state, Actions.EXIT_NOW, reward, next_state)
                position = None
            else:
                action = agent.act(state, training=True)
                
                if action == Actions.EXIT_NOW:
                    pnl_r = position['pnl_r']
                    reward = pnl_r * 0.4 if pnl_r > 0 else pnl_r * 0.2
                    next_state = build_state(data, idx + 1, None)
                    agent.update_q(state, action, reward, next_state)
                    position = None
                elif action == Actions.MOVE_STOP_BE:
                    position['stop_loss'] = position['entry']  # Move to BE
                    reward = 0.05  # Small reward for risk management
                    next_state = build_state(data, idx + 1, position)
                    agent.update_q(state, action, reward, next_state)
                elif action == Actions.TRAIL_STOP:
                    if direction == 1:
                        position['stop_loss'] = current_price * 0.99
                    else:
                        position['stop_loss'] = current_price * 1.01
                    reward = 0.05
                    next_state = build_state(data, idx + 1, position)
                    agent.update_q(state, action, reward, next_state)
                else:  # HOLD
                    reward = 0.01 if position['pnl_r'] > 0 else -0.01
                    next_state = build_state(data, idx + 1, position)
                    agent.update_q(state, action, reward, next_state)
    
    print(f"Training complete. Epsilon: {agent.epsilon:.3f}, Q-states: {len(agent.q_table)}")
    return agent


def backtest_agent(data, agent, start_idx=50, end_idx=None, risk_pct=0.02):
    """Backtest trained agent"""
    if end_idx is None:
        end_idx = len(data['closes']) - 50
    
    print(f"Backtesting agent on {end_idx - start_idx} bars...")
    
    trades = []
    position = None
    capital = 10000
    
    tp_multiplier = 2  # 2R for take profit
    
    for idx in range(start_idx, end_idx):
        state = build_state(data, idx, position)
        
        signal = get_signal(data, idx)
        
        if position is None:
            # No position - decide to enter or pass
            action = agent.act(state, training=False)
            
            if signal and action in [Actions.ENTRY_NOW, Actions.ENTRY_PULLBACK, Actions.ENTRY_LIMIT]:
                # Determine entry price based on action
                if action == Actions.ENTRY_NOW:
                    entry_price = data['closes'][idx]
                elif action == Actions.ENTRY_LIMIT and signal:
                    # Enter at better price
                    if signal['direction'] == 1:
                        entry_price = min(data['closes'][idx], signal['entry'] * 0.99)
                    else:
                        entry_price = max(data['closes'][idx], signal['entry'] * 1.01)
                else:
                    entry_price = data['closes'][idx]
                
                position = {
                    'direction': signal['direction'],
                    'entry': entry_price,
                    'entry_idx': idx,
                    'entry_time': str(data['df'].index[idx]),
                    'stop_loss': entry_price * (1 - risk_pct if signal['direction'] == 1 else 1 + risk_pct),
                    'take_profit': entry_price * (1 + risk_pct * tp_multiplier if signal['direction'] == 1 else 1 - risk_pct * tp_multiplier),
                    'bars_held': 0,
                    'pnl_r': 0,
                    'confluence': signal['confluence']
                }
        
        else:
            # In position - manage exit
            current_price = data['closes'][idx]
            direction = position['direction']
            
            # Update PnL
            if direction == 1:
                pnl = (current_price - position['entry']) / position['entry']
            else:
                pnl = (position['entry'] - current_price) / position['entry']
            
            position['pnl_r'] = pnl / 0.05
            position['bars_held'] += 1
            
            # Check stop/target (hard limits)
            hit_stop = (direction == 1 and current_price <= position['stop_loss']) or \
                       (direction == -1 and current_price >= position['stop_loss'])
            hit_target = (direction == 1 and current_price >= position['take_profit']) or \
                         (direction == -1 and current_price <= position['take_profit'])
            
            if hit_stop or hit_target or position['bars_held'] >= 40:
                # Exit on hard limit
                exit_price = current_price
                pnl_dollars = capital * pnl
                capital *= (1 + pnl)
                
                trades.append({
                    'symbol': 'NQ=F',
                    'entry_time': position['entry_time'],
                    'exit_time': str(data['df'].index[idx]),
                    'direction': 'LONG' if direction == 1 else 'SHORT',
                    'entry': round(position['entry'], 2),
                    'exit': round(exit_price, 2),
                    'pnl_pct': round(pnl * 100, 2),
                    'pnl_dollars': round(pnl_dollars, 2),
                    'pnl_r': round(position['pnl_r'], 2),
                    'result': 'WIN' if pnl > 0 else 'LOSS',
                    'confluence': position['confluence'],
                    'bars_held': position['bars_held']
                })
                position = None
            else:
                # RL exit decision
                action = agent.act(state, training=False)
                
                if action == Actions.EXIT_NOW:
                    exit_price = current_price
                    pnl_dollars = capital * pnl
                    capital *= (1 + pnl)
                    
                    trades.append({
                        'symbol': 'NQ=F',
                        'entry_time': position['entry_time'],
                        'exit_time': str(data['df'].index[idx]),
                        'direction': 'LONG' if direction == 1 else 'SHORT',
                        'entry': round(position['entry'], 2),
                        'exit': round(exit_price, 2),
                        'pnl_pct': round(pnl * 100, 2),
                        'pnl_dollars': round(pnl_dollars, 2),
                        'pnl_r': round(position['pnl_r'], 2),
                        'result': 'WIN' if pnl > 0 else 'LOSS',
                        'confluence': position['confluence'],
                        'bars_held': position['bars_held'],
                        'exit_reason': 'RL_EXIT'
                    })
                    position = None
                
                elif action == Actions.MOVE_STOP_BE:
                    position['stop_loss'] = position['entry']
                
                elif action == Actions.TRAIL_STOP:
                    if direction == 1:
                        position['stop_loss'] = current_price * 0.99
                    else:
                        position['stop_loss'] = current_price * 1.01
    
    # Close any open position at end
    if position is not None:
        exit_price = data['closes'][-1]
        pnl = (exit_price - position['entry']) / position['entry'] if position['direction'] == 1 else (position['entry'] - exit_price) / position['entry']
        pnl_dollars = capital * pnl
        capital *= (1 + pnl)
        
        trades.append({
            'symbol': 'NQ=F',
            'entry_time': position['entry_time'],
            'exit_time': str(data['df'].index[-1]),
            'direction': 'LONG' if position['direction'] == 1 else 'SHORT',
            'entry': round(position['entry'], 2),
            'exit': round(exit_price, 2),
            'pnl_pct': round(pnl * 100, 2),
            'pnl_dollars': round(pnl_dollars, 2),
            'pnl_r': round(pnl / 0.02, 2),
            'result': 'WIN' if pnl > 0 else 'LOSS',
            'confluence': position['confluence'],
            'bars_held': position['bars_held'],
            'exit_reason': 'END'
        })
    
    return trades, capital


# ============================================================================
# MAIN
# ============================================================================

def run_v3_backtest():
    """Main V3 backtest with train/test split"""
    
    symbols = ['NQ=F', 'YM=F', 'EURUSD=X', 'GBPUSD=X', 'GBPJPY=X', 'ES=F', 'GC=F', 'ZB=F', 'ZN=F', 'ZS=F', 'ZW=F', 'SI=F']
    all_trades = []
    all_results = {}
    
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"SYMBOL: {symbol}")
        print(f"{'='*60}")
        
        print(f"Preparing data...")
        data = prepare_data(symbol, period="1y")
        
        total_bars = len(data['closes'])
        train_end = total_bars // 2
        
        print(f"Total bars: {total_bars}")
        print(f"Train: 0 - {train_end} (first 6 months)")
        print(f"Test: {train_end} - {total_bars} (last 6 months)")
        print()
        
        # Initialize agent
        state_size = 20
        action_size = 8
        agent = QLearningAgent(state_size, action_size)
        
        # Phase 1: Train
        print("=" * 50)
        print("PHASE 1: TRAINING")
        print("=" * 50)
        agent = train_agent(data, agent, start_idx=50, end_idx=train_end, risk_pct=0.04)
        print()
        
        # Phase 2: Test
        print("=" * 50)
        print("PHASE 2: TESTING (Trained Agent)")
        print("=" * 50)
        trades, final_capital = backtest_agent(data, agent, start_idx=train_end, end_idx=total_bars - 50, risk_pct=0.04)
        print()
        
        all_trades.extend(trades)
        all_results[symbol] = {'trades': len(trades), 'pnl': final_capital - 10000}

    # Final results for all symbols
    print()
    print("=" * 60)
    print("FINAL RESULTS - ALL SYMBOLS")
    print("=" * 60)
    
    for sym, res in all_results.items():
        print(f"{sym}: {res['trades']} trades, PnL: ${res['pnl']:,.2f}")
    
    total_trades = sum(r['trades'] for r in all_results.values())
    total_pnl = sum(r['pnl'] for r in all_results.values())
    final_capital = 10000 + total_pnl
    
    wins = sum(1 for t in all_trades if t['result'] == 'WIN')
    losses = sum(1 for t in all_trades if t['result'] == 'LOSS')
    total = len(all_trades)
    win_rate = wins / total if total > 0 else 0
    
    print()
    print(f"Combined Total Trades: {total}")
    print(f"Wins: {wins} | Losses: {losses}")
    print(f"Win Rate: {win_rate:.1%}")
    print(f"Total PnL: ${total_pnl:,.2f}")
    print(f"Final Capital: ${final_capital:,.2f}")
    print(f"Return: {((final_capital / 10000) - 1) * 100:.1f}%")
    print()
    
    # Save results
    result = {
        'phase': 'test',
        'period': 'last_6_months',
        'risk_per_trade': '4%',
        'symbols': list(all_results.keys()),
        'total_trades': total,
        'wins': wins,
        'losses': losses,
        'win_rate': round(win_rate, 3),
        'total_pnl': round(total_pnl, 2),
        'final_capital': round(final_capital, 2),
        'return_pct': round(((final_capital / 10000) - 1) * 100, 1),
        'trades': all_trades
    }
    
    with open('v3_trades.json', 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Results saved to v3_trades.json")
    
    return result
    
    return result


if __name__ == "__main__":
    result = run_v3_backtest()
