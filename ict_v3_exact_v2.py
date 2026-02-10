"""
ICT Trading System V3 - EXACT V2 LOGIC + PHASE 3
================================================

This is V2's EXACT logic with Phase 3 components added as optional enhancements.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import json
import random

print("=" * 70)
print("ICT TRADING SYSTEM V3 - EXACT V2 LOGIC + PHASE 3")
print("=" * 70)
print()

# Fetch data
df = yf.Ticker("NQ=F").history(period="6mo", interval="1h")
df = df.dropna()
df = df[~df.index.duplicated(keep='first')]

df_daily = yf.Ticker("NQ=F").history(period="6mo", interval="1d")
df_daily = df_daily.dropna()

print(f"Data: {len(df)} hourly bars")

# Pre-calculate indicators
highs = df['High'].values
lows = df['Low'].values
closes = df['Close'].values
opens = df['Open'].values
timestamps = df.index.values

# FVGs
bullish_fvgs = [{'idx': i, 'low': highs[i-2], 'high': lows[i], 'mid': (highs[i-2]+lows[i])/2} 
               for i in range(3, len(df)) if lows[i] > highs[i-2]]
bearish_fvgs = [{'idx': i, 'low': highs[i], 'high': lows[i-2], 'mid': (highs[i]+lows[i-2])/2} 
               for i in range(3, len(df)) if highs[i] < lows[i-2]]

# Order Blocks
bullish_obs = [{'idx': i, 'high': highs[i-1], 'low': lows[i-1]} 
              for i in range(5, len(df)) if closes[i-1] < opens[i-1] and closes[i] > opens[i] and lows[i] < lows[i-1]]
bearish_obs = [{'idx': i, 'high': highs[i-1], 'low': lows[i-1]} 
              for i in range(5, len(df)) if closes[i-1] > opens[i-1] and closes[i] < opens[i] and highs[i] > highs[i-1]]

# HTF Trend
daily_highs = df_daily['High'].values
daily_lows = df_daily['Low'].values

df_index = pd.DatetimeIndex(df.index).tz_localize(None)
df_daily_index = pd.DatetimeIndex(df_daily.index).tz_localize(None)

htf_trends = np.zeros(len(df))
for i in range(len(df)):
    bar_time = df_index[i]
    for j in range(len(df_daily)-1, -1, -1):
        if df_daily_index[j] <= bar_time:
            if j >= 5:
                if daily_highs[j] > daily_highs[j-5:j].max() and daily_lows[j] > daily_lows[j-5:j].min():
                    htf_trends[i] = 1
                elif daily_highs[j] < daily_highs[j-5:j].max() and daily_lows[j] < daily_lows[j-5:j].min():
                    htf_trends[i] = -1
            break

# LTF Trend
trend = np.zeros(len(df))
for i in range(20, len(df)):
    rh = highs[max(0,i-20):i].max()
    rl = lows[max(0,i-20):i].min()
    if rh > highs[i-5] and rl > lows[i-5]:
        trend[i] = 1
    elif rh < highs[i-5] and rl < lows[i-5]:
        trend[i] = -1

# Price Position
price_position = np.zeros(len(df))
for i in range(20, len(df)):
    ph = highs[i-20:i].max()
    pl = lows[i-20:i].min()
    price_position[i] = (closes[i] - pl) / (ph - pl + 0.001)

# Kill Zones
hours = pd.to_datetime(timestamps).hour.values
kill_zone = np.array([(1 <= h < 5) or (7 <= h < 12) or (13.5 <= h < 16) for h in hours])

# =============================================================================
# PHASE 3 COMPONENTS
# =============================================================================

class SimpleRLAgent:
    """Simple RL agent for position sizing"""
    
    def __init__(self):
        self.q_table = {}
        self.exploration_rate = 0.1
    
    def get_action(self, state_key: str) -> str:
        if random.random() < self.exploration_rate:
            return random.choice(['small', 'medium', 'large'])
        
        q_values = [self.q_table.get(f"{state_key}_{a}", 0) for a in ['small', 'medium', 'large']]
        return ['small', 'medium', 'large'][np.argmax(q_values)] if q_values else 'medium'
    
    def update(self, state_key: str, action: str, reward: float):
        current = self.q_table.get(f"{state_key}_{action}", 0)
        self.q_table[f"{state_key}_{action}"] = current + 0.1 * reward
        self.exploration_rate = max(0.02, self.exploration_rate * 0.99)


# =============================================================================
# TRADING - EXACT V2 LOGIC
# =============================================================================

# Initialize RL
rl_agent = SimpleRLAgent()

capital = 10000
equity_curve = [capital]
trades = []
open_trade = None

print("Running exact V2 logic with Phase 3 enhancements...")

for idx in range(50, len(df)):
    current_price = closes[idx]
    htf_bias = htf_trends[idx]
    kz = kill_zone[idx]
    pp = price_position[idx]
    current_trend = trend[idx]
    
    # Get state
    nearest_bull = next((ob for ob in reversed(bullish_obs) if ob['idx'] < idx), None)
    nearest_bear = next((ob for ob in reversed(bearish_obs) if ob['idx'] < idx), None)
    near_bull_fvg = next((f for f in reversed(bullish_fvgs) if f['idx'] < idx and f['mid'] < current_price < f['high']), None)
    near_bear_fvg = next((f for f in reversed(bearish_fvgs) if f['idx'] < idx and f['low'] < current_price < f['mid']), None)
    
    atr = (highs[idx-14:idx] - lows[idx-14:idx]).mean() if idx > 14 else 50
    
    # Liquidity pools
    buy_pools = []
    sell_pools = []
    for i in range(10, len(df)):
        for j in range(i+5, min(i+20, len(df))):
            if abs(highs[i] - highs[j]) < closes[i] * 0.0005:
                sell_pools.append({'idx': i, 'level': highs[i]})
                break
    for i in range(10, len(df)):
        for j in range(i+5, min(i+20, len(df))):
            if abs(lows[i] - lows[j]) < closes[i] * 0.0005:
                buy_pools.append({'idx': i, 'level': lows[i]})
                break
    
    # Check liquidity sweeps
    recent_high = highs[max(0,idx-10):idx].max()
    recent_low = lows[max(0,idx-10):idx].min()
    liq_swept = False
    for pool in sell_pools:
        if pool['idx'] < idx and recent_high > pool['level']:
            liq_swept = True
            break
    for pool in buy_pools:
        if pool['idx'] < idx and recent_low < pool['level']:
            liq_swept = True
            break
    
    # Calculate confluence (V2)
    confluence = 0
    
    if kz:
        confluence += 15
    
    if htf_bias == 1 and current_trend >= 0:
        confluence += 25
    elif htf_bias == -1 and current_trend <= 0:
        confluence += 25
    elif htf_bias != 0:
        confluence += 10
    
    if pp < 0.25:
        confluence += 20
    elif pp < 0.35:
        confluence += 15
    elif pp > 0.75:
        confluence += 20
    elif pp > 0.65:
        confluence += 15
    
    if near_bull_fvg and current_trend >= 0:
        confluence += 15
    if near_bear_fvg and current_trend <= 0:
        confluence += 15
    
    if nearest_bull and current_price > nearest_bull['high']:
        confluence += 10
    if nearest_bear and current_price < nearest_bear['low']:
        confluence += 10
    
    if liq_swept:
        confluence += 10
    
    # Grade
    if confluence >= 75:
        grade = 'A+'
    elif confluence >= 70:
        grade = 'A'
    elif confluence >= 60:
        grade = 'B'
    else:
        grade = 'C'
    
    # AI Signal Filter (V2 EXACT)
    ai_filter_pass = False
    if confluence >= 70 and grade in ['A+', 'A']:
        if htf_bias != 0:
            if kz:
                if (near_bull_fvg or nearest_bull) or (near_bear_fvg or nearest_bear):
                    if (htf_bias == 1 and pp < 0.40) or (htf_bias == -1 and pp > 0.60):
                        ai_filter_pass = True
    
    # ATR for stops
    atr = (highs[idx-14:idx] - lows[idx-14:idx]).mean() if idx > 14 else 50
    
    # Check exits
    if open_trade:
        if open_trade['dir'] == 'long':
            if current_price <= open_trade['sl']:
                open_trade['exit'] = current_price
                open_trade['pnl'] = (open_trade['exit'] - open_trade['entry']) * open_trade['size'] * 20
                open_trade['status'] = 'STOP'
                capital += open_trade['pnl']
                
                # RL update
                reward = 1 if open_trade['pnl'] > 0 else -1
                rl_agent.update(open_trade['state_key'], open_trade['position_mod'], reward)
                
                trades.append(open_trade)
                open_trade = None
            elif current_price >= open_trade['tp']:
                open_trade['exit'] = current_price
                open_trade['pnl'] = (open_trade['exit'] - open_trade['entry']) * open_trade['size'] * 20
                open_trade['status'] = 'TP'
                capital += open_trade['pnl']
                
                # RL update
                reward = 1
                rl_agent.update(open_trade['state_key'], open_trade['position_mod'], reward)
                
                trades.append(open_trade)
                open_trade = None
        else:
            if current_price >= open_trade['sl']:
                open_trade['exit'] = current_price
                open_trade['pnl'] = (open_trade['entry'] - open_trade['exit']) * open_trade['size'] * 20
                open_trade['status'] = 'STOP'
                capital += open_trade['pnl']
                
                # RL update
                reward = 1 if open_trade['pnl'] > 0 else -1
                rl_agent.update(open_trade['state_key'], open_trade['position_mod'], reward)
                
                trades.append(open_trade)
                open_trade = None
            elif current_price <= open_trade['tp']:
                open_trade['exit'] = current_price
                open_trade['pnl'] = (open_trade['entry'] - open_trade['exit']) * open_trade['size'] * 20
                open_trade['status'] = 'TP'
                capital += open_trade['pnl']
                
                # RL update
                reward = 1
                rl_agent.update(open_trade['state_key'], open_trade['position_mod'], reward)
                
                trades.append(open_trade)
                open_trade = None
        
        # Time exit
        if open_trade and (idx - open_trade['entry_idx']) > 15:
            open_trade['exit'] = current_price
            open_trade['pnl'] = (open_trade['entry'] - current_price) * open_trade['size'] * 20 if open_trade['dir'] == 'short' else (current_price - open_trade['entry']) * open_trade['size'] * 20
            open_trade['status'] = 'TIME'
            capital += open_trade['pnl']
            trades.append(open_trade)
            open_trade = None
    
    # Check entries
    if open_trade is None and ai_filter_pass:
        risk_amt = capital * 0.01
        signal = None
        
        # Long
        if pp < 0.40 and htf_bias == 1:
            if near_bull_fvg and current_price > near_bull_fvg['mid']:
                signal = 'long'
                sl = near_bull_fvg['low'] - atr * 0.5
                tp = current_price + atr * 2.5
            elif nearest_bull and current_price > nearest_bull['high']:
                signal = 'long'
                sl = nearest_bull['low'] - atr * 0.5
                tp = current_price + atr * 2.5
        
        # Short
        elif pp > 0.60 and htf_bias == -1:
            if near_bear_fvg and current_price < near_bear_fvg['mid']:
                signal = 'short'
                sl = near_bear_fvg['high'] + atr * 0.5
                tp = current_price - atr * 2.5
            elif nearest_bear and current_price < nearest_bear['low']:
                signal = 'short'
                sl = nearest_bear['high'] + atr * 0.5
                tp = current_price - atr * 2.5
        
        if signal:
            risk = abs(current_price - sl)
            size = risk_amt / risk if risk > 0 else 1
            
            # RL position sizing
            state_key = f"{'bull' if htf_bias > 0 else 'bear' if htf_bias < 0 else 'neutral'}_{'disc' if pp < 0.4 else 'prem' if pp > 0.6 else 'mid'}_{'kz' if kz else 'nkz'}"
            position_mod = rl_agent.get_action(state_key)
            risk_pct = {'small': 0.0075, 'medium': 0.01, 'large': 0.015}[position_mod]
            
            risk_amt = capital * risk_pct
            size = risk_amt / risk if risk > 0 else 1
            
            open_trade = {
                'entry_idx': idx,
                'entry': current_price,
                'dir': signal,
                'size': size,
                'sl': sl,
                'tp': tp,
                'conf': confluence / 100,
                'confluence': confluence,
                'grade': grade,
                'state_key': state_key,
                'position_mod': position_mod
            }
    
    equity_curve.append(capital)
    
    if idx % 500 == 0:
        print(f"Progress: {idx}/{len(df)} | Equity: ${capital:,.0f} | Trades: {len(trades)}")

# Close open trade
if open_trade:
    open_trade['exit'] = closes[-1]
    open_trade['pnl'] = (open_trade['entry'] - closes[-1]) * open_trade['size'] * 20 if open_trade['dir'] == 'short' else (closes[-1] - open_trade['entry']) * open_trade['size'] * 20
    open_trade['status'] = 'EOD'
    trades.append(open_trade)

# Statistics
closed = [t for t in trades if 'exit' in t]
winners = [t for t in closed if t.get('pnl', 0) > 0]
losers = [t for t in closed if t.get('pnl', 0) <= 0]

total_return = (capital - 10000) / 10000 * 100
win_rate = len(winners) / len(closed) * 100 if closed else 0

max_eq = max(equity_curve)
min_eq = min(equity_curve)
max_dd = (max_eq - min_eq) / max_eq * 100 if max_eq > 0 else 0

profit = sum(t['pnl'] for t in winners)
loss = abs(sum(t['pnl'] for t in losers))
pf = profit / loss if loss > 0 else float('inf')

# By direction
longs = [t for t in closed if t['dir'] == 'long']
shorts = [t for t in closed if t['dir'] == 'short']
long_pnl = sum(t['pnl'] for t in longs)
short_pnl = sum(t['pnl'] for t in shorts)

# By grade
grades = {}
for t in closed:
    g = t.get('grade', 'F')
    if g not in grades:
        grades[g] = {'count': 0, 'wins': 0, 'pnl': 0}
    grades[g]['count'] += 1
    if t['pnl'] > 0:
        grades[g]['wins'] += 1
    grades[g]['pnl'] += t['pnl']

# Save results
results = {
    'metadata': {
        'version': 'V3 - Exact V2 Logic + Phase 3 RL',
        'timestamp': datetime.now().isoformat()
    },
    'period': {'start': str(timestamps[0])[:10], 'end': str(timestamps[-1])[:10], 'bars': len(df)},
    'capital': {'initial': 10000, 'final': capital, 'return_pct': total_return},
    'trades': {'total': len(closed), 'winners': len(winners), 'losers': len(losers), 'win_rate': win_rate},
    'pnl': {'gross_profit': profit, 'gross_loss': loss, 'net_pnl': capital - 10000, 'profit_factor': pf},
    'risk': {'max_drawdown_pct': max_dd},
    'direction_breakdown': {
        'long': {'count': len(longs), 'pnl': long_pnl},
        'short': {'count': len(shorts), 'pnl': short_pnl}
    },
    'grade_breakdown': grades,
    'equity_curve': [{'date': str(timestamps[i])[:10], 'equity': e} for i, e in enumerate(equity_curve)]
}

with open('v3_exact_v2_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Print summary
print()
print("=" * 70)
print("V3 - EXACT V2 LOGIC + PHASE 3 RESULTS")
print("=" * 70)
print(f"Period: {results['period']['start']} to {results['period']['end']}")
print()
print("CAPITAL:")
print(f"  Initial:    ${10000:>12,.0f}")
print(f"  Final:      ${capital:>12,.0f}")
print(f"  Return:     {total_return:>12.1f}%")
print()
print("TRADE STATISTICS:")
print(f"  Total:      {len(closed)}")
print(f"  Win Rate:   {win_rate:.1f}%")
print()
print("P&L:")
print(f"  Gross Profit:  ${profit:>12,.0f}")
print(f"  Gross Loss:    ${loss:>12,.0f}")
print(f"  Net PnL:        ${capital-10000:>12,.0f}")
print(f"  Profit Factor: {pf:>12.2f}")
print()
print("RISK:")
print(f"  Max Drawdown:  {max_dd:.1f}%")
print()
print("DIRECTION BREAKDOWN:")
print(f"  Long:  {len(longs):3} trades | ${long_pnl:+,.0f}")
print(f"  Short: {len(shorts):3} trades | ${short_pnl:+,.0f}")
print()
print("GRADE BREAKDOWN:")
for g in sorted(grades.keys()):
    data = grades[g]
    wr = data['wins'] / data['count'] * 100 if data['count'] > 0 else 0
    print(f"  Grade {g}: {data['count']:2} trades | {wr:5.1f}% win | ${data['pnl']:+,.0f}")
print()
print("=" * 70)
