"""
ICT Trading System V3 - OPTIMIZED FIXED (V2 Logic + Phase 3)
=============================================================

Key fix: Use V2's exact AI filter logic, add Phase 3 as enhancements
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
import json
import random

print("=" * 70)
print("ICT TRADING SYSTEM V3 - FIXED (V2 Logic + Phase 3)")
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
# PHASE 3 COMPONENTS (Simple implementations)
# =============================================================================

class SimpleAISignalFilter:
    """Simplified AI filter matching V2's logic exactly"""
    
    def __init__(self):
        self.threshold = 0.65
        self.learning_rate = 0.01
        self.pattern_weights = {
            'fvg': 0.8,
            'ob': 0.7,
            'structure': 0.9,
            'kill_zone': 0.6,
            'htf_alignment': 0.95,
            'discount': 0.75,
            'premium': 0.75
        }
    
    def filter(self, signal: Dict, state: Dict) -> Dict:
        """
        Match V2's exact AI filter logic:
        - confluence >= 70 and grade in ['A+', 'A']
        - htf_bias != 0 (must have HTF alignment)
        - kz (must be in kill zone)
        - (near_bull_fvg or nearest_bull) or (near_bear_fvg or nearest_bear)
        - (htf_bias == 1 and pp < 0.40) or (htf_bias == -1 and pp > 0.60)
        """
        if signal is None:
            return {'decision': 'reject', 'confidence': 0, 'reason': 'No signal'}
        
        confluence = signal.get('confluence', 0)
        grade = signal.get('grade', 'F')
        htf_bias = state.get('htf_trend', 0)
        kz = state.get('kill_zone', False)
        pp = state.get('price_position', 0.5)
        has_fvg_or_ob = state.get('has_fvg_or_ob', False)
        
        # V2's exact conditions
        if confluence >= 70 and grade in ['A+', 'A']:
            if htf_bias != 0:
                if kz:
                    if has_fvg_or_ob:
                        if (htf_bias == 1 and pp < 0.40) or (htf_bias == -1 and pp > 0.60):
                            return {
                                'decision': 'accept',
                                'confidence': confluence / 100,
                                'reason': 'V2 criteria met'
                            }
        
        return {'decision': 'reject', 'confidence': confluence / 100, 'reason': 'Failed V2 criteria'}


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
# TRADING SIMULATION - V2 LOGIC EXACTLY
# =============================================================================

# Initialize
ai_filter = SimpleAISignalFilter()
rl_agent = SimpleRLAgent()

# Trading state
capital = 10000
equity_curve = [capital]
trades = []
open_trade = None
signals_evaluated = 0
signals_accepted = 0

print("Running V2 logic with Phase 3 enhancements...")

for idx in range(50, len(df)):
    current_price = closes[idx]
    htf_bias = htf_trends[idx]  # V2 naming convention
    kz = kill_zone[idx]
    pp = price_position[idx]
    current_trend = trend[idx]
    
    # Get state
    nearest_bull = next((ob for ob in reversed(bullish_obs) if ob['idx'] < idx), None)
    nearest_bear = next((ob for ob in reversed(bearish_obs) if ob['idx'] < idx), None)
    near_bull_fvg = next((f for f in reversed(bullish_fvgs) if f['idx'] < idx and f['mid'] < current_price < f['high']), None)
    near_bear_fvg = next((f for f in reversed(bearish_fvgs) if f['idx'] < idx and f['low'] < current_price < f['mid']), None)
    
    atr = (highs[idx-14:idx] - lows[idx-14:idx]).mean() if idx > 14 else 50
    
    # Calculate confluence (V2 scoring)
    confluence = 0
    factors = []
    
    if kz:
        confluence += 15
        factors.append('KillZone')
    
    if htf_bias == 1 and current_trend >= 0:
        confluence += 25
        factors.append('HTFBullish')
    elif htf_bias == -1 and current_trend <= 0:
        confluence += 25
        factors.append('HTFBearish')
    
    if pp < 0.25:
        confluence += 20
        factors.append('DeepDiscount')
    elif pp < 0.35:
        confluence += 15
        factors.append('Discount')
    elif pp > 0.75:
        confluence += 20
        factors.append('DeepPremium')
    elif pp > 0.65:
        confluence += 15
        factors.append('Premium')
    
    if near_bull_fvg and current_trend >= 0:
        confluence += 15
        factors.append('BullFVG')
    if near_bear_fvg and current_trend <= 0:
        confluence += 15
        factors.append('BearFVG')
    
    if nearest_bull and current_price > nearest_bull['high']:
        confluence += 10
        factors.append('BullOB')
    if nearest_bear and current_price < nearest_bear['low']:
        confluence += 10
        factors.append('BearOB')
    
    # Grade
    if confluence >= 75:
        grade = 'A+'
    elif confluence >= 70:
        grade = 'A'
    elif confluence >= 60:
        grade = 'B'
    else:
        grade = 'C'
    
    # Generate signal (V2 conditions - exact copy)
    signal = None
    
    # Long: discount + bullish HTF + LTF alignment
    if pp < 0.40 and htf_bias == 1:
        if near_bull_fvg and current_price > near_bull_fvg['mid']:
            signal = {
                'direction': 'long',
                'entry': current_price,
                'stop_loss': near_bull_fvg['low'] - atr * 0.5,
                'take_profit': current_price + atr * 2.5,
                'confluence': confluence,
                'grade': grade,
                'factors': factors
            }
        elif nearest_bull and current_price > nearest_bull['high']:
            signal = {
                'direction': 'long',
                'entry': current_price,
                'stop_loss': nearest_bull['low'] - atr * 0.5,
                'take_profit': current_price + atr * 2.5,
                'confluence': confluence,
                'grade': grade,
                'factors': factors
            }
    
    # Short: premium + bearish HTF + LTF alignment
    elif pp > 0.60 and htf_bias == -1:
        if near_bear_fvg and current_price < near_bear_fvg['mid']:
            signal = {
                'direction': 'short',
                'entry': current_price,
                'stop_loss': near_bear_fvg['high'] + atr * 0.5,
                'take_profit': current_price - atr * 2.5,
                'confluence': confluence,
                'grade': grade,
                'factors': factors
            }
        elif nearest_bear and current_price < nearest_bear['low']:
            signal = {
                'direction': 'short',
                'entry': current_price,
                'stop_loss': nearest_bear['high'] + atr * 0.5,
                'take_profit': current_price - atr * 2.5,
                'confluence': confluence,
                'grade': grade,
                'factors': factors
            }
    
    # Apply AI Filter (V2 logic)
    if signal:
        signals_evaluated += 1
        
        # V2 variable naming for consistency
        htf_bias_val = htf_bias
        kz_val = kz
        pp_val = pp
        
        state = {
            'htf_trend': htf_bias_val,
            'kill_zone': kz_val,
            'price_position': pp_val,
            'has_fvg_or_ob': (near_bull_fvg or nearest_bull or near_bear_fvg or nearest_bear)
        }
        
        filtered = ai_filter.filter(signal, state)
        
        if filtered['decision'] == 'accept':
            signals_accepted += 1
            
            # RL position sizing
            state_key = f"{'bull' if htf_bias > 0 else 'bear' if htf_bias < 0 else 'neutral'}_{'disc' if pp < 0.4 else 'prem' if pp > 0.6 else 'mid'}_{'kz' if kz else 'nkz'}"
            position_mod = rl_agent.get_action(state_key)
            
            # V2: 1% risk
            risk_pct = {'small': 0.0075, 'medium': 0.01, 'large': 0.015}[position_mod]
            
            risk_amt = capital * risk_pct
            risk = abs(current_price - signal['stop_loss'])
            size = risk_amt / risk if risk > 0 else 1
            
            open_trade = {
                'entry_idx': idx,
                'entry': signal['entry'],
                'dir': signal['direction'],
                'size': size,
                'sl': signal['stop_loss'],
                'tp': signal['take_profit'],
                'confluence': confluence,
                'grade': grade,
                'position_mod': position_mod
            }
    
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
                rl_agent.update(state_key, open_trade['position_mod'], reward)
                
                trades.append(open_trade)
                open_trade = None
                
            elif current_price >= open_trade['tp']:
                open_trade['exit'] = current_price
                open_trade['pnl'] = (open_trade['exit'] - open_trade['entry']) * open_trade['size'] * 20
                open_trade['status'] = 'TP'
                capital += open_trade['pnl']
                
                # RL update
                reward = 1
                rl_agent.update(state_key, open_trade['position_mod'], reward)
                
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
                rl_agent.update(state_key, open_trade['position_mod'], reward)
                
                trades.append(open_trade)
                open_trade = None
                
            elif current_price <= open_trade['tp']:
                open_trade['exit'] = current_price
                open_trade['pnl'] = (open_trade['entry'] - open_trade['exit']) * open_trade['size'] * 20
                open_trade['status'] = 'TP'
                capital += open_trade['pnl']
                
                # RL update
                reward = 1
                rl_agent.update(state_key, open_trade['position_mod'], reward)
                
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
        'version': 'V3 Fixed (V2 Logic + Phase 3)',
        'timestamp': datetime.now().isoformat()
    },
    'period': {'start': str(timestamps[0])[:10], 'end': str(timestamps[-1])[:10], 'bars': len(df)},
    'capital': {'initial': 10000, 'final': capital, 'return_pct': total_return},
    'trades': {'total': len(closed), 'winners': len(winners), 'losers': len(losers), 'win_rate': win_rate},
    'pnl': {'gross_profit': profit, 'gross_loss': loss, 'net_pnl': capital - 10000, 'profit_factor': pf},
    'risk': {'max_drawdown_pct': max_dd},
    'signal_stats': {'evaluated': signals_evaluated, 'accepted': signals_accepted, 'accept_rate': signals_accepted/signals_evaluated*100 if signals_evaluated > 0 else 0},
    'direction_breakdown': {
        'long': {'count': len(longs), 'pnl': long_pnl},
        'short': {'count': len(shorts), 'pnl': short_pnl}
    },
    'grade_breakdown': grades,
    'equity_curve': [{'date': str(timestamps[i])[:10], 'equity': e} for i, e in enumerate(equity_curve)]
}

with open('v3_fixed_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Print summary
print()
print("=" * 70)
print("V3 FIXED RESULTS (V2 Logic + Phase 3)")
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
print("SIGNAL FILTERING:")
print(f"  Evaluated:  {signals_evaluated}")
print(f"  Accepted:   {signals_accepted}")
print(f"  Accept Rate: {signals_accepted/signals_evaluated*100:.1f}%" if signals_evaluated > 0 else 0)
print()
print("P&L:")
print(f"  Gross Profit:  ${profit:>12,.0f}")
print(f"  Gross Loss:    ${loss:>12,.0f}")
print(f"  Net PnL:        ${capital-10000:>12,.0f}")
print(f"  Profit Factor: {pf:>12.2f}")
print()
print("DIRECTION BREAKDOWN:")
print(f"  Long:  {len(longs):3} trades | ${long_pnl:+,.0f}")
print(f"  Short: {len(shorts):3} trades | ${short_pnl:+,.0f}")
print()
print("=" * 70)
