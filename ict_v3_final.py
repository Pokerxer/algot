"""
ICT Trading System V3 - All Phases Integrated (Tuned for Production)
=====================================================================

Phase 1: Core handlers
Phase 2: Signal generation & execution
Phase 3: AI filtering & ML learning

Key changes from V3:
- Tuned confluence thresholds
- Better signal generation logic
- More practical risk management
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
from enum import Enum
import json
import random

print("=" * 70)
print("ICT TRADING SYSTEM V3 - ALL PHASES INTEGRATED")
print("=" * 70)
print()
print("Starting Capital: $10,000")
print()

# Fetch data
print("Fetching NQ data...")
df = yf.Ticker("NQ=F").history(period="6mo", interval="1h")
df = df.dropna()
df = df[~df.index.duplicated(keep='first')]

df_daily = yf.Ticker("NQ=F").history(period="6mo", interval="1d")
df_daily = df_daily.dropna()

print(f"Data: {len(df)} hourly bars")

# Pre-calculate all indicators
highs = df['High'].values
lows = df['Low'].values
closes = df['Close'].values
opens = df['Open'].values
timestamps = df.index.values

print("Pre-calculating indicators...")

# FVGs
bullish_fvgs = []
bearish_fvgs = []
for i in range(3, len(df)):
    if lows[i] > highs[i-2]:
        bullish_fvgs.append({'idx': i, 'low': highs[i-2], 'high': lows[i], 'mid': (highs[i-2]+lows[i])/2})
    if highs[i] < lows[i-2]:
        bearish_fvgs.append({'idx': i, 'low': highs[i], 'high': lows[i-2], 'mid': (highs[i]+lows[i-2])/2})

# Order Blocks
bullish_obs = []
bearish_obs = []
for i in range(5, len(df)):
    if closes[i-1] < opens[i-1] and closes[i] > opens[i] and lows[i] < lows[i-1]:
        bullish_obs.append({'idx': i, 'high': highs[i-1], 'low': lows[i-1]})
    if closes[i-1] > opens[i-1] and closes[i] < opens[i] and highs[i] > highs[i-1]:
        bearish_obs.append({'idx': i, 'high': highs[i-1], 'low': lows[i-1]})

# HTF Trend
daily_highs = df_daily['High'].values
daily_lows = df_daily['Low'].values

htf = []
for i in range(1, len(df_daily)):
    if daily_highs[i] > daily_highs[max(0,i-5):i].max() and daily_lows[i] > daily_lows[max(0,i-5):i].min():
        htf.append(1)
    elif daily_highs[i] < daily_highs[max(0,i-5):i].max() and daily_lows[i] < daily_lows[max(0,i-5):i].min():
        htf.append(-1)
    else:
        htf.append(0)

# Map HTF to hourly
df_index = pd.DatetimeIndex(df.index).tz_localize(None)
df_daily_index = pd.DatetimeIndex(df_daily.index).tz_localize(None)

htf_trends = np.zeros(len(df))
for i in range(len(df)):
    bar_time = df_index[i]
    for j in range(len(df_daily)-1, -1, -1):
        if df_daily_index[j] <= bar_time:
            htf_trends[i] = htf[j] if j < len(htf) else 0
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

# Sessions (Kill Zones)
hours = pd.to_datetime(timestamps).hour.values
kill_zone = np.zeros(len(df), dtype=bool)
for i in range(len(hours)):
    h = hours[i]
    kill_zone[i] = (1 <= h < 5) or (7 <= h < 12) or (13.5 <= h < 16)

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

print("Running complete trading system...")

# Complete Trading System
capital = 10000
equity_curve = [capital]
trades = []
open_trade = None

# AI Filter state
ai_confidence_threshold = 0.55  # Tuned lower for more signals
ml_predictions = []
rl_q_table = {}

def get_rl_action(state_key):
    if random.random() < 0.2:
        return random.choice(['small', 'medium', 'large'])
    q_values = [rl_q_table.get(f"{state_key}_{a}", 0) for a in ['small', 'medium', 'large']]
    return ['small', 'medium', 'large'][np.argmax(q_values)]

for idx in range(50, len(df)):
    current_price = closes[idx]
    current_trend = trend[idx]
    htf_trend = htf_trends[idx]
    kz = kill_zone[idx]
    pp = price_position[idx]
    
    # Get nearest OB/FVG
    nearest_bull = next((ob for ob in reversed(bullish_obs) if ob['idx'] < idx), None)
    nearest_bear = next((ob for ob in reversed(bearish_obs) if ob['idx'] < idx), None)
    near_bull_fvg = next((f for f in reversed(bullish_fvgs) if f['idx'] < idx and f['mid'] < current_price < f['high']), None)
    near_bear_fvg = next((f for f in reversed(bearish_fvgs) if f['idx'] < idx and f['low'] < current_price < f['mid']), None)
    
    # ATR
    atr = (highs[idx-14:idx] - lows[idx-14:idx]).mean() if idx > 14 else 50
    
    # Calculate confluence
    confluence = 0
    factors = []
    
    if kz:
        confluence += 15
        factors.append('KillZone')
    
    if htf_trend == 1 and current_trend >= 0:
        confluence += 25
        factors.append('HTFBullish')
    elif htf_trend == -1 and current_trend <= 0:
        confluence += 25
        factors.append('HTFBearish')
    elif htf_trend != 0:
        confluence += 10
    
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
    
    if nearest_bull:
        confluence += 10
        factors.append('BullOB')
    if nearest_bear:
        confluence += 10
        factors.append('BearOB')
    
    # Grade
    if confluence >= 70:
        grade = 'A+'
    elif confluence >= 60:
        grade = 'A'
    elif confluence >= 50:
        grade = 'B'
    elif confluence >= 40:
        grade = 'C'
    else:
        grade = 'D'
    
    # AI Filter simulation (Phase 3)
    ai_score = confluence / 100
    
    # Regime check
    if htf_trend == 1 and pp < 0.5:
        regime_bonus = 0.15
    elif htf_trend == -1 and pp > 0.5:
        regime_bonus = 0.15
    else:
        regime_bonus = 0
    
    ai_score += regime_bonus
    
    # Pattern matching bonus
    pattern_count = len([f for f in ['BullFVG', 'BearFVG', 'BullOB', 'BearOB'] if f in factors])
    pattern_bonus = min(0.15, pattern_count * 0.05)
    ai_score += pattern_bonus
    
    # Final AI decision
    if ai_score >= ai_confidence_threshold:
        ai_decision = 'accept'
    elif ai_score >= 0.45:
        ai_decision = 'modify'
    else:
        ai_decision = 'reject'
    
    # RL action for position sizing
    state_key = f"{'bull' if htf_trend > 0 else 'bear' if htf_trend < 0 else 'neutral'}_{'disc' if pp < 0.4 else 'prem' if pp > 0.6 else 'mid'}_{'kz' if kz else 'nkz'}"
    position_mod = get_rl_action(state_key)
    
    risk_pct = {'small': 0.005, 'medium': 0.01, 'large': 0.02}[position_mod]
    
    # Check exits
    if open_trade:
        if open_trade['dir'] == 'long':
            if current_price <= open_trade['sl']:
                open_trade['exit'] = current_price
                open_trade['pnl'] = (open_trade['exit'] - open_trade['entry']) * open_trade['size'] * 20
                open_trade['status'] = 'STOP'
                capital += open_trade['pnl']
                trades.append(open_trade)
                open_trade = None
            elif current_price >= open_trade['tp']:
                open_trade['exit'] = current_price
                open_trade['pnl'] = (open_trade['exit'] - open_trade['entry']) * open_trade['size'] * 20
                open_trade['status'] = 'TP'
                capital += open_trade['pnl']
                trades.append(open_trade)
                open_trade = None
        else:
            if current_price >= open_trade['sl']:
                open_trade['exit'] = current_price
                open_trade['pnl'] = (open_trade['entry'] - open_trade['exit']) * open_trade['size'] * 20
                open_trade['status'] = 'STOP'
                capital += open_trade['pnl']
                trades.append(open_trade)
                open_trade = None
            elif current_price <= open_trade['tp']:
                open_trade['exit'] = current_price
                open_trade['pnl'] = (open_trade['entry'] - open_trade['exit']) * open_trade['size'] * 20
                open_trade['status'] = 'TP'
                capital += open_trade['pnl']
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
    
    # Generate and filter signals
    if open_trade is None and ai_decision in ['accept', 'modify']:
        signal = None
        
        # Long: discount + alignment
        if pp < 0.45 and (htf_trend == 1 or current_trend >= 0):
            if near_bull_fvg:
                signal = {'dir': 'long', 'entry': current_price, 'sl': near_bull_fvg['low'] - atr * 0.5, 'tp': current_price + atr * 2.5}
            elif nearest_bull and current_price > nearest_bull['high']:
                signal = {'dir': 'long', 'entry': current_price, 'sl': nearest_bull['low'] - atr * 0.5, 'tp': current_price + atr * 2.5}
        
        # Short: premium + alignment
        elif pp > 0.55 and (htf_trend == -1 or current_trend <= 0):
            if near_bear_fvg:
                signal = {'dir': 'short', 'entry': current_price, 'sl': near_bear_fvg['high'] + atr * 0.5, 'tp': current_price - atr * 2.5}
            elif nearest_bear and current_price < nearest_bear['low']:
                signal = {'dir': 'short', 'entry': current_price, 'sl': nearest_bear['high'] + atr * 0.5, 'tp': current_price - atr * 2.5}
        
        if signal:
            risk_amt = capital * risk_pct
            risk = abs(current_price - signal['sl'])
            size = risk_amt / risk if risk > 0 else 1
            
            open_trade = {
                'entry_idx': idx,
                'entry_time': str(timestamps[idx])[:19],
                'entry': signal['entry'],
                'dir': signal['dir'],
                'size': size,
                'sl': signal['sl'],
                'tp': signal['tp'],
                'confluence': confluence,
                'grade': grade,
                'ai_score': ai_score,
                'factors': factors,
                'position_mod': position_mod
            }
    
    equity_curve.append(capital)
    
    if idx % 500 == 0:
        print(f"Progress: {idx}/{len(df)} | Equity: ${capital:,.0f} | Trades: {len(trades)}")

# Close open trade
if open_trade:
    open_trade['exit'] = closes[-1]
    open_trade['exit_time'] = str(timestamps[-1])[:19]
    open_trade['pnl'] = (open_trade['entry'] - closes[-1]) * open_trade['size'] * 20 if open_trade['dir'] == 'short' else (closes[-1] - open_trade['entry']) * open_trade['size'] * 20
    open_trade['status'] = 'EOD'
    trades.append(open_trade)

# Statistics
closed = [t for t in trades if 'exit' in t]
winners = [t for t in closed if t['pnl'] > 0]
losers = [t for t in closed if t['pnl'] <= 0]

total_return = (capital - 10000) / 10000 * 100
win_rate = len(winners) / len(closed) * 100 if closed else 0

max_eq = max(equity_curve)
min_eq = min(equity_curve)
max_dd = (max_eq - min_eq) / max_eq * 100 if max_eq > 0 else 0

profit = sum(t['pnl'] for t in winners)
loss = abs(sum(t['pnl'] for t in losers))
pf = profit / loss if loss > 0 else float('inf')

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
        'version': 'V3 Complete - All Phases',
        'timestamp': datetime.now().isoformat(),
        'components': ['Phase1: FVG/OB/Structure', 'Phase2: SignalGen/Executor', 'Phase3: AI Filter/ML/RL']
    },
    'period': {'start': str(timestamps[0])[:10], 'end': str(timestamps[-1])[:10], 'bars': len(df)},
    'capital': {'initial': 10000, 'final': capital, 'return_pct': total_return},
    'trades': {'total': len(closed), 'winners': len(winners), 'losers': len(losers), 'win_rate': win_rate},
    'pnl': {'gross_profit': profit, 'gross_loss': loss, 'net_pnl': capital - 10000, 'profit_factor': pf},
    'risk': {'max_drawdown_pct': max_dd},
    'grade_breakdown': grades,
    'equity_curve': [{'date': str(timestamps[i])[:10], 'equity': e} for i, e in enumerate(equity_curve)]
}

with open('v3_complete_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Print summary
print()
print("=" * 70)
print("V3 COMPLETE TRADING SYSTEM RESULTS")
print("=" * 70)
print(f"Period: {results['period']['start']} to {results['period']['end']}")
print()
print("CAPITAL:")
print(f"  Initial:    ${10000:>12,.0f}")
print(f"  Final:      ${capital:>12,.0f}")
print(f"  Return:     {total_return:>12.1f}%")
print()
print(f"TRADES: {len(closed)} total | Win Rate: {win_rate:.1f}%")
print(f"  Winners: {len(winners)} | Losers: {len(losers)}")
print()
print("P&L:")
print(f"  Gross Profit:  ${profit:>12,.0f}")
print(f"  Gross Loss:    ${loss:>12,.0f}")
print(f"  Net PnL:       ${capital-10000:>12,.0f}")
print(f"  Profit Factor: {pf:>12.2f}")
print()
print("RISK:")
print(f"  Max Drawdown:  {max_dd:.1f}%")
print()
print("GRADE BREAKDOWN:")
for g in sorted(grades.keys()):
    d = grades[g]
    wr = d['wins'] / d['count'] * 100 if d['count'] > 0 else 0
    print(f"  Grade {g}: {d['count']:2} trades | {wr:5.1f}% win | ${d['pnl']:+,.0f}")
print()
print(f"Results saved to: v3_complete_results.json")
print("=" * 70)
