"""
ICT Unified Handler - 6 Month Backtest (Optimized)
===================================================
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List
from enum import Enum
import json

print("=" * 70)
print("ICT UNIFIED HANDLER - 6 MONTH BACKTEST (OPTIMIZED)")
print("=" * 70)
print("Starting Capital: $10,000")
print()

# Fetch data
print("Fetching NQ data...")
df = yf.Ticker("NQ=F").history(period="6mo", interval="1h")
df = df.dropna()
df = df[~df.index.duplicated(keep='first')]
print(f"Data: {len(df)} bars")

# Vectorized indicators
highs = df['High'].values
lows = df['Low'].values
closes = df['Close'].values
opens = df['Open'].values
timestamps = df.index.values

# Pre-calculate all indicators
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

# Trend
trend = np.zeros(len(df))
for i in range(20, len(df)):
    rh = highs[max(0,i-20):i].max()
    rl = lows[max(0,i-20):i].min()
    if rh > highs[i-5] and rl > lows[i-5]:
        trend[i] = 1
    elif rh < highs[i-5] and rl < lows[i-5]:
        trend[i] = -1

# PD Arrays
price_position = np.zeros(len(df))
for i in range(20, len(df)):
    ph = highs[i-20:i].max()
    pl = lows[i-20:i].min()
    price_position[i] = (closes[i] - pl) / (ph - pl + 0.001)

# Sessions
hours = pd.to_datetime(timestamps).hour.values
kill_zone = np.zeros(len(df), dtype=bool)
for i in range(len(hours)):
    h = hours[i]
    kill_zone[i] = (1 <= h < 5) or (7 <= h < 12) or (13.5 <= h < 16)

print("Running simulation...")

# Trading simulation
capital = 10000
equity_curve = [capital]
trades = []
open_trade = None

for idx in range(50, len(df)):
    current_price = closes[idx]
    current_trend = trend[idx]
    kz = kill_zone[idx]
    pp = price_position[idx]
    
    # Get nearest OB
    nearest_bull = next((ob for ob in reversed(bullish_obs) if ob['idx'] < idx), None)
    nearest_bear = next((ob for ob in reversed(bearish_obs) if ob['idx'] < idx), None)
    
    # Get nearest FVG
    near_bull_fvg = next((f for f in reversed(bullish_fvgs) if f['idx'] < idx and f['mid'] < current_price < f['high']), None)
    near_bear_fvg = next((f for f in reversed(bearish_fvgs) if f['idx'] < idx and f['low'] < current_price < f['mid']), None)
    
    atr = (highs[idx-14:idx] - lows[idx-14:idx]).mean() if idx > 14 else 50
    
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
        
        if open_trade and (idx - open_trade['entry_idx']) > 20:
            open_trade['exit'] = current_price
            open_trade['pnl'] = (open_trade['entry'] - current_price) * open_trade['size'] * 20 if open_trade['dir'] == 'short' else (current_price - open_trade['entry']) * open_trade['size'] * 20
            open_trade['status'] = 'TIME'
            capital += open_trade['pnl']
            trades.append(open_trade)
            open_trade = None
    
    # Check entries
    if open_trade is None and kz:
        risk_amt = capital * 0.02
        signal = None
        conf = 0
        
        # Long: discount + bullish
        if pp < 0.35 and (current_trend >= 0 or near_bull_fvg):
            if near_bull_fvg:
                signal = 'long'
                sl = near_bull_fvg['low'] - atr * 0.5
                tp = current_price + atr * 2
                conf = 0.70
            elif nearest_bull and current_price > nearest_bull['high']:
                signal = 'long'
                sl = nearest_bull['low'] - atr * 0.5
                tp = current_price + atr * 2
                conf = 0.75
        
        # Short: premium + bearish
        elif pp > 0.65 and (current_trend <= 0 or near_bear_fvg):
            if near_bear_fvg:
                signal = 'short'
                sl = near_bear_fvg['high'] + atr * 0.5
                tp = current_price - atr * 2
                conf = 0.70
            elif nearest_bear and current_price < nearest_bear['low']:
                signal = 'short'
                sl = nearest_bear['high'] + atr * 0.5
                tp = current_price - atr * 2
                conf = 0.75
        
        if signal and conf > 0.5:
            risk = abs(current_price - sl)
            size = risk_amt / risk if risk > 0 else 1
            open_trade = {
                'entry_idx': idx,
                'entry_time': str(timestamps[idx])[:19],
                'entry': current_price,
                'dir': signal,
                'size': size,
                'sl': sl,
                'tp': tp,
                'conf': conf
            }
    
    equity_curve.append(capital)
    
    if idx % 500 == 0:
        print(f"Progress: {idx}/{len(df)} | Equity: ${capital:,.0f}")

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

# By direction
longs = [t for t in closed if t['dir'] == 'long']
shorts = [t for t in closed if t['dir'] == 'short']
long_wins = len([t for t in longs if t['pnl'] > 0])
short_wins = len([t for t in shorts if t['pnl'] > 0])

# Save results
results = {
    'metadata': {'handler': 'ICTUnifiedHandler Phase 1', 'timestamp': datetime.now().isoformat()},
    'period': {'start': str(timestamps[0])[:10], 'end': str(timestamps[-1])[:10], 'bars': len(df)},
    'capital': {'initial': 10000, 'final': capital, 'return_pct': total_return},
    'trades': {'total': len(closed), 'winners': len(winners), 'losers': len(losers), 'win_rate': win_rate},
    'pnl': {'gross_profit': profit, 'gross_loss': loss, 'net_pnl': capital - 10000, 'profit_factor': pf},
    'risk': {'max_drawdown_pct': max_dd},
    'direction_breakdown': {
        'long': {'count': len(longs), 'wins': long_wins, 'win_rate': long_wins/len(longs)*100 if longs else 0, 'pnl': sum(t['pnl'] for t in longs)},
        'short': {'count': len(shorts), 'wins': short_wins, 'win_rate': short_wins/len(shorts)*100 if shorts else 0, 'pnl': sum(t['pnl'] for t in shorts)}
    },
    'equity_curve': [{'date': str(timestamps[i])[:10], 'equity': e} for i, e in enumerate(equity_curve)]
}

with open('unified_backtest_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Print summary
print()
print("=" * 70)
print("BACKTEST RESULTS - ICT UNIFIED HANDLER (PHASE 1)")
print("=" * 70)
print(f"Period: {results['period']['start']} to {results['period']['end']}")
print(f"Bars: {results['period']['bars']}")
print()
print("CAPITAL:")
print(f"  Initial:    ${10000:>10,.0f}")
print(f"  Final:      ${capital:>10,.0f}")
print(f"  Return:     {total_return:>10.2f}%")
print()
print(f"TRADES: {len(closed)} total | Win Rate: {win_rate:.1f}%")
print(f"  Winners: {len(winners)} | Losers: {len(losers)}")
print()
print("P&L:")
print(f"  Gross Profit:  ${profit:>10,.0f}")
print(f"  Gross Loss:    ${loss:>10,.0f}")
print(f"  Net PnL:       ${capital-10000:>10,.0f}")
print(f"  Profit Factor: {pf:>10.2f}")
print()
print("RISK:")
print(f"  Max Drawdown:  {max_dd:.2f}%")
print()
print("DIRECTION BREAKDOWN:")
ld = results['direction_breakdown']['long']
sd = results['direction_breakdown']['short']
print(f"  Long:  {ld['count']:3} trades | {ld['win_rate']:5.1f}% win | ${ld['pnl']:+,.0f}")
print(f"  Short: {sd['count']:3} trades | {sd['win_rate']:5.1f}% win | ${sd['pnl']:+,.0f}")
print()
print(f"Results saved to: unified_backtest_results.json")
print("=" * 70)
