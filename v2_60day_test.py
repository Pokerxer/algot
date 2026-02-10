"""
ICT V2 - Recent 60 Days Test (No Trailing Stops)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import json

print("=" * 70)
print("ICT V2 - RECENT 60 DAYS TEST")
print("=" * 70)

symbol = "NQ=F"
capital = 10000

print(f"Fetching {symbol} data (60 days, 1h)...")
df = yf.Ticker(symbol).history(period="60d", interval="1h")
df = df.dropna()
df = df[~df.index.duplicated(keep='first')]
print(f"Data: {len(df)} bars from {df.index[0]} to {df.index[-1]}")

df_daily = yf.Ticker(symbol).history(period="60d", interval="1d")

highs = df['High'].values
lows = df['Low'].values
closes = df['Close'].values
opens = df['Open'].values
timestamps = df.index.values

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
df_daily_index = pd.DatetimeIndex(df_daily.index).tz_localize(None)
df_index = pd.DatetimeIndex(df.index).tz_localize(None)

htf_trend = []
for i in range(1, len(df_daily)):
    if len(df_daily) < 5:
        htf_trend.append(0)
    elif daily_highs[i] > daily_highs[max(0,i-5):i].max() and daily_lows[i] > daily_lows[max(0,i-5):i].min():
        htf_trend.append(1)
    elif daily_highs[i] < daily_highs[max(0,i-5):i].max() and daily_lows[i] < daily_lows[max(0,i-5):i].min():
        htf_trend.append(-1)
    else:
        htf_trend.append(0)

htf_trend_hourly = np.zeros(len(df))
for i in range(len(df)):
    bar_time = df_index[i]
    for j in range(len(df_daily)-1, -1, -1):
        if df_daily_index[j] <= bar_time:
            htf_trend_hourly[i] = htf_trend[j] if j < len(htf_trend) else 0
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
kill_zone = np.zeros(len(df), dtype=bool)
for i in range(len(hours)):
    h = hours[i]
    kill_zone[i] = (1 <= h < 5) or (7 <= h < 12) or (13.5 <= h < 16)

print("Running simulation...")

# Trading
equity_curve = [capital]
trades = []
open_trade = None

for idx in range(50, len(df)):
    current_price = closes[idx]
    current_trend = trend[idx]
    htf_bias = htf_trend_hourly[idx]
    kz = kill_zone[idx]
    pp = price_position[idx]
    
    nearest_bull = next((ob for ob in reversed(bullish_obs) if ob['idx'] < idx), None)
    nearest_bear = next((ob for ob in reversed(bearish_obs) if ob['idx'] < idx), None)
    near_bull_fvg = next((f for f in reversed(bullish_fvgs) if f['idx'] < idx and f['mid'] < current_price < f['high']), None)
    near_bear_fvg = next((f for f in reversed(bearish_fvgs) if f['idx'] < idx and f['low'] < current_price < f['mid']), None)
    
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
    
    grade = 'F'
    if confluence >= 75:
        grade = 'A+'
    elif confluence >= 70:
        grade = 'A'
    
    ai_filter_pass = False
    if confluence >= 70 and grade in ['A+', 'A']:
        if htf_bias != 0 and kz:
            if (near_bull_fvg or nearest_bull) or (near_bear_fvg or nearest_bear):
                if (htf_bias == 1 and pp < 0.40) or (htf_bias == -1 and pp > 0.60):
                    ai_filter_pass = True
    
    atr = (highs[idx-14:idx] - lows[idx-14:idx]).mean() if idx > 14 else 50
    
    # Exit
    if open_trade:
        if open_trade['dir'] == 'long':
            if current_price <= open_trade['sl']:
                open_trade['exit'] = current_price
                open_trade['pnl'] = (open_trade['exit'] - open_trade['entry']) * open_trade['size'] * 20
                capital += open_trade['pnl']
                trades.append(open_trade)
                open_trade = None
            elif current_price >= open_trade['tp']:
                open_trade['exit'] = current_price
                open_trade['pnl'] = (open_trade['exit'] - open_trade['entry']) * open_trade['size'] * 20
                capital += open_trade['pnl']
                trades.append(open_trade)
                open_trade = None
        else:
            if current_price >= open_trade['sl']:
                open_trade['exit'] = current_price
                open_trade['pnl'] = (open_trade['entry'] - open_trade['exit']) * open_trade['size'] * 20
                capital += open_trade['pnl']
                trades.append(open_trade)
                open_trade = None
            elif current_price <= open_trade['tp']:
                open_trade['exit'] = current_price
                open_trade['pnl'] = (open_trade['entry'] - open_trade['exit']) * open_trade['size'] * 20
                capital += open_trade['pnl']
                trades.append(open_trade)
                open_trade = None
        
        if open_trade and (idx - open_trade['entry_idx']) > 15:
            open_trade['exit'] = current_price
            open_trade['pnl'] = (open_trade['entry'] - current_price) * open_trade['size'] * 20 if open_trade['dir'] == 'short' else (current_price - open_trade['entry']) * open_trade['size'] * 20
            capital += open_trade['pnl']
            trades.append(open_trade)
            open_trade = None
    
    # Entry
    if open_trade is None and ai_filter_pass:
        risk_amt = capital * 0.01
        signal = None
        
        if pp < 0.40 and htf_bias == 1:
            if near_bull_fvg and current_price > near_bull_fvg['mid']:
                signal = 'long'
                sl = near_bull_fvg['low'] - atr * 0.5
                tp = current_price + atr * 2.5
            elif nearest_bull and current_price > nearest_bull['high']:
                signal = 'long'
                sl = nearest_bull['low'] - atr * 0.5
                tp = current_price + atr * 2.5
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
            open_trade = {
                'entry_idx': idx,
                'entry_time': str(timestamps[idx])[:19],
                'entry': current_price,
                'dir': signal,
                'size': size,
                'sl': sl,
                'tp': tp,
                'confluence': confluence,
                'grade': grade
            }
    
    equity_curve.append(capital)

if open_trade:
    open_trade['exit'] = closes[-1]
    open_trade['pnl'] = (open_trade['entry'] - closes[-1]) * open_trade['size'] * 20 if open_trade['dir'] == 'short' else (closes[-1] - open_trade['entry']) * open_trade['size'] * 20
    trades.append(open_trade)

closed = [t for t in trades if 'exit' in t]
winners = [t for t in closed if t['pnl'] > 0]
losers = [t for t in closed if t['pnl'] <= 0]

total_return = (capital - 10000) / 10000 * 100
win_rate = len(winners) / len(closed) * 100 if closed else 0

profit = sum(t['pnl'] for t in winners)
loss = abs(sum(t['pnl'] for t in losers))
pf = profit / loss if loss > 0 else float('inf')

longs = [t for t in closed if t['dir'] == 'long']
shorts = [t for t in closed if t['dir'] == 'short']

print()
print("=" * 70)
print("RESULTS - ICT V2 60 DAYS")
print("=" * 70)
print(f"Period: {str(timestamps[0])[:10]} to {str(timestamps[-1])[:10]}")
print()
print(f"Capital: $10,000 -> ${capital:,.0f} ({total_return:+.1f}%)")
print(f"Trades: {len(closed)} | Win Rate: {win_rate:.1f}%")
print(f"Profit Factor: {pf:.2f}")
print()
print(f"Long:  {len(longs)} trades | PnL: ${sum(t['pnl'] for t in longs):+,.0f}")
print(f"Short: {len(shorts)} trades | PnL: ${sum(t['pnl'] for t in shorts):+,.0f}")
print("=" * 70)
