"""
ICT V2 - XAUUSD/Gold Trading
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import json

print("=" * 70)
print("ICT V2 - XAUUSD (GOLD) ANALYSIS")
print("=" * 70)

symbol = "GC=F"  # Gold Futures on Yahoo Finance

# Fetch data
print(f"Fetching {symbol} data...")
df = yf.Ticker(symbol).history(period="30d", interval="1h")
df_daily = yf.Ticker(symbol).history(period="60d", interval="1d")

if df.empty:
    # Try XAUUSD
    symbol = "XAUUSD=X"
    print(f"Fetching {symbol} data...")
    df = yf.Ticker(symbol).history(period="30d", interval="1h")
    df_daily = yf.Ticker(symbol).history(period="60d", interval="1d")

print(f"Data: {len(df)} hourly bars, {len(df_daily)} daily bars")
print(f"Current Price: ${df['Close'].iloc[-1]:.2f}")

# Calculate indicators
highs = df['High'].values
lows = df['Low'].values
closes = df['Close'].values
opens = df['Open'].values

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
    range_size = ph - pl
    if range_size < 0.001:
        range_size = 0.001
    price_position[i] = (closes[i] - pl) / range_size

# Kill Zones (Gold trades 24h, optimal sessions)
hours = pd.to_datetime(df.index).hour.values
kill_zone = np.array([(1 <= h < 5) or (7 <= h < 12) or (13.5 <= h < 16) for h in hours])

# Analyze current bar
idx = len(closes) - 1
current_price = closes[idx]

print(f"\n{'='*70}")
print("CURRENT MARKET STATE")
print(f"{'='*70}")
print(f"Price: ${current_price:.2f}")
print(f"HTF Trend: {htf_trend_hourly[idx]:.0f} (1=bull, -1=bear, 0=neutral)")
print(f"LTF Trend: {trend[idx]:.0f}")
print(f"Price Position: {price_position[idx]:.2f} (<0.4=buy zone, >0.6=sell zone)")
print(f"Kill Zone: {'YES' if kill_zone[idx] else 'NO'}")

# Calculate confluence
confluence = 0
if kill_zone[idx]:
    confluence += 15
    print(f"+15 Kill Zone")
    
if htf_trend_hourly[idx] == 1 and trend[idx] >= 0:
    confluence += 25
    print(f"+25 HTF Bullish")
elif htf_trend_hourly[idx] == -1 and trend[idx] <= 0:
    confluence += 25
    print(f"+25 HTF Bearish")
elif htf_trend_hourly[idx] != 0:
    confluence += 10
    print(f"+10 HTF Some Bias")

if price_position[idx] < 0.25:
    confluence += 20
    print(f"+20 Deep Discount")
elif price_position[idx] < 0.35:
    confluence += 15
    print(f"+15 Discount")
elif price_position[idx] > 0.75:
    confluence += 20
    print(f"+20 Deep Premium")
elif price_position[idx] > 0.65:
    confluence += 15
    print(f"+15 Premium")

# Check levels
nearest_bull = next((ob for ob in reversed(bullish_obs) if ob['idx'] < idx), None)
nearest_bear = next((ob for ob in reversed(bearish_obs) if ob['idx'] < idx), None)
near_bull_fvg = next((f for f in reversed(bullish_fvgs) if f['idx'] < idx and f['mid'] < current_price < f['high']), None)
near_bear_fvg = next((f for f in reversed(bearish_fvgs) if f['idx'] < idx and f['low'] < current_price < f['mid']), None)

if near_bull_fvg and trend[idx] >= 0:
    confluence += 15
    print(f"+15 Bullish FVG")
if near_bear_fvg and trend[idx] <= 0:
    confluence += 15
    print(f"+15 Bearish FVG")
if nearest_bull and current_price > nearest_bull['high']:
    confluence += 10
    print(f"+10 OB Broken")
if nearest_bear and current_price < nearest_bear['low']:
    confluence += 10
    print(f"+10 OB Broken")

# Grade
if confluence >= 75:
    grade = 'A+'
elif confluence >= 70:
    grade = 'A'
elif confluence >= 60:
    grade = 'B'
elif confluence >= 50:
    grade = 'C'
else:
    grade = 'F'

print(f"\n{'='*70}")
print(f"CONFLUENCE SCORE: {confluence}")
print(f"GRADE: {grade}")
print(f"{'='*70}")

# Signal decision
signal_ready = False
direction = None

if confluence >= 70 and grade in ['A+', 'A']:
    if htf_trend_hourly[idx] != 0 and kill_zone[idx]:
        if price_position[idx] < 0.40 and htf_trend_hourly[idx] == 1:
            if near_bull_fvg or nearest_bull:
                signal_ready = True
                direction = 'LONG'
        elif price_position[idx] > 0.60 and htf_trend_hourly[idx] == -1:
            if near_bear_fvg or nearest_bear:
                signal_ready = True
                direction = 'SHORT'

print(f"\nSIGNAL: {'YES - ' + direction if signal_ready else 'NO'}")

if signal_ready:
    atr = (highs[idx-14:idx] - lows[idx-14:idx]).mean() if idx > 14 else 5
    
    if direction == 'LONG':
        sl = (near_bull_fvg['low'] if near_bull_fvg else nearest_bull['low']) - atr * 0.5
        tp = current_price + atr * 2.5
    else:
        sl = (near_bear_fvg['high'] if near_bear_fvg else nearest_bear['high']) + atr * 0.5
        tp = current_price - atr * 2.5
    
    print(f"\nTRADE SETUP:")
    print(f"  Direction: {direction}")
    print(f"  Entry: ${current_price:.2f}")
    print(f"  Stop Loss: ${sl:.2f}")
    print(f"  Take Profit: ${tp:.2f}")
    print(f"  Risk/Reward: 1:2.5")

print(f"\n{'='*70}")
print("BACKTEST NOTES FOR GOLD")
print("="*70)
print("Gold typically has:")
print("  - Higher volatility (larger ATR)")
print("  - Strong trends during Asian session")
print("  - Optimal entries during London/New York overlap")
print("  - Watch for 50-70 confluence on Gold (slightly lower threshold)")
print("="*70)
