"""
ICT V2 - Check for Signals in Past 2 Days (NQ, YM, EURUSD)
Shows all A-grade signals and what trades would have been taken
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

print("=" * 80)
print("ICT V2 - SIGNAL ANALYSIS (PAST 2 DAYS)")
print("=" * 80)

def analyze_symbol(symbol, days=2):
    print(f"\n{'='*80}")
    print(f"SYMBOL: {symbol}")
    print(f"{'='*80}")
    
    # Fetch data
    df = yf.Ticker(symbol).history(period=f"{days}d", interval="1h")
    if df.empty:
        print(f"No data available")
        return None
    
    df = df.dropna()
    df_daily = yf.Ticker(symbol).history(period=f"{days}d", interval="1d")
    
    print(f"Data: {len(df)} hourly bars")
    print(f"Period: {df.index[0]} to {df.index[-1]}")
    print(f"Current Price: {df['Close'].iloc[-1]:.2f}")
    
    # Extract data
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    opens = df['Open'].values
    timestamps = df.index.values
    
    # Calculate indicators
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
    daily_highs = df_daily['High'].values if len(df_daily) > 0 else []
    daily_lows = df_daily['Low'].values if len(df_daily) > 0 else []
    
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
    
    df_daily_index = pd.DatetimeIndex(df_daily.index).tz_localize(None) if len(df_daily) > 0 else pd.DatetimeIndex([])
    df_index = pd.DatetimeIndex(df.index).tz_localize(None)
    
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
        range_size = ph - pl if (ph - pl) > 0.001 else 0.001
        price_position[i] = (closes[i] - pl) / range_size
    
    # Kill Zones
    hours = pd.to_datetime(timestamps).hour.values
    kill_zone = np.array([(1 <= h < 5) or (7 <= h < 12) or (13.5 <= h < 16) for h in hours])
    
    # Analyze each bar
    signals_found = []
    trades_taken = []
    
    for idx in range(20, len(df)):
        current_price = closes[idx]
        current_trend = trend[idx]
        htf_bias = htf_trend_hourly[idx]
        kz = kill_zone[idx]
        pp = price_position[idx]
        high = highs[idx]
        low = lows[idx]
        
        # Get nearest OB/FVG
        nearest_bull = next((ob for ob in reversed(bullish_obs) if ob['idx'] < idx), None)
        nearest_bear = next((ob for ob in reversed(bearish_obs) if ob['idx'] < idx), None)
        near_bull_fvg = next((f for f in reversed(bullish_fvgs) if f['idx'] < idx and f['mid'] < current_price < f['high']), None)
        near_bear_fvg = next((f for f in reversed(bearish_fvgs) if f['idx'] < idx and f['low'] < current_price < f['mid']), None)
        
        # Calculate confluence
        confluence = 0
        components = []
        
        if kz:
            confluence += 15
            components.append('KillZone(+15)')
        
        if htf_bias == 1 and current_trend >= 0:
            confluence += 25
            components.append('HTF_Bull(+25)')
        elif htf_bias == -1 and current_trend <= 0:
            confluence += 25
            components.append('HTF_Bear(+25)')
        elif htf_bias != 0:
            confluence += 10
            components.append('HTF_Weak(+10)')
        
        if pp < 0.25:
            confluence += 20
            components.append('Deep_Disc(+20)')
        elif pp < 0.35:
            confluence += 15
            components.append('Discount(+15)')
        elif pp > 0.75:
            confluence += 20
            components.append('Deep_Prem(+20)')
        elif pp > 0.65:
            confluence += 15
            components.append('Premium(+15)')
        
        if near_bull_fvg and current_trend >= 0:
            confluence += 15
            components.append('Bull_FVG(+15)')
        if near_bear_fvg and current_trend <= 0:
            confluence += 15
            components.append('Bear_FVG(+15)')
        if nearest_bull and current_price > nearest_bull['high']:
            confluence += 10
            components.append('Bull_OB(+10)')
        if nearest_bear and current_price < nearest_bear['low']:
            confluence += 10
            components.append('Bear_OB(+10)')
        
        grade = 'F'
        if confluence >= 75:
            grade = 'A+'
        elif confluence >= 70:
            grade = 'A'
        elif confluence >= 60:
            grade = 'B'
        elif confluence >= 50:
            grade = 'C'
        
        # Check for signal
        ai_filter_pass = False
        if confluence >= 70 and grade in ['A+', 'A']:
            if htf_bias != 0 and kz:
                if (near_bull_fvg or nearest_bull) or (near_bear_fvg or nearest_bear):
                    if (htf_bias == 1 and pp < 0.40) or (htf_bias == -1 and pp > 0.60):
                        ai_filter_pass = True
        
        if ai_filter_pass:
            atr = (highs[idx-14:idx] - lows[idx-14:idx]).mean() if idx > 14 else 50
            
            signal_info = {
                'time': str(timestamps[idx])[:16],
                'price': current_price,
                'confluence': confluence,
                'grade': grade,
                'htf': htf_bias,
                'pp': pp,
                'kz': kz,
                'components': components
            }
            
            # Check for trade
            if pp < 0.40 and htf_bias == 1:
                if near_bull_fvg and current_price > near_bull_fvg['mid']:
                    signal_info['direction'] = 'LONG'
                    signal_info['sl'] = near_bull_fvg['low'] - atr * 0.5
                    signal_info['tp'] = current_price + atr * 2.5
                    trades_taken.append(signal_info)
                elif nearest_bull and current_price > nearest_bull['high']:
                    signal_info['direction'] = 'LONG'
                    signal_info['sl'] = nearest_bull['low'] - atr * 0.5
                    signal_info['tp'] = current_price + atr * 2.5
                    trades_taken.append(signal_info)
            elif pp > 0.60 and htf_bias == -1:
                if near_bear_fvg and current_price < near_bear_fvg['mid']:
                    signal_info['direction'] = 'SHORT'
                    signal_info['sl'] = near_bear_fvg['high'] + atr * 0.5
                    signal_info['tp'] = current_price - atr * 2.5
                    trades_taken.append(signal_info)
                elif nearest_bear and current_price < nearest_bear['low']:
                    signal_info['direction'] = 'SHORT'
                    signal_info['sl'] = nearest_bear['high'] + atr * 0.5
                    signal_info['tp'] = current_price - atr * 2.5
                    trades_taken.append(signal_info)
            
            signals_found.append(signal_info)
    
    # Print results
    print(f"\nBARS ANALYZED: {len(df) - 20}")
    print(f"A-GRADE SIGNALS FOUND: {len(signals_found)}")
    print(f"TRADES THAT WOULD HAVE BEEN TAKEN: {len(trades_taken)}")
    
    if signals_found:
        print(f"\n{'='*80}")
        print("ALL SIGNALS FOUND:")
        print(f"{'='*80}")
        print(f"{'Time':<16} | {'Conf':<4} | {'Grade':<5} | {'HTF':<4} | {'PP':<5} | {'Components'}")
        print("-" * 80)
        for s in signals_found:
            print(f"{s['time']:<16} | {s['confluence']:<4} | {s['grade']:<5} | {s['htf']:<4.0f} | {s['pp']:<5.2f} | {', '.join(s['components'][:3])}")
    
    if trades_taken:
        print(f"\n{'='*80}")
        print("TRADES THAT WOULD HAVE BEEN EXECUTED:")
        print(f"{'='*80}")
        print(f"{'#':<3} | {'Time':<16} | {'Dir':<6} | {'Entry':<8} | {'SL':<8} | {'TP':<8} | {'Conf'}")
        print("-" * 80)
        for i, t in enumerate(trades_taken):
            print(f"{i+1:<3} | {t['time']:<16} | {t.get('direction', 'NONE'):<6} | {t['price']:<8.2f} | {t.get('sl', 0):<8.1f} | {t.get('tp', 0):<8.1f} | {t['confluence']}")
    else:
        print("\nNo trades would have been taken (no A-grade setups with OB/FVG entry)")
    
    return {
        'symbol': symbol,
        'signals': len(signals_found),
        'trades': len(trades_taken),
        'signal_details': signals_found,
        'trade_details': trades_taken
    }

# Run for all symbols
symbols = ['NQ=F', 'YM=F', 'EURUSD=X']
results = []

for symbol in symbols:
    result = analyze_symbol(symbol)
    if result:
        results.append(result)

# Summary
print(f"\n{'='*80}")
print("SUMMARY - PAST 2 DAYS")
print(f"{'='*80}")
print(f"{'Symbol':<15} | {'Signals':<8} | {'Trades':<8}")
print("-" * 40)
for r in results:
    print(f"{r['symbol']:<15} | {r['signals']:<8} | {r['trades']:<8}")
print(f"{'='*80}")

# Save results
output = {
    'timestamp': datetime.now().isoformat(),
    'period': 'past_2_days',
    'symbols': results
}

with open('v2_2day_signals_analysis.json', 'w') as f:
    json.dump(output, f, indent=2)

print("\nDetailed analysis saved to: v2_2day_signals_analysis.json")
print("="*80)
