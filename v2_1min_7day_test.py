"""
ICT V2 - 1min Timeframe Backtest (Max 7 Days - Yahoo Limit)
NQ, YM, EURUSD with Max 1 Contract
Note: Yahoo Finance limits 1min data to approximately 7 days
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import json

print("=" * 80)
print("ICT V2 - 1MIN TIMEFRAME BACKTEST")
print("Note: Yahoo limits 1min data to ~7 days max")
print("=" * 80)

def run_1min_backtest(symbol, days=7):
    print(f"\n{'='*80}")
    print(f"BACKTEST: {symbol} (1min, max {days} days - Yahoo limit)")
    print(f"{'='*80}")
    
    # Fetch 1min data (Yahoo limits to 7 days)
    print(f"Fetching {symbol} 1min data...")
    try:
        # Try to get max available (usually 7 days for 1min)
        df = yf.Ticker(symbol).history(period="7d", interval="1m")
        if df.empty or len(df) < 100:
            print(f"Not enough 1min data for {symbol}")
            return None
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None
    
    df = df.dropna()
    df = df[~df.index.duplicated(keep='first')]
    
    # Get hourly for HTF trend
    df_hourly = yf.Ticker(symbol).history(period="7d", interval="1h")
    
    print(f"Data: {len(df)} 1min bars from {df.index[0]} to {df.index[-1]}")
    print(f"This covers approximately {len(df)/390:.1f} trading days (390 min/day)")
    
    # Extract data
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    opens = df['Open'].values
    timestamps = df.index.values
    
    # Calculate indicators on 1min
    # FVGs (3 bar separation)
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
    
    # HTF Trend from hourly
    df_hourly_index = pd.DatetimeIndex(df_hourly.index).tz_localize(None)
    df_index = pd.DatetimeIndex(df.index).tz_localize(None)
    
    hourly_trend = np.zeros(len(df_hourly))
    if len(df_hourly) >= 5:
        h_highs = df_hourly['High'].values
        h_lows = df_hourly['Low'].values
        for i in range(1, len(df_hourly)):
            if h_highs[i] > h_highs[max(0,i-5):i].max() and h_lows[i] > h_lows[max(0,i-5):i].min():
                hourly_trend[i] = 1
            elif h_highs[i] < h_highs[max(0,i-5):i].max() and h_lows[i] < h_lows[max(0,i-5):i].min():
                hourly_trend[i] = -1
    
    htf_trend = np.zeros(len(df))
    for i in range(len(df)):
        bar_time = df_index[i]
        for j in range(len(df_hourly)-1, -1, -1):
            if df_hourly_index[j] <= bar_time:
                htf_trend[i] = hourly_trend[j] if j < len(hourly_trend) else 0
                break
    
    # LTF Trend (20 bar lookback on 1min = 20 minutes)
    trend = np.zeros(len(df))
    for i in range(20, len(df)):
        rh = highs[max(0,i-20):i].max()
        rl = lows[max(0,i-20):i].min()
        if rh > highs[i-5] and rl > lows[i-5]:
            trend[i] = 1
        elif rh < highs[i-5] and rl < lows[i-5]:
            trend[i] = -1
    
    # Price Position (20 bar lookback)
    price_position = np.zeros(len(df))
    for i in range(20, len(df)):
        ph = highs[i-20:i].max()
        pl = lows[i-20:i].min()
        range_size = ph - pl
        if range_size < 0.001:
            range_size = 0.001
        price_position[i] = (closes[i] - pl) / range_size
    
    # Kill Zones
    hours = pd.to_datetime(timestamps).hour.values
    minutes = pd.to_datetime(timestamps).minute.values
    kill_zone = np.zeros(len(df), dtype=bool)
    for i in range(len(hours)):
        h = hours[i]
        m = minutes[i]
        time_decimal = h + m/60
        kill_zone[i] = (1 <= time_decimal < 5) or (7 <= time_decimal < 12) or (13.5 <= time_decimal < 16)
    
    print("Running 1min simulation...")
    
    # Trading
    trades = []
    open_trade = None
    MAX_CONTRACTS = 1
    signals_found = 0
    bars_analyzed = 0
    
    for idx in range(30, len(df)):
        current_price = closes[idx]
        current_trend = trend[idx]
        htf_bias = htf_trend[idx]
        kz = kill_zone[idx]
        pp = price_position[idx]
        high = highs[idx]
        low = lows[idx]
        
        # Skip if outside market hours (9:30 AM - 4:00 PM ET for futures)
        h = hours[idx]
        if h < 9 or h >= 16:
            continue
        
        bars_analyzed += 1
        
        # Get nearest levels
        nearest_bull = next((ob for ob in reversed(bullish_obs) if ob['idx'] < idx), None)
        nearest_bear = next((ob for ob in reversed(bearish_obs) if ob['idx'] < idx), None)
        near_bull_fvg = next((f for f in reversed(bullish_fvgs) if f['idx'] < idx and f['mid'] < current_price < f['high']), None)
        near_bear_fvg = next((f for f in reversed(bearish_fvgs) if f['idx'] < idx and f['low'] < current_price < f['mid']), None)
        
        # Calculate confluence
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
        
        # Count all A-grade signals
        if confluence >= 70:
            signals_found += 1
        
        # V2 Filter
        ai_filter_pass = False
        if confluence >= 70 and grade in ['A+', 'A']:
            if htf_bias != 0 and kz:
                if (near_bull_fvg or nearest_bull) or (near_bear_fvg or nearest_bear):
                    if (htf_bias == 1 and pp < 0.40) or (htf_bias == -1 and pp > 0.60):
                        ai_filter_pass = True
        
        # ATR on 1min (14 bars = 14 minutes)
        atr = (highs[max(0,idx-14):idx] - lows[max(0,idx-14):idx]).mean() if idx > 14 else 5
        
        # Exit check
        if open_trade:
            if open_trade['dir'] == 'long':
                if low <= open_trade['sl']:
                    open_trade['exit'] = open_trade['sl']
                    open_trade['exit_time'] = str(timestamps[idx])[:19]
                    open_trade['pnl'] = (open_trade['exit'] - open_trade['entry']) * MAX_CONTRACTS * 20
                    open_trade['status'] = 'STOP_HIT'
                    trades.append(open_trade)
                    open_trade = None
                elif high >= open_trade['tp']:
                    open_trade['exit'] = open_trade['tp']
                    open_trade['exit_time'] = str(timestamps[idx])[:19]
                    open_trade['pnl'] = (open_trade['exit'] - open_trade['entry']) * MAX_CONTRACTS * 20
                    open_trade['status'] = 'TP_HIT'
                    trades.append(open_trade)
                    open_trade = None
            else:
                if high >= open_trade['sl']:
                    open_trade['exit'] = open_trade['sl']
                    open_trade['exit_time'] = str(timestamps[idx])[:19]
                    open_trade['pnl'] = (open_trade['entry'] - open_trade['exit']) * MAX_CONTRACTS * 20
                    open_trade['status'] = 'STOP_HIT'
                    trades.append(open_trade)
                    open_trade = None
                elif low <= open_trade['tp']:
                    open_trade['exit'] = open_trade['tp']
                    open_trade['exit_time'] = str(timestamps[idx])[:19]
                    open_trade['pnl'] = (open_trade['entry'] - open_trade['exit']) * MAX_CONTRACTS * 20
                    open_trade['status'] = 'TP_HIT'
                    trades.append(open_trade)
                    open_trade = None
            
            # Time exit (180 bars = 3 hours on 1min)
            if open_trade and (idx - open_trade['entry_idx']) > 180:
                open_trade['exit'] = closes[idx]
                open_trade['exit_time'] = str(timestamps[idx])[:19]
                open_trade['pnl'] = (closes[idx] - open_trade['entry']) * MAX_CONTRACTS * 20 if open_trade['dir'] == 'long' else (open_trade['entry'] - closes[idx]) * MAX_CONTRACTS * 20
                open_trade['status'] = 'TIME_EXIT'
                trades.append(open_trade)
                open_trade = None
        
        # Entry
        if open_trade is None and ai_filter_pass:
            signal = None
            sl = None
            tp = None
            
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
            
            if signal and sl and tp:
                # Check not immediately stopped
                would_sl = False
                if signal == 'long' and low <= sl:
                    would_sl = True
                elif signal == 'short' and high >= sl:
                    would_sl = True
                
                if not would_sl:
                    open_trade = {
                        'entry_idx': idx,
                        'entry_time': str(timestamps[idx])[:19],
                        'entry': current_price,
                        'dir': signal,
                        'size': MAX_CONTRACTS,
                        'sl': round(sl, 2),
                        'tp': round(tp, 2),
                        'confluence': confluence,
                        'grade': grade,
                        'atr': round(atr, 2)
                    }
        
        if idx % 1000 == 0:
            print(f"  Progress: {idx}/{len(df)} | Signals: {signals_found} | Trades: {len(trades)}")
    
    # Close open
    if open_trade:
        open_trade['exit'] = closes[-1]
        open_trade['exit_time'] = str(timestamps[-1])[:19]
        open_trade['pnl'] = (closes[-1] - open_trade['entry']) * MAX_CONTRACTS * 20 if open_trade['dir'] == 'long' else (open_trade['entry'] - closes[-1]) * MAX_CONTRACTS * 20
        open_trade['status'] = 'EOD'
        trades.append(open_trade)
    
    # Stats
    closed = [t for t in trades if 'exit' in t]
    winners = [t for t in closed if t['pnl'] > 0]
    losers = [t for t in closed if t['pnl'] <= 0]
    
    total_pnl = sum(t['pnl'] for t in closed)
    win_rate = len(winners) / len(closed) * 100 if closed else 0
    pf = sum(t['pnl'] for t in winners) / abs(sum(t['pnl'] for t in losers)) if losers else float('inf')
    
    longs = [t for t in closed if t['dir'] == 'long']
    shorts = [t for t in closed if t['dir'] == 'short']
    
    print(f"\nRESULTS - {symbol} (1min)")
    print(f"Bars Analyzed (market hours): {bars_analyzed}")
    print(f"A-Grade Signals Found: {signals_found}")
    print(f"Trades Taken: {len(closed)} | Win Rate: {win_rate:.1f}% | PnL: ${total_pnl:+,.0f} | PF: {pf:.2f}")
    print(f"Long: {len(longs)} (${sum(t['pnl'] for t in longs):+,.0f}) | Short: {len(shorts)} (${sum(t['pnl'] for t in shorts):+,.0f})")
    
    return {
        'symbol': symbol,
        'timeframe': '1min',
        'bars_analyzed': bars_analyzed,
        'signals_found': signals_found,
        'trades_count': len(closed),
        'win_rate': win_rate,
        'pnl': total_pnl,
        'profit_factor': pf,
        'trade_details': closed
    }


# Run for all symbols
print("="*80)
print("STARTING 1MIN BACKTESTS")
print("NOTE: Yahoo limits 1min data to ~7 days")
print("="*80)

symbols = ['NQ=F', 'YM=F', 'EURUSD=X']
all_results = []

for symbol in symbols:
    result = run_1min_backtest(symbol, days=7)
    if result:
        all_results.append(result)

# Summary
print(f"\n{'='*80}")
print("1MIN TIMEFRAME SUMMARY (MAX 7 DAYS - YAHOO LIMIT)")
print(f"{'='*80}")
print(f"{'Symbol':<15} | {'Signals':<10} | {'Trades':<8} | {'Win%':<8} | {'PnL':<12} | {'PF':<6}")
print("-" * 75)
total_signals = 0
total_trades = 0
total_pnl = 0
for r in all_results:
    print(f"{r['symbol']:<15} | {r['signals_found']:<10} | {r['trades_count']:<8} | {r['win_rate']:<8.1f} | ${r['pnl']:+>11,.0f} | {r['profit_factor']:<6.2f}")
    total_signals += r['signals_found']
    total_trades += r['trades_count']
    total_pnl += r['pnl']

print(f"{'='*80}")
print(f"TOTAL: {total_signals} signals | {total_trades} trades | ${total_pnl:+,.0f}")
print(f"{'='*80}")

# Save results
output = {
    'metadata': {
        'strategy': 'ICT V2 (1min, Max 1 Contract)',
        'timeframe': '1min',
        'period': 'past_7_days_max',
        'data_limit': 'Yahoo limits 1min to 7 days',
        'timestamp': datetime.now().isoformat()
    },
    'summary': {
        'total_signals': total_signals,
        'total_trades': total_trades,
        'total_pnl': total_pnl
    },
    'results': all_results
}

with open('v2_1min_7day_trades.json', 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nDetailed trades saved to: v2_1min_7day_trades.json")

# Show trade details
if all_results:
    print(f"\n{'='*100}")
    print("TRADE DETAILS")
    print(f"{'='*100}")
    for r in all_results:
        if r['trade_details']:
            print(f"\n{r['symbol']}:")
            print(f"{'#':<3} | {'Entry Time':<16} | {'Dir':<5} | {'Entry':<10} | {'SL':<10} | {'TP':<10} | {'Exit':<10} | {'PnL':>10} | {'Status'}")
            print("-" * 105)
            for i, t in enumerate(r['trade_details']):
                print(f"{i+1:<3} | {t['entry_time'][:10]} {t['entry_time'][11:16]:<6} | {t['dir'].upper():<5} | {t['entry']:<10.2f} | {t['sl']:<10.2f} | {t['tp']:<10.2f} | {t.get('exit',0):<10.2f} | ${t['pnl']:+>9,.0f} | {t['status']}")
