"""
ICT V2 - 2 Day Backtest on NQ, YM, EURUSD (Max 1 Contract)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import json
import sys

def run_backtest(symbol, days=2):
    print(f"\n{'='*70}")
    print(f"ICT V2 BACKTEST - {symbol} ({days} DAYS)")
    print(f"{'='*70}")
    
    capital = 10000
    
    print(f"Fetching {symbol} data ({days} days, 1h)...")
    df = yf.Ticker(symbol).history(period=f"{days}d", interval="1h")
    if df.empty:
        print(f"No data for {symbol}")
        return None
    
    df = df.dropna()
    df = df[~df.index.duplicated(keep='first')]
    print(f"Data: {len(df)} bars from {df.index[0]} to {df.index[-1]}")
    
    df_daily = yf.Ticker(symbol).history(period=f"{days}d", interval="1d")
    
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
        range_size = ph - pl
        if range_size < 0.001:
            range_size = 0.001
        price_position[i] = (closes[i] - pl) / range_size
    
    # Kill Zones
    hours = pd.to_datetime(timestamps).hour.values
    kill_zone = np.zeros(len(df), dtype=bool)
    for i in range(len(hours)):
        h = hours[i]
        kill_zone[i] = (1 <= h < 5) or (7 <= h < 12) or (13.5 <= h < 16)
    
    # Trading
    trades = []
    open_trade = None
    MAX_CONTRACTS = 1
    
    for idx in range(50, len(df)):
        current_price = closes[idx]
        current_trend = trend[idx]
        htf_bias = htf_trend_hourly[idx]
        kz = kill_zone[idx]
        pp = price_position[idx]
        high = highs[idx]
        low = lows[idx]
        
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
                if low <= open_trade['sl']:
                    open_trade['exit'] = low
                    open_trade['pnl'] = (open_trade['exit'] - open_trade['entry']) * MAX_CONTRACTS * 20
                    open_trade['status'] = 'STOP_HIT'
                    trades.append(open_trade)
                    open_trade = None
                elif high >= open_trade['tp']:
                    open_trade['exit'] = high
                    open_trade['pnl'] = (open_trade['exit'] - open_trade['entry']) * MAX_CONTRACTS * 20
                    open_trade['status'] = 'TP_HIT'
                    trades.append(open_trade)
                    open_trade = None
            else:
                if high >= open_trade['sl']:
                    open_trade['exit'] = high
                    open_trade['pnl'] = (open_trade['entry'] - open_trade['exit']) * MAX_CONTRACTS * 20
                    open_trade['status'] = 'STOP_HIT'
                    trades.append(open_trade)
                    open_trade = None
                elif low <= open_trade['tp']:
                    open_trade['exit'] = low
                    open_trade['pnl'] = (open_trade['entry'] - open_trade['exit']) * MAX_CONTRACTS * 20
                    open_trade['status'] = 'TP_HIT'
                    trades.append(open_trade)
                    open_trade = None
            
            if open_trade and (idx - open_trade['entry_idx']) > 20:
                open_trade['exit'] = closes[idx]
                open_trade['pnl'] = (open_trade['exit'] - open_trade['entry']) * MAX_CONTRACTS * 20 if open_trade['dir'] == 'long' else (open_trade['entry'] - open_trade['exit']) * MAX_CONTRACTS * 20
                open_trade['status'] = 'TIME_EXIT'
                trades.append(open_trade)
                open_trade = None
        
        # Entry
        if open_trade is None and ai_filter_pass:
            if pp < 0.40 and htf_bias == 1:
                if near_bull_fvg and current_price > near_bull_fvg['mid']:
                    open_trade = {
                        'entry_idx': idx,
                        'entry_time': str(timestamps[idx])[:19],
                        'entry': current_price,
                        'dir': 'long',
                        'size': MAX_CONTRACTS,
                        'sl': near_bull_fvg['low'] - atr * 0.5,
                        'tp': current_price + atr * 2.5,
                        'confluence': confluence,
                        'grade': grade
                    }
                elif nearest_bull and current_price > nearest_bull['high']:
                    open_trade = {
                        'entry_idx': idx,
                        'entry_time': str(timestamps[idx])[:19],
                        'entry': current_price,
                        'dir': 'long',
                        'size': MAX_CONTRACTS,
                        'sl': nearest_bull['low'] - atr * 0.5,
                        'tp': current_price + atr * 2.5,
                        'confluence': confluence,
                        'grade': grade
                    }
            elif pp > 0.60 and htf_bias == -1:
                if near_bear_fvg and current_price < near_bear_fvg['mid']:
                    open_trade = {
                        'entry_idx': idx,
                        'entry_time': str(timestamps[idx])[:19],
                        'entry': current_price,
                        'dir': 'short',
                        'size': MAX_CONTRACTS,
                        'sl': near_bear_fvg['high'] + atr * 0.5,
                        'tp': current_price - atr * 2.5,
                        'confluence': confluence,
                        'grade': grade
                    }
                elif nearest_bear and current_price < nearest_bear['low']:
                    open_trade = {
                        'entry_idx': idx,
                        'entry_time': str(timestamps[idx])[:19],
                        'entry': current_price,
                        'dir': 'short',
                        'size': MAX_CONTRACTS,
                        'sl': nearest_bear['high'] + atr * 0.5,
                        'tp': current_price - atr * 2.5,
                        'confluence': confluence,
                        'grade': grade
                    }
    
    # Close open
    if open_trade:
        open_trade['exit'] = closes[-1]
        open_trade['pnl'] = (open_trade['exit'] - open_trade['entry']) * MAX_CONTRACTS * 20 if open_trade['dir'] == 'long' else (open_trade['entry'] - open_trade['exit']) * MAX_CONTRACTS * 20
        open_trade['status'] = 'EOD'
        trades.append(open_trade)
    
    # Stats
    closed = [t for t in trades if 'exit' in t]
    winners = [t for t in closed if t['pnl'] > 0]
    losers = [t for t in closed if t['pnl'] <= 0]
    
    total_pnl = sum(t['pnl'] for t in closed)
    win_rate = len(winners) / len(closed) * 100 if closed else 0
    pf = sum(t['pnl'] for t in winners) / abs(sum(t['pnl'] for t in losers)) if losers else float('inf')
    
    print(f"\nRESULTS - {symbol}")
    print(f"Period: {str(timestamps[0])[:10]} to {str(timestamps[-1])[:10]}")
    print(f"Trades: {len(closed)} | Win Rate: {win_rate:.1f}% | PnL: ${total_pnl:+,.0f} | PF: {pf:.2f}")
    
    return {
        'symbol': symbol,
        'period': {'start': str(timestamps[0])[:10], 'end': str(timestamps[-1])[:10]},
        'trades': len(closed),
        'win_rate': win_rate,
        'pnl': total_pnl,
        'profit_factor': pf,
        'trade_details': closed
    }


# Run for all symbols
print("="*70)
print("ICT V2 - 2 DAY BACKTEST (NQ, YM, EURUSD)")
print("="*70)

symbols = ['NQ=F', 'YM=F', 'EURUSD=X']
all_results = []

for symbol in symbols:
    result = run_backtest(symbol, days=2)
    if result:
        all_results.append(result)

# Save combined results
combined = {
    'metadata': {
        'strategy': 'ICT V2 (Max 1 Contract)',
        'timestamp': datetime.now().isoformat()
    },
    'results': all_results,
    'summary': {
        'total_trades': sum(r['trades'] for r in all_results),
        'total_pnl': sum(r['pnl'] for r in all_results)
    }
}

with open('v2_2day_multisymbol_trades.json', 'w') as f:
    json.dump(combined, f, indent=2)

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
for r in all_results:
    print(f"{r['symbol']}: {r['trades']} trades | ${r['pnl']:+,.0f} | WR: {r['win_rate']:.1f}%")
print(f"\nTotal Trades: {combined['summary']['total_trades']}")
print(f"Total PnL: ${combined['summary']['total_pnl']:+,.0f}")
print(f"\nSaved to: v2_2day_multisymbol_trades.json")
