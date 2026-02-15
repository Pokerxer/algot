"""
ICT V2 - 6 Month Backtest on NQ (Max 1 Contract)
Entry at bar close, SL/TP checked intrabar
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import json

print("=" * 70)
print("ICT V2 - NQ 6 MONTHS (MAX 1 CONTRACT)")
print("=" * 70)

symbol = "NQ=F"
capital = 10000

print(f"Fetching {symbol} data (6 months, 1h)...")
df = yf.Ticker(symbol).history(period="6mo", interval="1h")
df = df.dropna()
df = df[~df.index.duplicated(keep='first')]
print(f"Data: {len(df)} bars from {df.index[0]} to {df.index[-1]}")

df_daily = yf.Ticker(symbol).history(period="6mo", interval="1d")

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

print("Running simulation...")

# Trading simulation
equity_curve = [capital]
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
    
    # Confluence
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
    elif confluence >= 60:
        grade = 'B'
    elif confluence >= 50:
        grade = 'C'
    
    ai_filter_pass = False
    if confluence >= 70 and grade in ['A+', 'A']:
        if htf_bias != 0 and kz:
            if (near_bull_fvg or nearest_bull) or (near_bear_fvg or nearest_bear):
                if (htf_bias == 1 and pp < 0.40) or (htf_bias == -1 and pp > 0.60):
                    ai_filter_pass = True
    
    atr = (highs[idx-14:idx] - lows[idx-14:idx]).mean() if idx > 14 else 50
    
    # Exit check with intrabar precision
    if open_trade:
        sl_hit = False
        tp_hit = False
        exit_price = None
        exit_idx = idx
        exit_time = str(timestamps[idx])[:19]
        
        if open_trade['dir'] == 'long':
            # Check if SL or TP was hit intrabar
            if low <= open_trade['sl']:
                sl_hit = True
                exit_price = open_trade['sl']
            elif high >= open_trade['tp']:
                tp_hit = True
                exit_price = open_trade['tp']
        else:
            if high >= open_trade['sl']:
                sl_hit = True
                exit_price = open_trade['sl']
            elif low <= open_trade['tp']:
                tp_hit = True
                exit_price = open_trade['tp']
        
        if sl_hit or tp_hit:
            pnl = (exit_price - open_trade['entry']) * MAX_CONTRACTS * 20 if open_trade['dir'] == 'long' else (open_trade['entry'] - exit_price) * MAX_CONTRACTS * 20
            
            open_trade['exit_idx'] = exit_idx
            open_trade['exit_time'] = exit_time
            open_trade['exit'] = exit_price
            open_trade['pnl'] = pnl
            open_trade['status'] = 'TP_HIT' if tp_hit else 'STOP_HIT'
            
            capital += pnl
            trades.append(open_trade)
            open_trade = None
        
        # Time exit after 20 bars
        elif (idx - open_trade['entry_idx']) > 20:
            exit_price = closes[idx]
            pnl = (exit_price - open_trade['entry']) * MAX_CONTRACTS * 20 if open_trade['dir'] == 'long' else (open_trade['entry'] - exit_price) * MAX_CONTRACTS * 20
            
            open_trade['exit_idx'] = idx
            open_trade['exit_time'] = str(timestamps[idx])[:19]
            open_trade['exit'] = exit_price
            open_trade['pnl'] = pnl
            open_trade['status'] = 'TIME_EXIT'
            
            capital += pnl
            trades.append(open_trade)
            open_trade = None
    
    # Entry
    if open_trade is None and ai_filter_pass:
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
            # Check if SL/TP already hit in this bar before entering
            would_sl_hit = False
            would_tp_hit = False
            
            if signal == 'long':
                if low <= sl:
                    would_sl_hit = True
                elif high >= tp:
                    would_tp_hit = True
            else:
                if high >= sl:
                    would_sl_hit = True
                elif low <= tp:
                    would_tp_hit = True
            
            # Only enter if not immediately stopped out
            if not would_sl_hit and not would_tp_hit:
                open_trade = {
                    'signal_idx': idx,
                    'signal_time': str(timestamps[idx])[:19],
                    'entry_idx': idx,
                    'entry_time': str(timestamps[idx])[:19],
                    'entry': current_price,
                    'dir': signal,
                    'size': MAX_CONTRACTS,
                    'sl': sl,
                    'tp': tp,
                    'confluence': confluence,
                    'grade': grade
                }
    
    equity_curve.append(capital)
    
    if idx % 500 == 0:
        print(f"Progress: {idx}/{len(df)} | Equity: ${capital:,.0f} | Trades: {len(trades)}")

# Close open positions at end
if open_trade:
    exit_price = closes[-1]
    pnl = (exit_price - open_trade['entry']) * MAX_CONTRACTS * 20 if open_trade['dir'] == 'long' else (open_trade['entry'] - exit_price) * MAX_CONTRACTS * 20
    
    open_trade['exit_idx'] = len(df) - 1
    open_trade['exit_time'] = str(timestamps[-1])[:19]
    open_trade['exit'] = exit_price
    open_trade['pnl'] = pnl
    open_trade['status'] = 'EOD'
    
    capital += pnl
    trades.append(open_trade)

# Statistics
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

exit_reasons = {}
for t in closed:
    reason = t.get('status', 'UNKNOWN')
    exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

# Max drawdown
equity_arr = np.array(equity_curve)
max_equity = np.maximum.accumulate(equity_arr)
drawdowns = (max_equity - equity_arr) / max_equity * 100
max_dd = np.max(drawdowns)

# Save results
results = {
    'metadata': {
        'symbol': symbol,
        'period': {'start': str(timestamps[0])[:10], 'end': str(timestamps[-1])[:10]},
        'strategy': 'ICT V2 (Max 1 Contract)',
        'timestamp': datetime.now().isoformat()
    },
    'capital': {'initial': 10000, 'final': capital, 'return_pct': total_return},
    'risk': {'max_drawdown_pct': max_dd},
    'trades': {'total': len(closed), 'winners': len(winners), 'losers': len(losers), 'win_rate': win_rate},
    'pnl': {'gross_profit': profit, 'gross_loss': loss, 'net_pnl': capital - 10000, 'profit_factor': pf},
    'direction': {
        'long': {'count': len(longs), 'wins': len([t for t in longs if t['pnl'] > 0]), 'pnl': sum(t['pnl'] for t in longs)},
        'short': {'count': len(shorts), 'wins': len([t for t in shorts if t['pnl'] > 0]), 'pnl': sum(t['pnl'] for t in shorts)}
    },
    'exit_reasons': exit_reasons,
    'trade_details': closed,
    'equity_curve': [{'date': str(timestamps[i])[:19], 'equity': e} for i, e in enumerate(equity_curve)]
}

with open('v2_nq_6month_1contract_trades.json', 'w') as f:
    json.dump(results, f, indent=2)

# Print summary
print()
print("=" * 70)
print("RESULTS - ICT V2 6 MONTHS (MAX 1 CONTRACT)")
print("=" * 70)
print(f"Period: {str(timestamps[0])[:10]} to {str(timestamps[-1])[:10]}")
print()
print(f"Capital: $10,000 -> ${capital:,.2f} ({total_return:+.1f}%)")
print(f"Max Drawdown: {max_dd:.1f}%")
print(f"Trades: {len(closed)} | Win Rate: {win_rate:.1f}%")
print(f"Profit Factor: {pf:.2f}")
print()
print("DIRECTION:")
print(f"  Long:  {len(longs)} trades | PnL: ${sum(t['pnl'] for t in longs):+,.0f}")
print(f"  Short: {len(shorts)} trades | PnL: ${sum(t['pnl'] for t in shorts):+,.0f}")
print()
print("EXIT REASONS:")
for reason, count in exit_reasons.items():
    print(f"  {reason}: {count}")
print()
print(f"Trade data saved to: v2_nq_6month_1contract_trades.json")
print("=" * 70)
