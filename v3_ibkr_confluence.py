"""
V3 Backtest with IBKR Data + V5 Confluence Filter
==================================================
V3 strategy with:
- V5's IBKR data fetching
- V5's confluence calculation (only take trades with confluence >= 60)
"""

import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import sys
sys.path.insert(0, '/Users/mac/Documents/Algot')

import json
from datetime import datetime
import pandas as pd
import numpy as np

exec(open('/Users/mac/Documents/Algot/ict_v5_ibkr.py').read().split('def run_backtest')[0])

def calculate_confluence_v5(htf, ltf, kz, pp, near_bull_fvg, near_bear_fvg):
    """V5 confluence calculation."""
    confluence = 0
    if kz:
        confluence += 15
    if htf == 1 and ltf >= 0:
        confluence += 25
    elif htf == -1 and ltf <= 0:
        confluence += 25
    if pp < 0.25:
        confluence += 20
    elif pp > 0.75:
        confluence += 20
    if near_bull_fvg and ltf >= 0:
        confluence += 15
    if near_bear_fvg and ltf <= 0:
        confluence += 15
    return confluence


def run_v3_confluence_backtest(symbols, days=180, initial_capital=10000, risk_per_trade=0.02, confluence_threshold=60):
    """Run V3 backtest with V5 confluence filtering."""
    
    print(f"\n{'='*80}")
    print(f"ICT V3 Backtest - V5 Confluence Filter (>= {confluence_threshold})")
    print(f"Initial Capital: ${initial_capital:,}")
    print(f"Risk per Trade: {risk_per_trade*100}%")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"{'='*80}\n")
    
    all_data = {}
    
    for symbol in symbols:
        print(f"Loading data for {symbol}...")
        data = prepare_data_ibkr(symbol)
        
        if data is not None and len(data.get('closes', [])) >= 50:
            print(f"  Loaded {len(data['closes'])} rows")
            all_data[symbol] = data
        else:
            print(f"  No data for {symbol}")
    
    if not all_data:
        print("No data loaded!")
        return None
    
    balance = initial_capital
    positions = {}
    all_trades = []
    symbol_stats = {s: {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0} for s in all_data.keys()}
    
    all_timestamps = set()
    for data in all_data.values():
        all_timestamps.update(data['df'].index.tolist())
    
    sorted_timestamps = sorted(all_timestamps)
    
    print(f"\nRunning backtest on {len(sorted_timestamps)} timestamps...")
    
    for i, timestamp in enumerate(sorted_timestamps):
        if i % 5000 == 0:
            print(f"  Progress: {i}/{len(sorted_timestamps)}")
        
        for symbol, position in list(positions.items()):
            data = all_data[symbol]
            
            if timestamp not in data['df'].index:
                continue
            
            idx = data['df'].index.get_loc(timestamp)
            if idx >= len(data['closes']) - 1:
                continue
            
            next_low = data['lows'][idx + 1]
            next_high = data['highs'][idx + 1]
            
            exit_price = None
            exit_reason = None
            
            if position['direction'] == 1:
                if next_low <= position['stop_loss']:
                    exit_price = position['stop_loss']
                    exit_reason = 'stop_loss'
                elif next_high >= position['take_profit']:
                    exit_price = position['take_profit']
                    exit_reason = 'take_profit'
            else:
                if next_high >= position['stop_loss']:
                    exit_price = position['stop_loss']
                    exit_reason = 'stop_loss'
                elif next_low <= position['take_profit']:
                    exit_price = position['take_profit']
                    exit_reason = 'take_profit'
            
            if exit_price:
                if position['direction'] == 1:
                    price_change = exit_price - position['entry_price']
                else:
                    price_change = position['entry_price'] - exit_price
                
                pnl = price_change * position['position_size']
                balance += pnl
                
                trade_record = {
                    'symbol': symbol,
                    'entry_price': round(position['entry_price'], 4),
                    'exit_price': round(exit_price, 4),
                    'direction': 'LONG' if position['direction'] == 1 else 'SHORT',
                    'position_size': round(position['position_size'], 4),
                    'pnl': round(pnl, 2),
                    'result': 'WIN' if pnl > 0 else 'LOSS',
                    'exit_reason': exit_reason,
                    'bars_held': idx - position['entry_idx'],
                    'confluence': position['confluence'],
                    'balance_after': round(balance, 2),
                    'exit_time': str(timestamp)
                }
                all_trades.append(trade_record)
                
                symbol_stats[symbol]['trades'] += 1
                symbol_stats[symbol]['pnl'] += pnl
                if pnl > 0:
                    symbol_stats[symbol]['wins'] += 1
                else:
                    symbol_stats[symbol]['losses'] += 1
                
                del positions[symbol]
        
        for symbol, data in all_data.items():
            if symbol in positions:
                continue
            
            if timestamp not in data['df'].index:
                continue
            
            idx = data['df'].index.get_loc(timestamp)
            if idx < 50 or idx >= len(data['closes']) - 1:
                continue
            
            current_price = data['closes'][idx]
            htf = data['htf_trend'][idx]
            ltf = data['ltf_trend'][idx]
            kz = data['kill_zone'][idx]
            pp = data['price_position'][idx]
            highs = data['highs']
            lows = data['lows']
            
            near_bull_fvg = next((f for f in reversed(data['bullish_fvgs']) 
                                   if f['idx'] < idx and f['mid'] < current_price < f['high']), None)
            near_bear_fvg = next((f for f in reversed(data['bearish_fvgs']) 
                                  if f['idx'] < idx and f['low'] < current_price < f['mid']), None)
            
            confluence = calculate_confluence_v5(htf, ltf, kz, pp, near_bull_fvg, near_bear_fvg)
            
            if confluence < confluence_threshold:
                continue
            
            if htf == 1 and ltf >= 0:
                direction = 1
                stop = lows[idx]
                target = current_price + (current_price - stop) * 2
            elif htf == -1 and ltf <= 0:
                direction = -1
                stop = highs[idx]
                target = current_price - (stop - current_price) * 2
            else:
                continue
            
            fixed_risk_amount = balance * risk_per_trade
            stop_distance = abs(current_price - stop)
            
            if stop_distance > 0:
                position_size = fixed_risk_amount / stop_distance
                max_position_pct = 0.25
                max_position_size = (balance * max_position_pct) / current_price
                position_size = min(position_size, max_position_size)
                
                if position_size > 0:
                    positions[symbol] = {
                        'entry_price': current_price,
                        'stop_loss': stop,
                        'take_profit': target,
                        'direction': direction,
                        'entry_idx': idx,
                        'position_size': position_size,
                        'entry_time': str(timestamp),
                        'confluence': confluence
                    }
    
    total_pnl = balance - initial_capital
    total_return_pct = (total_pnl / initial_capital) * 100
    total_trades = len(all_trades)
    wins = len([t for t in all_trades if t['result'] == 'WIN'])
    losses = len([t for t in all_trades if t['result'] == 'LOSS'])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    peak = initial_capital
    max_dd = 0
    running_balance = initial_capital
    
    for trade in all_trades:
        running_balance += trade['pnl']
        if running_balance > peak:
            peak = running_balance
        dd = (peak - running_balance) / peak * 100 if peak > 0 else 0
        if dd > max_dd:
            max_dd = dd
    
    results = {
        'config': {
            'symbols': list(all_data.keys()),
            'days': days,
            'initial_capital': initial_capital,
            'risk_per_trade': risk_per_trade,
            'confluence_threshold': confluence_threshold,
            'data_source': 'IBKR',
            'timestamp': datetime.now().isoformat()
        },
        'summary': {
            'initial_capital': initial_capital,
            'final_capital': round(balance, 2),
            'total_pnl': round(total_pnl, 2),
            'total_return_pct': round(total_return_pct, 2),
            'max_drawdown_pct': round(max_dd, 2),
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': round(win_rate, 2)
        },
        'symbol_stats': symbol_stats,
        'trades': all_trades
    }
    
    print(f"\n{'='*80}")
    print(f"V3 BACKTEST SUMMARY - IBKR + V5 Confluence (>= {confluence_threshold})")
    print(f"{'='*80}")
    print(f"Total PnL: ${total_pnl:,.2f}")
    print(f"Total Return: {total_return_pct:.1f}%")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {win_rate:.1f}%")
    print(f"Wins: {wins} | Losses: {losses}")
    print(f"\nSymbol     Trades   Wins   Losses   Win%     PnL")
    print("-" * 55)
    for symbol, stats in sorted(symbol_stats.items()):
        wr = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
        print(f"{symbol:<10} {stats['trades']:<7} {stats['wins']:<5} {stats['losses']:<7} {wr:5.1f}%   ${stats['pnl']:>9,.2f}")
    
    return results


if __name__ == '__main__':
    symbols = ['BTCUSD', 'ETHUSD', 'SOLUSD', 'LTCUSD', 'LINKUSD', 'UNIUSD']
    
    results = run_v3_confluence_backtest(
        symbols=symbols,
        days=180,
        initial_capital=10000,
        risk_per_trade=0.02,
        confluence_threshold=60
    )
    
    if results:
        print(f"\nFinal Capital: ${results['summary']['final_capital']:,.2f}")
