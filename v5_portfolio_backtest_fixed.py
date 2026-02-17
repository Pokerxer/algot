"""
V5 Backtest - Fixed PnL Calculation
====================================
Proper position sizing and PnL calculation with shared capital.
"""

import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import sys
sys.path.insert(0, '/Users/mac/Documents/Algot')

import json
from datetime import datetime
import pandas as pd
import numpy as np

# Import from ict_v5_ibkr
exec(open('/Users/mac/Documents/Algot/ict_v5_ibkr.py').read().split('def run_backtest')[0])

def run_portfolio_backtest_fixed(symbols, days=180, initial_capital=10000, risk_per_trade=0.02, use_ibkr=True):
    """Run portfolio backtest with proper PnL calculation."""
    
    print(f"\n{'='*80}")
    print(f"ICT V5 Portfolio Backtest - Fixed PnL Calculation")
    print(f"Initial Capital: ${initial_capital:,}")
    print(f"Risk per Trade: {risk_per_trade*100}%")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"{'='*80}\n")
    
    # Load data for all symbols
    all_data = {}
    for symbol in symbols:
        print(f"Loading data for {symbol}...")
        if use_ibkr:
            data = prepare_data_ibkr(symbol)
        else:
            data = prepare_data(symbol)
        
        if data is not None and len(data.get('closes', [])) >= 50:
            # Create DataFrame with timestamp index
            df = pd.DataFrame({
                'close': data['closes'],
                'high': data['highs'],
                'low': data['lows'],
                'htf_trend': data['htf_trend'],
                'ltf_trend': data['ltf_trend'],
                'kill_zone': data['kill_zone'],
                'price_position': data['price_position'],
            }, index=data['df'].index)
            
            df.attrs['bullish_fvgs'] = data['bullish_fvgs']
            df.attrs['bearish_fvgs'] = data['bearish_fvgs']
            df.attrs['symbol'] = symbol
            
            all_data[symbol] = df
            print(f"  Loaded {len(df)} rows")
        else:
            print(f"  No data for {symbol}")
    
    if not all_data:
        print("No data loaded!")
        return None
    
    # Combine all timestamps
    all_timestamps = set()
    for df in all_data.values():
        all_timestamps.update(df.index)
    
    sorted_timestamps = sorted(list(all_timestamps))
    print(f"\nProcessing {len(sorted_timestamps)} timestamps...")
    
    # Initialize portfolio
    balance = initial_capital
    positions = {}
    all_trades = []
    symbol_stats = {symbol: {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0} for symbol in all_data.keys()}
    
    # FIXED: Use fixed dollar risk amount instead of percentage of growing balance
    # This prevents exponential position growth
    fixed_risk_amount = initial_capital * risk_per_trade  # $200 risk per trade (2% of $10K)
    max_position_pct = 0.50  # Maximum 50% of capital in one trade
    
    # Process each timestamp
    for i, timestamp in enumerate(sorted_timestamps):
        if i % 1000 == 0 and i > 0:
            print(f"  Processed {i}/{len(sorted_timestamps)} timestamps, Balance: ${balance:,.2f}")
        
        # Check exits first
        for symbol, position in list(positions.items()):
            df = all_data[symbol]
            
            if timestamp not in df.index:
                continue
            
            idx = df.index.get_loc(timestamp)
            if idx >= len(df) - 1:
                continue
            
            # Get next bar's prices
            next_bar = df.iloc[idx + 1]
            next_low = next_bar['low']
            next_high = next_bar['high']
            next_close = next_bar['close']
            
            exit_price = None
            exit_reason = None
            
            # Check stop/target on next bar
            if position['direction'] == 1:  # Long
                if next_low <= position['stop_loss']:
                    exit_price = position['stop_loss']
                    exit_reason = 'stop_loss'
                elif next_high >= position['take_profit']:
                    exit_price = position['take_profit']
                    exit_reason = 'take_profit'
            else:  # Short
                if next_high >= position['stop_loss']:
                    exit_price = position['stop_loss']
                    exit_reason = 'stop_loss'
                elif next_low <= position['take_profit']:
                    exit_price = position['take_profit']
                    exit_reason = 'take_profit'
            
            if exit_price:
                # FIXED: Calculate PnL properly
                # For long: (exit - entry) * position_size
                # For short: (entry - exit) * position_size
                if position['direction'] == 1:
                    price_change = exit_price - position['entry_price']
                else:
                    price_change = position['entry_price'] - exit_price
                
                # FIXED: PnL = price_change * position_size
                pnl = price_change * position['position_size']
                
                # Update balance
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
                
                # Update stats
                symbol_stats[symbol]['trades'] += 1
                symbol_stats[symbol]['pnl'] += pnl
                if pnl > 0:
                    symbol_stats[symbol]['wins'] += 1
                else:
                    symbol_stats[symbol]['losses'] += 1
                
                del positions[symbol]
        
        # Check for new entries
        for symbol, df in all_data.items():
            if symbol in positions:
                continue
            
            if timestamp not in df.index:
                continue
            
            idx = df.index.get_loc(timestamp)
            if idx < 50 or idx >= len(df) - 1:
                continue
            
            row = df.iloc[idx]
            htf = row['htf_trend']
            ltf = row['ltf_trend']
            kz = row['kill_zone']
            pp = row['price_position']
            current_price = row['close']
            
            bullish_fvgs = df.attrs['bullish_fvgs']
            bearish_fvgs = df.attrs['bearish_fvgs']
            
            near_bull_fvg = next((f for f in reversed(bullish_fvgs) if f['idx'] < idx and f['mid'] < current_price < f['high']), None)
            near_bear_fvg = next((f for f in reversed(bearish_fvgs) if f['idx'] < idx and f['low'] < current_price < f['mid']), None)
            
            # Calculate confluence
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
            
            # Entry signal
            if confluence >= 60:
                if htf == 1 and ltf >= 0:
                    direction = 1
                    stop = row['low']
                    target = current_price + (current_price - stop) * 2
                elif htf == -1 and ltf <= 0:
                    direction = -1
                    stop = row['high']
                    target = current_price - (stop - current_price) * 2
                else:
                    continue
                
                # FIXED: Calculate position size properly
                stop_distance = abs(current_price - stop)
                if stop_distance > 0:
                    # Use fixed risk amount, not percentage of balance
                    # This prevents exponential growth
                    position_size = fixed_risk_amount / stop_distance
                    
                    # Cap position size at max_position_pct of current balance
                    max_position_size = (balance * max_position_pct) / current_price
                    position_size = min(position_size, max_position_size)
                    
                    # Ensure minimum position size
                    if position_size > 0:
                        positions[symbol] = {
                            'entry_price': current_price,
                            'stop_loss': stop,
                            'take_profit': target,
                            'direction': direction,
                            'entry_idx': idx,
                            'confluence': confluence,
                            'position_size': position_size,
                            'entry_time': str(timestamp),
                            'risk_amount': fixed_risk_amount  # Track actual risk
                        }
    
    # Calculate final results
    total_pnl = balance - initial_capital
    total_return_pct = (total_pnl / initial_capital) * 100
    total_trades = len(all_trades)
    wins = len([t for t in all_trades if t['result'] == 'WIN'])
    losses = len([t for t in all_trades if t['result'] == 'LOSS'])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    # Calculate max drawdown
    peak = initial_capital
    max_dd = 0
    running_balance = initial_capital
    
    for trade in all_trades:
        running_balance += trade['pnl']
        if running_balance > peak:
            peak = running_balance
        dd = (peak - running_balance) / peak * 100
        if dd > max_dd:
            max_dd = dd
    
    results = {
        'backtest_config': {
            'symbols': list(all_data.keys()),
            'days': days,
            'initial_capital': initial_capital,
            'risk_per_trade': risk_per_trade,
            'fixed_risk_amount': fixed_risk_amount,
            'max_position_pct': max_position_pct,
            'data_source': 'IBKR' if use_ibkr else 'Yahoo',
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
    
    return results


# Test with all symbols
symbols = ['SOLUSD', 'ETHUSD', 'BTCUSD', 'LINKUSD', 'LTCUSD', 'SI', 'UNIUSD', 'NG', 'NQ', 'GC', 'CL', 'ES']

results = run_portfolio_backtest_fixed(symbols, days=180, initial_capital=10000, risk_per_trade=0.02, use_ibkr=True)

if results:
    # Save to JSON
    output_file = '/Users/mac/Documents/Algot/v5_portfolio_backtest_fixed.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("PORTFOLIO BACKTEST SUMMARY - FIXED")
    print(f"{'='*80}")
    print(f"Initial Capital: ${results['summary']['initial_capital']:,}")
    print(f"Final Capital: ${results['summary']['final_capital']:,}")
    print(f"Total PnL: ${results['summary']['total_pnl']:,.2f}")
    print(f"Total Return: {results['summary']['total_return_pct']:.1f}%")
    print(f"Max Drawdown: {results['summary']['max_drawdown_pct']:.1f}%")
    print(f"\nTotal Trades: {results['summary']['total_trades']}")
    print(f"Win Rate: {results['summary']['win_rate']:.1f}%")
    print(f"Wins: {results['summary']['wins']} | Losses: {results['summary']['losses']}")
    print(f"\nResults saved to: {output_file}")
    print(f"{'='*80}")
    
    # Print symbol breakdown
    print("\nSymbol Breakdown:")
    print(f"{'Symbol':<10} {'Trades':<8} {'Wins':<6} {'Losses':<8} {'Win%':<8} {'PnL':<15}")
    print("-" * 60)
    for symbol, stats in sorted(results['symbol_stats'].items(), key=lambda x: x[1]['pnl'], reverse=True):
        win_pct = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
        print(f"{symbol:<10} {stats['trades']:<8} {stats['wins']:<6} {stats['losses']:<8} {win_pct:<8.1f} ${stats['pnl']:<14.2f}")
    
    # Print recent trades
    print("\n\nRecent 10 Trades:")
    print(f"{'#':<4} {'Symbol':<10} {'Direction':<8} {'Entry':<12} {'Exit':<12} {'PnL':<10} {'Result':<6}")
    print("-" * 75)
    for i, trade in enumerate(results['trades'][-10:], 1):
        print(f"{i:<4} {trade['symbol']:<10} {trade['direction']:<8} ${trade['entry_price']:<11.2f} ${trade['exit_price']:<11.2f} ${trade['pnl']:<9.2f} {trade['result']:<6}")
    
    # Print JSON summary only (not all trades)
    print("\n\nJSON SUMMARY:")
    summary_json = {
        'config': results['backtest_config'],
        'summary': results['summary'],
        'symbol_stats': results['symbol_stats']
    }
    print(json.dumps(summary_json, indent=2))
