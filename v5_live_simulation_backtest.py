"""
V5 Live Trading Simulation Backtest
=====================================
Properly simulates how the live bot would trade:
- Uses bracket orders (entry + SL + TP simultaneously)
- Proper position sizing with contract multipliers
- Bar-by-bar processing like live trading
- Shared portfolio capital (not per-symbol)
"""

import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import sys
sys.path.insert(0, '/Users/mac/Documents/Algot')

import json
import pandas as pd
import numpy as np
from datetime import datetime

# Import functions from ict_v5_ibkr
import importlib.util
spec = importlib.util.spec_from_file_location("ict_v5_ibkr", "/Users/mac/Documents/Algot/ict_v5_ibkr.py")
ict_v5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ict_v5)

# Import the functions we need
fetch_ibkr_data = ict_v5.fetch_ibkr_data
prepare_data_ibkr = ict_v5.prepare_data_ibkr
prepare_data = ict_v5.prepare_data
get_signal = ict_v5.get_signal
get_ibkr_contract = ict_v5.get_ibkr_contract
get_contract_multiplier = ict_v5.get_contract_multiplier
calculate_position_size = ict_v5.calculate_position_size


def run_live_simulation_backtest(symbols, days=30, initial_capital=50000, risk_per_trade=0.02, use_ibkr=True):
    """
    Backtest that simulates live trading behavior.
    
    Key features:
    - Bar-by-bar processing (like live loop)
    - Shared portfolio capital
    - Proper position sizing with multipliers
    - Simulates bracket orders
    - Only one position per symbol
    """
    
    print(f"\n{'='*80}")
    print(f"ICT V5 Live Trading Simulation")
    print(f"Initial Capital: ${initial_capital:,}")
    print(f"Risk per Trade: {risk_per_trade*100}%")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"{'='*80}\n")
    
    # Load data for all symbols
    all_data = {}
    for symbol in symbols:
        print(f"Loading {symbol}...", end=' ')
        if use_ibkr:
            data = prepare_data_ibkr(symbol)
        else:
            data = prepare_data(symbol)
        
        if data is not None and len(data.get('closes', [])) >= 50:
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
            print(f"✓ {len(df)} bars")
        else:
            print(f"✗ No data")
    
    if not all_data:
        print("No data loaded!")
        return None
    
    # Combine all timestamps and sort
    all_timestamps = sorted(set().union(*[set(df.index) for df in all_data.values()]))
    print(f"\nProcessing {len(all_timestamps)} timestamps across {len(all_data)} symbols...")
    
    # Portfolio state
    balance = initial_capital
    positions = {}  # symbol -> position dict
    active_trades = []  # All completed trades
    daily_balances = []  # Track balance over time
    
    # Processing state
    last_processed_bar = {}  # Track last processed bar per symbol
    
    for i, timestamp in enumerate(all_timestamps):
        # Record daily balance (every 24 bars approx)
        if i % 24 == 0:
            daily_balances.append({'timestamp': str(timestamp), 'balance': balance})
        
        if i % 1000 == 0 and i > 0:
            print(f"  [{i}/{len(all_timestamps)}] Balance: ${balance:,.2f} | Positions: {len(positions)}")
        
        # Process each symbol at this timestamp
        for symbol, df in all_data.items():
            if timestamp not in df.index:
                continue
            
            idx = df.index.get_loc(timestamp)
            if idx < 50 or idx >= len(df) - 1:
                continue
            
            # Skip if we've already processed this bar (simulates the bar completion check)
            if symbol in last_processed_bar and last_processed_bar[symbol] == timestamp:
                continue
            last_processed_bar[symbol] = timestamp
            
            current_price = df.iloc[idx]['close']
            row = df.iloc[idx]
            
            # Check if position exists and should be closed (simulates bracket order SL/TP)
            if symbol in positions:
                pos = positions[symbol]
                next_bar = df.iloc[idx + 1]
                next_low = next_bar['low']
                next_high = next_bar['high']
                
                exit_price = None
                exit_reason = None
                
                # Check stop loss and take profit (like bracket orders)
                if pos['direction'] == 1:  # Long
                    if next_low <= pos['stop']:
                        exit_price = pos['stop']
                        exit_reason = 'stop_loss'
                    elif next_high >= pos['target']:
                        exit_price = pos['target']
                        exit_reason = 'take_profit'
                else:  # Short
                    if next_high >= pos['stop']:
                        exit_price = pos['stop']
                        exit_reason = 'stop_loss'
                    elif next_low <= pos['target']:
                        exit_price = pos['target']
                        exit_reason = 'take_profit'
                
                if exit_price:
                    # Calculate PnL with proper contract multiplier
                    contract_info = get_contract_multiplier(symbol)
                    
                    if pos['direction'] == 1:
                        price_change = exit_price - pos['entry']
                    else:
                        price_change = pos['entry'] - exit_price
                    
                    # Debug output for forex
                    if symbol in ['EURUSD', 'GBPUSD'] and len(active_trades) < 5:
                        print(f"DEBUG {symbol}: Entry={pos['entry']:.6f}, Exit={exit_price:.6f}, Change={price_change:.6f}")
                        print(f"DEBUG {symbol}: Qty={pos['qty']}, Contract type={contract_info['type']}")
                    
                    # Apply multiplier for futures
                    if contract_info['type'] == 'futures':
                        pnl = price_change * pos['qty'] * contract_info['multiplier']
                    elif contract_info['type'] == 'forex':
                        # For forex: PnL = price_change_in_pips * pip_value * lots
                        pip_value = contract_info['pip_value']
                        if 'JPY' in symbol:
                            pips = price_change * 100  # 2 decimal places
                        else:
                            pips = price_change * 10000  # 4 decimal places
                        lots = pos['qty'] / 100000  # Convert units back to lots
                        pnl = pips * pip_value * lots
                        if symbol in ['EURUSD', 'GBPUSD'] and len(active_trades) < 5:
                            print(f"DEBUG {symbol}: Pips={pips:.2f}, Lots={lots:.2f}, PipValue={pip_value}, PnL={pnl:.2f}")
                    elif contract_info['type'] == 'crypto':
                        # For crypto: direct price change * quantity
                        pnl = price_change * pos['qty']
                    else:
                        pnl = price_change * pos['qty']
                    
                    balance += pnl
                    
                    trade_record = {
                        'symbol': symbol,
                        'direction': 'LONG' if pos['direction'] == 1 else 'SHORT',
                        'entry_price': round(pos['entry'], 4),
                        'exit_price': round(exit_price, 4),
                        'qty': pos['qty'],
                        'entry_time': str(pos['entry_time']),
                        'exit_time': str(timestamp),
                        'bars_held': idx - pos['entry_idx'],
                        'pnl': round(pnl, 2),
                        'result': 'WIN' if pnl > 0 else 'LOSS',
                        'exit_reason': exit_reason,
                        'confluence': pos['confluence'],
                        'balance_after': round(balance, 2)
                    }
                    active_trades.append(trade_record)
                    del positions[symbol]
            
            # Check for new entry signal (only if no position exists)
            elif symbol not in positions:
                htf = row['htf_trend']
                ltf = row['ltf_trend']
                kz = row['kill_zone']
                pp = row['price_position']
                
                bullish_fvgs = df.attrs['bullish_fvgs']
                bearish_fvgs = df.attrs['bearish_fvgs']
                
                near_bull_fvg = next((f for f in reversed(bullish_fvgs) 
                                     if f['idx'] < idx and f['mid'] < current_price < f['high']), None)
                near_bear_fvg = next((f for f in reversed(bearish_fvgs) 
                                     if f['idx'] < idx and f['low'] < current_price < f['mid']), None)
                
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
                    
                    stop_distance = abs(current_price - stop)
                    if stop_distance > 0:
                        # FIXED: Use FIXED dollar risk, not percentage of growing balance
                        # This is more realistic for live trading and prevents exponential position growth
                        fixed_risk_amount = initial_capital * risk_per_trade  # $1,000 for $50K account
                        qty, risk_per_unit = calculate_position_size(
                            symbol, initial_capital, risk_per_trade, stop_distance, current_price
                        )
                        
                        if qty > 0:
                            positions[symbol] = {
                                'entry': current_price,
                                'stop': stop,
                                'target': target,
                                'direction': direction,
                                'qty': qty,
                                'entry_idx': idx,
                                'entry_time': timestamp,
                                'confluence': confluence
                            }
    
    # Calculate statistics
    total_trades = len(active_trades)
    wins = len([t for t in active_trades if t['result'] == 'WIN'])
    losses = total_trades - wins
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    total_pnl = balance - initial_capital
    total_return_pct = (total_pnl / initial_capital) * 100
    
    # Calculate max drawdown
    peak = initial_capital
    max_dd = 0
    running_balance = initial_capital
    
    for trade in active_trades:
        running_balance += trade['pnl']
        if running_balance > peak:
            peak = running_balance
        dd = (peak - running_balance) / peak * 100
        if dd > max_dd:
            max_dd = dd
    
    # Symbol breakdown
    symbol_stats = {}
    for symbol in all_data.keys():
        symbol_trades = [t for t in active_trades if t['symbol'] == symbol]
        symbol_wins = len([t for t in symbol_trades if t['result'] == 'WIN'])
        symbol_stats[symbol] = {
            'trades': len(symbol_trades),
            'wins': symbol_wins,
            'losses': len(symbol_trades) - symbol_wins,
            'win_rate': (symbol_wins / len(symbol_trades) * 100) if symbol_trades else 0,
            'pnl': sum(t['pnl'] for t in symbol_trades),
            'avg_pnl': (sum(t['pnl'] for t in symbol_trades) / len(symbol_trades)) if symbol_trades else 0
        }
    
    # Profit factor
    gross_profit = sum(t['pnl'] for t in active_trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in active_trades if t['pnl'] < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    results = {
        'backtest_config': {
            'symbols': list(all_data.keys()),
            'days': days,
            'initial_capital': initial_capital,
            'risk_per_trade': risk_per_trade,
            'data_source': 'IBKR' if use_ibkr else 'Yahoo',
            'timestamp': datetime.now().isoformat()
        },
        'summary': {
            'initial_capital': initial_capital,
            'final_capital': round(balance, 2),
            'total_pnl': round(total_pnl, 2),
            'total_return_pct': round(total_return_pct, 2),
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': round(win_rate, 2),
            'max_drawdown_pct': round(max_dd, 2),
            'profit_factor': round(profit_factor, 2),
            'avg_trade_pnl': round(total_pnl / total_trades, 2) if total_trades > 0 else 0
        },
        'symbol_stats': symbol_stats,
        'trades': active_trades,
        'daily_balances': daily_balances
    }
    
    return results


# Run the backtest
if __name__ == "__main__":
    # Backtest symbols
    symbols = ['SOLUSD', 'LINKUSD', 'LTCUSD', 'UNIUSD', 'BTCUSD', 'ETHUSD', 'SI', 'NQ', 'ES', 'GC']
    
    print("="*80)
    print("ICT V5 - Live Trading Simulation Backtest")
    print("="*80)
    
    results = run_live_simulation_backtest(
        symbols=symbols,
        days=30,
        initial_capital=50000,
        risk_per_trade=0.02,
        use_ibkr=True
    )
    
    if results:
        # Save to JSON
        output_file = 'v5_live_simulation_results.json'
        try:
            with open(output_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"\nResults saved to: {output_file}")
        except PermissionError:
            print(f"\nWarning: Could not save to {output_file} (permission denied)")
            print("Results displayed above.")
        
        # Print summary
        print(f"\n{'='*80}")
        print("BACKTEST RESULTS")
        print(f"{'='*80}")
        print(f"Initial Capital: ${results['summary']['initial_capital']:,}")
        print(f"Final Capital: ${results['summary']['final_capital']:,}")
        print(f"Total PnL: ${results['summary']['total_pnl']:,.2f}")
        print(f"Total Return: {results['summary']['total_return_pct']:.2f}%")
        print(f"Max Drawdown: {results['summary']['max_drawdown_pct']:.2f}%")
        print(f"Profit Factor: {results['summary']['profit_factor']:.2f}")
        print(f"\nTotal Trades: {results['summary']['total_trades']}")
        print(f"Win Rate: {results['summary']['win_rate']:.1f}%")
        print(f"Wins: {results['summary']['wins']} | Losses: {results['summary']['losses']}")
        print(f"Avg Trade: ${results['summary']['avg_trade_pnl']:.2f}")
        print(f"\nResults saved to: {output_file}")
        print(f"{'='*80}")
        
        # Symbol breakdown
        print("\nSymbol Performance:")
        print(f"{'Symbol':<10} {'Trades':<8} {'Wins':<6} {'Losses':<8} {'Win%':<8} {'PnL':<15}")
        print("-" * 60)
        for symbol, stats in sorted(results['symbol_stats'].items(), key=lambda x: x[1]['pnl'], reverse=True):
            print(f"{symbol:<10} {stats['trades']:<8} {stats['wins']:<6} {stats['losses']:<8} {stats['win_rate']:<8.1f} ${stats['pnl']:<14.2f}")
        
        # Recent trades
        if results['trades']:
            print("\n\nRecent 15 Trades:")
            print(f"{'#':<4} {'Time':<20} {'Symbol':<10} {'Dir':<6} {'Entry':<12} {'Exit':<12} {'PnL':<10} {'Result':<6}")
            print("-" * 90)
            for i, trade in enumerate(results['trades'][-15:], 1):
                exit_time = trade['exit_time'][:16] if len(trade['exit_time']) > 16 else trade['exit_time']
                print(f"{i:<4} {exit_time:<20} {trade['symbol']:<10} {trade['direction']:<6} "
                      f"${trade['entry_price']:<11.4f} ${trade['exit_price']:<11.4f} "
                      f"${trade['pnl']:<9.2f} {trade['result']:<6}")
        
        # Print JSON summary
        print("\n\nSUMMARY JSON:")
        summary = {
            'config': results['backtest_config'],
            'summary': results['summary'],
            'symbol_stats': results['symbol_stats']
        }
        print(json.dumps(summary, indent=2))
