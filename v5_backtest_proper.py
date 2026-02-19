"""
V5 Backtest with Proper PnL Calculation - JSON Output
======================================================
"""

import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import sys
sys.path.insert(0, '/Users/mac/Documents/Algot')

import json
from datetime import datetime

# Import from ict_v5_ibkr
exec(open('/Users/mac/Documents/Algot/ict_v5_ibkr.py').read().split('def run_backtest')[0])

def run_backtest_json(symbols, days=180, risk_per_trade=0.02, use_ibkr=True):
    """Run backtest with proper PnL calculation and return JSON."""
    
    results = {
        'backtest_config': {
            'symbols': symbols,
            'days': days,
            'risk_per_trade': risk_per_trade,
            'data_source': 'IBKR' if use_ibkr else 'Yahoo',
            'timestamp': datetime.now().isoformat()
        },
        'symbols': {},
        'summary': {
            'total_trades': 0,
            'total_wins': 0,
            'total_losses': 0,
            'win_rate': 0,
            'total_pnl': 0,
            'total_return_pct': 0
        }
    }
    
    for symbol in symbols:
        print(f"\nTesting {symbol}...")
        
        if use_ibkr:
            data = prepare_data_ibkr(symbol)
        else:
            data = prepare_data(symbol)
        
        if data is None or len(data.get('closes', [])) < 50:
            print(f"  No data for {symbol}")
            continue
        
        # Initialize with $10,000 per symbol
        initial_balance = 10000
        balance = initial_balance
        position = None
        trades = []
        
        closes = data['closes']
        highs = data['highs']
        lows = data['lows']
        htf_trend = data['htf_trend']
        ltf_trend = data['ltf_trend']
        kill_zone = data['kill_zone']
        price_position = data['price_position']
        bullish_fvgs = data['bullish_fvgs']
        bearish_fvgs = data['bearish_fvgs']
        
        for idx in range(50, len(closes) - 1):
            htf = htf_trend[idx]
            ltf = ltf_trend[idx]
            kz = kill_zone[idx]
            pp = price_position[idx]
            current_price = closes[idx]
            
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
            if position is None and confluence >= 60:
                if htf == 1 and ltf >= 0:
                    direction = 1  # Long
                    stop = lows[idx]
                    target = current_price + (current_price - stop) * 2
                elif htf == -1 and ltf <= 0:
                    direction = -1  # Short
                    stop = highs[idx]
                    target = current_price - (stop - current_price) * 2
                else:
                    continue
                
                # Calculate position size based on risk
                stop_distance = abs(current_price - stop)
                if stop_distance > 0:
                    risk_amount = balance * risk_per_trade
                    # For futures/contracts, assume 1 contract = $1 per point
                    # For forex, standard lot = $10 per pip
                    if symbol in ['EURUSD', 'GBPUSD', 'USDJPY']:
                        contract_value = 100000  # Standard forex lot
                        pip_value = 10  # $10 per pip for standard lot
                        pips_risk = stop_distance * 10000  # Convert to pips
                        position_size = risk_amount / (pips_risk * pip_value)
                    else:
                        # For futures and crypto - simplified
                        position_size = risk_amount / stop_distance
                    
                    position = {
                        'entry_price': current_price,
                        'stop_loss': stop,
                        'take_profit': target,
                        'direction': direction,
                        'entry_idx': idx,
                        'confluence': confluence,
                        'position_size': position_size,
                        'entry_time': str(data['df'].index[idx])
                    }
            
            # Check exits
            elif position:
                next_low = lows[idx + 1]
                next_high = highs[idx + 1]
                
                exit_price = None
                exit_reason = None
                
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
                    # Calculate PnL
                    if position['direction'] == 1:
                        price_change = exit_price - position['entry_price']
                    else:
                        price_change = position['entry_price'] - exit_price
                    
                    pnl = price_change * position['position_size']
                    balance += pnl
                    
                    trade_record = {
                        'entry_price': round(position['entry_price'], 4),
                        'exit_price': round(exit_price, 4),
                        'direction': 'LONG' if position['direction'] == 1 else 'SHORT',
                        'position_size': round(position['position_size'], 4),
                        'pnl': round(pnl, 2),
                        'result': 'WIN' if pnl > 0 else 'LOSS',
                        'exit_reason': exit_reason,
                        'bars_held': idx - position['entry_idx'],
                        'confluence': position['confluence'],
                        'balance_after': round(balance, 2)
                    }
                    trades.append(trade_record)
                    position = None
        
        # Calculate stats
        wins = len([t for t in trades if t['result'] == 'WIN'])
        losses = len([t for t in trades if t['result'] == 'LOSS'])
        total_trades = len(trades)
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        total_pnl = sum(t['pnl'] for t in trades)
        return_pct = (total_pnl / initial_balance * 100) if initial_balance > 0 else 0
        
        results['symbols'][symbol] = {
            'initial_balance': initial_balance,
            'final_balance': round(balance, 2),
            'total_pnl': round(total_pnl, 2),
            'return_pct': round(return_pct, 2),
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': round(win_rate, 2),
            'trades': trades
        }
        
        print(f"  {symbol}: ${balance:,.2f} | {total_trades} trades | {win_rate:.1f}% win rate | PnL: ${total_pnl:,.2f}")
    
    # Calculate overall summary
    all_trades = []
    total_pnl_all = 0
    total_wins = 0
    total_losses = 0
    
    for symbol, data in results['symbols'].items():
        all_trades.extend(data['trades'])
        total_pnl_all += data['total_pnl']
        total_wins += data['wins']
        total_losses += data['losses']
    
    total_trades_all = len(all_trades)
    overall_win_rate = (total_wins / total_trades_all * 100) if total_trades_all > 0 else 0
    initial_capital = len(symbols) * 10000
    overall_return_pct = (total_pnl_all / initial_capital * 100) if initial_capital > 0 else 0
    
    results['summary'] = {
        'total_symbols': len(symbols),
        'symbols_tested': len(results['symbols']),
        'initial_capital': initial_capital,
        'final_capital': round(initial_capital + total_pnl_all, 2),
        'total_trades': total_trades_all,
        'total_wins': total_wins,
        'total_losses': total_losses,
        'win_rate': round(overall_win_rate, 2),
        'total_pnl': round(total_pnl_all, 2),
        'total_return_pct': round(overall_return_pct, 2)
    }
    
    return results


# Run the backtest
symbols = ['ES', 'NQ', 'GC', 'SI', 'CL', 'NG', 'YM', 'EURUSD', 'GBPUSD', 'USDJPY', 'BTCUSD', 'ETHUSD']
print("="*80)
print("ICT V5 Backtest - 6 Months (180 days)")
print("="*80)

results = run_backtest_json(symbols, days=180, risk_per_trade=0.02, use_ibkr=True)

# Save to JSON file
output_file = '/Users/mac/Documents/Algot/v5_backtest_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n{'='*80}")
print("BACKTEST SUMMARY")
print(f"{'='*80}")
print(f"Total Symbols: {results['summary']['symbols_tested']}")
print(f"Total Trades: {results['summary']['total_trades']}")
print(f"Win Rate: {results['summary']['win_rate']:.1f}%")
print(f"Total PnL: ${results['summary']['total_pnl']:,.2f}")
print(f"Total Return: {results['summary']['total_return_pct']:.1f}%")
print(f"\nResults saved to: {output_file}")
print(f"{'='*80}")

# Also print JSON to stdout
print("\nJSON OUTPUT:")
print(json.dumps(results, indent=2))
