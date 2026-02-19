"""
V5 EURUSD Only Backtest with Debug
===================================
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


print("Loading EURUSD data...")
data = prepare_data_ibkr('EURUSD')
print(f"Loaded {len(data['closes'])} bars")

initial_capital = 50000
risk_per_trade = 0.02
balance = initial_capital
positions = {}
trades = []

print("\nRunning backtest (first 10 trades only)...\n")

for idx in range(50, len(data['closes']) - 1):
    current_price = data['closes'][idx]
    
    # Check exits
    if 'EURUSD' in positions:
        pos = positions['EURUSD']
        next_low = data['lows'][idx + 1]
        next_high = data['highs'][idx + 1]
        
        exit_price = None
        if pos['direction'] == 1:  # Long
            if next_low <= pos['stop']:
                exit_price = pos['stop']
            elif next_high >= pos['target']:
                exit_price = pos['target']
        else:  # Short
            if next_high >= pos['stop']:
                exit_price = pos['stop']
            elif next_low <= pos['target']:
                exit_price = pos['target']
        
        if exit_price:
            # Calculate PnL
            contract_info = get_contract_multiplier('EURUSD')
            
            if pos['direction'] == 1:
                price_change = exit_price - pos['entry']
            else:
                price_change = pos['entry'] - exit_price
            
            # Debug output
            print(f"\n=== TRADE #{len(trades)+1} ===")
            print(f"Direction: {'LONG' if pos['direction'] == 1 else 'SHORT'}")
            print(f"Entry: {pos['entry']:.6f}")
            print(f"Exit: {exit_price:.6f}")
            print(f"Stop: {pos['stop']:.6f}")
            print(f"Target: {pos['target']:.6f}")
            print(f"Price Change: {price_change:.6f}")
            print(f"Quantity: {pos['qty']:,} units")
            
            # Forex PnL calculation
            pip_value = contract_info['pip_value']
            pips = price_change * 10000
            lots = pos['qty'] / 100000
            pnl = pips * pip_value * lots
            
            print(f"Pips: {pips:.2f}")
            print(f"Lots: {lots:.2f}")
            print(f"Pip Value: ${pip_value}")
            print(f"PnL: ${pnl:.2f}")
            
            balance += pnl
            trades.append({
                'direction': 'LONG' if pos['direction'] == 1 else 'SHORT',
                'entry': pos['entry'],
                'exit': exit_price,
                'pnl': pnl,
                'balance': balance
            })
            
            del positions['EURUSD']
            
            if len(trades) >= 10:
                break
    
    # Check entries
    elif 'EURUSD' not in positions:
        signal = get_signal(data, idx)
        
        if signal:
            # Calculate stop and position
            if signal['direction'] == 1:
                stop = data['lows'][idx]
                target = current_price + (current_price - stop) * 2
            else:
                stop = data['highs'][idx]
                target = current_price - (stop - current_price) * 2
            
            stop_distance = abs(current_price - stop)
            if stop_distance > 0:
                qty, risk_per = calculate_position_size('EURUSD', balance, risk_per_trade, stop_distance, current_price)
                
                if qty > 0:
                    positions['EURUSD'] = {
                        'entry': current_price,
                        'stop': stop,
                        'target': target,
                        'direction': signal['direction'],
                        'qty': qty,
                        'confluence': signal['confluence']
                    }
                    print(f"\n[ENTRY] {'LONG' if signal['direction'] == 1 else 'SHORT'} {qty:,} units @ {current_price:.6f}")
                    print(f"        Stop: {stop:.6f}, Target: {target:.6f}")

print(f"\n\n=== SUMMARY ===")
print(f"Initial: ${initial_capital:,.2f}")
print(f"Final: ${balance:,.2f}")
print(f"Trades: {len(trades)}")
print(f"Total PnL: ${balance - initial_capital:,.2f}")
