"""
V6 vs V8 3-Month Backtest Comparison
=====================================
Compare the performance of V6 (FVG+Gap) vs V8 (RL Agent) over 3 months.
"""

import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import sys
sys.path.insert(0, '/Users/mac/Documents/Algot')

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import os

# Import V5 base functions
import importlib.util
spec = importlib.util.spec_from_file_location("ict_v5", "/Users/mac/Documents/Algot/ict_v5_ibkr.py")
ict_v5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ict_v5)

fetch_ibkr_data = ict_v5.fetch_ibkr_data
prepare_data_ibkr = ict_v5.prepare_data_ibkr
get_signal = ict_v5.get_signal
calculate_position_size = ict_v5.calculate_position_size
get_contract_info = ict_v5.get_contract_info

# Import V6 signal generator
from fvg_handler import FVGHandler
from gap_handler import GapHandler

# Import V8 signal generator and RL
from v8_backtest import V8SignalGenerator


class V6SignalGenerator:
    """V6 Signal Generator for backtesting"""
    
    def __init__(self):
        self.fvg_handler = FVGHandler(
            sensitivity=0.0001,
            min_gap_size=0.0,
            track_body_respect=False,
            detect_volume_imbalances=False,
            detect_suspension_blocks=False
        )
    
    def generate_signal(self, data, idx):
        """Generate V6 signal with FVG analysis"""
        v5_signal = get_signal(data, idx)
        if not v5_signal:
            return None
        
        df = pd.DataFrame({
            'open': data['opens'][:idx+1],
            'high': data['highs'][:idx+1],
            'low': data['lows'][:idx+1],
            'close': data['closes'][:idx+1]
        })
        
        fvgs = self.fvg_handler.detect_all_fvgs(df)
        
        confluence_boost = 0
        current_price = data['closes'][idx]
        
        for fvg in fvgs:
            if fvg.status.value != 'active':
                continue
            
            distance = abs(current_price - fvg.consequent_encroachment)
            if distance < fvg.size * 2:
                if v5_signal['direction'] == 1 and fvg.gap_type in ['bullish', 'bisi']:
                    confluence_boost += 15
                    if fvg.is_high_probability:
                        confluence_boost += 10
                elif v5_signal['direction'] == -1 and fvg.gap_type in ['bearish', 'sibi']:
                    confluence_boost += 15
                    if fvg.is_high_probability:
                        confluence_boost += 10
        
        total_confluence = min(v5_signal['confluence'] + confluence_boost, 100)
        
        signal = v5_signal.copy()
        signal['confluence'] = total_confluence
        return signal


def run_backtest(signal_gen, all_data, initial_capital=100000, risk_per_trade=0.02, rr_ratio=2.0, name=""):
    """Run backtest for a signal generator"""
    
    balance = initial_capital
    positions = {}
    trades = []
    
    all_timestamps = sorted(set().union(*[set(df.index) for df in all_data.values()]))
    
    for i, timestamp in enumerate(all_timestamps):
        for symbol, df in all_data.items():
            if timestamp not in df.index:
                continue
            
            idx = df.index.get_loc(timestamp)
            if idx < 50 or idx >= len(df) - 1:
                continue
            
            current_price = df.iloc[idx]['close']
            
            # Check position exits
            if symbol in positions:
                pos = positions[symbol]
                next_bar = df.iloc[idx + 1]
                next_low = next_bar['low']
                next_high = next_bar['high']
                
                exit_price = None
                exit_reason = None
                
                if pos['direction'] == 1:
                    if next_low <= pos['stop']:
                        exit_price = pos['stop']
                        exit_reason = 'stop'
                    elif next_high >= pos['target']:
                        exit_price = pos['target']
                        exit_reason = 'target'
                else:
                    if next_high >= pos['stop']:
                        exit_price = pos['stop']
                        exit_reason = 'stop'
                    elif next_low <= pos['target']:
                        exit_price = pos['target']
                        exit_reason = 'target'
                
                if exit_price:
                    contract_info = get_contract_info(symbol)
                    
                    if pos['direction'] == 1:
                        price_change = exit_price - pos['entry']
                    else:
                        price_change = pos['entry'] - exit_price
                    
                    if contract_info['type'] == 'futures':
                        pnl = price_change * pos['qty'] * contract_info['multiplier']
                    else:
                        pnl = price_change * pos['qty']
                    
                    balance += pnl
                    
                    trades.append({
                        'symbol': symbol,
                        'direction': 'LONG' if pos['direction'] == 1 else 'SHORT',
                        'entry': pos['entry'],
                        'exit': exit_price,
                        'pnl': pnl,
                        'confluence': pos.get('confluence', 0),
                        'exit_reason': exit_reason
                    })
                    del positions[symbol]
            
            # Check for entries
            elif symbol not in positions:
                data = {
                    'opens': df['open'].values,
                    'highs': df['high'].values,
                    'lows': df['low'].values,
                    'closes': df['close'].values,
                    'htf_trend': df['htf_trend'].values,
                    'ltf_trend': df['ltf_trend'].values,
                    'kill_zone': df['kill_zone'].values,
                    'price_position': df['price_position'].values,
                    'bullish_fvgs': df.attrs.get('bullish_fvgs', []),
                    'bearish_fvgs': df.attrs.get('bearish_fvgs', [])
                }
                
                signal = signal_gen.generate_signal(data, idx)
                
                if signal and signal['confluence'] >= 60:
                    if signal['direction'] == 1:
                        stop = data['lows'][idx]
                        target = current_price + (current_price - stop) * rr_ratio
                    else:
                        stop = data['highs'][idx]
                        target = current_price - (stop - current_price) * rr_ratio
                    
                    stop_distance = abs(current_price - stop)
                    if stop_distance > 0:
                        qty, _ = calculate_position_size(
                            symbol, balance, risk_per_trade, stop_distance, current_price
                        )
                        
                        if qty > 0:
                            positions[symbol] = {
                                'entry': current_price,
                                'stop': stop,
                                'target': target,
                                'direction': signal['direction'],
                                'qty': qty,
                                'confluence': signal.get('confluence', 0)
                            }
    
    # Calculate stats
    total_trades = len(trades)
    wins = len([t for t in trades if t['pnl'] > 0])
    losses = total_trades - wins
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    gross_profit = sum(t['pnl'] for t in trades if t['pnl'] > 0)
    gross_loss = abs(sum(t['pnl'] for t in trades if t['pnl'] <= 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    net_pnl = balance - initial_capital
    total_return = (net_pnl / initial_capital) * 100
    
    return {
        'name': name,
        'initial': initial_capital,
        'final': round(balance, 2),
        'net_pnl': round(net_pnl, 2),
        'return_pct': round(total_return, 2),
        'trades': total_trades,
        'wins': wins,
        'losses': losses,
        'win_rate': round(win_rate, 2),
        'profit_factor': round(profit_factor, 2),
        'gross_profit': round(gross_profit, 2),
        'gross_loss': round(gross_loss, 2),
        'avg_win': round(gross_profit / wins, 2) if wins > 0 else 0,
        'avg_loss': round(gross_loss / losses, 2) if losses > 0 else 0,
        'trade_list': trades
    }


def main():
    # Configuration
    symbols = ['SOLUSD', 'LINKUSD', 'LTCUSD', 'UNIUSD', 'BTCUSD', 'ETHUSD', 'SI', 'NQ', 'ES', 'GC']
    initial_capital = 5000
    risk_per_trade = 0.02
    
    print("=" * 80)
    print("V6 vs V8 - 3 MONTH BACKTEST COMPARISON")
    print("=" * 80)
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Initial Capital: ${initial_capital:,}")
    print(f"Risk per Trade: {risk_per_trade*100}%")
    print("=" * 80)
    
    # Load data for all symbols (3 months = ~2160 hourly barsß)
    print("\nLoading data...")
    all_data = {}
    
    for symbol in symbols:
        print(f"  Loading {symbol}...", end=' ')
        try:
            data = prepare_data_ibkr(symbol)
            if data and len(data.get('closes', [])) >= 50:
                df = pd.DataFrame({
                    'close': data['closes'],
                    'high': data['highs'],
                    'low': data['lows'],
                    'open': data['opens'],
                    'htf_trend': data['htf_trend'],
                    'ltf_trend': data['ltf_trend'],
                    'kill_zone': data['kill_zone'],
                    'price_position': data['price_position'],
                }, index=data['df'].index)
                
                df.attrs['bullish_fvgs'] = data.get('bullish_fvgs', [])
                df.attrs['bearish_fvgs'] = data.get('bearish_fvgs', [])
                
                # Filter to last 3 months (~2160 hours)
                if len(df) > 2160:
                    df = df.iloc[-2160:]
                
                all_data[symbol] = df
                print(f"OK ({len(df)} bars)")
            else:
                print("FAILED - no data")
        except Exception as e:
            print(f"ERROR: {e}")
    
    if not all_data:
        print("No data loaded!")
        return
    
    # Run V6 backtest
    print("\n" + "=" * 80)
    print("Running V6 Backtest (FVG + Gap Analysis)...")
    print("=" * 80)
    
    v6_gen = V6SignalGenerator()
    v6_results = run_backtest(
        v6_gen, all_data, 
        initial_capital=initial_capital,
        risk_per_trade=risk_per_trade,
        rr_ratio=2.0,
        name="V6 (FVG+Gap)"
    )
    
    print(f"  Trades: {v6_results['trades']}")
    print(f"  Win Rate: {v6_results['win_rate']}%")
    print(f"  Return: {v6_results['return_pct']}%")
    
    # Run V8 backtest
    print("\n" + "=" * 80)
    print("Running V8 Backtest (RL Agent)...")
    print("=" * 80)
    
    # Try to load RL model
    v8_gen = V8SignalGenerator(use_rl=True)
    rl_model_path = '/Users/mac/Documents/Algot/v8_rl_model.pkl'
    if os.path.exists(rl_model_path):
        try:
            with open(rl_model_path, 'rb') as f:
                model_data = pickle.load(f)
            v8_gen.rl_agent = model_data['agent']
            print("  RL model loaded successfully")
        except Exception as e:
            print(f"  Warning: Could not load RL model: {e}")
    
    v8_results = run_backtest(
        v8_gen, all_data,
        initial_capital=initial_capital,
        risk_per_trade=risk_per_trade,
        rr_ratio=4.0,  # V8 uses 1:4 RR
        name="V8 (RL Agent)"
    )
    
    print(f"  Trades: {v8_results['trades']}")
    print(f"  Win Rate: {v8_results['win_rate']}%")
    print(f"  Return: {v8_results['return_pct']}%")
    
    # Print comparison
    print("\n" + "=" * 80)
    print("COMPARISON RESULTS (3 MONTHS)")
    print("=" * 80)
    print()
    print(f"{'Metric':<25} {'V6 (FVG+Gap)':>20} {'V8 (RL)':>20}")
    print("-" * 65)
    print(f"{'Initial Capital':<25} ${v6_results['initial']:>18,} ${v8_results['initial']:>18,}")
    print(f"{'Final Capital':<25} ${v6_results['final']:>18,.0f} ${v8_results['final']:>18,.0f}")
    print(f"{'Net P&L':<25} ${v6_results['net_pnl']:>18,.0f} ${v8_results['net_pnl']:>18,.0f}")
    print(f"{'Return %':<25} {v6_results['return_pct']:>19.1f}% {v8_results['return_pct']:>19.1f}%")
    print("-" * 65)
    print(f"{'Total Trades':<25} {v6_results['trades']:>20} {v8_results['trades']:>20}")
    print(f"{'Wins':<25} {v6_results['wins']:>20} {v8_results['wins']:>20}")
    print(f"{'Losses':<25} {v6_results['losses']:>20} {v8_results['losses']:>20}")
    print(f"{'Win Rate':<25} {v6_results['win_rate']:>19.1f}% {v8_results['win_rate']:>19.1f}%")
    print("-" * 65)
    print(f"{'Profit Factor':<25} {v6_results['profit_factor']:>20.2f} {v8_results['profit_factor']:>20.2f}")
    print(f"{'Avg Win':<25} ${v6_results['avg_win']:>18,.0f} ${v8_results['avg_win']:>18,.0f}")
    print(f"{'Avg Loss':<25} ${v6_results['avg_loss']:>18,.0f} ${v8_results['avg_loss']:>18,.0f}")
    
    # Determine winner
    print("\n" + "=" * 80)
    if v8_results['return_pct'] > v6_results['return_pct']:
        winner = "V8 (RL Agent)"
        diff = v8_results['return_pct'] - v6_results['return_pct']
    else:
        winner = "V6 (FVG+Gap)"
        diff = v6_results['return_pct'] - v8_results['return_pct']
    
    print(f"WINNER: {winner} (by {diff:.1f}% return)")
    print("=" * 80)
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'period': '3 months',
        'symbols': list(all_data.keys()),
        'initial_capital': initial_capital,
        'risk_per_trade': risk_per_trade,
        'v6': {k: v for k, v in v6_results.items() if k != 'trade_list'},
        'v8': {k: v for k, v in v8_results.items() if k != 'trade_list'},
        'winner': winner
    }
    
    with open('/Users/mac/Documents/Algot/v6_v8_3month_comparison.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to v6_v8_3month_comparison.json")


if __name__ == "__main__":
    main()
