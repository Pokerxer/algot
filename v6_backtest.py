"""
V6 Backtest with FVG + Gap Analysis
=====================================
Backtest the combined V5 + FVG + Gap strategy
"""

import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import sys
sys.path.insert(0, '/Users/mac/Documents/Algot')

import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List

# Import V5 base functions
import importlib.util
spec = importlib.util.spec_from_file_location("ict_v5", "/Users/mac/Documents/Algot/ict_v5_ibkr.py")
ict_v5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ict_v5)

# Import handlers
from fvg_handler import FVGHandler
from gap_handler import GapHandler

# Get V5 functions
fetch_ibkr_data = ict_v5.fetch_ibkr_data
prepare_data_ibkr = ict_v5.prepare_data_ibkr
prepare_data = ict_v5.prepare_data
get_signal = ict_v5.get_signal
calculate_position_size = ict_v5.calculate_position_size
get_contract_info = ict_v5.get_contract_info


class V6SignalGenerator:
    """V6 Signal Generator for backtesting - Optimized"""
    
    def __init__(self):
        self.fvg_handler = FVGHandler(
            sensitivity=0.0001,
            min_gap_size=0.0,
            track_body_respect=False,  # Disable for speed
            detect_volume_imbalances=False,  # Disable for speed
            detect_suspension_blocks=False  # Disable for speed
        )
        self.gap_handler = GapHandler(
            large_gap_pips_forex=40.0,
            large_gap_points_indices=50.0,
            keep_gaps_days=3
        )
        # Cache FVGs to avoid recomputing
        self.fvg_cache = {}
    
    def generate_signal(self, data: Dict, idx: int) -> Dict:
        """Generate V6 signal with FVG + Gap analysis"""
        # Get base V5 signal
        v5_signal = get_signal(data, idx)
        
        if not v5_signal:
            return None
        
        # Prepare DataFrame for analysis
        df = pd.DataFrame({
            'open': data['opens'][:idx+1],
            'high': data['highs'][:idx+1],
            'low': data['lows'][:idx+1],
            'close': data['closes'][:idx+1]
        })
        
        # FVG Analysis
        fvgs = self.fvg_handler.detect_all_fvgs(df)
        
        # Calculate confluence boost
        confluence_boost = 0
        fvg_info = None
        
        current_price = data['closes'][idx]
        
        # Check for relevant FVG
        for fvg in fvgs:
            if fvg.status.value != 'active':
                continue
            
            # Check if price is near FVG
            distance = abs(current_price - fvg.consequent_encroachment)
            if distance < fvg.size * 2:  # Within 2x FVG size
                # Direction alignment
                if v5_signal['direction'] == 1 and fvg.gap_type in ['bullish', 'bisi']:
                    confluence_boost += 15
                    fvg_info = f"BISI@{fvg.consequent_encroachment:.2f}"
                    if fvg.is_high_probability:
                        confluence_boost += 10
                elif v5_signal['direction'] == -1 and fvg.gap_type in ['bearish', 'sibi']:
                    confluence_boost += 15
                    fvg_info = f"SIBI@{fvg.consequent_encroachment:.2f}"
                    if fvg.is_high_probability:
                        confluence_boost += 10
        
        # Combine confluence
        total_confluence = min(v5_signal['confluence'] + confluence_boost, 100)
        
        # Create enhanced signal
        signal = v5_signal.copy()
        signal['confluence'] = total_confluence
        signal['v5_confluence'] = v5_signal['confluence']
        signal['fvg_boost'] = confluence_boost
        signal['fvg_info'] = fvg_info
        
        return signal


def run_v6_backtest_symbol(symbol, df, initial_capital, risk_per_trade):
    """Run V6 backtest for a single symbol"""
    
    signal_gen = V6SignalGenerator()
    balance = initial_capital
    position = None
    trades = []
    signal_check_interval = 5
    
    for idx in range(50, len(df) - 1):
        current_price = df.iloc[idx]['close']
        
        if position is not None:
            next_bar = df.iloc[idx + 1]
            next_low = next_bar['low']
            next_high = next_bar['high']
            
            exit_price = None
            if position['direction'] == 1:
                if next_low <= position['stop']:
                    exit_price = position['stop']
                elif next_high >= position['target']:
                    exit_price = position['target']
            else:
                if next_high >= position['stop']:
                    exit_price = position['stop']
                elif next_low <= position['target']:
                    exit_price = position['target']
            
            if exit_price:
                contract_info = get_contract_info(symbol)
                
                if position['direction'] == 1:
                    price_change = exit_price - position['entry']
                else:
                    price_change = position['entry'] - exit_price
                
                if contract_info['type'] == 'futures':
                    pnl = price_change * position['qty'] * contract_info['multiplier']
                else:
                    pnl = price_change * position['qty']
                
                balance += pnl
                
                trades.append({
                    'symbol': symbol,
                    'direction': 'LONG' if position['direction'] == 1 else 'SHORT',
                    'entry_price': position['entry'],
                    'exit_price': exit_price,
                    'qty': position['qty'],
                    'pnl': pnl,
                    'v5_confluence': position.get('v5_confluence', 0),
                    'fvg_boost': position.get('fvg_boost', 0)
                })
                position = None
        
        if position is None and idx % signal_check_interval == 0:
            data = {
                'opens': df['open'].values,
                'highs': df['high'].values,
                'lows': df['low'].values,
                'closes': df['close'].values,
                'htf_trend': df['htf_trend'].values,
                'ltf_trend': df['ltf_trend'].values,
                'kill_zone': df['kill_zone'].values,
                'price_position': df['price_position'].values,
                'bullish_fvgs': df.attrs['bullish_fvgs'],
                'bearish_fvgs': df.attrs['bearish_fvgs']
            }
            
            signal = signal_gen.generate_signal(data, idx)
            
            if signal and signal['confluence'] >= 60:
                if signal['direction'] == 1:
                    stop = data['lows'][idx]
                    target = current_price + (current_price - stop) * 2
                else:
                    stop = data['highs'][idx]
                    target = current_price - (stop - current_price) * 2
                
                stop_distance = abs(current_price - stop)
                if stop_distance > 0:
                    qty, _ = calculate_position_size(
                        symbol, initial_capital, risk_per_trade, stop_distance, current_price
                    )
                    
                    if qty > 0:
                        position = {
                            'entry': current_price,
                            'stop': stop,
                            'target': target,
                            'direction': signal['direction'],
                            'qty': qty,
                            'v5_confluence': signal.get('v5_confluence', 0),
                            'fvg_boost': signal.get('fvg_boost', 0),
                            'fvg_info': signal.get('fvg_info', '')
                        }
    
    return {
        'balance': balance,
        'trades': trades,
        'initial_balance': initial_capital
    }


def run_v6_backtest(symbols, days=30, initial_capital=5000, risk_per_trade=0.02, use_ibkr=True):
    """Run V6 backtest with FVG + Gap analysis"""
    
    print(f"\n{'='*80}")
    print(f"ICT V6 Backtest - FVG + Gap Analysis")
    print(f"Initial Capital: ${initial_capital:,}")
    print(f"Risk per Trade: {risk_per_trade*100}%")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"{'='*80}\n")
    
    per_symbol_capital = initial_capital / len(symbols)
    
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
                'open': data['opens'],
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
    
    print(f"\nProcessing {len(all_data)} symbols...")
    
    total_trades = []
    total_pnl = 0
    symbol_stats = {}
    
    for symbol, df in all_data.items():
        print(f"  Backtesting {symbol}...", end=' ')
        
        result = run_v6_backtest_symbol(symbol, df, per_symbol_capital, risk_per_trade)
        
        symbol_pnl = result['balance'] - result['initial_balance']
        total_pnl += symbol_pnl
        total_trades.extend(result['trades'])
        
        symbol_wins = len([t for t in result['trades'] if t['pnl'] > 0])
        symbol_stats[symbol] = {
            'trades': len(result['trades']),
            'wins': symbol_wins,
            'losses': len(result['trades']) - symbol_wins,
            'win_rate': (symbol_wins / len(result['trades']) * 100) if result['trades'] else 0,
            'pnl': symbol_pnl
        }
        print(f"{len(result['trades'])} trades, ${symbol_pnl:.2f}")
    
    balance = initial_capital + total_pnl
    
    all_trades = total_trades
    wins = len([t for t in all_trades if t['pnl'] > 0])
    losses = len(all_trades) - wins
    win_rate = (wins / len(all_trades) * 100) if all_trades else 0
    final_pnl = balance - initial_capital
    total_return_pct = (final_pnl / initial_capital) * 100
    
    results = {
        'backtest_config': {
            'symbols': list(all_data.keys()),
            'days': days,
            'initial_capital': initial_capital,
            'risk_per_trade': risk_per_trade,
            'data_source': 'IBKR' if use_ibkr else 'Yahoo',
            'timestamp': datetime.now().isoformat(),
            'version': 'V6'
        },
        'summary': {
            'initial_capital': initial_capital,
            'final_capital': round(balance, 2),
            'total_pnl': round(final_pnl, 2),
            'total_return_pct': round(total_return_pct, 2),
            'total_trades': len(all_trades),
            'wins': wins,
            'losses': losses,
            'win_rate': round(win_rate, 2),
            'avg_trade_pnl': round(final_pnl / len(all_trades), 2) if all_trades else 0
        },
        'symbol_stats': symbol_stats,
        'trades': all_trades
    }
    
    return results


if __name__ == "__main__":
    # Use fewer symbols for faster backtest
    symbols = ['SOLUSD','ETHUSD','BTCUSD','LINKUSD','LTCUSD','SI','UNIUSD','NG','NQ','GC','CL','ES']
    
    print("="*80)
    print("ICT V6 - FVG + Gap Backtest")
    print("="*80)
    
    results = run_v6_backtest(
        symbols=symbols,
        days=6,
        initial_capital=5000,
        risk_per_trade=0.02,
        use_ibkr=True
    )
    
    if results:
        print(f"\n{'='*80}")
        print("V6 BACKTEST RESULTS")
        print(f"{'='*80}")
        print(f"Initial Capital: ${results['summary']['initial_capital']:,}")
        print(f"Final Capital: ${results['summary']['final_capital']:,}")
        print(f"Total PnL: ${results['summary']['total_pnl']:,.2f}")
        print(f"Total Return: {results['summary']['total_return_pct']:.2f}%")
        print(f"\nTotal Trades: {results['summary']['total_trades']}")
        print(f"Win Rate: {results['summary']['win_rate']:.1f}%")
        print(f"Wins: {results['summary']['wins']} | Losses: {results['summary']['losses']}")
        print(f"Avg Trade: ${results['summary']['avg_trade_pnl']:.2f}")

        print(f"\n{'='*80}")
        
        print("\nSymbol Performance:")
        for symbol, stats in sorted(results['symbol_stats'].items(), key=lambda x: x[1]['pnl'], reverse=True):
            print(f"  {symbol}: {stats['trades']} trades, {stats['win_rate']:.1f}% WR, ${stats['pnl']:,.2f}")
