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
    """V6 Signal Generator for backtesting"""
    
    def __init__(self):
        self.fvg_handler = FVGHandler(
            sensitivity=0.0001,
            min_gap_size=0.0,
            track_body_respect=True,
            detect_volume_imbalances=True,
            detect_suspension_blocks=True
        )
        self.gap_handler = GapHandler(
            large_gap_pips_forex=40.0,
            large_gap_points_indices=50.0,
            keep_gaps_days=3
        )
    
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


def run_v6_backtest(symbols, days=30, initial_capital=50000, risk_per_trade=0.02, use_ibkr=True):
    """Run V6 backtest with FVG + Gap analysis"""
    
    print(f"\n{'='*80}")
    print(f"ICT V6 Backtest - FVG + Gap Analysis")
    print(f"Initial Capital: ${initial_capital:,}")
    print(f"Risk per Trade: {risk_per_trade*100}%")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"{'='*80}\n")
    
    # Initialize signal generator
    signal_gen = V6SignalGenerator()
    
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
    
    # Combine all timestamps
    all_timestamps = sorted(set().union(*[set(df.index) for df in all_data.values()]))
    print(f"\nProcessing {len(all_timestamps)} timestamps across {len(all_data)} symbols...")
    
    # Portfolio state
    balance = initial_capital
    positions = {}
    active_trades = []
    
    for i, timestamp in enumerate(all_timestamps):
        if i % 1000 == 0 and i > 0:
            print(f"  [{i}/{len(all_timestamps)}] Balance: ${balance:,.2f}")
        
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
                if pos['direction'] == 1:
                    if next_low <= pos['stop']:
                        exit_price = pos['stop']
                    elif next_high >= pos['target']:
                        exit_price = pos['target']
                else:
                    if next_high >= pos['stop']:
                        exit_price = pos['stop']
                    elif next_low <= pos['target']:
                        exit_price = pos['target']
                
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
                    
                    active_trades.append({
                        'symbol': symbol,
                        'direction': 'LONG' if pos['direction'] == 1 else 'SHORT',
                        'entry_price': pos['entry'],
                        'exit_price': exit_price,
                        'qty': pos['qty'],
                        'pnl': pnl,
                        'v5_confluence': pos.get('v5_confluence', 0),
                        'fvg_boost': pos.get('fvg_boost', 0)
                    })
                    del positions[symbol]
            
            # Check for entries
            elif symbol not in positions:
                # Prepare data for V6 signal
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
                            positions[symbol] = {
                                'entry': current_price,
                                'stop': stop,
                                'target': target,
                                'direction': signal['direction'],
                                'qty': qty,
                                'v5_confluence': signal.get('v5_confluence', 0),
                                'fvg_boost': signal.get('fvg_boost', 0),
                                'fvg_info': signal.get('fvg_info', '')
                            }
    
    # Calculate statistics
    total_trades = len(active_trades)
    wins = len([t for t in active_trades if t['pnl'] > 0])
    losses = total_trades - wins
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    total_pnl = balance - initial_capital
    total_return_pct = (total_pnl / initial_capital) * 100
    
    # Symbol stats
    symbol_stats = {}
    for symbol in all_data.keys():
        symbol_trades = [t for t in active_trades if t['symbol'] == symbol]
        symbol_wins = len([t for t in symbol_trades if t['pnl'] > 0])
        symbol_stats[symbol] = {
            'trades': len(symbol_trades),
            'wins': symbol_wins,
            'losses': len(symbol_trades) - symbol_wins,
            'win_rate': (symbol_wins / len(symbol_trades) * 100) if symbol_trades else 0,
            'pnl': sum(t['pnl'] for t in symbol_trades)
        }
    
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
            'total_pnl': round(total_pnl, 2),
            'total_return_pct': round(total_return_pct, 2),
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': round(win_rate, 2),
            'avg_trade_pnl': round(total_pnl / total_trades, 2) if total_trades > 0 else 0
        },
        'symbol_stats': symbol_stats,
        'trades': active_trades
    }
    
    return results


if __name__ == "__main__":
    symbols = ['SOLUSD', 'LINKUSD', 'LTCUSD', 'UNIUSD', 'BTCUSD', 'ETHUSD', 'SI', 'NQ', 'ES', 'GC']
    
    print("="*80)
    print("ICT V6 - FVG + Gap Backtest")
    print("="*80)
    
    results = run_v6_backtest(
        symbols=symbols,
        days=30,
        initial_capital=50000,
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
