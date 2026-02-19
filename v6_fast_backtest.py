"""
ICT V6 - Simplified with Core Concepts
======================================
Combines V5 + essential FVG + Liquidity + Order Block concepts
Optimized for faster backtesting
"""

import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import sys
sys.path.insert(0, '/Users/mac/Documents/Algot')

import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import importlib.util

# Import V5 base
spec = importlib.util.spec_from_file_location("ict_v5", "/Users/mac/Documents/Algot/ict_v5_ibkr.py")
ict_v5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ict_v5)

fetch_ibkr_data = ict_v5.fetch_ibkr_data
prepare_data_ibkr = ict_v5.prepare_data_ibkr
prepare_data = ict_v5.prepare_data
get_signal = ict_v5.get_signal
calculate_position_size = ict_v5.calculate_position_size
get_contract_info = ict_v5.get_contract_info


class V6SimplifiedSignal:
    """Fast V6 signal generator with core ICT concepts only"""
    
    def __init__(self, min_fvg_size=0.0):
        self.min_fvg_size = min_fvg_size
    
    def detect_fvgs_fast(self, highs, lows, closes, lookback=20):
        """Fast FVG detection - only last N bars"""
        fvgs = []
        n = len(closes)
        start = max(0, n - lookback)
        
        for i in range(max(2, start), n):
            # Bullish FVG
            if lows[i] > highs[i-2]:
                gap_size = lows[i] - highs[i-2]
                if gap_size >= self.min_fvg_size:
                    fvgs.append({
                        'type': 'bullish',
                        'high': lows[i],
                        'low': highs[i-2],
                        'ce': (lows[i] + highs[i-2]) / 2,
                        'idx': i
                    })
            
            # Bearish FVG
            if highs[i] < lows[i-2]:
                gap_size = lows[i-2] - highs[i]
                if gap_size >= self.min_fvg_size:
                    fvgs.append({
                        'type': 'bearish',
                        'high': lows[i-2],
                        'low': highs[i],
                        'ce': (lows[i-2] + highs[i]) / 2,
                        'idx': i
                    })
        
        return fvgs
    
    def find_liquidity_levels(self, highs, lows, window=5):
        """Find equal highs/lows for liquidity"""
        n = len(highs)
        if n < window * 2:
            return [], []
        
        buy_side = []
        sell_side = []
        tolerance = 0.001  # 0.1%
        
        for i in range(window, n - window):
            # Equal highs
            recent_highs = highs[i-window:i+window]
            max_high = np.max(recent_highs)
            touches = np.sum(np.abs(recent_highs - max_high) / max_high < tolerance)
            if touches >= 2:
                buy_side.append(max_high)
            
            # Equal lows
            recent_lows = lows[i-window:i+window]
            min_low = np.min(recent_lows)
            touches = np.sum(np.abs(recent_lows - min_low) / min_low < tolerance)
            if touches >= 2:
                sell_side.append(min_low)
        
        return list(set(buy_side))[:3], list(set(sell_side))[:3]  # Top 3 only
    
    def find_order_blocks(self, opens, highs, lows, closes, lookback=10):
        """Find simple order blocks"""
        obs = []
        n = len(closes)
        start = max(1, n - lookback)
        
        for i in range(start, n - 1):
            prev_close = closes[i-1]
            curr_close = closes[i]
            next_close = closes[i+1] if i+1 < n else curr_close
            
            body = abs(curr_close - opens[i])
            
            # Bullish OB: down close, followed by up move
            if curr_close < opens[i] and next_close > curr_close * 1.001:
                obs.append({
                    'type': 'bullish',
                    'open': opens[i],
                    'close': curr_close,
                    'high': highs[i],
                    'low': lows[i],
                    'idx': i
                })
            
            # Bearish OB: up close, followed by down move
            elif curr_close > opens[i] and next_close < curr_close * 0.999:
                obs.append({
                    'type': 'bearish',
                    'open': opens[i],
                    'close': curr_close,
                    'high': highs[i],
                    'low': lows[i],
                    'idx': i
                })
        
        return obs
    
    def generate_signal(self, data: Dict, idx: int) -> Optional[Dict]:
        """Generate V6 signal with core confluence"""
        # Base V5 signal
        v5_signal = get_signal(data, idx)
        if not v5_signal:
            return None
        
        # Only process if V5 confluence >= 50
        if v5_signal['confluence'] < 50:
            return None
        
        current_price = data['closes'][idx]
        
        # Fast FVG detection (last 20 bars only)
        fvgs = self.detect_fvgs_fast(
            data['highs'][:idx+1],
            data['lows'][:idx+1],
            data['closes'][:idx+1],
            lookback=20
        )
        
        # Check FVG alignment
        fvg_boost = 0
        fvg_info = None
        for fvg in reversed(fvgs):  # Check most recent first
            if fvg['type'] == 'bullish' and v5_signal['direction'] == 1:
                # Price near FVG CE
                if abs(current_price - fvg['ce']) < fvg['high'] - fvg['low']:
                    fvg_boost = 15
                    fvg_info = f"BISI@{fvg['ce']:.2f}"
                    break
            elif fvg['type'] == 'bearish' and v5_signal['direction'] == -1:
                if abs(current_price - fvg['ce']) < fvg['high'] - fvg['low']:
                    fvg_boost = 15
                    fvg_info = f"SIBI@{fvg['ce']:.2f}"
                    break
        
        # Find liquidity levels
        buy_side, sell_side = self.find_liquidity_levels(
            data['highs'][:idx+1],
            data['lows'][:idx+1],
            window=5
        )
        
        liquidity_info = None
        if v5_signal['direction'] == 1 and sell_side:
            # Check if we're near sell-side liquidity (support)
            for level in sell_side:
                if abs(current_price - level) / level < 0.005:  # Within 0.5%
                    liquidity_info = f"SSL@{level:.2f}"
                    fvg_boost += 5
                    break
        elif v5_signal['direction'] == -1 and buy_side:
            for level in buy_side:
                if abs(current_price - level) / level < 0.005:
                    liquidity_info = f"BSL@{level:.2f}"
                    fvg_boost += 5
                    break
        
        # Combine
        total_confluence = min(v5_signal['confluence'] + fvg_boost, 100)
        
        # Determine confidence
        if total_confluence >= 75:
            confidence = 'HIGH'
        elif total_confluence >= 60:
            confidence = 'MEDIUM'
        else:
            confidence = 'LOW'
        
        return {
            'direction': v5_signal['direction'],
            'confluence': total_confluence,
            'v5_confluence': v5_signal['confluence'],
            'fvg_boost': fvg_boost,
            'confidence': confidence,
            'fvg_info': fvg_info,
            'liquidity_info': liquidity_info
        }


def run_v6_backtest_fast(symbols, days=30, initial_capital=50000, risk_per_trade=0.02):
    """Fast V6 backtest"""
    
    print(f"\n{'='*80}")
    print(f"V6 Fast Backtest - Core ICT Concepts")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Capital: ${initial_capital:,} | Risk: {risk_per_trade*100}%")
    print(f"{'='*80}\n")
    
    signal_gen = V6SimplifiedSignal(min_fvg_size=0.0)
    
    # Load data
    all_data = {}
    for symbol in symbols:
        print(f"Loading {symbol}...", end=' ')
        data = prepare_data_ibkr(symbol)
        if data and len(data.get('closes', [])) >= 50:
            all_data[symbol] = data
            print(f"✓ {len(data['closes'])} bars")
        else:
            print(f"✗")
    
    if not all_data:
        print("No data loaded!")
        return None
    
    # Get all timestamps
    all_timestamps = sorted(set().union(*[set(data['df'].index) for data in all_data.values()]))
    print(f"\nProcessing {len(all_timestamps)} timestamps...")
    
    # Trading state
    balance = initial_capital
    positions = {}
    trades = []
    
    # Process each timestamp
    for i, timestamp in enumerate(all_timestamps):
        if i % 1000 == 0 and i > 0:
            print(f"  [{i}/{len(all_timestamps)}] Balance: ${balance:,.2f} | Trades: {len(trades)}")
        
        for symbol, data in all_data.items():
            if timestamp not in data['df'].index:
                continue
            
            idx = data['df'].index.get_loc(timestamp)
            if idx < 50 or idx >= len(data['closes']) - 1:
                continue
            
            current_price = data['closes'][idx]
            
            # Check exits
            if symbol in positions:
                pos = positions[symbol]
                next_bar = data['df'].iloc[idx + 1]
                
                exit_price = None
                if pos['direction'] == 1:
                    if next_bar['low'] <= pos['stop']:
                        exit_price = pos['stop']
                    elif next_bar['high'] >= pos['target']:
                        exit_price = pos['target']
                else:
                    if next_bar['high'] >= pos['stop']:
                        exit_price = pos['stop']
                    elif next_bar['low'] <= pos['target']:
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
                    trades.append({
                        'symbol': symbol,
                        'direction': 'LONG' if pos['direction'] == 1 else 'SHORT',
                        'entry': pos['entry'],
                        'exit': exit_price,
                        'pnl': pnl,
                        'confidence': pos.get('confidence', 'MEDIUM')
                    })
                    del positions[symbol]
            
            # Check entries
            elif symbol not in positions:
                signal = signal_gen.generate_signal(data, idx)
                
                if signal and signal['confluence'] >= 60:
                    # Calculate stops/targets
                    if signal['direction'] == 1:
                        stop = data['lows'][idx]
                        target = current_price + (current_price - stop) * 2
                    else:
                        stop = data['highs'][idx]
                        target = current_price - (stop - current_price) * 2
                    
                    stop_distance = abs(current_price - stop)
                    if stop_distance > 0:
                        qty, _ = calculate_position_size(
                            symbol, initial_capital, risk_per_trade, 
                            stop_distance, current_price
                        )
                        
                        if qty > 0:
                            positions[symbol] = {
                                'entry': current_price,
                                'stop': stop,
                                'target': target,
                                'direction': signal['direction'],
                                'qty': qty,
                                'confidence': signal['confidence'],
                                'fvg_info': signal.get('fvg_info'),
                                'liquidity_info': signal.get('liquidity_info')
                            }
    
    # Results
    total_trades = len(trades)
    wins = len([t for t in trades if t['pnl'] > 0])
    losses = total_trades - wins
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    total_pnl = balance - initial_capital
    return_pct = (total_pnl / initial_capital) * 100
    
    # Symbol stats
    symbol_stats = {}
    for symbol in all_data.keys():
        symbol_trades = [t for t in trades if t['symbol'] == symbol]
        symbol_wins = len([t for t in symbol_trades if t['pnl'] > 0])
        symbol_stats[symbol] = {
            'trades': len(symbol_trades),
            'wins': symbol_wins,
            'losses': len(symbol_trades) - symbol_wins,
            'win_rate': (symbol_wins / len(symbol_trades) * 100) if symbol_trades else 0,
            'pnl': sum(t['pnl'] for t in symbol_trades)
        }
    
    print(f"\n{'='*80}")
    print("V6 BACKTEST RESULTS")
    print(f"{'='*80}")
    print(f"Initial: ${initial_capital:,}")
    print(f"Final: ${balance:,.2f}")
    print(f"Return: {return_pct:.2f}%")
    print(f"\nTrades: {total_trades} | Win Rate: {win_rate:.1f}%")
    print(f"Wins: {wins} | Losses: {losses}")
    print(f"Avg Trade: ${total_pnl/total_trades:.2f}" if total_trades > 0 else "N/A")
    print(f"\nSymbol Performance:")
    for symbol, stats in sorted(symbol_stats.items(), key=lambda x: x[1]['pnl'], reverse=True):
        print(f"  {symbol}: {stats['trades']}T {stats['win_rate']:.0f}%WR ${stats['pnl']:,.0f}")
    print(f"{'='*80}\n")
    
    return {
        'summary': {
            'initial': initial_capital,
            'final': balance,
            'return_pct': return_pct,
            'trades': total_trades,
            'win_rate': win_rate,
            'wins': wins,
            'losses': losses
        },
        'symbol_stats': symbol_stats,
        'trades': trades
    }


if __name__ == "__main__":
    symbols = ['BTCUSD', 'ETHUSD', 'ES', 'NQ', 'GC']
    
    results = run_v6_backtest_fast(
        symbols=symbols,
        days=30,
        initial_capital=50000,
        risk_per_trade=0.02
    )
    
    if results:
        with open('v6_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print("Results saved to v6_results.json")
