"""
ICT V7 - With Premium/Discount Arrays
======================================
Combines V6 + PD Array analysis for optimal trade entries
"""

import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import sys
sys.path.insert(0, '/Users/mac/Documents/Algot')

import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import importlib.util

# Import V5 base
spec = importlib.util.spec_from_file_location("ict_v5", "/Users/mac/Documents/Algot/ict_v5_ibkr.py")
ict_v5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ict_v5)

fetch_ibkr_data = ict_v5.fetch_ibkr_data
prepare_data_ibkr = ict_v5.prepare_data_ibkr
get_signal = ict_v5.get_signal
calculate_position_size = ict_v5.calculate_position_size
get_contract_info = ict_v5.get_contract_info


class V7SignalGenerator:
    """V7 Signal Generator with PD Array analysis"""
    
    def __init__(self):
        self.min_fvg_size = 0.0
    
    def calculate_daily_quadrants(self, highs, lows, lookback=24):
        """Calculate daily range quadrants (ICT: Grade everything)"""
        if len(highs) < lookback:
            lookback = len(highs)
        
        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]
        
        daily_high = np.max(recent_highs)
        daily_low = np.min(recent_lows)
        range_size = daily_high - daily_low
        
        return {
            'high': daily_high,
            'low': daily_low,
            'ce': daily_low + (range_size * 0.5),  # 50%
            'upper_quad': daily_low + (range_size * 0.75),  # 75%
            'lower_quad': daily_low + (range_size * 0.25),  # 25%
            'ote_high': daily_low + (range_size * 0.79),  # 79% - OTE
            'ote_low': daily_low + (range_size * 0.62),  # 62% - OTE
            'range_size': range_size
        }
    
    def get_pd_zone(self, price, quadrants):
        """Determine Premium/Discount zone for price"""
        if price >= quadrants['upper_quad']:
            return 'extreme_premium'  # 75-100% - Sell zone
        elif price >= quadrants['ce']:
            return 'premium'  # 50-75% - Sell zone
        elif price <= quadrants['lower_quad']:
            return 'extreme_discount'  # 0-25% - Buy zone
        elif price <= quadrants['ce']:
            return 'discount'  # 25-50% - Buy zone
        else:
            return 'equilibrium'
    
    def is_in_ote(self, price, quadrants, direction):
        """Check if price is in Optimal Trade Entry zone (62-79%)"""
        if direction == 1:  # Long
            # For longs, OTE is in discount (62-79% of range from bottom)
            return quadrants['ote_low'] <= price <= quadrants['ote_high']
        else:  # Short
            # For shorts, OTE is in premium (21-38% from top, or 62-79% from bottom)
            ote_short_high = quadrants['high'] - (quadrants['range_size'] * 0.62)
            ote_short_low = quadrants['high'] - (quadrants['range_size'] * 0.79)
            return ote_short_low <= price <= ote_short_high
    
    def detect_fvgs_fast(self, highs, lows, closes, lookback=20):
        """Fast FVG detection"""
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
        """Find equal highs/lows"""
        n = len(highs)
        if n < window * 2:
            return [], []
        
        buy_side = []
        sell_side = []
        tolerance = 0.001
        
        for i in range(window, n - window):
            recent_highs = highs[i-window:i+window]
            max_high = np.max(recent_highs)
            touches = np.sum(np.abs(recent_highs - max_high) / max_high < tolerance)
            if touches >= 2:
                buy_side.append(max_high)
            
            recent_lows = lows[i-window:i+window]
            min_low = np.min(recent_lows)
            touches = np.sum(np.abs(recent_lows - min_low) / min_low < tolerance)
            if touches >= 2:
                sell_side.append(min_low)
        
        return list(set(buy_side))[:3], list(set(sell_side))[:3]
    
    def generate_signal(self, data: Dict, idx: int) -> Optional[Dict]:
        """Generate V7 signal with PD Array analysis"""
        # Base V5 signal
        v5_signal = get_signal(data, idx)
        if not v5_signal:
            return None
        
        # Only process if V5 confluence >= 50
        if v5_signal['confluence'] < 50:
            return None
        
        current_price = data['closes'][idx]
        
        # Calculate daily quadrants
        quadrants = self.calculate_daily_quadrants(
            data['highs'][:idx+1],
            data['lows'][:idx+1],
            lookback=24
        )
        
        # Get PD zone
        pd_zone = self.get_pd_zone(current_price, quadrants)
        
        # ICT Rule: When bullish, ONLY buy from discount
        # When bearish, ONLY sell from premium
        pd_boost = 0
        pd_valid = False
        
        if v5_signal['direction'] == 1:  # Long
            if pd_zone in ['discount', 'extreme_discount']:
                pd_boost = 20
                pd_valid = True
                # Extra boost if in OTE
                if self.is_in_ote(current_price, quadrants, 1):
                    pd_boost += 10
            elif pd_zone == 'equilibrium':
                pd_boost = 5  # Neutral
                pd_valid = True
            else:
                pd_boost = -10  # Wrong zone for longs
                pd_valid = False
        
        else:  # Short
            if pd_zone in ['premium', 'extreme_premium']:
                pd_boost = 20
                pd_valid = True
                if self.is_in_ote(current_price, quadrants, -1):
                    pd_boost += 10
            elif pd_zone == 'equilibrium':
                pd_boost = 5
                pd_valid = True
            else:
                pd_boost = -10
                pd_valid = False
        
        # Fast FVG detection
        fvgs = self.detect_fvgs_fast(
            data['highs'][:idx+1],
            data['lows'][:idx+1],
            data['closes'][:idx+1],
            lookback=20
        )
        
        # Check FVG alignment
        fvg_boost = 0
        fvg_info = None
        for fvg in reversed(fvgs):
            if fvg['type'] == 'bullish' and v5_signal['direction'] == 1:
                if abs(current_price - fvg['ce']) < fvg['high'] - fvg['low']:
                    fvg_boost = 10
                    fvg_info = f"BISI@{fvg['ce']:.2f}"
                    break
            elif fvg['type'] == 'bearish' and v5_signal['direction'] == -1:
                if abs(current_price - fvg['ce']) < fvg['high'] - fvg['low']:
                    fvg_boost = 10
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
            for level in sell_side:
                if abs(current_price - level) / level < 0.005:
                    liquidity_info = f"SSL@{level:.2f}"
                    fvg_boost += 5
                    break
        elif v5_signal['direction'] == -1 and buy_side:
            for level in buy_side:
                if abs(current_price - level) / level < 0.005:
                    liquidity_info = f"BSL@{level:.2f}"
                    fvg_boost += 5
                    break
        
        # Combine confluence
        total_confluence = v5_signal['confluence'] + pd_boost + fvg_boost
        total_confluence = max(0, min(total_confluence, 100))
        
        # Determine confidence (require PD zone validity for HIGH)
        if total_confluence >= 75 and pd_valid:
            confidence = 'HIGH'
        elif total_confluence >= 60:
            confidence = 'MEDIUM'
        elif total_confluence >= 45:
            confidence = 'LOW'
        else:
            confidence = 'LOW'
            if not pd_valid:
                return None  # Skip if in wrong PD zone
        
        return {
            'direction': v5_signal['direction'],
            'confluence': total_confluence,
            'v5_confluence': v5_signal['confluence'],
            'pd_boost': pd_boost,
            'fvg_boost': fvg_boost,
            'confidence': confidence,
            'pd_zone': pd_zone,
            'daily_ce': quadrants['ce'],
            'fvg_info': fvg_info,
            'liquidity_info': liquidity_info,
            'in_ote': self.is_in_ote(current_price, quadrants, v5_signal['direction'])
        }


def run_v7_backtest(symbols, days=30, initial_capital=50000, risk_per_trade=0.02):
    """Run V7 backtest with PD Array analysis"""
    
    print(f"\n{'='*80}")
    print(f"V7 Backtest - Premium/Discount Arrays + FVG + Liquidity")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Capital: ${initial_capital:,} | Risk: {risk_per_trade*100}%")
    print(f"{'='*80}\n")
    
    signal_gen = V7SignalGenerator()
    
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
                        'confidence': pos.get('confidence', 'MEDIUM'),
                        'pd_zone': pos.get('pd_zone')
                    })
                    del positions[symbol]
            
            # Check entries
            elif symbol not in positions:
                signal = signal_gen.generate_signal(data, idx)
                
                if signal and signal['confluence'] >= 55:  # Slightly lower threshold for V7
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
                                'pd_zone': signal['pd_zone'],
                                'fvg_info': signal.get('fvg_info'),
                                'in_ote': signal.get('in_ote')
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
    print("V7 BACKTEST RESULTS")
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
    
    results = run_v7_backtest(
        symbols=symbols,
        days=30,
        initial_capital=50000,
        risk_per_trade=0.02
    )
    
    if results:
        with open('v7_results.json', 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print("Results saved to v7_results.json")
