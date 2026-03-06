"""
V6 Backtest - Binance Crypto (Optimized)
=========================================
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
from datetime import datetime
from typing import Dict, List

from binance_client import fetch_binance_data
from fvg_handler import FVGHandler
import pandas as pd
import numpy as np


SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT']
RR_RATIO = 3.0
CONF_THRESHOLD = 40  # Lowered for crypto
RISK_PCT = 0.02
DAYS = 90


def generate_signal(closes, highs, lows, idx):
    """Generate V6 signal"""
    if idx < 50:
        return None
    
    current_price = closes[idx]
    
    # SMA trends
    sma_20 = np.mean(closes[idx-20:idx])
    sma_50 = np.mean(closes[idx-50:idx])
    
    # Price position
    high_50 = np.max(highs[idx-50:idx])
    low_50 = np.min(lows[idx-50:idx])
    price_pos = (current_price - low_50) / (high_50 - low_50) if high_50 != low_50 else 0.5
    
    confluence = 0
    direction = 0
    
    if sma_20 > sma_50:
        confluence += 25
        direction = 1
    elif sma_20 < sma_50:
        confluence += 25
        direction = -1
    
    if price_pos < 0.25:
        confluence += 20
    elif price_pos > 0.75:
        confluence += 20
    
    # ATR
    atr = np.mean([highs[i] - lows[i] for i in range(max(0, idx-14), idx)])
    
    if direction == 0:
        return None
    
    return {
        'direction': direction,
        'confluence': min(confluence, 100),
        'stop_distance': atr * 2
    }


def run_backtest():
    """Run backtest"""
    global DAYS
    
    print(f"ICT V6 Binance Backtest | {len(SYMBOLS)} symbols | {DAYS} days")
    
    all_data = {}
    
    # Load data
    for symbol in SYMBOLS:
        print(f"Loading {symbol}...", end=' ')
        data = fetch_binance_data(symbol, '1h', DAYS * 24)
        if data:
            all_data[symbol] = data
            print(f"✓ {len(data['closes'])} bars")
        else:
            print("✗ Failed")
    
    if not all_data:
        print("No data!")
        return
    
    # Backtest each symbol
    results = []
    total_pnl = 0
    
    for symbol, data in all_data.items():
        print(f"Backtesting {symbol}...", end=' ')
        
        closes = np.array(data['closes'])
        highs = np.array(data['highs'])
        lows = np.array(data['lows'])
        
        balance = 10000
        trades = 0
        wins = 0
        pnl = 0
        position = None
        
        for idx in range(50, len(closes) - 1):
            current_price = closes[idx]
            
            # Check exit
            if position:
                next_low = lows[idx + 1]
                next_high = highs[idx + 1]
                
                if position['direction'] == 1:  # LONG
                    if next_low <= position['stop']:
                        pnl += (position['stop'] - position['entry']) * position['qty']
                        trades += 1
                        if pnl > 0:
                            wins += 1
                        position = None
                    elif next_high >= position['target']:
                        pnl += (position['target'] - position['entry']) * position['qty']
                        trades += 1
                        wins += 1
                        position = None
                else:  # SHORT
                    if next_high >= position['stop']:
                        pnl += (position['entry'] - position['stop']) * position['qty']
                        trades += 1
                        if pnl > 0:
                            wins += 1
                        position = None
                    elif next_low <= position['target']:
                        pnl += (position['entry'] - position['target']) * position['qty']
                        trades += 1
                        wins += 1
                        position = None
            
            # Check entry
            if not position:
                signal = generate_signal(closes, highs, lows, idx)
                
                if signal and signal['confluence'] >= CONF_THRESHOLD:
                    stop_dist = signal['stop_distance']
                    risk = balance * RISK_PCT
                    qty = risk / stop_dist
                    
                    if signal['direction'] == 1:
                        stop = current_price - stop_dist
                        target = current_price + stop_dist * RR_RATIO
                    else:
                        stop = current_price + stop_dist
                        target = current_price - stop_dist * RR_RATIO
                    
                    position = {
                        'entry': current_price,
                        'stop': stop,
                        'target': target,
                        'direction': signal['direction'],
                        'qty': qty
                    }
        
        win_rate = (wins / trades * 100) if trades > 0 else 0
        
        results.append({
            'symbol': symbol,
            'trades': trades,
            'wins': wins,
            'win_rate': round(win_rate, 1),
            'pnl': round(pnl, 2)
        })
        
        total_pnl += pnl
        print(f"{trades} trades, WR {win_rate:.1f}%, P&L ${pnl:.2f}")
    
    # Summary
    total_trades = sum(r['trades'] for r in results)
    total_wins = sum(r['wins'] for r in results)
    wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
    
    print(f"\n{'='*50}")
    print(f"SUMMARY")
    print(f"{'='*50}")
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {wr:.1f}%")
    print(f"Total P&L: ${total_pnl:,.2f}")
    print(f"\nBy Symbol:")
    for r in sorted(results, key=lambda x: x['pnl'], reverse=True):
        print(f"  {r['symbol']:10} {r['trades']:3} trades | WR: {r['win_rate']:5.1f}% | ${r['pnl']:,.2f}")
    
    return results


if __name__ == "__main__":
    run_backtest()
