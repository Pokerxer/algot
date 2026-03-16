"""
V6 Backtest - Binance Crypto (Using Real V6 Signal)
====================================================
Uses the actual V6 signal generator with FVG, HTF/LTF, Kill Zone, Price Position
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
from datetime import datetime
from typing import Dict, List, Optional

from binance_client import fetch_binance_data
import numpy as np


SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'LTCUSDT', 'LINKUSDT', 
           'UNIUSDT', 'XRPUSDT', 'ADAUSDT', 'DOGEUSDT', 'DOTUSDT', 
           'AVAXUSDT', 'MATICUSDT']

RR_RATIO = 3.0
CONF_THRESHOLD = 50  # Lower threshold to match V5 approach
RISK_PCT = 0.02
DAYS = 90


def prepare_binance_data(symbol: str, days: int = 90) -> Optional[Dict]:
    """Prepare Binance data in V6 format"""
    data = fetch_binance_data(symbol, '1h', days * 24 + 100)
    if not data or len(data['closes']) < 100:
        return None
    
    highs = np.array(data['highs'])
    lows = np.array(data['lows'])
    closes = np.array(data['closes'])
    opens = np.array(data['opens'])
    
    n = len(closes)
    
    # FVGs (simple calculation)
    bullish_fvgs = []
    bearish_fvgs = []
    for i in range(3, n):
        if lows[i] > highs[i-2]:
            bullish_fvgs.append({'idx': i, 'mid': (highs[i-2] + lows[i]) / 2, 'high': lows[i]})
        if highs[i] < lows[i-2]:
            bearish_fvgs.append({'idx': i, 'mid': (highs[i] + lows[i-2]) / 2, 'low': highs[i]})
    
    # HTF trend (daily approximation)
    htf_trend = np.zeros(n)
    for i in range(24, n):
        if i % 24 == 0:
            lookback_start = max(0, i - 24 * 5)
            high_5d = np.max(highs[lookback_start:i])
            low_5d = np.min(lows[lookback_start:i])
            if closes[i] > high_5d * 0.99:
                htf_trend[i:] = 1
            elif closes[i] < low_5d * 1.01:
                htf_trend[i:] = -1
    
    # LTF trend
    ltf_trend = np.zeros(n)
    for i in range(20, n):
        momentum = closes[i] - closes[i-10]
        pct = momentum / closes[i-10] if closes[i-10] > 0 else 0
        if pct > 0.005:
            ltf_trend[i] = 1
        elif pct < -0.005:
            ltf_trend[i] = -1
    
    # Kill zone
    kill_zone = np.zeros(n, dtype=bool)
    for i in range(20, n):
        atr = np.mean([highs[i-j] - lows[i-j] for j in range(1, min(14, i))])
        avg_range = np.mean([highs[i-j] - lows[i-j] for j in range(20, min(50, i))])
        if avg_range > 0 and atr > avg_range * 1.3:
            kill_zone[i] = True
    
    # Price position
    price_position = np.zeros(n)
    for i in range(20, n):
        high_20 = np.max(highs[i-20:i])
        low_20 = np.min(lows[i-20:i])
        if high_20 != low_20:
            price_position[i] = (closes[i] - low_20) / (high_20 - low_20)
    
    return {
        'opens': opens,
        'highs': highs,
        'lows': lows,
        'closes': closes,
        'htf_trend': htf_trend,
        'ltf_trend': ltf_trend,
        'kill_zone': kill_zone,
        'price_position': price_position,
        'bullish_fvgs': bullish_fvgs,
        'bearish_fvgs': bearish_fvgs
    }


def analyze_v6(data: Dict, idx: int) -> Optional[Dict]:
    """Generate V6 signal with all factors"""
    if idx < 50:
        return None
    
    closes = data['closes']
    highs = data['highs']
    lows = data['lows']
    current_price = closes[idx]
    
    htf = data['htf_trend'][idx]
    ltf = data['ltf_trend'][idx]
    kz = data['kill_zone'][idx]
    pp = data['price_position'][idx]
    
    # FVGs
    near_bull = next((f for f in reversed(data['bullish_fvgs']) 
                     if f['idx'] < idx and f['mid'] < current_price), None)
    near_bear = next((f for f in reversed(data['bearish_fvgs']) 
                     if f['idx'] < idx and f['mid'] > current_price), None)
    
    confluence = 0
    direction = 0
    
    # Kill zone (+15)
    if kz:
        confluence += 15
    
    # HTF+LTF alignment (+25)
    if htf == 1 and ltf >= 0:
        confluence += 25
        direction = 1
    elif htf == -1 and ltf <= 0:
        confluence += 25
        direction = -1
    elif htf == 0 and ltf == 1:
        confluence += 15
        direction = 1
    elif htf == 0 and ltf == -1:
        confluence += 15
        direction = -1
    
    # Price position (+20)
    if pp < 0.25:
        confluence += 20
    elif pp > 0.75:
        confluence += 20
    
    # FVG confluence (+15)
    if near_bull and ltf >= 0:
        confluence += 15
    if near_bear and ltf <= 0:
        confluence += 15
    
    if direction == 0:
        return None
    
    # ATR for stops
    atr = np.mean([highs[i] - lows[i] for i in range(max(0, idx-14), idx)])
    
    return {
        'direction': direction,
        'confluence': min(confluence, 100),
        'stop_distance': atr * 2
    }


def check_exit(position: Dict, close_price: float) -> tuple:
    if position['direction'] == 1:
        if close_price <= position['stop']:
            return True, position['stop'], 'stop'
        elif close_price >= position['target']:
            return True, position['target'], 'target'
    else:
        if close_price >= position['stop']:
            return True, position['stop'], 'stop'
        elif close_price <= position['target']:
            return True, position['target'], 'target'
    return False, 0, ''


def run_backtest():
    """Run backtest"""
    
    print(f"ICT V6 Binance Backtest (Real V6 Signal)")
    print(f"Symbols: {len(SYMBOLS)} | Days: {DAYS}")
    print(f"RR: {RR_RATIO}:1 | Conf: {CONF_THRESHOLD} | Risk: {RISK_PCT*100}%")
    print("=" * 60)
    
    all_data = {}
    
    for symbol in SYMBOLS:
        print(f"Loading {symbol}...", end=' ')
        data = prepare_binance_data(symbol, DAYS)
        if data:
            all_data[symbol] = data
            print(f"✓ {len(data['closes'])} bars")
        else:
            print("✗ Failed")
    
    if not all_data:
        print("No data!")
        return
    
    results = []
    
    for symbol, data in all_data.items():
        print(f"Backtesting {symbol}...", end=' ')
        
        closes = data['closes']
        
        balance = 100
        trades = 0
        wins = 0
        pnl = 0
        position = None
        
        for idx in range(50, len(closes) - 1):
            current_close = closes[idx]
            
            # Check exit
            if position:
                exited, exit_price, reason = check_exit(position, current_close)
                
                if exited:
                    if position['direction'] == 1:
                        trade_pnl = (exit_price - position['entry']) * position['qty']
                    else:
                        trade_pnl = (position['entry'] - exit_price) * position['qty']
                    
                    pnl += trade_pnl
                    balance += trade_pnl
                    trades += 1
                    if trade_pnl > 0:
                        wins += 1
                    position = None
            
            # Check entry
            if not position:
                signal = analyze_v6(data, idx)
                
                if signal and signal['confluence'] >= CONF_THRESHOLD:
                    stop_dist = signal['stop_distance']
                    risk = balance * RISK_PCT
                    qty = risk / stop_dist
                    
                    if qty > 0:
                        if signal['direction'] == 1:
                            entry = current_close
                            stop = entry - stop_dist
                            target = entry + stop_dist * RR_RATIO
                        else:
                            entry = current_close
                            stop = entry + stop_dist
                            target = entry - stop_dist * RR_RATIO
                        
                        position = {
                            'entry': entry,
                            'stop': stop,
                            'target': target,
                            'direction': signal['direction'],
                            'qty': qty
                        }
        
        # Close at end
        if position:
            final_close = closes[-1]
            exited, exit_price, _ = check_exit(position, final_close)
            if exited:
                if position['direction'] == 1:
                    trade_pnl = (exit_price - position['entry']) * position['qty']
                else:
                    trade_pnl = (position['entry'] - exit_price) * position['qty']
                pnl += trade_pnl
                trades += 1
                if trade_pnl > 0:
                    wins += 1
        
        win_rate = (wins / trades * 100) if trades > 0 else 0
        
        results.append({
            'symbol': symbol,
            'trades': trades,
            'wins': wins,
            'win_rate': round(win_rate, 1),
            'pnl': round(pnl, 2)
        })
        
        print(f"{trades} trades, WR {win_rate:.1f}%, P&L ${pnl:,.2f}")
    
    # Summary
    total_trades = sum(r['trades'] for r in results)
    total_wins = sum(r['wins'] for r in results)
    total_pnl = sum(r['pnl'] for r in results)
    wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
    
    print("=" * 60)
    print(f"TOTAL: {total_trades} trades | WR: {wr:.1f}%")
    print(f"P&L: ${total_pnl:,.2f} ({total_pnl/10000*100:.1f}%)")
    
    # Save
    with open('v6_binance_backtest_results.json', 'w') as f:
        json.dump({
            'config': {'rr': RR_RATIO, 'conf': CONF_THRESHOLD, 'days': DAYS},
            'summary': {'trades': total_trades, 'wr': wr, 'pnl': total_pnl},
            'results': results
        }, f, indent=2)
    
    return results


if __name__ == "__main__":
    run_backtest()
