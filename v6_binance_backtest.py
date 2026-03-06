"""
V6 Backtest - Binance Crypto (Realistic)
========================================
Proper backtest without lookahead bias:
- Entry at close of signal bar
- Exit at close of next bar (not intrabar high/low)
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
CONF_THRESHOLD = 40
RISK_PCT = 0.02
DAYS = 90


def generate_signal(closes, highs, lows, idx: int) -> Optional[Dict]:
    """Generate V6 signal - NO LOOKAHEAD"""
    if idx < 50:
        return None
    
    current_price = closes[idx]
    
    # Use ONLY past data (no lookahead)
    past_closes = closes[:idx+1]
    past_highs = highs[:idx+1]
    past_lows = lows[:idx+1]
    
    # SMA trends (only past data)
    sma_20 = np.mean(past_closes[-20:])
    sma_50 = np.mean(past_closes[-50:]) if len(past_closes) >= 50 else sma_20
    
    # Price position (only past 50 bars)
    lookback = min(50, len(past_highs))
    high_50 = np.max(past_highs[-lookback:])
    low_50 = np.min(past_lows[-lookback:])
    price_pos = (current_price - low_50) / (high_50 - low_50) if high_50 != low_50 else 0.5
    
    confluence = 0
    direction = 0
    
    # Trend
    if sma_20 > sma_50:
        confluence += 25
        direction = 1
    elif sma_20 < sma_50:
        confluence += 25
        direction = -1
    
    # Price position
    if price_pos < 0.25:
        confluence += 20
    elif price_pos > 0.75:
        confluence += 20
    
    # ATR for stops (past data only)
    atr_lookback = min(14, len(past_lows) - 1)
    if atr_lookback > 0:
        atr = np.mean([past_highs[i] - past_lows[i] for i in range(-atr_lookback, 0)])
    else:
        atr = current_price * 0.02
    
    if direction == 0:
        return None
    
    return {
        'direction': direction,
        'confluence': min(confluence, 100),
        'stop_distance': atr * 2
    }


def check_exit(position: Dict, close_price: float) -> tuple:
    """
    Check if position should exit based on close price only.
    Returns (exited: bool, exit_price: float, reason: str)
    """
    if position['direction'] == 1:  # LONG
        if close_price <= position['stop']:
            return True, position['stop'], 'stop'
        elif close_price >= position['target']:
            return True, position['target'], 'target'
    else:  # SHORT
        if close_price >= position['stop']:
            return True, position['stop'], 'stop'
        elif close_price <= position['target']:
            return True, position['target'], 'target'
    
    return False, 0, ''


def run_backtest():
    """Run backtest with proper entry/exit logic"""
    
    print(f"ICT V6 Binance Backtest (Realistic)")
    print(f"Symbols: {len(SYMBOLS)} | Days: {DAYS}")
    print(f"RR: {RR_RATIO}:1 | Conf: {CONF_THRESHOLD} | Risk: {RISK_PCT*100}%")
    print("=" * 60)
    
    all_data = {}
    
    # Load all data first
    for symbol in SYMBOLS:
        print(f"Loading {symbol}...", end=' ')
        data = fetch_binance_data(symbol, '1h', DAYS * 24 + 100)  # Extra for warmup
        if data and len(data['closes']) > 100:
            all_data[symbol] = {
                'closes': np.array(data['closes']),
                'highs': np.array(data['highs']),
                'lows': np.array(data['lows']),
            }
            print(f"✓ {len(data['closes'])} bars")
        else:
            print(f"✗ Failed")
    
    if not all_data:
        print("No data loaded!")
        return
    
    # Backtest each symbol
    results = []
    all_trades = []
    
    for symbol, data in all_data.items():
        closes = data['closes']
        highs = data['highs']
        lows = data['lows']
        
        balance = 10000
        trades = 0
        wins = 0
        pnl = 0
        position = None
        
        # Start from index 50 (after warmup)
        for idx in range(50, len(closes) - 1):
            current_close = closes[idx]
            
            # Check exit at close of current bar (NO LOOKAHEAD)
            if position:
                exited, exit_price, reason = check_exit(position, current_close)
                
                if exited:
                    # Calculate P&L
                    if position['direction'] == 1:
                        trade_pnl = (exit_price - position['entry']) * position['qty']
                    else:
                        trade_pnl = (position['entry'] - exit_price) * position['qty']
                    
                    pnl += trade_pnl
                    balance += trade_pnl
                    trades += 1
                    
                    if trade_pnl > 0:
                        wins += 1
                    
                    all_trades.append({
                        'symbol': symbol,
                        'entry': position['entry'],
                        'exit': exit_price,
                        'direction': 'LONG' if position['direction'] == 1 else 'SHORT',
                        'pnl': trade_pnl,
                        'reason': reason
                    })
                    
                    position = None
            
            # Check for new entry at close of current bar
            if not position:
                signal = generate_signal(closes, highs, lows, idx)
                
                if signal and signal['confluence'] >= CONF_THRESHOLD:
                    stop_dist = signal['stop_distance']
                    
                    # Position size based on risk
                    risk = balance * RISK_PCT
                    qty = risk / stop_dist
                    
                    if qty > 0:
                        if signal['direction'] == 1:  # LONG
                            entry = current_close
                            stop = entry - stop_dist
                            target = entry + stop_dist * RR_RATIO
                        else:  # SHORT
                            entry = current_close
                            stop = entry + stop_dist
                            target = entry - stop_dist * RR_RATIO
                        
                        position = {
                            'entry': entry,
                            'stop': stop,
                            'target': target,
                            'direction': signal['direction'],
                            'qty': qty,
                            'confluence': signal['confluence']
                        }
        
        # Close any open position at last close
        if position:
            final_close = closes[-1]
            exited, exit_price, reason = check_exit(position, final_close)
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
            'losses': trades - wins,
            'win_rate': round(win_rate, 1),
            'pnl': round(pnl, 2),
            'return_pct': round((pnl / 10000) * 100, 1)
        })
        
        print(f"{symbol:10} {trades:3} trades | WR: {win_rate:5.1f}% | P&L: ${pnl:>8,.2f} ({pnl/10000*100:>6.1f}%)")
    
    # Summary
    total_trades = sum(r['trades'] for r in results)
    total_wins = sum(r['wins'] for r in results)
    total_pnl = sum(r['pnl'] for r in results)
    wr = (total_wins / total_trades * 100) if total_trades > 0 else 0
    
    print("=" * 60)
    print(f"TOTAL: {total_trades} trades | WR: {wr:.1f}%")
    print(f"P&L: ${total_pnl:,.2f} ({total_pnl/10000*100:.1f}%)")
    print("=" * 60)
    
    # Best/worst
    best = max(results, key=lambda x: x['pnl'])
    worst = min(results, key=lambda x: x['pnl'])
    print(f"Best: {best['symbol']} +${best['pnl']:.2f}")
    print(f"Worst: {worst['symbol']} ${worst['pnl']:.2f}")
    
    # Save results
    output = {
        'config': {
            'symbols': SYMBOLS,
            'days': DAYS,
            'rr_ratio': RR_RATIO,
            'confluence': CONF_THRESHOLD,
            'risk_pct': RISK_PCT,
            'entry': 'close',
            'exit': 'close_next_bar'
        },
        'summary': {
            'trades': total_trades,
            'wins': total_wins,
            'losses': total_trades - total_wins,
            'win_rate': round(wr, 1),
            'pnl': round(total_pnl, 2),
            'return_pct': round((total_pnl / 10000) * 100, 1)
        },
        'results': results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('v6_binance_backtest_results.json', 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nResults saved to v6_binance_backtest_results.json")
    
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--days', type=int, default=DAYS)
    args = parser.parse_args()
    
    if args.days != DAYS:
        DAYS = args.days
    
    run_backtest()
