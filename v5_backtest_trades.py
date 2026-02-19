"""
V5 Backtest with Trade Logging
==============================
Run backtest and log all trades taken.
"""

import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import sys
sys.path.insert(0, '/Users/mac/Documents/Algot')

# Import from ict_v5_ibkr
exec(open('/Users/mac/Documents/Algot/ict_v5_ibkr.py').read().split('def run_backtest')[0])

def run_backtest_with_logs(symbols, days=180, use_ibkr=True):
    """Run backtest and log all trades."""
    print(f"\n{'='*80}")
    print(f"ICT V5 Backtest with Trade Logs - {days} days")
    print(f"Data source: {'IBKR' if use_ibkr else 'Yahoo'}")
    print(f"{'='*80}\n")
    
    all_trades = []
    results = []
    
    for symbol in symbols:
        print(f"\nTesting {symbol}...")
        
        if use_ibkr:
            data = prepare_data_ibkr(symbol)
        else:
            data = prepare_data(symbol)
        
        if data is None or len(data.get('closes', [])) < 50:
            print(f"  No data for {symbol}")
            continue
        
        balance = 10000
        position = None
        trades = 0
        wins = 0
        losses = 0
        symbol_trades = []
        
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
            
            if position is None and confluence >= 60:
                if htf == 1 and ltf >= 0:
                    direction = 1
                    stop = lows[idx]
                    target = current_price + (current_price - stop) * 2
                elif htf == -1 and ltf <= 0:
                    direction = -1
                    stop = highs[idx]
                    target = current_price - (stop - current_price) * 2
                else:
                    continue
                
                position = {
                    'entry': current_price,
                    'stop': stop,
                    'target': target,
                    'direction': direction,
                    'entry_idx': idx,
                    'confluence': confluence
                }
            
            elif position:
                next_close = closes[idx + 1]
                next_low = lows[idx + 1]
                next_high = highs[idx + 1]
                
                exit_price = None
                pnl = 0
                result = None
                
                exit_price = None
                pnl = 0
                result = None
                
                # Store position info before clearing it
                pos_entry = position['entry']
                pos_direction = position['direction']
                pos_entry_idx = position['entry_idx']
                pos_confluence = position['confluence']
                
                if position['direction'] == 1:
                    if next_low <= position['stop']:
                        exit_price = position['stop']
                        pnl = -100
                        losses += 1
                        trades += 1
                        result = 'LOSS'
                        position = None
                    elif next_high >= position['target']:
                        exit_price = position['target']
                        pnl = 200
                        wins += 1
                        trades += 1
                        result = 'WIN'
                        position = None
                else:
                    if next_high >= position['stop']:
                        exit_price = position['stop']
                        pnl = -100
                        losses += 1
                        trades += 1
                        result = 'LOSS'
                        position = None
                    elif next_low <= position['target']:
                        exit_price = position['target']
                        pnl = 200
                        wins += 1
                        trades += 1
                        result = 'WIN'
                        position = None
                
                if exit_price:
                    trade_log = {
                        'symbol': symbol,
                        'entry': pos_entry,
                        'exit': exit_price,
                        'direction': 'LONG' if pos_direction == 1 else 'SHORT',
                        'pnl': pnl,
                        'result': result,
                        'bars_held': idx - pos_entry_idx,
                        'confluence': pos_confluence
                    }
                    symbol_trades.append(trade_log)
                    all_trades.append(trade_log)
        
        win_rate = wins / trades * 100 if trades > 0 else 0
        result = {
            'symbol': symbol,
            'balance': balance,
            'trades': trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate
        }
        results.append(result)
        
        print(f"  {symbol}: ${balance:,.0f} | {trades} trades | {win_rate:.0f}% win rate")
        
        # Print trade log for this symbol
        if symbol_trades:
            print(f"\n  Trade Log for {symbol}:")
            print(f"  {'#':<4} {'Direction':<8} {'Entry':<12} {'Exit':<12} {'PnL':<10} {'Result':<6} {'Bars':<6} {'Conf':<6}")
            print(f"  {'-'*76}")
            for i, t in enumerate(symbol_trades, 1):
                print(f"  {i:<4} {t['direction']:<8} ${t['entry']:<11.2f} ${t['exit']:<11.2f} ${t['pnl']:<9.0f} {t['result']:<6} {t['bars_held']:<6} {t['confluence']:<6}")
    
    total_return = sum(r['balance'] for r in results) - (10000 * len(results))
    total_pct = total_return / (10000 * len(results)) * 100 if results else 0
    
    print(f"\n{'='*80}")
    print(f"SUMMARY - Total Return: ${total_return:,.0f} ({total_pct:.1f}%)")
    print(f"{'='*80}\n")
    
    # Print all trades summary
    if all_trades:
        wins = len([t for t in all_trades if t['result'] == 'WIN'])
        losses = len([t for t in all_trades if t['result'] == 'LOSS'])
        total_pnl = sum(t['pnl'] for t in all_trades)
        print(f"ALL TRADES SUMMARY:")
        print(f"Total Trades: {len(all_trades)}")
        print(f"Wins: {wins} ({wins/len(all_trades)*100:.0f}%)")
        print(f"Losses: {losses} ({losses/len(all_trades)*100:.0f}%)")
        print(f"Total PnL: ${total_pnl:,.0f}")
    
    return results, all_trades


# Run the backtest
symbols = ['ES', 'NQ', 'GC', 'SI', 'CL', 'NG', 'YM', 'EURUSD', 'GBPUSD', 'USDJPY', 'BTCUSD', 'ETHUSD']
results, trades = run_backtest_with_logs(symbols, days=180, use_ibkr=True)
