"""
V3 Backtest with IBKR Data
==========================
Run V3 strategy backtest using Interactive Brokers data.
"""

import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import sys
sys.path.insert(0, '/Users/mac/Documents/Algot')

import json
from datetime import datetime
import pandas as pd
import numpy as np

# Import V3 functions
exec(open('/Users/mac/Documents/Algot/ict_v3_live.py').read().split('def run_live_trading')[0])

def run_v3_backtest_ibkr(symbols, days=180, initial_capital=10000, risk_per_trade=0.02, use_ibkr=True):
    """Run V3 backtest with IBKR data."""
    
    print(f"\n{'='*80}")
    print(f"ICT V3 Backtest with IBKR Data")
    print(f"Initial Capital: ${initial_capital:,}")
    print(f"Risk per Trade: {risk_per_trade*100}%")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"{'='*80}\n")
    
    # Setup agents
    agents = {}
    combined_agent = QLearningAgent(20, 8)
    if combined_agent.load(Q_TABLE_FILE):
        print(f"Loaded Q-table: {len(combined_agent.q_table)} states")
    else:
        print("No Q-table found, starting fresh")
    
    for symbol in symbols:
        agents[symbol] = combined_agent
    
    # Load data for all symbols
    all_data = {}
    for symbol in symbols:
        print(f"Loading data for {symbol}...")
        
        # Try IBKR first, then Yahoo
        if use_ibkr:
            try:
                from ib_insync import IB, util
                df = fetch_ibkr_data_v3(symbol, days=days)
            except:
                df = None
        else:
            df = None
        
        if df is None or len(df) < 50:
            # Fallback to Yahoo
            print(f"  Using Yahoo Finance for {symbol}")
            yahoo_symbol = symbol.replace('USD', '-USD') if 'USD' in symbol else symbol
            df = yf.Ticker(yahoo_symbol).history(period=f"{days}d", interval="1h")
            df = df.dropna()
        
        if df is not None and len(df) >= 50:
            # Prepare V3 format data
            data = prepare_data_from_df(df, symbol=symbol, use_ibkr=use_ibkr)
            if data:
                all_data[symbol] = data
                print(f"  Loaded {len(data['closes'])} rows")
        else:
            print(f"  No data for {symbol}")
    
    if not all_data:
        print("No data loaded!")
        return None
    
    # Combine timestamps
    all_timestamps = set()
    for data in all_data.values():
        all_timestamps.update(data['df'].index)
    
    sorted_timestamps = sorted(list(all_timestamps))
    print(f"\nProcessing {len(sorted_timestamps)} timestamps...")
    
    # Portfolio tracking
    balance = initial_capital
    positions = {}
    all_trades = []
    symbol_stats = {symbol: {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0} for symbol in all_data.keys()}
    
    # Risk management
    fixed_risk_amount = initial_capital * risk_per_trade
    max_position_pct = 0.50
    
    # Process each timestamp
    for i, timestamp in enumerate(sorted_timestamps):
        if i % 1000 == 0 and i > 0:
            print(f"  Processed {i}/{len(sorted_timestamps)} timestamps, Balance: ${balance:,.2f}")
        
        # Check exits
        for symbol, position in list(positions.items()):
            data = all_data[symbol]
            
            if timestamp not in data['df'].index:
                continue
            
            idx = data['df'].index.get_loc(timestamp)
            if idx >= len(data['closes']) - 1:
                continue
            
            next_low = data['lows'][idx + 1]
            next_high = data['highs'][idx + 1]
            
            exit_price = None
            exit_reason = None
            
            if position['direction'] == 1:
                if next_low <= position['stop_loss']:
                    exit_price = position['stop_loss']
                    exit_reason = 'stop_loss'
                elif next_high >= position['take_profit']:
                    exit_price = position['take_profit']
                    exit_reason = 'take_profit'
            else:
                if next_high >= position['stop_loss']:
                    exit_price = position['stop_loss']
                    exit_reason = 'stop_loss'
                elif next_low <= position['take_profit']:
                    exit_price = position['take_profit']
                    exit_reason = 'take_profit'
            
            if exit_price:
                if position['direction'] == 1:
                    price_change = exit_price - position['entry_price']
                else:
                    price_change = position['entry_price'] - exit_price
                
                pnl = price_change * position['position_size']
                balance += pnl
                
                trade_record = {
                    'symbol': symbol,
                    'entry_price': round(position['entry_price'], 4),
                    'exit_price': round(exit_price, 4),
                    'direction': 'LONG' if position['direction'] == 1 else 'SHORT',
                    'position_size': round(position['position_size'], 4),
                    'pnl': round(pnl, 2),
                    'result': 'WIN' if pnl > 0 else 'LOSS',
                    'exit_reason': exit_reason,
                    'bars_held': idx - position['entry_idx'],
                    'balance_after': round(balance, 2),
                    'exit_time': str(timestamp)
                }
                all_trades.append(trade_record)
                
                symbol_stats[symbol]['trades'] += 1
                symbol_stats[symbol]['pnl'] += pnl
                if pnl > 0:
                    symbol_stats[symbol]['wins'] += 1
                else:
                    symbol_stats[symbol]['losses'] += 1
                
                del positions[symbol]
        
        # Check entries (V3 uses 2R ratio like V5)
        for symbol, data in all_data.items():
            if symbol in positions:
                continue
            
            if timestamp not in data['df'].index:
                continue
            
            idx = data['df'].index.get_loc(timestamp)
            if idx < 50 or idx >= len(data['closes']) - 1:
                continue
            
            # Get signal using V3 logic
            current_price = data['closes'][idx]
            htf = data['htf_trend'][idx]
            ltf = data['ltf_trend'][idx]
            kz = data['kill_zone'][idx]
            
            # V3: Only trade in kill zones
            if not kz:
                continue
            
            # DEBUG: Log why no trades (first few times)
            if len(all_trades) == 0 and idx < 100:
                if idx % 20 == 0:
                    print(f"  Bar {idx}: KZ={kz}, HTF={htf}, LTF={ltf} - Waiting for trend alignment...")
            
            # Simple V3 signal - RELAXED: Allow LTF=0 (neutral) to trade
            if htf >= 0 and ltf >= 0:  # Relaxed: HTF can be neutral too
                direction = 1
                stop = data['lows'][idx]
                target = current_price + (current_price - stop) * 2
            elif htf <= 0 and ltf <= 0:  # Relaxed: HTF can be neutral too
                direction = -1
                stop = data['highs'][idx]
                target = current_price - (stop - current_price) * 2
            else:
                continue
            
            stop_distance = abs(current_price - stop)
            if stop_distance > 0:
                position_size = fixed_risk_amount / stop_distance
                max_position_size = (balance * max_position_pct) / current_price
                position_size = min(position_size, max_position_size)
                
                if position_size > 0:
                    positions[symbol] = {
                        'entry_price': current_price,
                        'stop_loss': stop,
                        'take_profit': target,
                        'direction': direction,
                        'entry_idx': idx,
                        'position_size': position_size,
                        'entry_time': str(timestamp)
                    }
    
    # Calculate results
    total_pnl = balance - initial_capital
    total_return_pct = (total_pnl / initial_capital) * 100
    total_trades = len(all_trades)
    wins = len([t for t in all_trades if t['result'] == 'WIN'])
    losses = len([t for t in all_trades if t['result'] == 'LOSS'])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    # Max drawdown
    peak = initial_capital
    max_dd = 0
    running_balance = initial_capital
    
    for trade in all_trades:
        running_balance += trade['pnl']
        if running_balance > peak:
            peak = running_balance
        dd = (peak - running_balance) / peak * 100
        if dd > max_dd:
            max_dd = dd
    
    results = {
        'backtest_config': {
            'symbols': list(all_data.keys()),
            'days': days,
            'initial_capital': initial_capital,
            'risk_per_trade': risk_per_trade,
            'data_source': 'IBKR' if use_ibkr else 'Yahoo',
            'timestamp': datetime.now().isoformat()
        },
        'summary': {
            'initial_capital': initial_capital,
            'final_capital': round(balance, 2),
            'total_pnl': round(total_pnl, 2),
            'total_return_pct': round(total_return_pct, 2),
            'max_drawdown_pct': round(max_dd, 2),
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': round(win_rate, 2)
        },
        'symbol_stats': symbol_stats,
        'trades': all_trades
    }
    
    return results


def fetch_ibkr_data_v3(symbol, days=30):
    """Fetch data from IBKR for V3."""
    try:
        from ib_insync import IB, Crypto, Forex, Future, util
    except ImportError:
        return None
    
    # Map symbol to IBKR contract
    contract = None
    symbol_upper = symbol.upper()
    
    # Crypto
    crypto_map = {'BTCUSD': 'BTC', 'ETHUSD': 'ETH', 'SOLUSD': 'SOL', 
                  'LTCUSD': 'LTC', 'LINKUSD': 'LINK', 'UNIUSD': 'UNI'}
    if symbol_upper in crypto_map:
        contract = Crypto(crypto_map[symbol_upper], exchange='PAXOS', currency='USD')
    
    # Futures
    elif symbol_upper in ['ES', 'NQ']:
        contract = Future(symbol_upper, exchange='CME', currency='USD')
    elif symbol_upper in ['GC', 'SI', 'CL', 'NG']:
        contract = Future(symbol_upper, exchange='COMEX' if symbol_upper in ['GC', 'SI'] else 'NYMEX', currency='USD')
    
    if not contract:
        return None
    
    ib = IB()
    try:
        ib.connect('127.0.0.1', 7497, clientId=np.random.randint(1000, 50000))
    except:
        return None
    
    try:
        bars = ib.reqHistoricalData(
            contract,
            endDateTime='',
            durationStr=f'{days} D',
            barSizeSetting='1 hour',
            whatToShow='MIDPOINT',
            useRTH=False,
            formatDate=2
        )
    except:
        ib.disconnect()
        return None
    
    ib.disconnect()
    
    if not bars:
        return None
    
    df = util.df(bars)
    df.set_index('date', inplace=True)
    df.index = pd.to_datetime(df.index)
    
    return df


def fetch_daily_data(symbol, days=60):
    """Fetch daily data for HTF trend calculation."""
    try:
        from ib_insync import IB, Crypto, Future, util
        
        # Map symbol to IBKR contract
        contract = None
        symbol_upper = symbol.upper()
        
        # Crypto
        crypto_map = {'BTCUSD': 'BTC', 'ETHUSD': 'ETH', 'SOLUSD': 'SOL', 
                      'LTCUSD': 'LTC', 'LINKUSD': 'LINK', 'UNIUSD': 'UNI'}
        if symbol_upper in crypto_map:
            contract = Crypto(crypto_map[symbol_upper], exchange='PAXOS', currency='USD')
        
        # Futures
        elif symbol_upper in ['ES', 'NQ']:
            contract = Future(symbol_upper, exchange='CME', currency='USD', lastTradeDateOrContractMonth='202506')
        elif symbol_upper in ['GC', 'SI']:
            contract = Future(symbol_upper, exchange='COMEX', currency='USD', lastTradeDateOrContractMonth='202506')
        elif symbol_upper in ['CL', 'NG']:
            contract = Future(symbol_upper, exchange='NYMEX', currency='USD', lastTradeDateOrContractMonth='202506')
        
        if not contract:
            return None
        
        ib = IB()
        try:
            ib.connect('127.0.0.1', 7497, clientId=np.random.randint(1000, 50000))
        except:
            return None
        
        try:
            bars = ib.reqHistoricalData(
                contract,
                endDateTime='',
                durationStr=f'{days} D',
                barSizeSetting='1 day',
                whatToShow='MIDPOINT',
                useRTH=False,
                formatDate=2
            )
        except:
            ib.disconnect()
            return None
        
        ib.disconnect()
        
        if not bars:
            return None
        
        df = util.df(bars)
        df.set_index('date', inplace=True)
        df.index = pd.to_datetime(df.index)
        
        return df
    except:
        return None


def prepare_data_from_df(df, symbol=None, use_ibkr=True):
    """Prepare V3 format data from DataFrame."""
    if len(df) < 50:
        return None
    
    # Handle both Yahoo and IBKR column names
    if 'Close' in df.columns:
        closes = df['Close'].values
        highs = df['High'].values
        lows = df['Low'].values
    else:
        closes = df['close'].values
        highs = df['high'].values
        lows = df['low'].values
    
    # Calculate FVGs
    bullish_fvgs = []
    bearish_fvgs = []
    for i in range(3, len(df)):
        if lows[i] > highs[i-2]:
            bullish_fvgs.append({'idx': i, 'mid': (highs[i-2] + lows[i]) / 2, 'high': lows[i]})
        if highs[i] < lows[i-2]:
            bearish_fvgs.append({'idx': i, 'mid': (highs[i] + lows[i-2]) / 2, 'low': highs[i]})
    
    # Calculate HTF trend from daily data (like original V3)
    htf_trend = np.zeros(len(df))
    if symbol:
        # Try to get daily data
        df_daily = None
        if use_ibkr:
            df_daily = fetch_daily_data(symbol, days=60)
        
        if df_daily is None or len(df_daily) < 5:
            # Fallback to Yahoo for daily data
            try:
                yahoo_symbol = symbol.replace('USD', '-USD') if 'USD' in symbol else symbol
                df_daily = yf.Ticker(yahoo_symbol).history(period="60d", interval="1d")
            except:
                df_daily = None
        
        if df_daily is not None and len(df_daily) >= 5:
            print(f"    Using daily data for HTF trend: {len(df_daily)} days")
            # Handle column names
            if 'Close' in df_daily.columns:
                daily_highs = df_daily['High'].values
                daily_lows = df_daily['Low'].values
            else:
                daily_highs = df_daily['high'].values
                daily_lows = df_daily['low'].values
            
            # Calculate daily trend
            htf = []
            for i in range(1, len(df_daily)):
                if daily_highs[i] > np.max(daily_highs[max(0,i-5):i]) and daily_lows[i] > np.min(daily_lows[max(0,i-5):i]):
                    htf.append(1)
                elif daily_highs[i] < np.max(daily_highs[max(0,i-5):i]) and daily_lows[i] < np.min(daily_lows[max(0,i-5):i]):
                    htf.append(-1)
                else:
                    htf.append(0)
            
            # Map daily trend to hourly bars
            df_daily_index = pd.DatetimeIndex(df_daily.index).tz_localize(None)
            df_index = pd.DatetimeIndex(df.index).tz_localize(None)
            for i in range(len(df)):
                bar_time = df_index[i]
                for j in range(len(df_daily) - 1, -1, -1):
                    if df_daily_index[j] <= bar_time:
                        htf_trend[i] = htf[j] if j < len(htf) else 0
                        break
        else:
            print(f"    Warning: No daily data available for HTF trend")
    
    # Calculate LTF trend from intraday data
    ltf_trend = np.zeros(len(df))
    for i in range(20, len(df)):
        rh = np.max(highs[max(0,i-20):i])
        rl = np.min(lows[max(0,i-20):i])
        if rh > highs[i-5] and rl > lows[i-5]:
            ltf_trend[i] = 1
        elif rh < highs[i-5] and rl < lows[i-5]:
            ltf_trend[i] = -1
    
    # Kill zone (V3: strict kill zone only in New York time)
    # Convert timestamps to America/New_York timezone
    try:
        import pytz
        ny_tz = pytz.timezone('America/New_York')
        
        # Ensure index is timezone-aware, then convert to NY
        if df.index.tz is None:
            # Assume UTC if no timezone
            ny_index = df.index.tz_localize('UTC').tz_convert(ny_tz)
        else:
            # Convert to NY time
            ny_index = df.index.tz_convert(ny_tz)
        
        hours = ny_index.hour + ny_index.minute / 60.0  # Include minutes for 13.5 (1:30 PM)
    except ImportError:
        # Fallback: assume data is already in NY time or use raw hours
        hours = pd.to_datetime(df.index).hour + pd.to_datetime(df.index).minute / 60.0
    
    kill_zone = np.zeros(len(df), dtype=bool)
    for i in range(len(hours)):
        h = hours[i]
        # Kill zones: 1:00-5:00 AM, 7:00 AM-12:00 PM, 1:30-4:00 PM (NY time)
        kill_zone[i] = (1 <= h < 5) or (7 <= h < 12) or (13.5 <= h < 16)
    
    return {
        'df': df, 'highs': highs, 'lows': lows, 'closes': closes,
        'bullish_fvgs': bullish_fvgs, 'bearish_fvgs': bearish_fvgs,
        'htf_trend': htf_trend, 'ltf_trend': ltf_trend,
        'kill_zone': kill_zone
    }


# Run V3 backtest
symbols = ['BTCUSD', 'ETHUSD', 'SOLUSD', 'LTCUSD', 'LINKUSD', 'UNIUSD', 'ES', 'NQ', 'GC', 'SI', 'NG', 'CL']

results = run_v3_backtest_ibkr(symbols, days=180, initial_capital=10000, risk_per_trade=0.02, use_ibkr=True)

if results:
    output_file = '/Users/mac/Documents/Algot/v3_backtest_ibkr_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*80}")
    print("V3 BACKTEST SUMMARY - IBKR")
    print(f"{'='*80}")
    print(f"Initial Capital: ${results['summary']['initial_capital']:,}")
    print(f"Final Capital: ${results['summary']['final_capital']:,}")
    print(f"Total PnL: ${results['summary']['total_pnl']:,.2f}")
    print(f"Total Return: {results['summary']['total_return_pct']:.1f}%")
    print(f"Max Drawdown: {results['summary']['max_drawdown_pct']:.1f}%")
    print(f"\nTotal Trades: {results['summary']['total_trades']}")
    print(f"Win Rate: {results['summary']['win_rate']:.1f}%")
    print(f"Wins: {results['summary']['wins']} | Losses: {results['summary']['losses']}")
    print(f"\nResults saved to: {output_file}")
    print(f"{'='*80}")
    
    print("\nSymbol Breakdown:")
    print(f"{'Symbol':<10} {'Trades':<8} {'Wins':<6} {'Losses':<8} {'Win%':<8} {'PnL':<15}")
    print("-" * 60)
    for symbol, stats in sorted(results['symbol_stats'].items(), key=lambda x: x[1]['pnl'], reverse=True):
        win_pct = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
        print(f"{symbol:<10} {stats['trades']:<8} {stats['wins']:<6} {stats['losses']:<8} {win_pct:<8.1f} ${stats['pnl']:<14.2f}")
    
    print("\nJSON SUMMARY:")
    summary_json = {
        'config': results['backtest_config'],
        'summary': results['summary'],
        'symbol_stats': results['symbol_stats']
    }
    print(json.dumps(summary_json, indent=2))
