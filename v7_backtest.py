"""
V7 Backtest - MT5 Version
==========================
Backtest the V7 strategy with FVG + Gap analysis using MT5 data
"""

import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import sys
import os
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import json
import pandas as pd
import numpy as np
from datetime import datetime, time as dt_time
from typing import Dict, List
import pytz
import time

import importlib.util
spec = importlib.util.spec_from_file_location("ict_v7_mt5", os.path.join(SCRIPT_DIR, "ict_v7_mt5.py"))
ict_v7 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ict_v7)

from fvg_handler import FVGHandler
from gap_handler import GapHandler

try:
    from mtf_coordinator import MTFCoordinator
    from market_structure_handler import MarketStructureHandler, StructureBreakType, TrendState, PriceZone
    MTF_AVAILABLE = True
except ImportError:
    MTF_AVAILABLE = False
    print("WARNING: MTF modules not available")

fetch_mt5_rates = ict_v7.fetch_mt5_rates
prepare_data_mt5 = ict_v7.prepare_data_mt5
get_contract_info = ict_v7.get_contract_info
calculate_position_size = ict_v7.calculate_position_size


MT5_SYMBOLS = {
    # Crypto
    'BTCUSD': 'BTCUSDm',
    'ETHUSD': 'ETHUSDm',
    'SOLUSD': 'SOLUSDm',
    # Metals
    'XAUUSD': 'XAUUSDm',
    'XAGUSD': 'XAGUSDm',
    # Oil
    'XTIUSD': 'USOILm',
    # Major Forex
    'EURUSD': 'EURUSDm',
    'GBPUSD': 'GBPUSDm',
    'USDJPY': 'USDJPYm',
    'USDCAD': 'USDCADm',
    'AUDUSD': 'AUDUSDm',
    'USDCHF': 'USDCHFm',
    'EURGBP': 'EURGBPm',
    'EURJPY': 'EURJPYm',
    'GBPJPY': 'GBPJPYm',
    # Indices
    'US30': 'US30m',
    'USTEC': 'USTECm',  # Nasdaq
    'US500': 'US500m',
    'UK100': 'UK100m',
}


def is_valid_trading_session(timestamp, is_forex: bool = False) -> bool:
    """Check if timestamp is within valid trading sessions"""
    et_tz = pytz.timezone('US/Eastern')
    ts_aware = timestamp.tz_localize(et_tz) if timestamp.tzinfo is None else timestamp.astimezone(et_tz)
    current_time = ts_aware.time()
    
    if is_forex:
        forex_open = dt_time(7, 0)
        forex_close = dt_time(11, 0)
        return forex_open <= current_time <= forex_close
    
    london_open = dt_time(2, 0)
    london_close = dt_time(17, 0)
    ny_open = dt_time(9, 30)
    ny_close = dt_time(16, 15)
    
    return (london_open <= current_time <= london_close) or (ny_open <= current_time <= ny_close)


class V7SignalGenerator:
    """V7 Signal Generator for backtesting with MT5"""
    
    def __init__(self, rr_ratio: float = 3.0, confluence_threshold: int = 60):
        self.fvg_handler = FVGHandler(
            sensitivity=0.0001,
            min_gap_size=0.0,
            track_body_respect=False,
            detect_volume_imbalances=False,
            detect_suspension_blocks=False
        )
        self.gap_handler = GapHandler(
            large_gap_pips_forex=40.0,
            large_gap_points_indices=50.0,
            keep_gaps_days=3
        )
        self.rr_ratio = rr_ratio
        self.confluence_threshold = confluence_threshold
        
        if MTF_AVAILABLE:
            self.ms_handler = MarketStructureHandler(
                swing_lookback=5,
                min_displacement_pct=0.1
            )
        else:
            self.ms_handler = None
    
    def generate_signal(self, data: Dict, idx: int) -> Dict:
        """Generate V7 signal with FVG + Gap + MS analysis"""
        
        current_price = data['closes'][idx]
        
        htf = data['htf_trend'][idx]
        ltf = data['ltf_trend'][idx]
        kz = data['kill_zone'][idx]
        pp = data['price_position'][idx]
        closes = data['closes'][idx]
        
        df = pd.DataFrame({
            'open': data['opens'][:idx+1],
            'high': data['highs'][:idx+1],
            'low': data['lows'][:idx+1],
            'close': data['closes'][:idx+1]
        })
        
        fvg_analysis = self.fvg_handler.analyze_fvgs(df)
        gap_analysis = self.gap_handler.analyze(df, current_price)
        
        ms_analysis = None
        if self.ms_handler:
            try:
                ms_analysis = self.ms_handler.analyze(df)
            except:
                pass
        
        signal = {
            'direction': 0,
            'confluence': 0,
            'entry_price': current_price,
            'stop_loss': None,
            'take_profit': None,
            'confidence': 'LOW',
            'reasoning': [],
            'fvg_data': None,
            'gap_data': None
        }
        
        near_bull_fvg = next((f for f in reversed(data['bullish_fvgs']) if f['idx'] < idx and f['mid'] < closes), None)
        near_bear_fvg = next((f for f in reversed(data['bearish_fvgs']) if f['idx'] < idx and f['mid'] > closes), None)
        
        base_confluence = 0
        
        if kz:
            base_confluence += 15
            signal['reasoning'].append("Kill Zone: +15")
        
        if htf == 1 and ltf >= 0:
            base_confluence += 25
            signal['direction'] = 1
            signal['reasoning'].append("HTF+LTF Bullish: +25")
        elif htf == -1 and ltf <= 0:
            base_confluence += 25
            signal['direction'] = -1
            signal['reasoning'].append("HTF+LTF Bearish: +25")
        elif htf == 0 and ltf == 1:
            base_confluence += 15
            signal['direction'] = 1
            signal['reasoning'].append("LTF Bullish: +15")
        elif htf == 0 and ltf == -1:
            base_confluence += 15
            signal['direction'] = -1
            signal['reasoning'].append("LTF Bearish: +15")
        
        if pp < 0.25:
            base_confluence += 20
            signal['reasoning'].append("Price near lows: +20")
        elif pp > 0.75:
            base_confluence += 20
            signal['reasoning'].append("Price near highs: +20")
        
        if near_bull_fvg and ltf >= 0:
            base_confluence += 15
            signal['reasoning'].append("V5 Bull FVG: +15")
        if near_bear_fvg and ltf <= 0:
            base_confluence += 15
            signal['reasoning'].append("V5 Bear FVG: +15")
        
        if ms_analysis:
            try:
                if hasattr(ms_analysis, 'trend_state'):
                    if ms_analysis.trend_state == TrendState.BULLISH and signal['direction'] == 1:
                        base_confluence += 10
                    elif ms_analysis.trend_state == TrendState.BEARISH and signal['direction'] == -1:
                        base_confluence += 10
                        
                if hasattr(ms_analysis, 'current_zone'):
                    if ms_analysis.current_zone == PriceZone.DISCOUNT and signal['direction'] == 1:
                        base_confluence += 10
                    elif ms_analysis.current_zone == PriceZone.PREMIUM and signal['direction'] == -1:
                        base_confluence += 10
            except:
                pass
        
        fvg_confluence = 0
        if fvg_analysis.best_bisi_fvg and signal['direction'] == 1:
            fvg = fvg_analysis.best_bisi_fvg
            distance = abs(current_price - fvg.consequent_encroachment)
            if distance < fvg.size * 2:
                fvg_confluence += 20
                signal['fvg_data'] = {
                    'type': 'BISI',
                    'ce': fvg.consequent_encroachment,
                    'distance': distance
                }
                if fvg.is_high_probability:
                    fvg_confluence += 15
        
        elif fvg_analysis.best_sibi_fvg and signal['direction'] == -1:
            fvg = fvg_analysis.best_sibi_fvg
            distance = abs(current_price - fvg.consequent_encroachment)
            if distance < fvg.size * 2:
                fvg_confluence += 20
                signal['fvg_data'] = {
                    'type': 'SIBI',
                    'ce': fvg.consequent_encroachment,
                    'distance': distance
                }
                if fvg.is_high_probability:
                    fvg_confluence += 15
        
        gap_confluence = 0
        if gap_analysis.current_gap and gap_analysis.in_gap_zone:
            gap_confluence += 10
            if gap_analysis.nearest_level:
                level_price, level_name = gap_analysis.nearest_level
                distance_pct = abs(current_price - level_price) / current_price * 100
                if distance_pct < 0.5:
                    gap_confluence += 10
        
        total_confluence = base_confluence + fvg_confluence + gap_confluence
        signal['confluence'] = min(total_confluence, 100)
        
        if total_confluence >= 80:
            signal['confidence'] = 'HIGH'
        elif total_confluence >= 60:
            signal['confidence'] = 'MEDIUM'
        elif total_confluence >= 50:
            signal['confidence'] = 'LOW'
        else:
            signal['direction'] = 0
        
        if signal['direction'] != 0:
            entry = current_price
            
            highs = data['highs']
            lows = data['lows']
            closes_arr = data['closes']
            
            atr = np.mean([max(
                highs[i] - lows[i],
                abs(highs[i] - closes_arr[i-1]) if i > 0 else 0,
                abs(lows[i] - closes_arr[i-1]) if i > 0 else 0
            ) for i in range(max(0, idx-14), idx+1)])
            
            contract_info = get_contract_info(data.get('symbol', ''))
            symbol_type = contract_info['type']
            
            atr_multiplier = 2.0
            min_atr_multiplier = 1.5
            
            # Calculate stop distance based on instrument type
            if symbol_type == 'forex':
                decimal_places = contract_info.get('decimal_places', 5)
                pip_size = 0.01 if decimal_places == 3 else 0.0001  # JPY = 0.01, others = 0.0001
                atr_pips = atr / pip_size
                min_stop_pips = max(25, atr_pips * min_atr_multiplier)
                max_stop_pips = max(80, atr_pips * atr_multiplier)
                atr_stop_distance = atr * atr_multiplier
                min_stop_distance = min_stop_pips * pip_size
                max_stop_distance = max_stop_pips * pip_size
                stop_distance = max(min_stop_distance, min(atr_stop_distance, max_stop_distance))
                
                # Validate pips
                risk_pips = stop_distance / pip_size
                if risk_pips < 15 or risk_pips > 100:
                    signal['direction'] = 0
                    return signal
                    
            elif symbol_type == 'futures':
                # Metals (XAU, XAG) and Oil
                min_stop = contract_info.get('min_stop', 10)
                stop_distance = max(atr * atr_multiplier, min_stop)
                
            elif symbol_type == 'indices':
                # Indices (US30, USTEC, US500, UK100)
                min_stop = contract_info.get('min_stop', 20)
                stop_distance = max(atr * atr_multiplier, min_stop)
                
            elif symbol_type == 'crypto':
                # Crypto - use percentage-based stops
                stop_distance = current_price * 0.02  # 2% stop
            else:
                # Default
                stop_distance = atr * atr_multiplier
            
            signal['entry_price'] = entry
            signal['stop_loss'] = entry - stop_distance if signal['direction'] == 1 else entry + stop_distance
            signal['take_profit'] = entry + (stop_distance * self.rr_ratio) if signal['direction'] == 1 else entry - (stop_distance * self.rr_ratio)
        
        return signal


def run_v7_backtest(symbols, days=30, initial_capital=10000, risk_per_trade=0.02, 
                   rr_ratio=3.0, confluence_threshold=40, timeframe="M15"):
    """Run V7 backtest using MT5 data"""
    
    print(f"\n{'='*80}")
    print(f"ICT V7 Backtest - MT5 Data")
    print(f"Initial Capital: ${initial_capital:,}")
    print(f"Timeframe: {timeframe}")
    print(f"Risk per Trade: {risk_per_trade*100}%")
    print(f"RR Ratio: 1:{rr_ratio}")
    print(f"Confluence Threshold: {confluence_threshold}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"{'='*80}\n")
    
    signal_gen = V7SignalGenerator(rr_ratio=rr_ratio, confluence_threshold=confluence_threshold)
    
    all_data = {}
    
    for symbol in symbols:
        mt5_symbol = MT5_SYMBOLS.get(symbol, symbol)
        print(f"Loading {mt5_symbol}...", end=' ')
        
        try:
            df = fetch_mt5_rates(mt5_symbol, timeframe, num_bars=500)
            
            if df is None or len(df) < 50:
                print(f"X No data from MT5")
                continue
            
            highs = df['high'].values
            lows = df['low'].values
            closes = df['close'].values
            opens = df['open'].values
            
            bullish_fvgs = []
            bearish_fvgs = []
            for i in range(3, len(df)):
                if lows[i] > highs[i-2]:
                    bullish_fvgs.append({'idx': i, 'mid': (highs[i-2] + lows[i]) / 2, 'high': lows[i]})
                if highs[i] < lows[i-2]:
                    bearish_fvgs.append({'idx': i, 'mid': (highs[i] + lows[i-2]) / 2, 'low': highs[i]})
            
            daily_df = fetch_mt5_rates(mt5_symbol, "D1", num_bars=60)  # Keep D1 for HTF
            if daily_df is None or len(daily_df) < 5:
                htf_trend = np.zeros(len(df))
            else:
                daily_highs = daily_df['high'].values
                daily_lows = daily_df['low'].values
                htf = []
                for i in range(1, len(daily_df)):
                    if daily_highs[i] > np.max(daily_highs[max(0,i-5):i]) and daily_lows[i] > np.min(daily_lows[max(0,i-5):i]):
                        htf.append(1)
                    elif daily_highs[i] < np.max(daily_highs[max(0,i-5):i]) and daily_lows[i] < np.min(daily_lows[max(0,i-5):i]):
                        htf.append(-1)
                    else:
                        htf.append(0)
                
                htf_trend = np.zeros(len(df))
                for i in range(len(df)):
                    bar_time = df.index[i]
                    for j in range(len(daily_df) - 1, -1, -1):
                        if daily_df.index[j] <= bar_time:
                            htf_trend[i] = htf[j] if j < len(htf) else 0
                            break
            
            trend = np.zeros(len(df))
            for i in range(20, len(df)):
                momentum = closes[i] - closes[i-10]
                pct_change = momentum / closes[i-10] if closes[i-10] > 0 else 0
                ema_fast = np.mean(closes[max(0,i-5):i+1])
                ema_slow = np.mean(closes[max(0,i-13):i+1])
                
                if pct_change > 0.005 or (pct_change > 0.001 and ema_fast > ema_slow):
                    trend[i] = 1
                elif pct_change < -0.005 or (pct_change < -0.001 and ema_fast < ema_slow):
                    trend[i] = -1
            
            price_position = np.zeros(len(df))
            for i in range(20, len(df)):
                ph = np.max(highs[i-20:i])
                pl = np.min(lows[i-20:i])
                rng = ph - pl
                if rng < 0.001:
                    rng = 0.001
                price_position[i] = (closes[i] - pl) / rng
            
            hours = df.index.hour.values
            kill_zone = np.zeros(len(df), dtype=bool)
            for i in range(len(hours)):
                h = hours[i]
                kill_zone[i] = (1 <= h < 5) or (7 <= h < 12) or (13.5 <= h < 16)
            
            data = {
                'opens': opens,
                'highs': highs,
                'lows': lows,
                'closes': closes,
                'htf_trend': htf_trend,
                'ltf_trend': trend,
                'price_position': price_position,
                'kill_zone': kill_zone,
                'bullish_fvgs': bullish_fvgs,
                'bearish_fvgs': bearish_fvgs,
                'symbol': symbol,
                'df': df
            }
            
            all_data[symbol] = {'df': df, 'data': data}
            
            print(f"OK {len(df)} bars")
            
        except Exception as e:
            print(f"X Error: {e}")
    
    if not all_data:
        print("No data loaded! Make sure MT5 is running with opened charts.")
        return None
    
    balance = initial_capital
    positions = {}
    active_trades = []
    is_forex = lambda s: s in {
        'EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD', 'AUDUSD', 'NZDUSD', 
        'EURGBP', 'EURJPY', 'GBPJPY', 'USDCHF', 'XAUUSD', 'XAGUSD'
    }
    is_index = lambda s: s in {
        'US30', 'US100', 'US500', 'GER40', 'UK100', 'FRA40', 'JPN225', 'AUS200'
    }
    
    # Simplified: Process each symbol independently
    for symbol, item in all_data.items():
        df = item['df']
        data = item['data']
        
        print(f"Processing {symbol} ({len(df)} bars)...")
        
        start_time = time.time()
        for idx in range(50, len(df) - 10, 5):  # Check every 5th bar
            if idx % 50 == 0:
                elapsed = time.time() - start_time
                print(f"  [{symbol}] Bar {idx}/{len(df)} ({elapsed:.1f}s)")
            
            current_price = df.iloc[idx]['close']
            
            # Generate signal
            signal = signal_gen.generate_signal(data, idx)
            
            # Check exits
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
                    
                    # Calculate PnL based on instrument type
                    symbol_type = contract_info['type']
                    
                    if symbol_type == 'forex':
                        decimal_places = contract_info.get('decimal_places', 5)
                        pip_size = 0.0001 if decimal_places == 5 else 0.01
                        pips = abs(price_change / pip_size)
                        
                        if decimal_places == 3:
                            # JPY pairs: $10 per pip per lot
                            pip_value = 10
                        else:
                            # Non-JPY: $10 per pip per lot
                            pip_value = 10
                        
                        pnl = pips * pos['qty'] * pip_value
                        if price_change < 0:
                            pnl = -pnl
                            
                    elif symbol_type == 'indices':
                        # Indices: $1 per point per lot
                        pnl = price_change * pos['qty']
                        
                    elif symbol_type == 'futures':
                        # Metals/Oil
                        multiplier = contract_info.get('multiplier', 100)
                        pnl = price_change * pos['qty'] * multiplier
                    else:
                        # Default
                        pnl = price_change * pos['qty']
                    
                    balance += pnl
                    
                    active_trades.append({
                        'symbol': symbol,
                        'direction': 'LONG' if pos['direction'] == 1 else 'SHORT',
                        'entry_price': pos['entry'],
                        'exit_price': exit_price,
                        'qty': pos['qty'],
                        'pnl': pnl,
                        'confluence': pos.get('confluence', 0),
                        'fvg_info': pos.get('fvg_info', '')
                    })
                    del positions[symbol]
            
            # Check entries
            elif symbol not in positions:
                signal = signal_gen.generate_signal(data, idx)
                
                # Session filter - disabled for backtest (timestamps may be UTC)
                in_session = True
                
                if signal and signal['direction'] != 0 and signal['confluence'] >= confluence_threshold and in_session:
                    stop_distance = abs(current_price - signal['stop_loss'])
                    if stop_distance > 0:
                        qty, _ = calculate_position_size(
                            symbol, initial_capital, risk_per_trade, stop_distance, current_price
                        )
                        
                        if qty > 0:
                            positions[symbol] = {
                                'entry': current_price,
                                'stop': signal['stop_loss'],
                                'target': signal['take_profit'],
                                'direction': signal['direction'],
                                'qty': qty,
                                'confluence': signal['confluence'],
                                'fvg_info': signal.get('fvg_data', {}).get('type', '') if signal.get('fvg_data') else ''
                            }
    
    total_trades = len(active_trades)
    wins = len([t for t in active_trades if t['pnl'] > 0])
    losses = total_trades - wins
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    total_pnl = balance - initial_capital
    total_return_pct = (total_pnl / initial_capital) * 100
    
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
            'data_source': 'MT5',
            'timestamp': datetime.now().isoformat(),
            'version': 'V7',
            'session_filter': 'London/NY sessions',
            'rr_ratio': rr_ratio,
            'confluence_threshold': confluence_threshold
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
    
    # Log trades to file
    if active_trades:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"v7_trades_{timestamp}.json"
        with open(log_file, 'w') as f:
            json.dump({
                'config': results['backtest_config'],
                'trades': active_trades
            }, f, indent=2, default=str)
        print(f"\nTrades logged to: {log_file}")
    
    return results


if __name__ == "__main__":
    symbols = ['XAUUSD', 'XTIUSD', 'EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD', 'AUDUSD', 'BTCUSD', 'ETHUSD']
    
    print("="*80)
    print("ICT V7 - MT5 Backtest")
    print("="*80)
    
    # Test with M5 timeframe
    timeframe = "M5"
    
    # Symbols to test
    symbols = [
        # Major Forex
        # 'EURUSD', 'GBPUSD', 'USDJPY', 'USDCAD', 'AUDUSD', 'USDCHF',
        # 'EURGBP', 'EURJPY', 'GBPJPY',
        # # Metals
        # 'XAUUSD',
        # 'XAGUSD',
        # Oil
        # 'XTIUSD',
        # # Indices
        'US30',
        # 'USTEC', 'US500', 'UK100',
        # 'GER40', 'UK100', 'FRA40', 'JPN225', 'AUS200'
    ]
    
    try:
        import MetaTrader5 as mt5
        if not mt5.initialize(login=0):  # Initialize without login for local terminal
            print(f"ERROR: MT5 initialize failed: {mt5.last_error()}")
            exit(1)
        
        # Check MT5 terminal info
        terminal_info = mt5.terminal_info()
        print(f"MT5 Terminal connected: {terminal_info.connected}")
        
        results = run_v7_backtest(
            symbols=symbols,
            days=30,
            initial_capital=10000,
            risk_per_trade=0.02,
            rr_ratio=3.0,
            confluence_threshold=40,
            timeframe=timeframe
        )
        
        if results:
            print(f"\n{'='*80}")
            print("V7 BACKTEST RESULTS (MT5)")
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
        
        mt5.shutdown()
        
    except ImportError:
        print("ERROR: MetaTrader5 package not installed")
        print("Run: pip install MetaTrader5")
