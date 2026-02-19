"""
ICT V6 Trading Bot - Combined FVG + Gap Analysis
=================================================
Combines V5 live trading with comprehensive FVG and Gap handlers
"""

import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import sys
import os

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

# Import V5 components using relative path
v5_path = os.path.join(SCRIPT_DIR, 'ict_v5_ibkr.py')
with open(v5_path, 'r') as f:
    v5_code = f.read().split('if __name__')[0]
    exec(v5_code)

# Import FVG and Gap handlers
from fvg_handler import FVGHandler, FairValueGap, FVGStatus
from gap_handler import GapHandler, Gap, GapType, GapDirection

import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple


class V6SignalGenerator:
    """Enhanced signal generator combining V5 ICT with FVG and Gap analysis"""
    
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
        self.last_analysis = {}
    
    def analyze_symbol(self, symbol: str, data: Dict, current_price: float) -> Dict:
        """Comprehensive analysis combining V5, FVG, and Gap signals"""
        
        # Get V5 base signal
        idx = len(data['closes']) - 1
        v5_signal = get_signal(data, idx)
        
        # Prepare DataFrame for FVG/Gap analysis
        df = pd.DataFrame({
            'open': data['opens'],
            'high': data['highs'],
            'low': data['lows'],
            'close': data['closes']
        })
        
        # FVG Analysis
        fvgs = self.fvg_handler.detect_all_fvgs(df)
        fvg_analysis = self.fvg_handler.analyze_fvgs(df)
        
        # Gap Analysis
        gap_analysis = self.gap_handler.analyze(df, current_price)
        
        # Combine signals
        combined_signal = self._combine_signals(
            symbol, v5_signal, fvg_analysis, gap_analysis, 
            current_price, data, idx
        )
        
        # Store analysis
        self.last_analysis[symbol] = {
            'timestamp': datetime.now().isoformat(),
            'v5_confluence': v5_signal['confluence'] if v5_signal else 0,
            'fvg_count': len(fvgs),
            'active_fvgs': len(fvg_analysis.active_fvgs),
            'high_prob_fvgs': len(fvg_analysis.high_prob_fvgs),
            'gap_levels': len(gap_analysis.all_levels)
        }
        
        return combined_signal
    
    def _combine_signals(self, symbol: str, v5_signal: Optional[Dict], 
                        fvg_analysis, gap_analysis, current_price: float,
                        data: Dict, idx: int) -> Dict:
        """Combine V5, FVG, and Gap signals into unified trading signal"""
        
        signal = {
            'symbol': symbol,
            'direction': 0,
            'confluence': 0,
            'entry_price': current_price,
            'stop_loss': None,
            'take_profit': None,
            'confidence': 'LOW',
            'reasoning': [],
            'v5_signal': v5_signal,
            'fvg_data': None,
            'gap_data': None
        }
        
        # Base confluence from V5
        base_confluence = v5_signal['confluence'] if v5_signal else 0
        signal['confluence'] = base_confluence
        
        if v5_signal:
            signal['direction'] = v5_signal['direction']
            signal['reasoning'].append(f"V5 Signal: Confluence {base_confluence}/100")
        
        # Add FVG confluence
        fvg_confluence = 0
        if fvg_analysis.best_bisi_fvg and signal['direction'] == 1:
            # Check if price is near FVG
            fvg = fvg_analysis.best_bisi_fvg
            distance = abs(current_price - fvg.consequent_encroachment)
            if distance < fvg.size * 2:  # Within 2x FVG size
                fvg_confluence += 20
                signal['fvg_data'] = {
                    'type': 'BISI',
                    'ce': fvg.consequent_encroachment,
                    'distance': distance
                }
                signal['reasoning'].append(f"FVG BISI at {fvg.consequent_encroachment:.4f}")
                
                if fvg.is_high_probability:
                    fvg_confluence += 15
                    signal['reasoning'].append("High Probability FVG")
        
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
                signal['reasoning'].append(f"FVG SIBI at {fvg.consequent_encroachment:.4f}")
                
                if fvg.is_high_probability:
                    fvg_confluence += 15
                    signal['reasoning'].append("High Probability FVG")
        
        # Add Gap confluence
        gap_confluence = 0
        if gap_analysis.current_gap:
            gap = gap_analysis.current_gap
            
            # Check if we're in a gap zone
            if gap_analysis.in_gap_zone:
                gap_confluence += 10
                signal['reasoning'].append(f"In {gap.gap_type.value} gap zone")
                
                # Check CE proximity
                if gap.quadrants:
                    ce_distance = abs(current_price - gap.quadrants.ce)
                    if ce_distance < (gap.quadrants.range_size * 0.1):
                        gap_confluence += 15
                        signal['reasoning'].append("At Gap CE (50%)")
                        signal['gap_data'] = {
                            'type': gap.gap_type.value,
                            'ce': gap.quadrants.ce,
                            'direction': gap.direction.value
                        }
        
        # Check nearest gap level
        if gap_analysis.nearest_level:
            level_price, level_name = gap_analysis.nearest_level
            distance_pct = abs(current_price - level_price) / current_price * 100
            if distance_pct < 0.5:  # Within 0.5%
                gap_confluence += 10
                signal['reasoning'].append(f"Near {level_name}")
        
        # Calculate total confluence
        total_confluence = base_confluence + fvg_confluence + gap_confluence
        signal['confluence'] = min(total_confluence, 100)
        
        # Set confidence level
        if total_confluence >= 80:
            signal['confidence'] = 'HIGH'
        elif total_confluence >= 60:
            signal['confidence'] = 'MEDIUM'
        elif total_confluence >= 40:
            signal['confidence'] = 'LOW'
        else:
            signal['direction'] = 0  # No trade
        
        # Calculate stop/target if we have a signal
        if signal['direction'] != 0:
            htf = data['htf_trend'][idx]
            ltf = data['ltf_trend'][idx]
            
            # Use FVG CE as entry refinement if available
            entry = current_price
            if signal['fvg_data']:
                # Blend current price with FVG CE
                fvg_ce = signal['fvg_data']['ce']
                if signal['direction'] == 1 and current_price > fvg_ce:
                    entry = fvg_ce  # Better entry at FVG CE for longs
                elif signal['direction'] == -1 and current_price < fvg_ce:
                    entry = fvg_ce  # Better entry at FVG CE for shorts
            
            signal['entry_price'] = entry
            
            # Set stop based on data
            if signal['direction'] == 1:
                signal['stop_loss'] = data['lows'][idx]
                risk = entry - signal['stop_loss']
                signal['take_profit'] = entry + (risk * 2)
            else:
                signal['stop_loss'] = data['highs'][idx]
                risk = signal['stop_loss'] - entry
                signal['take_profit'] = entry - (risk * 2)
        
        return signal


class V6LiveTrader(LiveTrader):
    """V6 Live Trader with FVG and Gap analysis"""
    
    def __init__(self, ib, symbols, risk_pct=0.02, poll_interval=30):
        super().__init__(ib, symbols, risk_pct, poll_interval)
        self.signal_generator = V6SignalGenerator()
    
    def _on_realtime_bar(self, symbol, bar):
        """Enhanced real-time bar handler with V6 signals"""
        if not self.historical_data.get(symbol):
            return
        
        data = self.historical_data[symbol]
        current_price = bar.close
        
        # Generate V6 signal
        signal = self.signal_generator.analyze_symbol(symbol, data, current_price)
        
        # Update Telegram with enhanced data
        if tn:
            try:
                tn.update_market_data(symbol, {
                    'price': current_price,
                    'confluence': signal['confluence'],
                    'confidence': signal['confidence'],
                    'v5_confluence': signal['v5_signal']['confluence'] if signal['v5_signal'] else 0,
                    'fvg_count': signal.get('fvg_data', {}).get('type', 'None'),
                    'last_update': datetime.now().isoformat()
                })
            except:
                pass
        
        # Check position or signal
        if symbol in self.positions:
            self._check_position_exit(symbol, current_price)
        else:
            # Use V6 signal for entry
            if signal['direction'] != 0 and signal['confluence'] >= 60:
                self._enter_trade_v6(symbol, signal, current_price)
    
    def _enter_trade_v6(self, symbol, signal, current_price):
        """Enter trade using V6 signal"""
        try:
            entry_price = signal['entry_price']
            stop_price = signal['stop_loss']
            target_price = signal['take_profit']
            
            stop_distance = abs(entry_price - stop_price)
            if stop_distance <= 0:
                return
            
            qty, _ = calculate_position_size(symbol, self.account_value, self.risk_pct, stop_distance, entry_price)
            if qty <= 0:
                return
            
            contract = get_ibkr_contract(symbol)
            bracket = place_bracket_order(self.ib, contract, signal['direction'], qty, stop_price, target_price)
            
            if bracket:
                parent_trade, sl_trade, tp_trade = bracket
                filled, fill_price, filled_qty = wait_for_fill(self.ib, parent_trade, timeout=10)
                
                if filled:
                    self.positions[symbol] = {
                        'entry': fill_price,
                        'stop': stop_price,
                        'target': target_price,
                        'direction': signal['direction'],
                        'qty': filled_qty,
                        'confluence': signal['confluence'],
                        'confidence': signal['confidence'],
                        'entry_time': datetime.now(),
                        'reasoning': signal['reasoning']
                    }
                    
                    self.active_orders[symbol] = {
                        'sl_order_id': sl_trade.order.orderId,
                        'tp_order_id': tp_trade.order.orderId
                    }
                    
                    print(f"[{symbol}] V6 ENTRY: {'LONG' if signal['direction'] == 1 else 'SHORT'} x {filled_qty} @ {fill_price:.4f}")
                    print(f"  Confidence: {signal['confidence']} | Confluence: {signal['confluence']}/100")
                    print(f"  Reasoning: {' | '.join(signal['reasoning'][:3])}")
                    
                    self.trade_count += 1
                    
                    if tn:
                        try:
                            tn.send_trade_entry(symbol, signal['direction'], filled_qty, 
                                              fill_price, signal['confluence'], target_price, stop_price)
                        except:
                            pass
        except Exception as e:
            print(f"[{symbol}] V6 Error entering trade: {e}")
    
    def poll_historical_symbols(self):
        """Poll with V6 signal generation"""
        current_time = time.time()
        
        for symbol in self.historical_symbols:
            last_poll = self.last_poll_time.get(symbol, 0)
            if current_time - last_poll < self.poll_interval:
                continue
            
            self.last_poll_time[symbol] = current_time
            
            try:
                data = prepare_data_ibkr(symbol, ib=self.ib, use_cache=True)
                if data is None or len(data.get('closes', [])) < 50:
                    continue
                
                self.historical_data[symbol] = data
                idx = len(data['closes']) - 1
                current_price = data['closes'][idx]
                
                # Generate V6 signal
                signal = self.signal_generator.analyze_symbol(symbol, data, current_price)
                
                if tn:
                    try:
                        htf = data['htf_trend'][idx]
                        ltf = data['ltf_trend'][idx]
                        tn.update_market_data(symbol, {
                            'price': current_price,
                            'htf_trend': htf,
                            'ltf_trend': ltf,
                            'confluence': signal['confluence'],
                            'confidence': signal['confidence']
                        })
                    except:
                        pass
                
                if symbol in self.positions:
                    self._check_position_exit(symbol, current_price)
                else:
                    if signal['direction'] != 0 and signal['confluence'] >= 60:
                        self._enter_trade_v6(symbol, signal, current_price)
                        
            except Exception as e:
                print(f"[{symbol}] Error polling: {e}")


def run_v6_trading(symbols, interval=30, risk_pct=0.02, port=7497):
    """Run V6 trading with FVG + Gap analysis"""
    try:
        from ib_insync import IB
    except ImportError:
        print("ERROR: ib_insync not installed")
        return
    
    ib = IB()
    
    try:
        ib.connect('127.0.0.1', port, clientId=1)
        print(f"V6 Bot Connected to IBKR (port {port})")
    except Exception as e:
        print(f"ERROR: {e}")
        return
    
    account_value = 100000
    try:
        for av in ib.accountValues():
            if av.tag == 'NetLiquidation' and av.currency == 'USD':
                account_value = float(av.value)
                print(f"Account: ${account_value:,.2f}")
                break
    except:
        pass
    
    print(f"\nICT V6 - FVG + Gap Trading")
    print(f"Symbols: {symbols}")
    print(f"Risk: {risk_pct*100}%")
    print("-" * 50)
    
    if tn:
        try:
            message = f"""
🚀 <b>V6 Trading Bot Started</b>
━━━━━━━━━━━━━━━━━━━━
<b>FVG + Gap Analysis</b>
<b>Symbols:</b> {', '.join(symbols)}
<b>Risk:</b> {risk_pct*100}%
<b>Account:</b> ${account_value:,.0f}
━━━━━━━━━━━━━━━━━━━━
"""
            tn.send_message(message)
        except:
            pass
    
    trader = V6LiveTrader(ib, symbols, risk_pct, poll_interval=interval)
    trader.account_value = account_value
    
    try:
        trader.start()
        
        iteration = 0
        while True:
            ib.sleep(1)
            iteration += 1
            
            if iteration % interval == 0:
                trader.poll_historical_symbols()
            
            if iteration % 60 == 0:
                try:
                    for av in ib.accountValues():
                        if av.tag == 'NetLiquidation' and av.currency == 'USD':
                            trader.account_value = float(av.value)
                            break
                except:
                    pass
                
    except KeyboardInterrupt:
        print("\n\nShutdown...")
    finally:
        trader.stop()
        ib.disconnect()
        print(f"\nTrades: {trader.trade_count} | Final: ${trader.account_value:,.2f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ICT V6 - FVG + Gap Trading')
    parser.add_argument("--symbols", default="BTCUSD,ETHUSD,SOLUSD,ES,NQ,GC", 
                        help="Comma-separated symbols")
    parser.add_argument("--interval", type=int, default=30)
    parser.add_argument("--risk", type=float, default=0.02)
    parser.add_argument("--port", type=int, default=7497)
    
    args = parser.parse_args()
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    run_v6_trading(symbols, args.interval, args.risk, args.port)
