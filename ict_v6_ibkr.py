"""
ICT V6 Trading Bot - Combined FVG + Gap Analysis
=================================================
Combines V5 live trading with comprehensive FVG and Gap handlers.
Full Telegram integration for commands and notifications.

Usage:
    python3 ict_v6_ibkr.py --symbols "BTCUSD,ETHUSD,GC,CL" --port 7497
"""

import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import sys
import os
import time

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

# Telegram notifications
try:
    import telegram_notify as tn
    # Initialize Telegram bot
    if tn and hasattr(tn, 'init_bot'):
        try:
            tn.init_bot()
            print("Telegram bot initialized")
        except Exception as e:
            print(f"Telegram bot init failed: {e}")
except ImportError:
    tn = None
    print("WARNING: telegram_notify not installed")


def is_trading_paused() -> bool:
    """Check if trading is paused via Telegram."""
    if tn and hasattr(tn, 'is_trading_paused'):
        return tn.is_trading_paused()
    return False


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
    """V6 Live Trader with FVG and Gap analysis and Telegram integration"""
    
    def __init__(self, ib, symbols, risk_pct=0.02, poll_interval=30):
        super().__init__(ib, symbols, risk_pct, poll_interval)
        self.signal_generator = V6SignalGenerator()
        self.mode = 'paper'  # Default mode, can be 'shadow', 'paper', 'live'
        self.use_rl = False  # V6 doesn't use RL
        self.port = 7497     # Default port
        self.daily_pnl = 0.0
        self.last_signals = {}  # Track last signals for Telegram /signals command
    
    def _on_realtime_bar(self, symbol, bar):
        """Enhanced real-time bar handler with V6 signals"""
        if not self.historical_data.get(symbol):
            return
        
        data = self.historical_data[symbol]
        current_price = bar.close
        
        # Generate V6 signal
        signal = self.signal_generator.analyze_symbol(symbol, data, current_price)
        
        # Store signal for Telegram /signals command
        if signal and signal['direction'] != 0:
            self.last_signals[symbol] = {
                'direction': signal['direction'],
                'confluence': signal['confluence'],
                'confidence': signal['confidence'],
                'pd_zone': signal.get('fvg_data', {}).get('type', '') or signal.get('gap_data', {}).get('type', ''),
                'entry': signal['entry_price'],
                'timestamp': datetime.now().isoformat()
            }
        
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
            self._check_position_exit_v6(symbol, current_price)
        else:
            # Use V6 signal for entry
            if signal['direction'] != 0 and signal['confluence'] >= 60:
                self._enter_trade_v6(symbol, signal, current_price)
    
    def _enter_trade_v6(self, symbol, signal, current_price):
        """Enter trade using V6 signal"""
        # Check if trading is paused via Telegram
        if is_trading_paused():
            print(f"[{symbol}] Signal found but trading is PAUSED")
            return
        
        try:
            entry_price = signal['entry_price']
            stop_price = signal['stop_loss']
            target_price = signal['take_profit']
            
            stop_distance = abs(entry_price - stop_price)
            if stop_distance <= 0:
                return
            
            qty, risk_amount = calculate_position_size(symbol, self.account_value, self.risk_pct, stop_distance, entry_price)
            if qty <= 0:
                return
            
            direction_str = 'LONG' if signal['direction'] == 1 else 'SHORT'
            
            # Get FVG/Gap info for logging
            fvg_info = signal.get('fvg_data', {})
            gap_info = signal.get('gap_data', {})
            pd_zone = None
            if fvg_info:
                pd_zone = f"FVG {fvg_info.get('type', '')}"
            elif gap_info:
                pd_zone = f"Gap {gap_info.get('type', '')}"
            
            # Shadow mode - just log, don't execute
            if self.mode == 'shadow':
                print(f"[{symbol}] V6 SHADOW SIGNAL: {direction_str} @ {current_price:.4f}")
                print(f"  Stop: {stop_price:.4f} | Target: {target_price:.4f}")
                print(f"  Confluence: {signal['confluence']}/100 | {pd_zone or 'No PD'}")
                print(f"  Risk: ${risk_amount:.2f} | Qty: {qty}")
                
                # Send signal alert to Telegram
                if tn:
                    try:
                        tn.send_signal_alert(
                            symbol=symbol,
                            direction=signal['direction'],
                            confluence=signal['confluence'],
                            pd_zone=pd_zone or '',
                            current_price=current_price
                        )
                    except:
                        pass
                return
            
            # Paper or Live mode - execute trade
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
                        'reasoning': signal['reasoning'],
                        'bars_held': 0,
                        'current_price': fill_price
                    }
                    
                    self.active_orders[symbol] = {
                        'sl_order_id': sl_trade.order.orderId,
                        'tp_order_id': tp_trade.order.orderId
                    }
                    
                    print(f"[{symbol}] V6 ENTRY: {direction_str} x {filled_qty} @ {fill_price:.4f}")
                    print(f"  Confidence: {signal['confidence']} | Confluence: {signal['confluence']}/100")
                    print(f"  Stop: {stop_price:.4f} | Target: {target_price:.4f}")
                    if signal['reasoning']:
                        print(f"  Reasoning: {' | '.join(signal['reasoning'][:3])}")
                    
                    self.trade_count += 1
                    
                    if tn:
                        try:
                            tn.send_trade_entry(
                                symbol, signal['direction'], filled_qty, 
                                fill_price, signal['confluence'], target_price, stop_price,
                                pd_zone=pd_zone
                            )
                        except Exception as e:
                            print(f"[{symbol}] Telegram notification error: {e}")
        except Exception as e:
            print(f"[{symbol}] V6 Error entering trade: {e}")
    
    def _check_position_exit_v6(self, symbol, current_price):
        """Check if position hit stop/target and handle exit with Telegram notification."""
        try:
            if symbol not in self.positions:
                return
            
            pos = self.positions[symbol]
            pos['bars_held'] = pos.get('bars_held', 0) + 1
            pos['current_price'] = current_price
            
            # Query IBKR to see if position still exists
            ibkr_pos = self.ib.positions()
            has_position = False
            
            for p in ibkr_pos:
                p_symbol = p.contract.symbol
                if p.contract.secType == 'CASH':
                    p_symbol = f"{p.contract.symbol}{p.contract.currency}"
                elif p.contract.secType == 'CRYPTO':
                    p_symbol = f"{p.contract.symbol}USD"
                
                if p_symbol.upper() == symbol.upper() and abs(p.position) > 0:
                    has_position = True
                    break
            
            if not has_position:
                # Position was closed by bracket order
                self._handle_position_closed_v6(symbol, current_price)
                
        except Exception as e:
            print(f"[{symbol}] Error checking position: {e}")
    
    def _handle_position_closed_v6(self, symbol, exit_price):
        """Handle position closure with PnL calculation and Telegram notification."""
        try:
            if symbol not in self.positions:
                return
            
            pos = self.positions[symbol]
            
            # Calculate PnL
            if pos['direction'] == 1:
                pnl = (exit_price - pos['entry']) * pos['qty']
            else:
                pnl = (pos['entry'] - exit_price) * pos['qty']
            
            # Apply contract multiplier for futures
            contract_info = get_contract_info(symbol)
            if contract_info['type'] == 'futures':
                pnl *= contract_info['multiplier']
            
            self.daily_pnl += pnl
            
            # Determine exit reason
            if pos['direction'] == 1:
                if exit_price <= pos['stop']:
                    exit_reason = 'stop'
                else:
                    exit_reason = 'target'
            else:
                if exit_price >= pos['stop']:
                    exit_reason = 'stop'
                else:
                    exit_reason = 'target'
            
            direction_str = 'LONG' if pos['direction'] == 1 else 'SHORT'
            pnl_str = f"+${pnl:.2f}" if pnl > 0 else f"-${abs(pnl):.2f}"
            
            print(f"[{symbol}] V6 EXIT ({exit_reason}): {direction_str} @ {exit_price:.4f} | P&L: {pnl_str}")
            print(f"  Daily P&L: ${self.daily_pnl:.2f}")
            
            # Telegram notification
            if tn:
                try:
                    tn.send_trade_exit(
                        symbol, pos['direction'], pnl, exit_reason,
                        pos['entry'], exit_price, pos.get('bars_held', 0)
                    )
                except:
                    pass
            
            # Clean up
            del self.positions[symbol]
            if symbol in self.active_orders:
                del self.active_orders[symbol]
        
        except Exception as e:
            print(f"[{symbol}] Error handling position close: {e}")
    
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
                
                # Store signal for Telegram /signals command
                if signal and signal['direction'] != 0:
                    self.last_signals[symbol] = {
                        'direction': signal['direction'],
                        'confluence': signal['confluence'],
                        'confidence': signal['confidence'],
                        'pd_zone': signal.get('fvg_data', {}).get('type', '') or signal.get('gap_data', {}).get('type', ''),
                        'entry': signal['entry_price'],
                        'timestamp': datetime.now().isoformat()
                    }
                
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
                    self._check_position_exit_v6(symbol, current_price)
                else:
                    if signal['direction'] != 0 and signal['confluence'] >= 60:
                        self._enter_trade_v6(symbol, signal, current_price)
                        
            except Exception as e:
                print(f"[{symbol}] Error polling: {e}")


def run_v6_trading(symbols, interval=30, risk_pct=0.02, port=7497, mode='paper'):
    """Run V6 trading with FVG + Gap analysis and Telegram integration"""
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
    print(f"Mode: {mode.upper()}")
    print(f"Symbols: {symbols}")
    print(f"Risk: {risk_pct*100}%")
    print("-" * 50)
    
    # Send startup notification to Telegram
    if tn:
        try:
            tn.send_startup(
                symbols=symbols,
                risk_pct=risk_pct,
                interval=interval,
                mode=f"V6 {mode.upper()}"
            )
        except Exception as e:
            print(f"Telegram startup notification failed: {e}")
    
    trader = V6LiveTrader(ib, symbols, risk_pct, poll_interval=interval)
    trader.account_value = account_value
    trader.mode = mode  # Add mode attribute for Telegram status
    trader.use_rl = False  # V6 doesn't use RL
    trader.port = port  # Store port for potential reconnection
    
    # Set live trader reference for Telegram commands
    if tn and hasattr(tn, 'set_live_trader'):
        tn.set_live_trader(trader)
        print("Live trader registered with Telegram")
    
    # Start Telegram polling for commands (non-blocking)
    if tn and hasattr(tn, 'start_polling_background'):
        try:
            tn.start_polling_background()
            print("Telegram command polling started")
        except Exception as e:
            print(f"Failed to start Telegram polling: {e}")
    
    try:
        trader.start()
        
        iteration = 0
        while True:
            ib.sleep(1)
            iteration += 1
            
            if iteration % interval == 0:
                trader.poll_historical_symbols()
            
            # Update account value every minute
            if iteration % 60 == 0:
                try:
                    for av in ib.accountValues():
                        if av.tag == 'NetLiquidation' and av.currency == 'USD':
                            trader.account_value = float(av.value)
                            break
                except:
                    pass
            
            # Check price alerts every 30 seconds
            if iteration % 30 == 0 and tn and hasattr(tn, 'check_price_alerts'):
                try:
                    # Build current prices dict from historical data
                    current_prices = {}
                    for symbol in symbols:
                        if hasattr(trader, 'historical_data') and symbol in trader.historical_data:
                            closes = trader.historical_data[symbol].get('closes', [])
                            if len(closes) > 0:
                                current_prices[symbol] = closes[-1]
                    if current_prices:
                        tn.check_price_alerts(current_prices)
                except Exception as e:
                    print(f"Price alert check error: {e}")
                
    except KeyboardInterrupt:
        print("\n\nShutdown...")
    finally:
        trader.stop()
        ib.disconnect()
        print(f"\nTrades: {trader.trade_count} | Final: ${trader.account_value:,.2f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ICT V6 - FVG + Gap Trading with Telegram')
    parser.add_argument("--symbols", default="BTCUSD,ETHUSD,SOLUSD,GC,CL", 
                        help="Comma-separated symbols")
    parser.add_argument("--interval", type=int, default=30,
                        help="Poll interval in seconds")
    parser.add_argument("--risk", type=float, default=0.02,
                        help="Risk per trade (e.g., 0.02 for 2%)")
    parser.add_argument("--port", type=int, default=7497,
                        help="IBKR port (7497=paper, 7496=live)")
    parser.add_argument("--mode", type=str, default="paper",
                        choices=["shadow", "paper", "live"],
                        help="Trading mode")
    
    args = parser.parse_args()
    symbols = [s.strip().upper() for s in args.symbols.split(',')]
    
    print("="*60)
    print("ICT V6 Trading Bot - FVG + Gap Analysis")
    print("="*60)
    print(f"Mode: {args.mode.upper()}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Risk: {args.risk*100}%")
    print(f"Port: {args.port}")
    print("="*60)
    
    run_v6_trading(symbols, args.interval, args.risk, args.port, args.mode)
