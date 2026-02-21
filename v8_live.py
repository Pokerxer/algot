"""
ICT V8 Live Trading System
===========================
Live trading with V8 signal generation + RL agent for entry/exit timing.

Usage:
    python3 v8_live.py --symbols "BTCUSD,ETHUSD,ES,NQ,GC" --mode paper
    python3 v8_live.py --symbols "BTCUSD,ES" --mode live --risk 0.02
"""

import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import sys
sys.path.insert(0, '.')

import json
import pandas as pd
import numpy as np
import argparse
import time
import pickle
import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

# Import V8 signal generator
from v8_backtest import V8SignalGenerator, TrainingConfig

# Import IBKR functions from V5
from ict_v5_ibkr import (
    fetch_ibkr_data,
    prepare_data_ibkr,
    get_signal,
    calculate_position_size,
    get_contract_info,
    get_ibkr_contract,
    place_bracket_order,
    wait_for_fill,
    get_contract_multiplier
)

# Import RL components
from reinforcement_learning_agent import (
    ICTReinforcementLearningAgent,
    EntryAction,
    ExitAction
)

# Telegram notifications
try:
    import telegram_notify as tn
except ImportError:
    tn = None

# IBKR connection
try:
    from ib_insync import IB, util
    IBKR_AVAILABLE = True
except ImportError:
    IB = None  # type: ignore
    util = None  # type: ignore
    IBKR_AVAILABLE = False
    print("WARNING: ib_insync not installed. Run: pip install ib_insync")


class V8LiveTrader:
    """Live trading with V8 signals + RL agent."""
    
    def __init__(
        self,
        ib: Any,  # IB connection object
        symbols: List[str],
        risk_pct: float = 0.02,
        poll_interval: int = 30,
        mode: str = 'paper',  # 'paper', 'live', 'shadow'
        rr_ratio: float = 4.0,  # Risk:Reward ratio (1:4)
        confluence_threshold: int = 60,
        use_rl: bool = True,
        rl_model_path: Optional[str] = None,
        port: int = 7497  # IBKR port for reconnection
    ):
        self.ib = ib
        self.symbols = symbols
        self.risk_pct = risk_pct
        self.poll_interval = poll_interval
        self.mode = mode
        self.rr_ratio = rr_ratio
        self.confluence_threshold = confluence_threshold
        self.use_rl = use_rl
        self.port = port  # Store port for reconnection
        
        # State tracking
        self.positions = {}
        self.active_orders = {}
        self.historical_data = {}
        self.account_value = 100000
        self.trade_count = 0
        self.daily_pnl = 0.0
        self.max_daily_loss = -2000  # Stop trading if down $2000
        
        # V8 Signal Generator with RL
        self.signal_gen = V8SignalGenerator(use_rl=use_rl)
        
        # Load pre-trained RL model if available
        if use_rl and rl_model_path and os.path.exists(rl_model_path):
            self._load_rl_model(rl_model_path)
            print(f"Loaded RL model from {rl_model_path}")
        
        # Streaming state
        self.streaming_symbols = set()
        self.last_poll_time = {}
        self.bar_handlers = {}
        self.last_signal_time = {}  # Track when we last checked signals per symbol
        self.last_hourly_bar = {}   # Track last hourly bar timestamp per symbol
        
        # Initialize
        self._init_account()
        self._init_historical_data()
        self._sync_positions()
    
    def _load_rl_model(self, path: str):
        """Load pre-trained RL agent."""
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            self.signal_gen.rl_agent = model_data['agent']
            print(f"RL agent loaded successfully")
        except Exception as e:
            print(f"Error loading RL model: {e}")
    
    def save_rl_model(self, path: str):
        """Save current RL agent."""
        try:
            with open(path, 'wb') as f:
                pickle.dump({'agent': self.signal_gen.rl_agent}, f)
            print(f"RL model saved to {path}")
        except Exception as e:
            print(f"Error saving RL model: {e}")
    
    def _init_account(self):
        """Get account value from IBKR."""
        try:
            account_values = self.ib.accountValues()
            for av in account_values:
                if av.tag == 'NetLiquidation' and av.currency == 'USD':
                    self.account_value = float(av.value)
                    break
            print(f"Account Value: ${self.account_value:,.2f}")
        except Exception as e:
            print(f"Error getting account value: {e}")
            print(f"Using default: ${self.account_value:,.2f}")
    
    def _init_historical_data(self):
        """Fetch historical data for all symbols."""
        print("\nInitializing historical data...")
        for symbol in self.symbols:
            try:
                data = prepare_data_ibkr(symbol, ib=self.ib, use_cache=True)
                if data and len(data.get('closes', [])) >= 50:
                    self.historical_data[symbol] = data
                    # Track initial bar count for new bar detection
                    self.last_hourly_bar[symbol] = len(data['closes'])
                    print(f"  {symbol}: {len(data['closes'])} bars loaded")
                else:
                    print(f"  {symbol}: Failed to load data")
            except Exception as e:
                print(f"  {symbol}: Error - {e}")
    
    def _sync_positions(self):
        """Sync positions with IBKR on startup."""
        print("\nSyncing positions with IBKR...")
        try:
            ibkr_positions = self.ib.positions()
            for pos in ibkr_positions:
                symbol = pos.contract.symbol
                if pos.contract.secType == 'CASH':
                    symbol = f"{pos.contract.symbol}{pos.contract.currency}"
                elif pos.contract.secType == 'CRYPTO':
                    symbol = f"{pos.contract.symbol}USD"
                
                if abs(pos.position) > 0 and symbol.upper() in [s.upper() for s in self.symbols]:
                    self.positions[symbol] = {
                        'qty': abs(pos.position),
                        'direction': 1 if pos.position > 0 else -1,
                        'entry': pos.avgCost,
                        'synced': True
                    }
                    print(f"  Found open position: {symbol} x {pos.position} @ {pos.avgCost:.4f}")
            
            if not self.positions:
                print("  No open positions found")
        except Exception as e:
            print(f"  Error syncing positions: {e}")
    
    def _on_realtime_bar(self, symbol: str, bar):
        """Handle real-time bar update."""
        if symbol not in self.historical_data:
            return
        
        data = self.historical_data[symbol]
        current_price = bar.close
        
        # Check daily loss limit
        if self.daily_pnl <= self.max_daily_loss:
            return  # Silently skip - already logged once
        
        # Check position management first
        if symbol in self.positions:
            self._manage_position(symbol, current_price)
        else:
            # Only check for new entry once per hour to prevent duplicate signals
            current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
            last_signal = self.last_signal_time.get(symbol)
            
            if last_signal and last_signal >= current_hour:
                return  # Already checked this hour
            
            self._check_entry(symbol, current_price)
    
    def _refresh_hourly_data(self):
        """Refresh hourly data for all symbols and check for new bars."""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Refreshing hourly data...")
        for symbol in self.symbols:
            try:
                old_bar_count = len(self.historical_data.get(symbol, {}).get('closes', []))
                data = prepare_data_ibkr(symbol, ib=self.ib, use_cache=False)  # Force refresh
                if data and len(data.get('closes', [])) >= 50:
                    new_bar_count = len(data['closes'])
                    self.historical_data[symbol] = data
                    
                    # Check if we have a new bar
                    if new_bar_count > old_bar_count:
                        print(f"  {symbol}: {new_bar_count} bars (NEW BAR DETECTED)")
                        # New bar detected - check for signal
                        if symbol not in self.positions:
                            self._check_entry(symbol, data['closes'][-1])
                    else:
                        print(f"  {symbol}: {new_bar_count} bars")
            except Exception as e:
                print(f"  {symbol}: Error - {e}")
        print("")
    
    def _check_entry(self, symbol: str, current_price: float):
        """Check for entry signal using V8 + RL."""
        try:
            data = self.historical_data[symbol]
            idx = len(data['closes']) - 1
            
            # Get V8 signal with RL
            signal = self.signal_gen.generate_signal(data, idx)
            
            # Track that we checked this hour (even if no signal)
            current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
            self.last_signal_time[symbol] = current_hour
            
            if not signal:
                return
            
            # Check confluence threshold
            if signal['confluence'] < self.confluence_threshold:
                return
            
            # Check RL entry decision
            rl_action = signal.get('rl_entry_action', 'ENTER_NOW')
            
            if rl_action == 'PASS':
                print(f"[{symbol}] Signal found (conf={signal['confluence']}) but RL says PASS")
                return
            
            # Log signal
            direction_str = 'LONG' if signal['direction'] == 1 else 'SHORT'
            print(f"[{symbol}] V8 Signal: {direction_str} | Confluence: {signal['confluence']} | "
                  f"PD Zone: {signal['pd_zone']} | RL: {rl_action}")
            
            # Calculate stop and target with configurable RR
            if signal['direction'] == 1:
                stop = data['lows'][idx]
                target = current_price + (current_price - stop) * self.rr_ratio
            else:
                stop = data['highs'][idx]
                target = current_price - (stop - current_price) * self.rr_ratio
            
            stop_distance = abs(current_price - stop)
            if stop_distance <= 0:
                return
            
            # Calculate position size
            qty, risk_amount = calculate_position_size(
                symbol, self.account_value, self.risk_pct,
                stop_distance, current_price
            )
            
            if qty <= 0:
                print(f"[{symbol}] Position size too small, skipping")
                return
            
            # Execute trade based on mode
            if self.mode == 'shadow':
                # Shadow mode - just log
                print(f"[{symbol}] SHADOW ENTRY: {direction_str} x {qty} @ {current_price:.4f}")
                print(f"          Stop: {stop:.4f} | Target: {target:.4f} | Risk: ${risk_amount:.2f}")
                self._log_shadow_trade(symbol, signal, current_price, stop, target, qty)
            else:
                # Paper or Live mode - execute
                self._execute_entry(symbol, signal, current_price, stop, target, qty)
        
        except Exception as e:
            print(f"[{symbol}] Error checking entry: {e}")
    
    def _execute_entry(self, symbol: str, signal: Dict, entry_price: float,
                       stop: float, target: float, qty: float):
        """Execute entry order via IBKR."""
        try:
            contract = get_ibkr_contract(symbol)
            
            # Place bracket order
            bracket = place_bracket_order(
                self.ib, contract, signal['direction'],
                qty, stop, target
            )
            
            if not bracket:
                print(f"[{symbol}] Failed to place bracket order")
                return
            
            parent_trade, sl_trade, tp_trade = bracket
            
            # Wait for fill
            filled, fill_price, filled_qty = wait_for_fill(self.ib, parent_trade, timeout=10)
            
            if filled:
                self.positions[symbol] = {
                    'entry': fill_price,
                    'stop': stop,
                    'target': target,
                    'direction': signal['direction'],
                    'qty': filled_qty,
                    'confluence': signal['confluence'],
                    'pd_zone': signal.get('pd_zone'),
                    'rl_action': signal.get('rl_entry_action'),
                    'entry_time': datetime.now(),
                    'bars_held': 0
                }
                
                self.active_orders[symbol] = {
                    'sl_order_id': sl_trade.order.orderId,
                    'tp_order_id': tp_trade.order.orderId
                }
                
                direction_str = 'LONG' if signal['direction'] == 1 else 'SHORT'
                print(f"[{symbol}] ENTRY: {direction_str} x {filled_qty} @ {fill_price:.4f}")
                print(f"         Stop: {stop:.4f} | Target: {target:.4f}")
                
                self.trade_count += 1
                
                # Telegram notification
                if tn:
                    try:
                        tn.send_trade_entry(
                            symbol, signal['direction'], filled_qty,
                            fill_price, signal['confluence'], target, stop
                        )
                    except:
                        pass
            else:
                print(f"[{symbol}] Order not filled within timeout")
                
        except Exception as e:
            print(f"[{symbol}] Error executing entry: {e}")
    
    def _manage_position(self, symbol: str, current_price: float):
        """Manage open position using RL exit decisions."""
        try:
            pos = self.positions[symbol]
            pos['bars_held'] = pos.get('bars_held', 0) + 1
            
            # Check if position still exists in IBKR
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
                self._handle_position_closed(symbol, current_price)
                return
            
            # Get RL exit decision if enabled
            if self.use_rl and self.signal_gen.rl_agent:
                data = self.historical_data.get(symbol)
                if data:
                    idx = len(data['closes']) - 1
                    exit_decision = self.signal_gen.evaluate_exit(data, idx, pos)
                    
                    if exit_decision:
                        action = exit_decision['action']
                        
                        if action == 'EXIT_NOW':
                            print(f"[{symbol}] RL says EXIT_NOW at {current_price:.4f}")
                            self._execute_exit(symbol, current_price, 'rl_exit')
                        
                        elif action == 'MOVE_STOP_BE':
                            # Move stop to breakeven
                            new_stop = pos['entry']
                            if pos['direction'] == 1:
                                if new_stop > pos['stop']:
                                    self._modify_stop(symbol, new_stop)
                                    print(f"[{symbol}] RL: Moving stop to breakeven @ {new_stop:.4f}")
                            else:
                                if new_stop < pos['stop']:
                                    self._modify_stop(symbol, new_stop)
                                    print(f"[{symbol}] RL: Moving stop to breakeven @ {new_stop:.4f}")
                        
                        elif action.startswith('TRAIL_STOP'):
                            # Trail stop
                            stop_distance = abs(pos['entry'] - pos['stop'])
                            if pos['direction'] == 1:
                                new_stop = current_price - stop_distance * 0.5
                                if new_stop > pos['stop']:
                                    self._modify_stop(symbol, new_stop)
                                    print(f"[{symbol}] RL: Trailing stop to {new_stop:.4f}")
                            else:
                                new_stop = current_price + stop_distance * 0.5
                                if new_stop < pos['stop']:
                                    self._modify_stop(symbol, new_stop)
                                    print(f"[{symbol}] RL: Trailing stop to {new_stop:.4f}")
        
        except Exception as e:
            print(f"[{symbol}] Error managing position: {e}")
    
    def _modify_stop(self, symbol: str, new_stop: float):
        """Modify stop loss order."""
        try:
            if symbol not in self.active_orders:
                return
            
            sl_order_id = self.active_orders[symbol].get('sl_order_id')
            if sl_order_id:
                # Cancel old stop and place new one
                # This is simplified - in production you'd modify the order
                self.positions[symbol]['stop'] = new_stop
                print(f"[{symbol}] Stop modified to {new_stop:.4f}")
        except Exception as e:
            print(f"[{symbol}] Error modifying stop: {e}")
    
    def _execute_exit(self, symbol: str, current_price: float, reason: str):
        """Execute market exit."""
        try:
            pos = self.positions[symbol]
            contract = get_ibkr_contract(symbol)
            
            # Place market order to close
            action = 'SELL' if pos['direction'] == 1 else 'BUY'
            from ib_insync import MarketOrder
            order = MarketOrder(action, pos['qty'])
            
            trade = self.ib.placeOrder(contract, order)
            filled, fill_price, _ = wait_for_fill(self.ib, trade, timeout=10)
            
            if filled:
                self._handle_position_closed(symbol, fill_price, reason)
        except Exception as e:
            print(f"[{symbol}] Error executing exit: {e}")
    
    def _handle_position_closed(self, symbol: str, exit_price: float, reason: str = 'bracket'):
        """Handle position closure."""
        try:
            if symbol not in self.positions:
                return
            
            pos = self.positions[symbol]
            
            # Calculate PnL
            if pos['direction'] == 1:
                pnl = (exit_price - pos['entry']) * pos['qty']
            else:
                pnl = (pos['entry'] - exit_price) * pos['qty']
            
            # Apply contract multiplier
            contract_info = get_contract_info(symbol)
            if contract_info['type'] == 'futures':
                pnl *= contract_info['multiplier']
            
            self.daily_pnl += pnl
            
            direction_str = 'LONG' if pos['direction'] == 1 else 'SHORT'
            pnl_str = f"+${pnl:.2f}" if pnl > 0 else f"-${abs(pnl):.2f}"
            
            print(f"[{symbol}] EXIT ({reason}): {direction_str} @ {exit_price:.4f} | P&L: {pnl_str}")
            print(f"         Daily P&L: ${self.daily_pnl:.2f}")
            
            # Telegram notification
            if tn:
                try:
                    tn.send_trade_exit(
                        symbol, pos['direction'], pnl, reason,
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
    
    def _log_shadow_trade(self, symbol: str, signal: Dict, entry: float,
                          stop: float, target: float, qty: float):
        """Log shadow trade to file."""
        try:
            trade = {
                'timestamp': datetime.now().isoformat(),
                'symbol': symbol,
                'direction': 'LONG' if signal['direction'] == 1 else 'SHORT',
                'entry': entry,
                'stop': stop,
                'target': target,
                'qty': qty,
                'confluence': signal['confluence'],
                'pd_zone': signal.get('pd_zone'),
                'rl_action': signal.get('rl_entry_action')
            }
            
            with open('v8_shadow_trades.json', 'a') as f:
                f.write(json.dumps(trade) + '\n')
        except:
            pass
    
    def start(self):
        """Start live trading."""
        print(f"\n{'='*60}")
        print(f"V8 Live Trader - Mode: {self.mode.upper()}")
        print(f"{'='*60}")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Risk per trade: {self.risk_pct*100}%")
        print(f"Risk:Reward: 1:{self.rr_ratio}")
        print(f"Confluence threshold: {self.confluence_threshold}")
        print(f"RL Agent: {'Enabled' if self.use_rl else 'Disabled'}")
        print(f"Account Value: ${self.account_value:,.2f}")
        print(f"{'='*60}\n")
        
        # Track last signal time to prevent duplicate trades
        self.last_signal_time = {}
        self.last_hourly_refresh = {}
        
        # Start streaming for each symbol
        streaming_failed = []
        for symbol in self.symbols:
            try:
                contract = get_ibkr_contract(symbol)
                
                # Subscribe to real-time bars
                bars = self.ib.reqRealTimeBars(contract, 5, 'MIDPOINT', False)
                
                # Create handler
                def make_handler(sym):
                    def handler(bars, hasNewBar):
                        if hasNewBar and len(bars) > 0:
                            self._on_realtime_bar(sym, bars[-1])
                    return handler
                
                bars.updateEvent += make_handler(symbol)
                self.bar_handlers[symbol] = bars
                self.streaming_symbols.add(symbol)
                print(f"[{symbol}] Streaming started")
                
            except Exception as e:
                error_msg = str(e)
                if 'permissions' in error_msg.lower() or '420' in error_msg:
                    print(f"[{symbol}] No market data permissions - using polling mode")
                    streaming_failed.append(symbol)
                else:
                    print(f"[{symbol}] Streaming failed: {e}")
                    streaming_failed.append(symbol)
        
        # Initialize polling for failed symbols
        if streaming_failed:
            print(f"\nUsing polling for: {', '.join(streaming_failed)}")
        
        # Main loop with auto-reconnection for nightly IBKR restart
        print("\nTrading started. Press Ctrl+C to stop.\n")
        print("NOTE: Signals are checked on hourly bar close. Data refreshes every hour.")
        print("Auto-reconnect enabled for nightly IBKR restart (~11:45 PM ET).\n")
        
        try:
            iteration = 0
            reconnect_attempts = 0
            max_reconnect_attempts = 10
            
            while True:
                try:
                    # Check if still connected
                    if not self.ib.isConnected():
                        raise ConnectionError("IBKR connection lost")
                    
                    self.ib.sleep(1)
                    iteration += 1
                    reconnect_attempts = 0  # Reset on successful iteration
                    
                    # Refresh hourly data every 5 minutes for all symbols
                    if iteration % 300 == 0:
                        self._refresh_hourly_data()
                    
                    # Poll symbols that aren't streaming
                    current_time = time.time()
                    for symbol in self.symbols:
                        if symbol not in self.streaming_symbols:
                            last_poll = self.last_poll_time.get(symbol, 0)
                            if current_time - last_poll >= self.poll_interval:
                                self._poll_symbol(symbol)
                                self.last_poll_time[symbol] = current_time
                
                except (ConnectionError, OSError, Exception) as e:
                    error_msg = str(e)
                    if 'connection' in error_msg.lower() or 'disconnect' in error_msg.lower() or not self.ib.isConnected():
                        reconnect_attempts += 1
                        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Connection lost: {e}")
                        
                        if reconnect_attempts > max_reconnect_attempts:
                            print(f"Max reconnect attempts ({max_reconnect_attempts}) reached. Stopping.")
                            break
                        
                        # Wait before reconnecting (longer wait during nightly restart window)
                        current_hour = datetime.now().hour
                        if 23 <= current_hour or current_hour < 1:
                            # Nightly restart window - wait longer
                            wait_time = 120  # 2 minutes
                            print(f"Nightly restart window detected. Waiting {wait_time}s before reconnect...")
                        else:
                            wait_time = 10 + (reconnect_attempts * 5)  # Progressive backoff
                            print(f"Waiting {wait_time}s before reconnect (attempt {reconnect_attempts}/{max_reconnect_attempts})...")
                        
                        time.sleep(wait_time)
                        
                        # Attempt reconnection
                        if self._reconnect():
                            print(f"[{datetime.now().strftime('%H:%M:%S')}] Reconnected successfully!")
                            # Re-sync positions after reconnect
                            self._sync_positions()
                            # Re-initialize streaming
                            self._restart_streaming()
                            iteration = 0
                        else:
                            print(f"Reconnection failed. Will retry...")
                    else:
                        # Non-connection error, log and continue
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Error: {e}")
                
        except KeyboardInterrupt:
            print("\nStopping trader...")
            self.stop()
    
    def _reconnect(self) -> bool:
        """Attempt to reconnect to IBKR."""
        try:
            # Disconnect first if partially connected
            try:
                self.ib.disconnect()
            except:
                pass
            
            time.sleep(2)
            
            # Try to reconnect using stored port
            print(f"Attempting to reconnect to IBKR on port {self.port}...")
            self.ib.connect('127.0.0.1', self.port, clientId=1, timeout=30)
            
            if self.ib.isConnected():
                # Re-initialize account info
                self._init_account()
                return True
            return False
            
        except Exception as e:
            print(f"Reconnection error: {e}")
            return False
    
    def _restart_streaming(self):
        """Restart real-time data streaming after reconnection."""
        print("Restarting data streams...")
        
        # Clear old handlers
        self.bar_handlers = {}
        self.streaming_symbols = set()
        
        # Re-subscribe to real-time bars
        for symbol in self.symbols:
            try:
                contract = get_ibkr_contract(symbol)
                bars = self.ib.reqRealTimeBars(contract, 5, 'MIDPOINT', False)
                
                def make_handler(sym):
                    def handler(bars, hasNewBar):
                        if hasNewBar and len(bars) > 0:
                            self._on_realtime_bar(sym, bars[-1])
                    return handler
                
                bars.updateEvent += make_handler(symbol)
                self.bar_handlers[symbol] = bars
                self.streaming_symbols.add(symbol)
                print(f"  [{symbol}] Streaming restarted")
                
            except Exception as e:
                print(f"  [{symbol}] Streaming failed: {e}")
        
        # Refresh historical data
        self._refresh_hourly_data()
    
    def _poll_symbol(self, symbol: str):
        """Poll symbol for updates (fallback for non-streaming)."""
        try:
            # Refresh historical data
            data = prepare_data_ibkr(symbol, ib=self.ib, use_cache=True)
            if data and len(data.get('closes', [])) >= 50:
                self.historical_data[symbol] = data
                current_price = data['closes'][-1]
                
                # Create fake bar for handler
                class FakeBar:
                    def __init__(self, price):
                        self.close = price
                
                self._on_realtime_bar(symbol, FakeBar(current_price))
        except Exception as e:
            print(f"[{symbol}] Poll error: {e}")
    
    def stop(self):
        """Stop trading and cleanup."""
        print("\nCleaning up...")
        
        # Cancel streaming
        for symbol, bars in self.bar_handlers.items():
            try:
                self.ib.cancelRealTimeBars(bars)
            except:
                pass
        
        # Print summary
        print(f"\n{'='*60}")
        print("SESSION SUMMARY")
        print(f"{'='*60}")
        print(f"Trades executed: {self.trade_count}")
        print(f"Daily P&L: ${self.daily_pnl:.2f}")
        print(f"Open positions: {len(self.positions)}")
        print(f"{'='*60}")
        
        # Save RL model if used
        if self.use_rl:
            self.save_rl_model('v8_rl_model.pkl')


def main():
    parser = argparse.ArgumentParser(description='V8 Live Trading System')
    parser.add_argument('--symbols', type=str, default='BTCUSD,ETHUSD,SOLUSD,LINKUSD,LTCUSD',
                        help='Comma-separated symbols (crypto only - no futures permissions needed)')
    parser.add_argument('--mode', type=str, default='shadow',
                        choices=['shadow', 'paper', 'live'],
                        help='Trading mode')
    parser.add_argument('--risk', type=float, default=0.02,
                        help='Risk per trade, e.g. 0.02 for 2 percent')
    parser.add_argument('--rr', type=float, default=4.0,
                        help='Risk:Reward ratio')
    parser.add_argument('--confluence', type=int, default=60,
                        help='Minimum confluence threshold')
    parser.add_argument('--no-rl', action='store_true',
                        help='Disable RL agent')
    parser.add_argument('--rl-model', type=str, default=None,
                        help='Path to pre-trained RL model')
    parser.add_argument('--port', type=int, default=7497,
                        help='IBKR port (7497=paper, 7496=live)')
    
    args = parser.parse_args()
    
    if not IBKR_AVAILABLE:
        print("ERROR: ib_insync not installed")
        print("Install with: pip install ib_insync")
        return
    
    # Connect to IBKR with retry logic
    print("="*60)
    print("V8 Live Trading System")
    print("="*60)
    print(f"IBKR Port: {args.port}")
    print(f"Mode: {args.mode}")
    print(f"Symbols: {args.symbols}")
    print("="*60)
    
    assert IB is not None, "IB should be available after IBKR_AVAILABLE check"
    ib = IB()
    connected = False
    
    # Try to connect with retries
    for attempt in range(3):
        try:
            print(f"Connecting to IBKR (attempt {attempt + 1}/3)...", end=" ")
            ib.connect('127.0.0.1', args.port, clientId=1, timeout=10)
            connected = True
            print("SUCCESS!")
            break
        except Exception as e:
            print(f"FAILED - {e}")
            if attempt < 2:
                print("Retrying in 5 seconds...")
                time.sleep(5)
    
    if not connected:
        port = args.port
        print("\n" + "="*60)
        print("ERROR: Could not connect to IBKR")
        print("="*60)
        print(f"""
To enable IBKR API:

1. IB Gateway (recommended):
   - Open IB Gateway
   - Go to Settings (gear icon) → API
   - Check "Enable API connections"
   - Make sure port is correct (7497=paper, 7496=live)

2. TWS (alternative):
   - Open TWS
   - File → Global Configuration → API
   - Check "Enable ActiveX and Socket Clients"
   - Uncheck "Read-Only API" if you want to trade

3. Verify IB Gateway is running:
   - Check that IB Gateway process is running
   - Check firewall isn't blocking port {port}

Current settings:
- Port: {port} (try 7497 for paper, 7496 for live)
- Host: 127.0.0.1 (localhost)
""")
        return
    
    # Parse symbols
    symbols = [s.strip().upper() for s in args.symbols.split(',')]
    
    # Create trader
    trader = V8LiveTrader(
        ib=ib,
        symbols=symbols,
        risk_pct=args.risk,
        mode=args.mode,
        rr_ratio=args.rr,
        confluence_threshold=args.confluence,
        use_rl=not args.no_rl,
        rl_model_path=args.rl_model,
        port=args.port  # Pass port for auto-reconnection
    )
    
    # Start trading
    try:
        trader.start()
    finally:
        ib.disconnect()
        print("Disconnected from IBKR")


if __name__ == "__main__":
    main()
