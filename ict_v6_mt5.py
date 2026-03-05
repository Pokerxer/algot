"""
ICT V6 Trading Bot - MetaTrader 5 Version
==========================================
Combines V6 signal generation with MT5 for data and trading.
Full Telegram integration for commands and notifications.

Usage:
    python3 ict_v6_mt5.py --symbols "EURUSD,GBPUSD,AUDUSD" --login 12345 --password "xxx" --server "MetaQuotes-Demo"
"""

import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import sys
import os
import time
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

import pandas as pd
import numpy as np

from fvg_handler import FVGHandler, FairValueGap, FVGStatus
from gap_handler import GapHandler, Gap, GapType, GapDirection

try:
    from mtf_coordinator import MTFCoordinator, TimeframePurpose, TimeframeRelation
    from market_structure_handler import (
        MarketStructureHandler, MarketStructureAnalysis, 
        StructureBreakType, TrendState, PriceZone
    )
    MTF_AVAILABLE = True
except ImportError as MTFError:
    MTF_AVAILABLE = False
    print(f"WARNING: MTF modules not available: {MTFError}")

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    print("WARNING: MetaTrader5 not installed. Run: pip install MetaTrader5")

try:
    import telegram_notify as tn
    if tn and hasattr(tn, 'init_bot'):
        try:
            tn.init_bot()
            print("Telegram bot initialized")
        except Exception as e:
            print(f"Telegram bot init failed: {e}")
except ImportError:
    tn = None
    print("WARNING: telegram_notify not installed")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def is_trading_paused() -> bool:
    """Check if trading is paused via Telegram."""
    if tn and hasattr(tn, 'is_trading_paused'):
        return tn.is_trading_paused()
    return False


class V6SignalGenerator:
    """Enhanced signal generator combining V5 ICT with FVG, Gap, MTF and Market Structure"""
    
    def __init__(self):
        self.fvg_handler = FVGHandler(
            sensitivity=0.0001,
            min_gap_size=0.0,
            track_body_respect=True,
            detect_volume_imbalances=True
        )
        self.gap_handler = GapHandler()
        
    def analyze_symbol(self, symbol: str, data: Dict, current_price: float) -> Dict:
        """Analyze symbol and generate trading signal"""
        signal = {
            'direction': 0,
            'confidence': 'LOW',
            'confluence': 0,
            'entry_price': current_price,
            'stop_loss': 0,
            'take_profit': 0,
            'reasoning': [],
            'fvg_data': {},
            'gap_data': {},
            'v5_signal': {}
        }
        
        try:
            closes = np.array(data.get('closes', []))
            highs = np.array(data.get('highs', []))
            lows = np.array(data.get('lows', []))
            opens = np.array(data.get('opens', []))
            
            if len(closes) < 50:
                return signal
            
            idx = len(closes) - 1
            
            fvg_signals = self.fvg_handler.detect_fvg(opens, highs, lows, closes)
            gap_signals = self.gap_handler.detect_gaps(highs, lows, closes)
            
            bullish_fvgs = [f for f in fvg_signals if f.status == FVGStatus.BULLISH]
            bearish_fvgs = [f for f in fvg_signals if f.status == FVGStatus.BEARISH]
            
            bullish_gaps = [g for g in gap_signals if g.gap_type == GapType.BULLISH]
            bearish_gaps = [g for g in gap_signals if g.gap_type == GapType.BEARISH]
            
            confluence = 0
            reasoning = []
            
            htf_trend = data.get('htf_trend', [0] * len(closes))[-1] if 'htf_trend' in data else 0
            ltf_trend = data.get('ltf_trend', [0] * len(closes))[-1] if 'ltf_trend' in data else 0
            kill_zone = data.get('kill_zone', [False] * len(closes))[-1] if 'kill_zone' in data else False
            price_position = data.get('price_position', [0.5] * len(closes))[-1] if 'price_position' in data else 0.5
            
            if kill_zone:
                confluence += 15
                reasoning.append("Kill Zone Active: +15")
            
            if htf_trend == 1 and ltf_trend == 1:
                confluence += 25
                reasoning.append("HTF+LTF Bullish: +25")
            elif htf_trend == -1 and ltf_trend == -1:
                confluence += 25
                reasoning.append("HTF+LTF Bearish: +25")
            elif htf_trend == 1 and ltf_trend == 0:
                confluence += 15
                reasoning.append("LTF Bullish (HTF flat): +15")
            elif htf_trend == -1 and ltf_trend == 0:
                confluence += 15
                reasoning.append("LTF Bearish (HTF flat): +15")
            
            if price_position > 0.7:
                confluence += 20
                reasoning.append("Price near highs: +20")
            elif price_position < 0.3:
                confluence += 20
                reasoning.append("Price near lows: +20")
            
            current_price = closes[-1]
            recent_high = np.max(highs[-20:-1])
            recent_low = np.min(lows[-20:-1])
            
            if bullish_fvgs or bullish_gaps:
                if current_price < recent_high * 0.99:
                    direction = 1
                    signal_type = "FVG" if bullish_fvgs else "Gap"
                    signal_type_detail = bullish_fvgs[0].type.value if bullish_fvgs else bullish_gaps[0].direction.value
                    confluence += 20
                    reasoning.append(f"{signal_type} Bullish ({signal_type_detail}): +20")
                    
                    stop_distance = (recent_high - current_price) * 0.5
                    signal['direction'] = direction
                    signal['confluence'] = min(confluence, 100)
                    signal['confidence'] = 'HIGH' if confluence >= 70 else 'MEDIUM' if confluence >= 50 else 'LOW'
                    signal['reasoning'] = reasoning
                    signal['entry_price'] = current_price
                    signal['stop_loss'] = current_price - stop_distance
                    signal['take_profit'] = current_price + (stop_distance * 2)
                    signal['fvg_data'] = {
                        'type': bullish_fvgs[0].type.value if bullish_fvgs else None,
                        'status': bullish_fvgs[0].status.value if bullish_fvgs else None
                    } if bullish_fvgs else {}
                    signal['gap_data'] = {
                        'type': bullish_gaps[0].direction.value if bullish_gaps else None,
                    } if bullish_gaps else {}
                    
            elif bearish_fvgs or bearish_gaps:
                if current_price > recent_low * 1.01:
                    direction = -1
                    signal_type = "FVG" if bearish_fvgs else "Gap"
                    signal_type_detail = bearish_fvgs[0].type.value if bearish_fvgs else bearish_gaps[0].direction.value
                    confluence += 20
                    reasoning.append(f"{signal_type} Bearish ({signal_type_detail}): +20")
                    
                    stop_distance = (current_price - recent_low) * 0.5
                    signal['direction'] = direction
                    signal['confluence'] = min(confluence, 100)
                    signal['confidence'] = 'HIGH' if confluence >= 70 else 'MEDIUM' if confluence >= 50 else 'LOW'
                    signal['reasoning'] = reasoning
                    signal['entry_price'] = current_price
                    signal['stop_loss'] = current_price + stop_distance
                    signal['take_profit'] = current_price - (stop_distance * 2)
                    signal['fvg_data'] = {
                        'type': bearish_fvgs[0].type.value if bearish_fvgs else None,
                        'status': bearish_fvgs[0].status.value if bearish_fvgs else None
                    } if bearish_fvgs else {}
                    signal['gap_data'] = {
                        'type': bearish_gaps[0].direction.value if bearish_gaps else None,
                    } if bearish_gaps else {}
            
        except Exception as e:
            logger.error(f"Error analyzing {symbol}: {e}")
        
        return signal


class MT5DataFetcher:
    """Fetch market data from MetaTrader 5"""
    
    TIMEFRAMES = {
        '1m': mt5.TIMEFRAME_M1 if MT5_AVAILABLE else None,
        '5m': mt5.TIMEFRAME_M5 if MT5_AVAILABLE else None,
        '15m': mt5.TIMEFRAME_M15 if MT5_AVAILABLE else None,
        '30m': mt5.TIMEFRAME_M30 if MT5_AVAILABLE else None,
        '1H': mt5.TIMEFRAME_H1 if MT5_AVAILABLE else None,
        '4H': mt5.TIMEFRAME_H4 if MT5_AVAILABLE else None,
        '1D': mt5.TIMEFRAME_D1 if MT5_AVAILABLE else None,
    }
    
    def __init__(self):
        self.connected = False
    
    def connect(self, login: Optional[int] = None, password: Optional[str] = None, 
                server: Optional[str] = None) -> bool:
        """Connect to MT5 terminal"""
        if not MT5_AVAILABLE:
            logger.error("MT5 not available")
            return False
        
        try:
            if not mt5.initialize():
                logger.error(f"MT5 init failed: {mt5.last_error()}")
                return False
            
            if login and password and server:
                authorized = mt5.login(login=login, password=password, server=server)
                if not authorized:
                    logger.error(f"MT5 login failed: {mt5.last_error()}")
                    return False
            
            self.connected = True
            logger.info("Connected to MT5")
            return True
            
        except Exception as e:
            logger.error(f"MT5 connection error: {e}")
            return False
    
    def disconnect(self):
        """Disconnect from MT5"""
        if MT5_AVAILABLE:
            mt5.shutdown()
        self.connected = False
    
    def get_symbols(self) -> List[str]:
        """Get available symbols"""
        if not self.connected:
            return []
        return [s.name for s in mt5.symbols_get()]
    
    def get_rates(self, symbol: str, timeframe: str = '1H', num_bars: int = 500) -> Optional[pd.DataFrame]:
        """Get historical rates for a symbol"""
        if not self.connected or not MT5_AVAILABLE:
            return None
        
        try:
            tf = self.TIMEFRAMES.get(timeframe)
            if tf is None:
                logger.error(f"Invalid timeframe: {timeframe}")
                return None
            
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, num_bars)
            if rates is None or len(rates) == 0:
                logger.warning(f"No data for {symbol}")
                return None
            
            df = pd.DataFrame(rates)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)
            return df
            
        except Exception as e:
            logger.error(f"Error getting rates for {symbol}: {e}")
            return None
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        if not self.connected:
            return None
        
        try:
            tick = mt5.symbol_info_tick(symbol)
            if tick:
                return (tick.bid + tick.ask) / 2
            return None
        except Exception as e:
            logger.error(f"Error getting price for {symbol}: {e}")
            return None
    
    def get_account_info(self) -> Dict:
        """Get account information"""
        if not self.connected:
            return {}
        
        try:
            info = mt5.account_info()
            if info:
                return {
                    'balance': info.balance,
                    'equity': info.equity,
                    'profit': info.profit,
                    'margin': info.margin,
                    'free_margin': info.margin_free,
                    'leverage': info.leverage,
                    'currency': info.currency
                }
        except Exception as e:
            logger.error(f"Error getting account info: {e}")
        return {}
    
    def get_positions(self) -> List[Dict]:
        """Get open positions"""
        if not self.connected:
            return []
        
        try:
            positions = mt5.positions_get()
            if positions is None:
                return []
            
            return [
                {
                    'symbol': p.symbol,
                    'type': 'buy' if p.type == 0 else 'sell',
                    'volume': p.volume,
                    'price': p.price_open,
                    'profit': p.profit,
                    'sl': p.sl,
                    'tp': p.tp,
                    'ticket': p.ticket
                }
                for p in positions
            ]
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def get_orders(self) -> List[Dict]:
        """Get pending orders"""
        if not self.connected:
            return []
        
        try:
            orders = mt5.orders_get()
            if orders is None:
                return []
            
            return [
                {
                    'symbol': o.symbol,
                    'type': 'buy' if o.type == 0 else 'sell',
                    'volume': o.volume,
                    'price': o.price_open,
                    'sl': o.sl,
                    'tp': o.tp,
                    'ticket': o.ticket,
                    'state': o.state
                }
                for o in orders
            ]
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return []


class MT5OrderExecutor:
    """Execute orders on MetaTrader 5"""
    
    def __init__(self, data_fetcher: MT5DataFetcher):
        self.data_fetcher = data_fetcher
        self.magic_number = 234000
    
    def place_bracket_order(self, symbol: str, direction: int, volume: float,
                           stop_loss: float, take_profit: float,
                           order_type: str = 'market') -> Tuple[bool, str]:
        """Place a bracket order (entry + SL + TP)"""
        if not self.data_fetcher.connected:
            return False, "Not connected to MT5"
        
        try:
            symbol_info = mt5.symbol_info(symbol)
            if not symbol_info:
                return False, f"Symbol {symbol} not found"
            
            if not symbol_info.visible:
                mt5.symbol_select(symbol, True)
            
            tick = mt5.symbol_info_tick(symbol)
            if not tick:
                return False, "Cannot get price"
            
            if direction == 1:
                action = mt5.TRADE_ACTION_DEAL
                order_type_mt5 = mt5.ORDER_TYPE_BUY
                price = tick.ask
                sl = stop_loss
                tp = take_profit
            else:
                action = mt5.TRADE_ACTION_DEAL
                order_type_mt5 = mt5.ORDER_TYPE_SELL
                price = tick.bid
                sl = stop_loss
                tp = take_profit
            
            request = {
                'action': action,
                'symbol': symbol,
                'volume': volume,
                'type': order_type_mt5,
                'price': price,
                'sl': sl,
                'tp': tp,
                'deviation': 20,
                'magic': self.magic_number,
                'comment': f'V6_{direction}',
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_IOC
            }
            
            result = mt5.order_send(request)
            
            if result is None:
                return False, "Order send failed"
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return False, f"Order failed: {result.comment}"
            
            return True, f"Order placed: {result.order}"
            
        except Exception as e:
            logger.error(f"Error placing order: {e}")
            return False, str(e)
    
    def close_position(self, ticket: int, volume: Optional[float] = None) -> Tuple[bool, str]:
        """Close a position"""
        if not self.data_fetcher.connected:
            return False, "Not connected to MT5"
        
        try:
            positions = mt5.positions_get(ticket=ticket)
            if not positions:
                return False, "Position not found"
            
            pos = positions[0]
            
            if volume is None:
                volume = pos.volume
            
            symbol_info = mt5.symbol_info(pos.symbol)
            if not symbol_info.visible:
                mt5.symbol_select(pos.symbol, True)
            
            tick = mt5.symbol_info_tick(pos.symbol)
            
            if pos.type == 0:
                order_type = mt5.ORDER_TYPE_SELLL
                price = tick.bid
            else:
                order_type = mt5.ORDER_TYPE_BUYL
                price = tick.ask
            
            request = {
                'action': mt5.TRADE_ACTION_DEAL,
                'symbol': pos.symbol,
                'volume': volume,
                'type': order_type,
                'position': ticket,
                'price': price,
                'deviation': 20,
                'magic': self.magic_number,
                'comment': 'V6_close',
                'type_time': mt5.ORDER_TIME_GTC,
                'type_filling': mt5.ORDER_FILLING_IOC
            }
            
            result = mt5.order_send(request)
            
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                return False, f"Close failed: {result.comment}"
            
            return True, f"Position {ticket} closed"
            
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return False, str(e)


class MT5LiveTrader:
    """MT5 Live Trader with V6 signals"""
    
    def __init__(self, mt5_data: MT5DataFetcher, symbols: List[str], 
                 risk_pct: float = 0.02, rr_ratio: float = 2.0,
                 confluence_threshold: int = 60, max_daily_loss: float = -2000):
        self.mt5_data = mt5_data
        self.executor = MT5OrderExecutor(mt5_data)
        self.symbols = symbols
        self.risk_pct = risk_pct
        self.rr_ratio = rr_ratio
        self.confluence_threshold = confluence_threshold
        self.max_daily_loss = max_daily_loss
        
        self.signal_generator = V6SignalGenerator()
        self.mode = 'paper'
        self.daily_pnl = 0.0
        self.last_signal_time = {}
        self.positions = {}
        self.account_value = mt5_data.get_account_info().get('equity', 100000)
        self.trade_count = 0
        self.running = False
        
        self.historical_data = {}
        self._load_historical_data()
    
    def _load_historical_data(self):
        """Load historical data for all symbols"""
        for symbol in self.symbols:
            df = self.mt5_data.get_rates(symbol, '1H', 500)
            if df is not None and len(df) > 100:
                self.historical_data[symbol] = {
                    'opens': df['open'].values,
                    'highs': df['high'].values,
                    'lows': df['low'].values,
                    'closes': df['close'].values,
                    'volumes': df['tick_volume'].values if 'tick_volume' in df.columns else np.zeros(len(df)),
                    'htf_trend': np.zeros(len(df)),
                    'ltf_trend': np.zeros(len(df)),
                    'kill_zone': np.zeros(len(df), dtype=bool),
                    'price_position': np.full(len(df), 0.5)
                }
                logger.info(f"Loaded {len(df)} bars for {symbol}")
    
    def _calculate_position_size(self, symbol: str, risk_amount: float, stop_distance: float) -> float:
        """Calculate position size based on risk"""
        try:
            info = mt5.symbol_info(symbol)
            if not info:
                return 0.0
            
            tick_size = info.point
            contract_size = info.contract_size
            
            risk_pips = stop_distance / tick_size
            if risk_pips > 0:
                position_size = (risk_amount * contract_size) / (risk_pips * tick_size * stop_distance)
                position_size = min(position_size, info.volume_max)
                position_size = max(position_size, info.volume_min)
                return round(position_size / info.volume_step) * info.volume_step
        
        except Exception as e:
            logger.error(f"Error calculating position size: {e}")
        
        return 0.0
    
    def check_signals(self):
        """Check for trading signals on all symbols"""
        for symbol in self.symbols:
            if symbol not in self.historical_data:
                continue
            
            current_price = self.mt5_data.get_current_price(symbol)
            if current_price is None:
                continue
            
            if symbol in self.positions:
                self._check_position_exit(symbol, current_price)
            else:
                self._check_entry_signal(symbol, current_price)
    
    def _check_entry_signal(self, symbol: str, current_price: float):
        """Check if there's an entry signal"""
        current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
        last_signal = self.last_signal_time.get(symbol)
        
        if last_signal and last_signal >= current_hour:
            return
        
        data = self.historical_data[symbol]
        signal = self.signal_generator.analyze_symbol(symbol, data, current_price)
        
        if signal['direction'] != 0 and signal['confluence'] >= self.confluence_threshold:
            self.last_signal_time[symbol] = current_hour
            self._enter_trade(symbol, signal, current_price)
    
    def _enter_trade(self, symbol: str, signal: Dict, current_price: float):
        """Enter a trade"""
        if is_trading_paused():
            logger.info(f"[{symbol}] Signal found but trading is PAUSED")
            return
        
        if self.daily_pnl <= self.max_daily_loss:
            logger.info(f"[{symbol}] Daily loss limit reached (${self.daily_pnl:.2f}), skipping trade")
            return
        
        positions = self.mt5_data.get_positions()
        for pos in positions:
            if pos['symbol'] == symbol:
                logger.info(f"[{symbol}] Already has open position, skipping entry")
                self.positions[symbol] = pos
                return
        
        try:
            entry_price = signal['entry_price']
            stop_price = signal['stop_loss']
            stop_distance = abs(entry_price - stop_price)
            
            if stop_distance <= 0:
                return
            
            risk_amount = self.account_value * self.risk_pct
            volume = self._calculate_position_size(symbol, risk_amount, stop_distance)
            
            if volume <= 0:
                return
            
            direction_str = "LONG" if signal['direction'] == 1 else "SHORT"
            target_price = signal['take_profit']
            
            if self.mode == 'shadow':
                logger.info(f"[{symbol}] V6 SHADOW SIGNAL: {direction_str} @ {current_price:.4f}")
                logger.info(f"  Stop: {stop_price:.4f} | Target: {target_price:.4f} (R:R 1:{self.rr_ratio})")
                logger.info(f"  Confluence: {signal['confluence']}/100")
                return
            
            success, msg = self.executor.place_bracket_order(
                symbol, signal['direction'], volume, stop_price, target_price
            )
            
            if success:
                self.positions[symbol] = {
                    'entry': entry_price,
                    'stop': stop_price,
                    'target': target_price,
                    'direction': signal['direction'],
                    'volume': volume,
                    'entry_time': datetime.now()
                }
                self.trade_count += 1
                logger.info(f"[{symbol}] V6 ENTRY: {direction_str} x {volume} @ {entry_price:.4f}")
                logger.info(f"  Stop: {stop_price:.4f} | Target: {target_price:.4f}")
                
                if tn:
                    try:
                        tn.send_trade_entry(
                            symbol, signal['direction'], volume,
                            entry_price, signal['confluence'], target_price, stop_price
                        )
                    except Exception as e:
                        logger.error(f"Telegram notification error: {e}")
            else:
                logger.error(f"[{symbol}] Order failed: {msg}")
        
        except Exception as e:
            logger.error(f"[{symbol}] Error entering trade: {e}")
    
    def _check_position_exit(self, symbol: str, current_price: float):
        """Check if position should be closed"""
        if symbol not in self.positions:
            return
        
        pos = self.positions[symbol]
        direction = pos['direction']
        entry = pos['entry']
        stop = pos['stop']
        target = pos['target']
        
        exit_reason = None
        
        if direction == 1:
            if current_price <= stop:
                exit_reason = 'stop'
            elif current_price >= target:
                exit_reason = 'target'
        else:
            if current_price >= stop:
                exit_reason = 'stop'
            elif current_price <= target:
                exit_reason = 'target'
        
        if exit_reason:
            self._close_position(symbol, exit_reason)
    
    def _close_position(self, symbol: str, reason: str):
        """Close a position"""
        try:
            positions = self.mt5_data.get_positions()
            for pos in positions:
                if pos['symbol'] == symbol:
                    success, msg = self.executor.close_position(pos['ticket'])
                    
                    if success:
                        pnl = pos['profit']
                        self.daily_pnl += pnl
                        
                        logger.info(f"[{symbol}] Position closed: {reason} | P&L: ${pnl:.2f}")
                        
                        if tn:
                            try:
                                tn.send_trade_exit(
                                    symbol, pos['type'], pnl, reason,
                                    pos['price'], pos.get('close_price', 0), 0
                                )
                            except Exception as e:
                                logger.error(f"Telegram error: {e}")
                        
                        del self.positions[symbol]
                    else:
                        logger.error(f"[{symbol}] Close failed: {msg}")
                    return
            
            logger.warning(f"[{symbol}] Position not found in MT5")
            del self.positions[symbol]
            
        except Exception as e:
            logger.error(f"[{symbol}] Error closing position: {e}")
    
    def sync_positions(self):
        """Sync positions with MT5"""
        positions = self.mt5_data.get_positions()
        self.positions = {p['symbol']: p for p in positions if p['symbol'] in self.symbols}
        logger.info(f"Synced {len(self.positions)} positions from MT5")
    
    def start(self, poll_interval: int = 30):
        """Start the trading loop"""
        self.running = True
        self.sync_positions()
        
        logger.info(f"Starting MT5 V6 trader for {self.symbols}")
        
        while self.running:
            try:
                self.check_signals()
                
                account_info = self.mt5_data.get_account_info()
                self.account_value = account_info.get('equity', self.account_value)
                
                self.daily_pnl = account_info.get('profit', 0)
                
                time.sleep(poll_interval)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(poll_interval)
        
        logger.info("Trading stopped")
    
    def stop(self):
        """Stop the trading loop"""
        self.running = False


def run_mt5_trading(symbols: List[str], poll_interval: int = 30,
                    risk_pct: float = 0.02, login: Optional[int] = None,
                    password: Optional[str] = None, server: Optional[str] = None,
                    mode: str = 'paper', rr_ratio: float = 2.0,
                    confluence_threshold: int = 60, max_daily_loss: float = -2000):
    """Run MT5 trading"""
    mt5_data = MT5DataFetcher()
    
    if not mt5_data.connect(login, password, server):
        logger.error("Failed to connect to MT5")
        return
    
    try:
        trader = MT5LiveTrader(
            mt5_data, symbols, risk_pct, rr_ratio,
            confluence_threshold, max_daily_loss
        )
        trader.mode = mode
        
        if tn and hasattr(tn, 'set_live_trader'):
            tn.set_live_trader(trader)
        
        trader.start(poll_interval)
        
    finally:
        mt5_data.disconnect()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='ICT V6 - MT5 Trading')
    parser.add_argument("--symbols", default="EURUSD,GBPUSD,USDJPY",
                        help="Comma-separated symbols")
    parser.add_argument("--interval", type=int, default=30,
                        help="Poll interval in seconds")
    parser.add_argument("--risk", type=float, default=0.02,
                        help="Risk per trade (e.g., 0.02 for 2%%)")
    parser.add_argument("--login", type=int, default=None,
                        help="MT5 login (account number)")
    parser.add_argument("--password", type=str, default=None,
                        help="MT5 password")
    parser.add_argument("--server", type=str, default=None,
                        help="MT5 server")
    parser.add_argument("--mode", type=str, default="paper",
                        choices=["shadow", "paper", "live"],
                        help="Trading mode")
    parser.add_argument("--rr", type=float, default=2.0,
                        help="Risk:Reward ratio")
    parser.add_argument("--confluence", type=int, default=60,
                        help="Minimum confluence threshold")
    parser.add_argument("--max-loss", type=float, default=-2000,
                        help="Max daily loss before stopping")
    
    args = parser.parse_args()
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    print("="*60)
    print("ICT V6 Trading Bot - MetaTrader 5")
    print("="*60)
    print(f"Mode: {args.mode.upper()}")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Risk: {args.risk*100}% | R:R 1:{args.rr}")
    print(f"Confluence: {args.confluence}+ | Max Loss: ${args.max_loss}")
    print("="*60)
    
    if not MT5_AVAILABLE:
        print("ERROR: MetaTrader5 not installed")
        print("Install with: pip install MetaTrader5")
        sys.exit(1)
    
    run_mt5_trading(
        symbols, args.interval, args.risk,
        args.login, args.password, args.server,
        args.mode, args.rr, args.confluence, args.max_loss
    )
