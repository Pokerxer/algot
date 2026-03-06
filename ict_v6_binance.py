"""
ICT V6 Binance Trading - Crypto Trading via Binance
===================================================
Use Binance API for crypto data and trading.

Usage:
    python3 ict_v6_binance.py --symbols "BTCUSDT,ETHUSDT" --mode paper
    python3 ict_v6_binance.py --symbols "BTCUSDT,ETHUSDT" --mode live --api-key XXX --api-secret YYY
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import time
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import threading

# Environment variables for API keys (more secure)
BINANCE_API_KEY = os.environ.get('BINANCE_API_KEY', '')
BINANCE_API_SECRET = os.environ.get('BINANCE_API_SECRET', '')

# Import V5 components
import importlib.util
v5_path = os.path.join(os.path.dirname(__file__), 'ict_v5_ibkr.py')
spec = importlib.util.spec_from_file_location("ict_v5", v5_path)
ict_v5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ict_v5)

# Import Binance client
from binance_client import (
    BinanceClient, fetch_binance_data, to_binance_symbol, 
    from_binance_symbol, calculate_position_size_binance
)

# Import handlers
from fvg_handler import FVGHandler
from gap_handler import GapHandler

# Telegram notifications
try:
    import telegram_notify as tn
    if tn and hasattr(tn, 'init_bot'):
        try:
            tn.init_bot()
        except:
            pass
except ImportError:
    tn = None


# Crypto symbols that use Binance
CRYPTO_SYMBOLS = {
    'BTCUSD': 'BTCUSDT',
    'ETHUSD': 'ETHUSDT', 
    'SOLUSD': 'SOLUSDT',
    'LTCUSD': 'LTCUSDT',
    'LINKUSD': 'LINKUSDT',
    'UNIUSD': 'UNIUSDT',
    'XRPUSD': 'XRPUSDT',
    'ADAUSD': 'ADAUSDT',
    'DOGEUSD': 'DOGEUSDT',
    'DOTUSD': 'DOTUSDT',
    'AVAXUSD': 'AVAXUSDT',
    'MATICUSD': 'MATICUSDT',
}

# Session filter - crypto trades 24/7 but best during these hours
def is_valid_crypto_session() -> Tuple[bool, str]:
    """Crypto is 24/7 but best liquidity during NY/London overlap"""
    from datetime import time as dt_time
    import pytz
    
    et_tz = pytz.timezone('US/Eastern')
    now_et = datetime.now(et_tz)
    current_time = now_et.time()
    
    # Best session: 8 AM - 8 PM ET (NY + London overlap)
    session_start = dt_time(8, 0)
    session_end = dt_time(20, 0)
    
    if session_start <= current_time <= session_end:
        return True, "HIGH_LIQUIDITY"
    
    return True, "LOW_LIQUIDITY"  # Crypto is 24/7


def is_trading_paused() -> bool:
    if tn and hasattr(tn, 'is_trading_paused'):
        return tn.is_trading_paused()
    return False


class V6BinanceTrader:
    """V6 Trading with Binance"""
    
    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        symbols: List[str] = None,
        testnet: bool = False,
        risk_pct: float = 0.02,
        rr_ratio: float = 3.0,
        confluence_threshold: int = 60,
        max_daily_loss: float = -2000,
        mode: str = 'paper'
    ):
        self.api_key = api_key
        self.api_secret = api_secret
        self.symbols = symbols or ['BTCUSDT', 'ETHUSDT']
        self.testnet = testnet
        self.risk_pct = risk_pct
        self.rr_ratio = rr_ratio
        self.confluence_threshold = confluence_threshold
        self.max_daily_loss = max_daily_loss
        self.mode = mode
        
        # Initialize Binance client
        self.client = None
        if api_key and api_secret:
            self.client = BinanceClient(api_key, api_secret, testnet=testnet)
        
        # State
        self.positions = {}
        self.active_orders = {}
        self.historical_data = {}
        self.account_balance = 10000.0  # Default
        self.daily_pnl = 0.0
        self.trade_count = 0
        self.last_signal_time = {}
        
        # Signal generator
        self.signal_gen = self._create_signal_generator()
        
        # Initialize
        self._init_account()
        self._init_data()
    
    def _create_signal_generator(self):
        """Create V6 signal generator"""
        return V6SignalGenerator()
    
    def _init_account(self):
        """Get account balance"""
        if self.client:
            try:
                self.account_balance = self.client.get_balance('USDT')
                print(f"Account Balance: ${self.account_balance:,.2f}")
            except Exception as e:
                print(f"Error getting balance: {e}")
    
    def _init_data(self):
        """Fetch initial data for all symbols"""
        print(f"\nFetching initial data for {len(self.symbols)} symbols...")
        for symbol in self.symbols:
            bn_symbol = to_binance_symbol(symbol)
            data = fetch_binance_data(bn_symbol, '1h', 200, testnet=self.testnet)
            if data:
                self.historical_data[symbol] = data
                print(f"  {symbol}: {len(data['closes'])} bars @ {data['closes'][-1]}")
            else:
                print(f"  {symbol}: Failed to fetch")
        
        # Sync existing positions
        self._sync_positions()
    
    def _sync_positions(self):
        """Sync positions with Binance"""
        if not self.client:
            return
        
        try:
            positions = self.client.get_positions()
            print(f"\nSynced {len(positions)} positions from Binance")
            for pos in positions:
                if isinstance(pos, dict) and 'symbol' in pos:
                    symbol = from_binance_symbol(pos['symbol'])
                    if symbol.upper() in [s.upper() for s in self.symbols]:
                        self.positions[symbol] = {
                            'qty': abs(pos['qty']),
                            'direction': 1 if pos.get('side') == 'LONG' else -1,
                            'entry': pos.get('entry_price', 0),
                            'stop': 0,
                            'target': 0,
                            'bars_held': 0
                        }
        except Exception as e:
            print(f"Error syncing positions: {e}")
    
    def analyze(self, symbol: str) -> Optional[Dict]:
        """Generate V6 signal for symbol"""
        if symbol not in self.historical_data:
            return None
        
        data = self.historical_data[symbol]
        idx = len(data['closes']) - 1
        current_price = data['closes'][idx]
        
        signal = self.signal_gen.generate_signal(data, idx)
        
        if signal and signal['confluence'] >= self.confluence_threshold:
            return {
                'symbol': symbol,
                'direction': signal['direction'],
                'confluence': signal['confluence'],
                'entry_price': current_price,
                'stop_loss': signal.get('stop_loss'),
                'take_profit': signal.get('take_profit'),
                'confidence': signal.get('confidence', 'MEDIUM')
            }
        
        return None
    
    def _enter_trade(self, symbol: str, signal: Dict, current_price: float) -> bool:
        """Enter trade"""
        if is_trading_paused():
            print(f"[{symbol}] Trading paused")
            return False
        
        if self.daily_pnl <= self.max_daily_loss:
            print(f"[{symbol}] Daily loss limit reached")
            return False
        
        # Session check
        in_session, session_name = is_valid_crypto_session()
        if not in_session:
            print(f"[{symbol}] Outside session: {session_name}")
            return False
        
        # Check if already in position
        if symbol in self.positions:
            print(f"[{symbol}] Already in position")
            return False
        
        # Calculate position size
        if signal['stop_loss']:
            stop_distance = abs(current_price - signal['stop_loss'])
            stop_pct = stop_distance / current_price
        else:
            stop_pct = 0.02  # Default 2%
        
        bn_symbol = to_binance_symbol(symbol)
        qty = calculate_position_size_binance(
            bn_symbol,
            self.account_balance,
            self.risk_pct,
            stop_pct
        )
        
        if qty <= 0:
            print(f"[{symbol}] Invalid quantity")
            return False
        
        # Calculate SL/TP
        if signal['direction'] == 1:  # LONG
            stop_price = current_price * (1 - stop_pct)
            target_price = current_price * (1 + stop_pct * self.rr_ratio)
            side = 'BUY'
        else:  # SHORT
            stop_price = current_price * (1 + stop_pct)
            target_price = current_price * (1 - stop_pct * self.rr_ratio)
            side = 'SELL'
        
        print(f"\n[{symbol}] Signal: {side} {qty} @ {current_price}")
        print(f"  Stop: {stop_price:.4f} | Target: {target_price:.4f}")
        
        # Execute or simulate
        if self.mode == 'live' and self.client:
            try:
                result = self.client.place_bracket_order(
                    bn_symbol, side, qty, stop_price, target_price
                )
                print(f"  Order placed: {result}")
                
                self.positions[symbol] = {
                    'qty': qty,
                    'direction': signal['direction'],
                    'entry': current_price,
                    'stop': stop_price,
                    'target': target_price,
                    'entry_time': datetime.now(),
                    'bars_held': 0
                }
                
                if tn:
                    tn.send_trade_entry(
                        symbol, signal['direction'], qty, current_price,
                        signal['confluence'], target_price, stop_price
                    )
                
                return True
            except Exception as e:
                print(f"Order error: {e}")
                return False
        
        elif self.mode in ['paper', 'shadow']:
            # Simulate
            self.positions[symbol] = {
                'qty': qty,
                'direction': signal['direction'],
                'entry': current_price,
                'stop': stop_price,
                'target': target_price,
                'entry_time': datetime.now(),
                'bars_held': 0
            }
            print(f"  [{self.mode.upper()}] Trade simulated")
            return True
        
        return False
    
    def _check_exit(self, symbol: str) -> bool:
        """Check if position should exit"""
        if symbol not in self.positions:
            return False
        
        if symbol not in self.historical_data:
            return False
        
        data = self.historical_data[symbol]
        current_price = data['closes'][-1]
        
        pos = self.positions[symbol]
        
        # Check SL/TP
        exited = False
        exit_reason = None
        exit_price = current_price
        
        if pos['direction'] == 1:  # LONG
            if current_price <= pos['stop']:
                exit_price = pos['stop']
                exit_reason = 'stop'
                exited = True
            elif current_price >= pos['target']:
                exit_price = pos['target']
                exit_reason = 'target'
                exited = True
        else:  # SHORT
            if current_price >= pos['stop']:
                exit_price = pos['stop']
                exit_reason = 'stop'
                exited = True
            elif current_price <= pos['target']:
                exit_price = pos['target']
                exit_reason = 'target'
                exited = True
        
        if exited:
            # Calculate PnL
            if pos['direction'] == 1:
                pnl = (exit_price - pos['entry']) * pos['qty']
            else:
                pnl = (pos['entry'] - exit_price) * pos['qty']
            
            self.daily_pnl += pnl
            pnl_str = f"+${pnl:.2f}" if pnl > 0 else f"-${abs(pnl):.2f}"
            
            print(f"\n[{symbol}] EXIT ({exit_reason}): {pnl_str}")
            print(f"  Daily P&L: ${self.daily_pnl:.2f}")
            
            if tn:
                tn.send_trade_exit(
                    symbol, pos['direction'], pnl, exit_reason,
                    pos['entry'], exit_price, pos.get('bars_held', 0)
                )
            
            del self.positions[symbol]
            return True
        
        return False
    
    def refresh_data(self):
        """Refresh data for all symbols"""
        for symbol in self.symbols:
            bn_symbol = to_binance_symbol(symbol)
            data = fetch_binance_data(bn_symbol, '1h', 200, testnet=self.testnet)
            if data:
                self.historical_data[symbol] = data
    
    def run(self, interval: int = 60):
        """Main trading loop"""
        print(f"\n{'='*60}")
        print(f"ICT V6 - Binance Trading")
        print(f"Mode: {self.mode.upper()}")
        print(f"Symbols: {', '.join(self.symbols)}")
        print(f"Risk: {self.risk_pct*100}% | R:R 1:{self.rr_ratio}")
        print(f"{'='*60}\n")
        
        iteration = 0
        
        while True:
            try:
                iteration += 1
                
                # Refresh data periodically
                if iteration % 5 == 0:
                    self.refresh_data()
                
                # Check positions
                for symbol in list(self.positions.keys()):
                    self._check_exit(symbol)
                
                # Check for signals (once per hour per symbol)
                current_hour = datetime.now().replace(minute=0, second=0, microsecond=0)
                
                for symbol in self.symbols:
                    if symbol in self.positions:
                        continue
                    
                    last_signal = self.last_signal_time.get(symbol)
                    if last_signal and last_signal >= current_hour:
                        continue
                    
                    signal = self.analyze(symbol)
                    if signal:
                        current_price = self.historical_data[symbol]['closes'][-1]
                        if self._enter_trade(symbol, signal, current_price):
                            self.last_signal_time[symbol] = current_hour
                
                # Update Telegram
                if tn and iteration % 60 == 0:
                    try:
                        tn.send_position_update(self.positions, self.daily_pnl)
                    except:
                        pass
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                print("\nStopping...")
                break
            except Exception as e:
                print(f"Error: {e}")
                time.sleep(10)


class V6SignalGenerator:
    """Simplified V6 signal generator for Binance"""
    
    def __init__(self):
        self.fvg_handler = FVGHandler(
            sensitivity=0.0001,
            min_gap_size=0.0,
            track_body_respect=False,
            detect_volume_imbalances=False,
            detect_suspension_blocks=False
        )
        self.gap_handler = GapHandler(
            large_gap_pips_forex=0.5,  # Crypto uses percentages
            large_ggap_points_indices=0.5,
            keep_gaps_days=1
        )
    
    def generate_signal(self, data: Dict, idx: int) -> Optional[Dict]:
        """Generate trading signal"""
        if idx < 50:
            return None
        
        closes = data['closes']
        highs = data['highs']
        lows = data['lows']
        
        current_price = closes[idx]
        
        # Simple trend detection
        sma_20 = sum(closes[idx-20:idx]) / 20
        sma_50 = sum(closes[idx-50:idx]) / 50
        
        # Price position
        high_50 = max(highs[idx-50:idx])
        low_50 = min(lows[idx-50:idx])
        price_pos = (current_price - low_50) / (high_50 - low_50) if high_50 != low_50 else 0.5
        
        # Calculate confluence
        confluence = 0
        direction = 0
        reasoning = []
        
        # Trend alignment
        if sma_20 > sma_50:
            confluence += 25
            direction = 1
            reasoning.append("Bullish trend")
        elif sma_20 < sma_50:
            confluence += 25
            direction = -1
            reasoning.append("Bearish trend")
        
        # Price position
        if price_pos < 0.25:
            confluence += 20
            reasoning.append("Near lows")
        elif price_pos > 0.75:
            confluence += 20
            reasoning.append("Near highs")
        
        # FVG detection
        import pandas as pd
        df = pd.DataFrame({
            'open': data['opens'][:idx+1],
            'high': data['highs'][:idx+1],
            'low': data['lows'][:idx+1],
            'close': data['closes'][:idx+1]
        })
        
        fvgs = self.fvg_handler.detect_all_fvgs(df)
        
        # Check for active FVGs near current price
        for fvg in fvgs[-5:]:
            if fvg.status.value != 'active':
                continue
            
            distance = abs(current_price - fvg.consequent_encroachment) / current_price
            if distance < 0.02:  # Within 2%
                if direction == 1 and fvg.gap_type in ['bullish', 'bisi']:
                    confluence += 15
                    reasoning.append(f"FVG bullish")
                elif direction == -1 and fvg.gap_type in ['bearish', 'sibi']:
                    confluence += 15
                    reasoning.append(f"FVG bearish")
        
        # Stop loss calculation
        atr = sum([highs[i] - lows[i] for i in range(max(0, idx-14), idx)]) / 14
        stop_distance = atr * 2
        
        if direction == 0:
            return None
        
        return {
            'direction': direction,
            'confluence': min(confluence, 100),
            'entry_price': current_price,
            'stop_loss': current_price - stop_distance if direction == 1 else current_price + stop_distance,
            'confidence': 'HIGH' if confluence >= 80 else 'MEDIUM' if confluence >= 60 else 'LOW',
            'reasoning': reasoning
        }


def run_binance_trading(
    symbols: List[str],
    api_key: str = "",
    api_secret: str = "",
    testnet: bool = False,
    mode: str = 'paper',
    risk_pct: float = 0.02,
    rr_ratio: float = 3.0,
    confluence_threshold: int = 60,
    max_daily_loss: float = -2000
):
    """Run Binance trading"""
    trader = V6BinanceTrader(
        api_key=api_key,
        api_secret=api_secret,
        symbols=symbols,
        testnet=testnet,
        risk_pct=risk_pct,
        rr_ratio=rr_ratio,
        confluence_threshold=confluence_threshold,
        max_daily_loss=max_daily_loss,
        mode=mode
    )
    
    if tn and hasattr(tn, 'set_live_trader'):
        tn.set_live_trader(trader)
    
    trader.run(interval=30)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ICT V6 - Binance Trading')
    parser.add_argument('--symbols', type=str, default='BTCUSDT,SOLUSDT,DOTUSDT,MATICUSDT,ETHUSDT',
                       help='Comma-separated symbols (use USDT format)')
    parser.add_argument('--api-key', type=str, default=BINANCE_API_KEY,
                       help='Binance API Key (or use BINANCE_API_KEY env var)')
    parser.add_argument('--api-secret', type=str, default=BINANCE_API_SECRET,
                       help='Binance API Secret (or use BINANCE_API_SECRET env var)')
    parser.add_argument('--testnet', action='store_true',
                       help='Use Binance testnet')
    parser.add_argument('--mode', type=str, default='paper',
                       choices=['paper', 'live', 'shadow'],
                       help='Trading mode')
    parser.add_argument('--risk', type=float, default=0.02,
                       help='Risk per trade (default 0.02 = 2%)')
    parser.add_argument('--rr', type=float, default=3.0,
                       help='Risk:Reward ratio (default 3.0)')
    parser.add_argument('--confluence', type=int, default=60,
                       help='Minimum confluence (default 60)')
    parser.add_argument('--max-loss', type=float, default=-2000,
                       help='Maximum daily loss')
    
    args = parser.parse_args()
    
    symbols = [s.strip().upper() for s in args.symbols.split(',')]
    
    run_binance_trading(
        symbols=symbols,
        api_key=args.api_key,
        api_secret=args.api_secret,
        testnet=args.testnet,
        mode=args.mode,
        risk_pct=args.risk,
        rr_ratio=args.rr,
        confluence_threshold=args.confluence,
        max_daily_loss=args.max_loss
    )
