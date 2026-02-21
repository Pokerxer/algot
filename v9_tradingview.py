"""
V9 TradingView Testing System
==============================
Test the V8 ICT trading bot using TradingView data instead of IBKR.
No broker connection required - perfect for testing signals and strategies.

Features:
- Real-time data from TradingView (via tvDatafeed)
- Fallback to Yahoo Finance for crypto
- Paper trading simulation
- Full V8 signal generation
- RL agent integration
- Telegram notifications

Usage:
    python3 v9_tradingview.py --symbols "BTCUSD,ETHUSD" --mode paper
    python3 v9_tradingview.py --symbols "AAPL,MSFT,SPY" --mode shadow
"""

import sys
import os
import time
import json
import pickle
import argparse
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import pandas as pd

# Try to import TradingView datafeed
try:
    from tvDatafeed import TvDatafeed, Interval
    TV_AVAILABLE = True
except ImportError:
    TV_AVAILABLE = False
    print("Note: tvDatafeed not installed. Install with: pip install tvDatafeed")

# Try to import yfinance as fallback
try:
    import yfinance as yf
    YF_AVAILABLE = True
except ImportError:
    YF_AVAILABLE = False
    print("Note: yfinance not installed. Install with: pip install yfinance")

# Import V8 components
try:
    from v8_backtest import V8SignalGenerator, TrainingConfig
    V8_AVAILABLE = True
except ImportError:
    V8_AVAILABLE = False
    print("ERROR: v8_backtest.py not found")

# Import RL components
try:
    from reinforcement_learning_agent import (
        ICTReinforcementLearningAgent,
        EntryAction,
        ExitAction
    )
    RL_AVAILABLE = True
except ImportError:
    RL_AVAILABLE = False
    print("Note: RL agent not available")

# Telegram notifications
try:
    import telegram_notify as tn
    TG_AVAILABLE = True
except ImportError:
    TG_AVAILABLE = False
    print("Note: Telegram notifications not available")


# =============================================================================
# DATA FETCHERS
# =============================================================================

class TradingViewDataFetcher:
    """Fetch data from TradingView"""
    
    def __init__(self, username: str = None, password: str = None):
        self.tv = None
        if TV_AVAILABLE:
            try:
                if username and password:
                    self.tv = TvDatafeed(username, password)
                else:
                    self.tv = TvDatafeed()  # Anonymous access
                print("TradingView connection established")
            except Exception as e:
                print(f"TradingView connection failed: {e}")
    
    def get_exchange(self, symbol: str) -> str:
        """Determine the exchange for a symbol"""
        symbol = symbol.upper()
        
        # Crypto
        if symbol in ['BTCUSD', 'ETHUSD', 'SOLUSD', 'LINKUSD', 'LTCUSD', 'UNIUSD', 'BTCUSDT', 'ETHUSDT']:
            return 'BINANCE'
        if symbol.endswith('USD') and len(symbol) <= 7:
            return 'BINANCE'
        
        # Futures
        if symbol in ['ES', 'NQ', 'GC', 'SI', 'CL']:
            return 'CME_MINI'
        
        # Forex
        if symbol in ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD']:
            return 'FX'
        
        # Default to stocks
        return 'NASDAQ'
    
    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol for TradingView"""
        symbol = symbol.upper()
        
        # Convert crypto symbols
        if symbol == 'BTCUSD':
            return 'BTCUSDT'
        if symbol == 'ETHUSD':
            return 'ETHUSDT'
        if symbol == 'SOLUSD':
            return 'SOLUSDT'
        if symbol == 'LINKUSD':
            return 'LINKUSDT'
        if symbol == 'LTCUSD':
            return 'LTCUSDT'
        if symbol == 'UNIUSD':
            return 'UNIUSDT'
        
        return symbol
    
    def fetch(self, symbol: str, interval: str = '1h', bars: int = 500) -> Optional[pd.DataFrame]:
        """Fetch historical data from TradingView"""
        if not self.tv:
            return None
        
        try:
            exchange = self.get_exchange(symbol)
            tv_symbol = self.normalize_symbol(symbol)
            
            # Map interval
            interval_map = {
                '1m': Interval.in_1_minute,
                '5m': Interval.in_5_minute,
                '15m': Interval.in_15_minute,
                '30m': Interval.in_30_minute,
                '1h': Interval.in_1_hour,
                '4h': Interval.in_4_hour,
                '1d': Interval.in_daily,
            }
            tv_interval = interval_map.get(interval, Interval.in_1_hour)
            
            df = self.tv.get_hist(
                symbol=tv_symbol,
                exchange=exchange,
                interval=tv_interval,
                n_bars=bars
            )
            
            if df is not None and len(df) > 0:
                # Rename columns to match our format
                df = df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })
                df.index.name = 'Date'
                return df
            
        except Exception as e:
            print(f"TradingView fetch error for {symbol}: {e}")
        
        return None


class YahooDataFetcher:
    """Fetch data from Yahoo Finance (fallback)"""
    
    def normalize_symbol(self, symbol: str) -> str:
        """Normalize symbol for Yahoo Finance"""
        symbol = symbol.upper()
        
        # Convert crypto symbols
        if symbol == 'BTCUSD':
            return 'BTC-USD'
        if symbol == 'ETHUSD':
            return 'ETH-USD'
        if symbol == 'SOLUSD':
            return 'SOL-USD'
        if symbol == 'LINKUSD':
            return 'LINK-USD'
        if symbol == 'LTCUSD':
            return 'LTC-USD'
        if symbol == 'UNIUSD':
            return 'UNI-USD'
        
        # Futures
        if symbol == 'ES':
            return 'ES=F'
        if symbol == 'NQ':
            return 'NQ=F'
        if symbol == 'GC':
            return 'GC=F'
        
        return symbol
    
    def fetch(self, symbol: str, interval: str = '1h', bars: int = 500) -> Optional[pd.DataFrame]:
        """Fetch historical data from Yahoo Finance"""
        if not YF_AVAILABLE:
            return None
        
        try:
            yf_symbol = self.normalize_symbol(symbol)
            
            # Calculate period based on bars and interval
            if interval == '1h':
                days = bars // 24 + 30
                period = f'{min(days, 730)}d'  # Max 2 years for hourly
            elif interval == '1d':
                days = bars + 30
                period = f'{min(days, 3650)}d'
            else:
                period = '60d'
            
            ticker = yf.Ticker(yf_symbol)
            df = ticker.history(period=period, interval=interval)
            
            if df is not None and len(df) > 0:
                # Ensure proper column names
                df = df.rename(columns={
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                })
                
                # Take last N bars
                df = df.tail(bars)
                return df
            
        except Exception as e:
            print(f"Yahoo Finance fetch error for {symbol}: {e}")
        
        return None


class DataManager:
    """Manage data fetching from multiple sources"""
    
    def __init__(self, tv_username: str = None, tv_password: str = None):
        self.tv_fetcher = TradingViewDataFetcher(tv_username, tv_password) if TV_AVAILABLE else None
        self.yf_fetcher = YahooDataFetcher() if YF_AVAILABLE else None
        self.cache: Dict[str, Dict] = {}
        self.cache_ttl = 60  # Cache TTL in seconds
    
    def fetch(self, symbol: str, interval: str = '1h', bars: int = 500) -> Optional[pd.DataFrame]:
        """Fetch data with caching and fallback"""
        cache_key = f"{symbol}_{interval}"
        
        # Check cache
        if cache_key in self.cache:
            cached = self.cache[cache_key]
            if time.time() - cached['timestamp'] < self.cache_ttl:
                return cached['data']
        
        df = None
        
        # Try TradingView first
        if self.tv_fetcher:
            df = self.tv_fetcher.fetch(symbol, interval, bars)
            if df is not None:
                print(f"[{symbol}] Fetched {len(df)} bars from TradingView")
        
        # Fallback to Yahoo Finance
        if df is None and self.yf_fetcher:
            df = self.yf_fetcher.fetch(symbol, interval, bars)
            if df is not None:
                print(f"[{symbol}] Fetched {len(df)} bars from Yahoo Finance")
        
        # Cache the result
        if df is not None:
            self.cache[cache_key] = {
                'data': df,
                'timestamp': time.time()
            }
        
        return df
    
    def get_current_price(self, symbol: str) -> Optional[float]:
        """Get current price for a symbol"""
        df = self.fetch(symbol, '1h', 10)
        if df is not None and len(df) > 0:
            return float(df['Close'].iloc[-1])
        return None


# =============================================================================
# PAPER TRADING ENGINE
# =============================================================================

class PaperPosition:
    """Represents a paper trading position"""
    
    def __init__(
        self,
        symbol: str,
        direction: int,
        entry_price: float,
        quantity: float,
        stop_loss: float,
        take_profit: float,
        entry_time: datetime,
        signal_data: Dict = None
    ):
        self.symbol = symbol
        self.direction = direction  # 1 = long, -1 = short
        self.entry_price = entry_price
        self.quantity = quantity
        self.stop_loss = stop_loss
        self.take_profit = take_profit
        self.entry_time = entry_time
        self.signal_data = signal_data or {}
        self.bars_held = 0
        self.unrealized_pnl = 0.0
        self.closed = False
        self.exit_price = None
        self.exit_time = None
        self.exit_reason = None
        self.realized_pnl = 0.0
    
    def update(self, current_price: float):
        """Update position with current price"""
        if self.closed:
            return
        
        self.bars_held += 1
        
        if self.direction == 1:
            self.unrealized_pnl = (current_price - self.entry_price) * self.quantity
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.quantity
    
    def check_exit(self, current_price: float) -> Optional[str]:
        """Check if position should be closed"""
        if self.closed:
            return None
        
        if self.direction == 1:  # Long
            if current_price <= self.stop_loss:
                return 'stop_loss'
            if current_price >= self.take_profit:
                return 'take_profit'
        else:  # Short
            if current_price >= self.stop_loss:
                return 'stop_loss'
            if current_price <= self.take_profit:
                return 'take_profit'
        
        return None
    
    def close(self, exit_price: float, reason: str):
        """Close the position"""
        self.closed = True
        self.exit_price = exit_price
        self.exit_time = datetime.now()
        self.exit_reason = reason
        
        if self.direction == 1:
            self.realized_pnl = (exit_price - self.entry_price) * self.quantity
        else:
            self.realized_pnl = (self.entry_price - exit_price) * self.quantity
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'symbol': self.symbol,
            'direction': 'LONG' if self.direction == 1 else 'SHORT',
            'entry_price': self.entry_price,
            'quantity': self.quantity,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'entry_time': self.entry_time.isoformat(),
            'bars_held': self.bars_held,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'closed': self.closed,
            'exit_reason': self.exit_reason
        }


class PaperTradingEngine:
    """Paper trading simulation engine"""
    
    def __init__(self, initial_balance: float = 100000.0, risk_per_trade: float = 0.02):
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.risk_per_trade = risk_per_trade
        
        self.positions: Dict[str, PaperPosition] = {}
        self.closed_trades: List[Dict] = []
        self.trade_history: List[Dict] = []
        
        # Stats
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        self.peak_balance = initial_balance
        self.max_drawdown = 0.0
    
    def calculate_position_size(
        self,
        entry_price: float,
        stop_loss: float,
        symbol: str = None
    ) -> float:
        """Calculate position size based on risk"""
        risk_amount = self.balance * self.risk_per_trade
        stop_distance = abs(entry_price - stop_loss)
        
        if stop_distance <= 0:
            return 0.0
        
        quantity = risk_amount / stop_distance
        
        # Round based on symbol type
        if symbol and ('BTC' in symbol or 'ETH' in symbol):
            quantity = round(quantity, 4)
        elif symbol and ('USD' in symbol):
            quantity = round(quantity, 2)
        else:
            quantity = round(quantity, 2)
        
        return quantity
    
    def open_position(
        self,
        symbol: str,
        direction: int,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        signal_data: Dict = None
    ) -> Optional[PaperPosition]:
        """Open a new paper position"""
        
        # Check if already have position in this symbol
        if symbol in self.positions:
            print(f"[{symbol}] Already have open position")
            return None
        
        # Calculate position size
        quantity = self.calculate_position_size(entry_price, stop_loss, symbol)
        
        if quantity <= 0:
            print(f"[{symbol}] Invalid position size")
            return None
        
        # Create position
        position = PaperPosition(
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=datetime.now(),
            signal_data=signal_data
        )
        
        self.positions[symbol] = position
        self.total_trades += 1
        
        # Record trade
        trade_record = {
            'type': 'OPEN',
            'symbol': symbol,
            'direction': 'LONG' if direction == 1 else 'SHORT',
            'entry_price': entry_price,
            'quantity': quantity,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'time': datetime.now().isoformat(),
            'signal_data': signal_data
        }
        self.trade_history.append(trade_record)
        
        print(f"[{symbol}] OPENED {'LONG' if direction == 1 else 'SHORT'} @ {entry_price:.2f}")
        print(f"  Qty: {quantity}, SL: {stop_loss:.2f}, TP: {take_profit:.2f}")
        
        return position
    
    def close_position(self, symbol: str, exit_price: float, reason: str) -> Optional[Dict]:
        """Close a paper position"""
        if symbol not in self.positions:
            return None
        
        position = self.positions[symbol]
        position.close(exit_price, reason)
        
        # Update stats
        pnl = position.realized_pnl
        self.balance += pnl
        self.total_pnl += pnl
        
        if pnl > 0:
            self.winning_trades += 1
        else:
            self.losing_trades += 1
        
        # Update peak and drawdown
        if self.balance > self.peak_balance:
            self.peak_balance = self.balance
        
        current_drawdown = (self.peak_balance - self.balance) / self.peak_balance
        if current_drawdown > self.max_drawdown:
            self.max_drawdown = current_drawdown
        
        # Record closed trade
        trade_record = {
            'type': 'CLOSE',
            'symbol': symbol,
            'direction': 'LONG' if position.direction == 1 else 'SHORT',
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'quantity': position.quantity,
            'pnl': pnl,
            'bars_held': position.bars_held,
            'exit_reason': reason,
            'time': datetime.now().isoformat()
        }
        self.trade_history.append(trade_record)
        self.closed_trades.append(trade_record)
        
        # Remove from active positions
        del self.positions[symbol]
        
        pnl_icon = "+" if pnl >= 0 else ""
        print(f"[{symbol}] CLOSED @ {exit_price:.2f} - {reason}")
        print(f"  P&L: {pnl_icon}${pnl:.2f}, Balance: ${self.balance:.2f}")
        
        return trade_record
    
    def update_positions(self, prices: Dict[str, float]):
        """Update all positions with current prices"""
        for symbol, position in list(self.positions.items()):
            if symbol not in prices:
                continue
            
            current_price = prices[symbol]
            position.update(current_price)
            
            # Check for stop/target hit
            exit_reason = position.check_exit(current_price)
            if exit_reason:
                self.close_position(symbol, current_price, exit_reason)
    
    def get_stats(self) -> Dict:
        """Get trading statistics"""
        win_rate = self.winning_trades / max(self.total_trades, 1) * 100
        
        avg_win = 0
        avg_loss = 0
        if self.winning_trades > 0:
            wins = [t['pnl'] for t in self.closed_trades if t['pnl'] > 0]
            avg_win = sum(wins) / len(wins)
        if self.losing_trades > 0:
            losses = [t['pnl'] for t in self.closed_trades if t['pnl'] <= 0]
            avg_loss = abs(sum(losses) / len(losses))
        
        profit_factor = avg_win * self.winning_trades / max(avg_loss * self.losing_trades, 1)
        
        return {
            'balance': self.balance,
            'total_pnl': self.total_pnl,
            'total_pnl_pct': (self.balance - self.initial_balance) / self.initial_balance * 100,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'max_drawdown': self.max_drawdown * 100,
            'open_positions': len(self.positions)
        }
    
    def print_stats(self):
        """Print trading statistics"""
        stats = self.get_stats()
        
        print("\n" + "=" * 60)
        print("PAPER TRADING STATISTICS")
        print("=" * 60)
        print(f"Balance:        ${stats['balance']:,.2f}")
        print(f"Total P&L:      ${stats['total_pnl']:+,.2f} ({stats['total_pnl_pct']:+.2f}%)")
        print(f"Total Trades:   {stats['total_trades']}")
        print(f"Win Rate:       {stats['win_rate']:.1f}%")
        print(f"Wins/Losses:    {stats['winning_trades']}/{stats['losing_trades']}")
        print(f"Avg Win:        ${stats['avg_win']:.2f}")
        print(f"Avg Loss:       ${stats['avg_loss']:.2f}")
        print(f"Profit Factor:  {stats['profit_factor']:.2f}")
        print(f"Max Drawdown:   {stats['max_drawdown']:.2f}%")
        print(f"Open Positions: {stats['open_positions']}")
        print("=" * 60)


# =============================================================================
# V9 TRADER
# =============================================================================

class V9TradingViewTrader:
    """V9 Trading system using TradingView data"""
    
    def __init__(
        self,
        symbols: List[str],
        mode: str = 'shadow',  # 'shadow' or 'paper'
        risk_pct: float = 0.02,
        rr_ratio: float = 4.0,
        confluence_threshold: int = 60,
        use_rl: bool = True,
        rl_model_path: str = None,
        poll_interval: int = 60,
        tv_username: str = None,
        tv_password: str = None
    ):
        self.symbols = symbols
        self.mode = mode
        self.risk_pct = risk_pct
        self.rr_ratio = rr_ratio
        self.confluence_threshold = confluence_threshold
        self.use_rl = use_rl
        self.poll_interval = poll_interval
        
        # Data manager
        self.data_manager = DataManager(tv_username, tv_password)
        
        # Paper trading engine
        self.paper_engine = PaperTradingEngine(
            initial_balance=100000.0,
            risk_per_trade=risk_pct
        )
        
        # V8 signal generator
        self.signal_gen = None
        self.confluence_threshold = confluence_threshold
        self.rr_ratio = rr_ratio
        if V8_AVAILABLE:
            self.signal_gen = V8SignalGenerator(use_rl=use_rl)
            
            # Load RL model if provided
            if use_rl and rl_model_path and os.path.exists(rl_model_path):
                try:
                    with open(rl_model_path, 'rb') as f:
                        rl_data = pickle.load(f)
                    self.signal_gen.rl_agent = rl_data.get('rl_agent')
                    print(f"Loaded RL model from {rl_model_path}")
                except Exception as e:
                    print(f"Failed to load RL model: {e}")
        
        # Historical data cache
        self.historical_data: Dict[str, Dict] = {}
        
        # State
        self.running = False
        self.last_signals: Dict[str, Dict] = {}
        
        # Telegram - disabled for V9 (can enable with --telegram flag)
        self.tg_enabled = False
    
    def _prepare_data(self, symbol: str) -> Optional[Dict]:
        """Prepare data for signal generation (matching IBKR format)"""
        # Fetch hourly data
        df_1h = self.data_manager.fetch(symbol, '1h', 500)
        if df_1h is None or len(df_1h) < 100:
            return None
        
        # Fetch daily data for HTF bias
        df_1d = self.data_manager.fetch(symbol, '1d', 100)
        
        opens = df_1h['Open'].values
        highs = df_1h['High'].values
        lows = df_1h['Low'].values
        closes = df_1h['Close'].values
        
        # Detect FVGs
        bullish_fvgs = []
        bearish_fvgs = []
        for i in range(3, len(df_1h)):
            if lows[i] > highs[i-2]:
                bullish_fvgs.append({'idx': i, 'mid': (highs[i-2] + lows[i]) / 2, 'high': lows[i]})
            if highs[i] < lows[i-2]:
                bearish_fvgs.append({'idx': i, 'mid': (highs[i] + lows[i-2]) / 2, 'low': highs[i]})
        
        # Calculate HTF trend from daily data
        if df_1d is not None and len(df_1d) >= 5:
            daily_highs = df_1d['High'].values
            daily_lows = df_1d['Low'].values
            htf = []
            for i in range(1, len(df_1d)):
                if daily_highs[i] > np.max(daily_highs[max(0,i-5):i]) and daily_lows[i] > np.min(daily_lows[max(0,i-5):i]):
                    htf.append(1)
                elif daily_highs[i] < np.max(daily_highs[max(0,i-5):i]) and daily_lows[i] < np.min(daily_lows[max(0,i-5):i]):
                    htf.append(-1)
                else:
                    htf.append(0)
            
            df_daily_index = pd.DatetimeIndex(df_1d.index).tz_localize(None)
            df_index = pd.DatetimeIndex(df_1h.index).tz_localize(None)
            htf_trend = np.zeros(len(df_1h))
            for i in range(len(df_1h)):
                bar_time = df_index[i]
                for j in range(len(df_1d) - 1, -1, -1):
                    if df_daily_index[j] <= bar_time:
                        htf_trend[i] = htf[j] if j < len(htf) else 0
                        break
        else:
            htf_trend = np.zeros(len(df_1h))
        
        # Calculate LTF trend
        trend = np.zeros(len(df_1h))
        for i in range(20, len(df_1h)):
            rh = np.max(highs[max(0,i-20):i])
            rl = np.min(lows[max(0,i-20):i])
            if rh > highs[i-5] and rl > lows[i-5]:
                trend[i] = 1
            elif rh < highs[i-5] and rl < lows[i-5]:
                trend[i] = -1
        
        # Calculate price position
        price_position = np.zeros(len(df_1h))
        for i in range(20, len(df_1h)):
            ph = np.max(highs[i-20:i])
            pl = np.min(lows[i-20:i])
            rng = ph - pl
            if rng < 0.001:
                rng = 0.001
            price_position[i] = (closes[i] - pl) / rng
        
        # Calculate kill zones
        hours = pd.to_datetime(df_1h.index).hour.values
        kill_zone = np.zeros(len(df_1h), dtype=bool)
        for i in range(len(hours)):
            h = hours[i]
            kill_zone[i] = (1 <= h < 5) or (7 <= h < 12) or (13.5 <= h < 16)
        
        # Calculate volatility (ATR-like)
        volatility = np.zeros(len(df_1h))
        for i in range(14, len(df_1h)):
            trs = []
            for j in range(max(0, i-14), i+1):
                tr = max(highs[j] - lows[j], abs(highs[j] - closes[j-1]), abs(lows[j] - closes[j-1])) if j > 0 else highs[j] - lows[j]
                trs.append(tr)
            volatility[i] = np.mean(trs) if trs else 0
        
        # Return in IBKR format
        return {
            'symbol': symbol,
            'df': df_1h,
            'opens': opens,
            'highs': highs,
            'lows': lows,
            'closes': closes,
            'bullish_fvgs': bullish_fvgs,
            'bearish_fvgs': bearish_fvgs,
            'htf_trend': htf_trend,
            'ltf_trend': trend,
            'price_position': price_position,
            'kill_zone': kill_zone,
            'volatility': volatility,
            'hours': hours
        }
    
    def _check_signal(self, symbol: str) -> Optional[Dict]:
        """Check for trading signal"""
        data = self._prepare_data(symbol)
        if data is None:
            return None
        
        self.historical_data[symbol] = data
        
        if self.signal_gen is None:
            return None
        
        try:
            idx = len(data['closes']) - 1
            signal = self.signal_gen.generate_signal(data, idx)
            
            if signal and signal.get('direction', 0) != 0:
                confluence = signal.get('confluence', 0)
                
                if confluence >= self.confluence_threshold:
                    return signal
        
        except Exception as e:
            print(f"[{symbol}] Signal generation error: {e}")
        
        return None
    
    def _execute_signal(self, symbol: str, signal: Dict):
        """Execute a trading signal"""
        direction = signal.get('direction', 0)
        entry = signal.get('entry', 0)
        stop = signal.get('stop', 0)
        target = signal.get('target', 0)
        confluence = signal.get('confluence', 0)
        
        if direction == 0 or entry == 0:
            return
        
        # Check if paused
        if TG_AVAILABLE and tn.is_trading_paused():
            print(f"[{symbol}] Trading paused - skipping signal")
            return
        
        # Shadow mode - just log
        if self.mode == 'shadow':
            dir_str = "LONG" if direction == 1 else "SHORT"
            print(f"\n[{symbol}] SHADOW SIGNAL: {dir_str}")
            print(f"  Entry: {entry:.4f}, Stop: {stop:.4f}, Target: {target:.4f}")
            print(f"  Confluence: {confluence}")
            
            # Send Telegram notification
            if self.tg_enabled:
                try:
                    tn.send_signal_alert(
                        symbol=symbol,
                        direction=direction,
                        confluence=confluence,
                        pd_zone=signal.get('pd_zone', 'unknown'),
                        current_price=entry,
                        rl_action=signal.get('rl_action', 'N/A')
                    )
                except Exception as e:
                    print(f"Telegram error: {e}")
            
            return
        
        # Paper mode - execute
        position = self.paper_engine.open_position(
            symbol=symbol,
            direction=direction,
            entry_price=entry,
            stop_loss=stop,
            take_profit=target,
            signal_data=signal
        )
        
        if position and self.tg_enabled:
            try:
                risk_amount = self.paper_engine.balance * self.risk_pct
                tn.send_trade_entry(
                    symbol=symbol,
                    direction=direction,
                    entry_price=entry,
                    stop_loss=stop,
                    take_profit=target,
                    quantity=position.quantity,
                    confluence=confluence,
                    pd_zone=signal.get('pd_zone', 'unknown'),
                    rl_action=signal.get('rl_action', 'ENTER'),
                    risk_amount=risk_amount
                )
            except Exception as e:
                print(f"Telegram error: {e}")
    
    def _update_positions(self):
        """Update paper positions"""
        if self.mode != 'paper':
            return
        
        # Get current prices
        prices = {}
        for symbol in self.paper_engine.positions:
            price = self.data_manager.get_current_price(symbol)
            if price:
                prices[symbol] = price
        
        # Check for closed positions before update
        closed_symbols = set(self.paper_engine.positions.keys())
        
        # Update positions
        self.paper_engine.update_positions(prices)
        
        # Check what was closed
        for symbol in closed_symbols:
            if symbol not in self.paper_engine.positions:
                # Position was closed
                trade = self.paper_engine.closed_trades[-1]
                
                if self.tg_enabled:
                    try:
                        tn.send_trade_exit(
                            symbol=symbol,
                            direction=1 if trade['direction'] == 'LONG' else -1,
                            entry_price=trade['entry_price'],
                            exit_price=trade['exit_price'],
                            quantity=trade['quantity'],
                            pnl=trade['pnl'],
                            exit_reason=trade['exit_reason'],
                            bars_held=trade['bars_held']
                        )
                    except Exception as e:
                        print(f"Telegram error: {e}")
    
    def _poll_cycle(self):
        """Run one polling cycle"""
        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Polling {len(self.symbols)} symbols...")
        
        for symbol in self.symbols:
            try:
                # Skip if already have position
                if symbol in self.paper_engine.positions:
                    continue
                
                # Check for signal
                signal = self._check_signal(symbol)
                
                if signal:
                    self._execute_signal(symbol, signal)
                    self.last_signals[symbol] = signal
                
            except Exception as e:
                print(f"[{symbol}] Error: {e}")
        
        # Update positions
        self._update_positions()
    
    def start(self):
        """Start the trading system"""
        self.running = True
        
        print("\n" + "=" * 60)
        print("V9 TRADINGVIEW TRADING SYSTEM")
        print("=" * 60)
        print(f"Mode:       {self.mode}")
        print(f"Symbols:    {', '.join(self.symbols)}")
        print(f"Risk:       {self.risk_pct * 100:.1f}%")
        print(f"R:R Ratio:  1:{self.rr_ratio}")
        print(f"Confluence: {self.confluence_threshold}")
        print(f"RL Agent:   {'Enabled' if self.use_rl and self.signal_gen and self.signal_gen.rl_agent else 'Disabled'}")
        print(f"Telegram:   {'Enabled' if self.tg_enabled else 'Disabled'}")
        print("=" * 60)
        
        # Send startup notification
        if self.tg_enabled:
            try:
                mode_emoji = "👀" if self.mode == 'shadow' else "📝"
                tn.send_notification(
                    f"{mode_emoji} <b>V9 TradingView Bot Started</b>\n\n"
                    f"Mode: {self.mode.upper()}\n"
                    f"Symbols: {', '.join(self.symbols)}\n"
                    f"Risk: {self.risk_pct * 100:.1f}%"
                )
            except:
                pass
        
        iteration = 0
        
        try:
            while self.running:
                self._poll_cycle()
                iteration += 1
                
                # Print stats periodically
                if iteration % 10 == 0 and self.mode == 'paper':
                    self.paper_engine.print_stats()
                
                # Send hourly update
                if iteration % 60 == 0 and self.tg_enabled:
                    try:
                        stats = self.paper_engine.get_stats()
                        tn.send_position_update(
                            positions=self.paper_engine.positions,
                            total_pnl=stats['total_pnl'],
                            win_rate=stats['win_rate'],
                            open_count=stats['open_positions']
                        )
                    except:
                        pass
                
                # Wait for next cycle
                time.sleep(self.poll_interval)
                
        except KeyboardInterrupt:
            print("\nStopping...")
        
        self.stop()
    
    def stop(self):
        """Stop the trading system"""
        self.running = False
        
        # Print final stats
        if self.mode == 'paper':
            self.paper_engine.print_stats()
        
        # Send shutdown notification
        if self.tg_enabled:
            try:
                stats = self.paper_engine.get_stats()
                tn.send_notification(
                    f"V9 Bot Stopped\n\n"
                    f"Final P&L: ${stats['total_pnl']:+,.2f}\n"
                    f"Trades: {stats['total_trades']}\n"
                    f"Win Rate: {stats['win_rate']:.1f}%"
                )
            except:
                pass


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='V9 TradingView Trading System')
    parser.add_argument('--symbols', type=str, default='BTCUSD,ETHUSD',
                        help='Comma-separated symbols')
    parser.add_argument('--mode', type=str, default='shadow',
                        choices=['shadow', 'paper'],
                        help='Trading mode (shadow=log only, paper=simulate)')
    parser.add_argument('--risk', type=float, default=0.02,
                        help='Risk per trade (0.02 = 2%%)')
    parser.add_argument('--rr', type=float, default=4.0,
                        help='Risk:Reward ratio')
    parser.add_argument('--confluence', type=int, default=60,
                        help='Minimum confluence threshold')
    parser.add_argument('--no-rl', action='store_true',
                        help='Disable RL agent')
    parser.add_argument('--rl-model', type=str, default='v8_rl_model.pkl',
                        help='Path to RL model')
    parser.add_argument('--interval', type=int, default=60,
                        help='Poll interval in seconds')
    parser.add_argument('--tv-user', type=str, default=None,
                        help='TradingView username')
    parser.add_argument('--tv-pass', type=str, default=None,
                        help='TradingView password')
    
    args = parser.parse_args()
    
    # Check dependencies
    if not TV_AVAILABLE and not YF_AVAILABLE:
        print("ERROR: No data source available!")
        print("Install tvDatafeed: pip install tvDatafeed")
        print("Or yfinance: pip install yfinance")
        return
    
    if not V8_AVAILABLE:
        print("ERROR: v8_backtest.py not found!")
        return
    
    # Parse symbols
    symbols = [s.strip().upper() for s in args.symbols.split(',')]
    
    # Create trader
    trader = V9TradingViewTrader(
        symbols=symbols,
        mode=args.mode,
        risk_pct=args.risk,
        rr_ratio=args.rr,
        confluence_threshold=args.confluence,
        use_rl=not args.no_rl,
        rl_model_path=args.rl_model,
        poll_interval=args.interval,
        tv_username=args.tv_user,
        tv_password=args.tv_pass
    )
    
    # Start trading
    trader.start()


if __name__ == "__main__":
    main()
