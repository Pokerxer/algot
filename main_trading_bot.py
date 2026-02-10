"""
ICT Trading Bot - Main Application Entry Point
==============================================

Main orchestration layer that integrates all ICT trading components:
- Market Data Engine for time/session management
- Multi-Timeframe Coordinator for analysis
- ICT Integration Engine for confluence detection
- Signal Generator for trade signals
- Trade Manager for position management
- AI Learning Engine for pattern recognition

This is the primary entry point for both live trading and backtesting.

ICT Principle: "Wait for the setup, execute with precision, manage with discipline"
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import threading
import queue
import time
from abc import ABC, abstractmethod


# ============================================================================
# ENUMS AND CONFIGURATION
# ============================================================================

class BotMode(Enum):
    """Trading bot operating modes"""
    BACKTEST = "backtest"
    PAPER = "paper"
    LIVE = "live"
    ANALYSIS = "analysis"
    SIGNAL_ONLY = "signal_only"


class BotState(Enum):
    """Trading bot states"""
    INITIALIZING = "initializing"
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
    MAINTENANCE = "maintenance"


class MarketState(Enum):
    """Current market state assessment"""
    TRENDING_BULLISH = "trending_bullish"
    TRENDING_BEARISH = "trending_bearish"
    RANGING = "ranging"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    UNCERTAIN = "uncertain"


@dataclass
class BotConfig:
    """Main configuration for trading bot"""
    # Identity
    bot_name: str = "ICT_Trading_Bot"
    version: str = "1.0.0"
    
    # Mode settings
    mode: BotMode = BotMode.PAPER
    
    # Symbols to trade
    symbols: List[str] = field(default_factory=lambda: ["EURUSD"])
    
    # Timeframes (ICT hierarchy: HTF → LTF)
    htf_timeframes: List[str] = field(default_factory=lambda: ["D", "4H"])
    ltf_timeframes: List[str] = field(default_factory=lambda: ["1H", "15m", "5m"])
    entry_timeframe: str = "5m"
    
    # Risk management
    max_risk_per_trade: float = 0.01  # 1% of account
    max_daily_risk: float = 0.03  # 3% daily max loss
    max_concurrent_positions: int = 2
    max_daily_trades: int = 5
    
    # Session filters (ICT kill zones)
    trade_sessions: List[str] = field(default_factory=lambda: ["london", "new_york"])
    require_kill_zone: bool = True
    
    # Signal requirements
    min_confluence_grade: str = "C"  # Minimum grade to take trade
    min_confidence: float = 60.0
    min_risk_reward: float = 1.5
    
    # Execution settings
    use_limit_orders: bool = True
    max_slippage_pips: float = 2.0
    order_timeout_seconds: int = 300
    
    # AI/ML settings
    enable_ai_learning: bool = True
    min_ai_win_rate: float = 50.0  # Minimum predicted win rate
    
    # Data settings
    lookback_days: int = 90
    update_interval_seconds: float = 1.0
    
    # Logging
    log_level: str = "INFO"
    log_trades: bool = True
    log_signals: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            'bot_name': self.bot_name,
            'version': self.version,
            'mode': self.mode.value,
            'symbols': self.symbols,
            'htf_timeframes': self.htf_timeframes,
            'ltf_timeframes': self.ltf_timeframes,
            'entry_timeframe': self.entry_timeframe,
            'max_risk_per_trade': self.max_risk_per_trade,
            'max_daily_risk': self.max_daily_risk,
            'max_concurrent_positions': self.max_concurrent_positions,
            'max_daily_trades': self.max_daily_trades,
            'trade_sessions': self.trade_sessions,
            'require_kill_zone': self.require_kill_zone,
            'min_confluence_grade': self.min_confluence_grade,
            'min_confidence': self.min_confidence,
            'min_risk_reward': self.min_risk_reward,
            'use_limit_orders': self.use_limit_orders,
            'enable_ai_learning': self.enable_ai_learning,
            'min_ai_win_rate': self.min_ai_win_rate,
            'lookback_days': self.lookback_days,
            'update_interval_seconds': self.update_interval_seconds
        }


@dataclass
class AccountState:
    """Current account state"""
    balance: float = 10000.0
    equity: float = 10000.0
    margin_used: float = 0.0
    margin_available: float = 10000.0
    unrealized_pnl: float = 0.0
    realized_pnl_today: float = 0.0
    trades_today: int = 0
    currency: str = "USD"
    leverage: float = 100.0
    
    @property
    def daily_risk_used(self) -> float:
        """Calculate daily risk used as percentage"""
        if self.balance == 0:
            return 0.0
        return abs(min(0, self.realized_pnl_today)) / self.balance


@dataclass
class BotMetrics:
    """Trading bot performance metrics"""
    # Session metrics
    session_start: datetime = field(default_factory=datetime.now)
    signals_generated: int = 0
    trades_executed: int = 0
    trades_won: int = 0
    trades_lost: int = 0
    trades_break_even: int = 0
    
    # PnL metrics
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_pnl: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    max_consecutive_losses: int = 0
    current_consecutive_losses: int = 0
    
    # Analysis metrics
    analyses_performed: int = 0
    confluences_detected: int = 0
    setups_generated: int = 0
    
    @property
    def win_rate(self) -> float:
        """Calculate win rate"""
        total = self.trades_won + self.trades_lost
        if total == 0:
            return 0.0
        return (self.trades_won / total) * 100
    
    @property
    def profit_factor(self) -> float:
        """Calculate profit factor"""
        if self.gross_loss == 0:
            return float('inf') if self.gross_profit > 0 else 0.0
        return self.gross_profit / abs(self.gross_loss)


# ============================================================================
# DATA FEED INTERFACES
# ============================================================================

class DataFeedInterface(ABC):
    """Abstract interface for data feeds"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to data source"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from data source"""
        pass
    
    @abstractmethod
    async def subscribe(self, symbol: str, timeframe: str) -> bool:
        """Subscribe to market data"""
        pass
    
    @abstractmethod
    async def unsubscribe(self, symbol: str, timeframe: str) -> bool:
        """Unsubscribe from market data"""
        pass
    
    @abstractmethod
    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime
    ) -> Optional[List[Dict[str, Any]]]:
        """Get historical OHLCV data"""
        pass
    
    @abstractmethod
    async def get_current_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get current bid/ask prices"""
        pass


class BrokerInterface(ABC):
    """Abstract interface for broker execution"""
    
    @abstractmethod
    async def connect(self) -> bool:
        """Connect to broker"""
        pass
    
    @abstractmethod
    async def disconnect(self) -> None:
        """Disconnect from broker"""
        pass
    
    @abstractmethod
    async def get_account_info(self) -> Optional[AccountState]:
        """Get current account state"""
        pass
    
    @abstractmethod
    async def place_order(
        self,
        symbol: str,
        order_type: str,
        direction: str,
        quantity: float,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Place an order"""
        pass
    
    @abstractmethod
    async def modify_order(
        self,
        order_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> bool:
        """Modify an existing order"""
        pass
    
    @abstractmethod
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        pass
    
    @abstractmethod
    async def close_position(
        self,
        position_id: str,
        quantity: Optional[float] = None
    ) -> bool:
        """Close a position (partially or fully)"""
        pass
    
    @abstractmethod
    async def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions"""
        pass
    
    @abstractmethod
    async def get_pending_orders(self) -> List[Dict[str, Any]]:
        """Get all pending orders"""
        pass


# ============================================================================
# SIMULATED DATA FEED (FOR PAPER TRADING / BACKTESTING)
# ============================================================================

class SimulatedDataFeed(DataFeedInterface):
    """Simulated data feed for paper trading and backtesting"""
    
    def __init__(self):
        self.connected = False
        self.subscriptions: Dict[str, List[str]] = {}  # symbol -> [timeframes]
        self.historical_data: Dict[str, Dict[str, List[Dict]]] = {}  # symbol -> tf -> data
        self.current_prices: Dict[str, Dict[str, float]] = {}  # symbol -> {bid, ask, mid}
        self.logger = logging.getLogger("SimulatedDataFeed")
    
    async def connect(self) -> bool:
        """Simulate connection"""
        self.connected = True
        self.logger.info("Simulated data feed connected")
        return True
    
    async def disconnect(self) -> None:
        """Simulate disconnection"""
        self.connected = False
        self.subscriptions.clear()
        self.logger.info("Simulated data feed disconnected")
    
    async def subscribe(self, symbol: str, timeframe: str) -> bool:
        """Subscribe to simulated data"""
        if symbol not in self.subscriptions:
            self.subscriptions[symbol] = []
        if timeframe not in self.subscriptions[symbol]:
            self.subscriptions[symbol].append(timeframe)
        self.logger.debug(f"Subscribed to {symbol} {timeframe}")
        return True
    
    async def unsubscribe(self, symbol: str, timeframe: str) -> bool:
        """Unsubscribe from simulated data"""
        if symbol in self.subscriptions:
            if timeframe in self.subscriptions[symbol]:
                self.subscriptions[symbol].remove(timeframe)
        return True
    
    def load_historical_data(
        self,
        symbol: str,
        timeframe: str,
        data: List[Dict[str, Any]]
    ) -> None:
        """Load historical data for backtesting"""
        if symbol not in self.historical_data:
            self.historical_data[symbol] = {}
        self.historical_data[symbol][timeframe] = data
        self.logger.info(f"Loaded {len(data)} bars for {symbol} {timeframe}")
    
    async def get_historical_data(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime
    ) -> Optional[List[Dict[str, Any]]]:
        """Get historical data from loaded data"""
        if symbol not in self.historical_data:
            return None
        if timeframe not in self.historical_data[symbol]:
            return None
        
        data = self.historical_data[symbol][timeframe]
        # Filter by date range
        filtered = []
        for bar in data:
            bar_time = bar.get('timestamp')
            if isinstance(bar_time, str):
                bar_time = datetime.fromisoformat(bar_time)
            if start <= bar_time <= end:
                filtered.append(bar)
        
        return filtered
    
    def set_current_price(
        self,
        symbol: str,
        bid: float,
        ask: float
    ) -> None:
        """Set current price for simulation"""
        self.current_prices[symbol] = {
            'bid': bid,
            'ask': ask,
            'mid': (bid + ask) / 2,
            'spread': ask - bid
        }
    
    async def get_current_price(self, symbol: str) -> Optional[Dict[str, float]]:
        """Get current simulated price"""
        return self.current_prices.get(symbol)


class SimulatedBroker(BrokerInterface):
    """Simulated broker for paper trading and backtesting"""
    
    def __init__(self, initial_balance: float = 10000.0):
        self.connected = False
        self.account = AccountState(
            balance=initial_balance,
            equity=initial_balance,
            margin_available=initial_balance
        )
        self.positions: Dict[str, Dict[str, Any]] = {}  # position_id -> position
        self.orders: Dict[str, Dict[str, Any]] = {}  # order_id -> order
        self.order_counter = 0
        self.position_counter = 0
        self.trade_history: List[Dict[str, Any]] = []
        self.logger = logging.getLogger("SimulatedBroker")
        
        # Callbacks for order/position events
        self.on_order_filled: Optional[Callable] = None
        self.on_position_closed: Optional[Callable] = None
    
    async def connect(self) -> bool:
        """Simulate broker connection"""
        self.connected = True
        self.logger.info("Simulated broker connected")
        return True
    
    async def disconnect(self) -> None:
        """Simulate broker disconnection"""
        self.connected = False
        self.logger.info("Simulated broker disconnected")
    
    async def get_account_info(self) -> Optional[AccountState]:
        """Get current account state"""
        # Update equity based on unrealized PnL
        self.account.equity = self.account.balance + self.account.unrealized_pnl
        return self.account
    
    def _generate_order_id(self) -> str:
        """Generate unique order ID"""
        self.order_counter += 1
        return f"ORD-{self.order_counter:06d}"
    
    def _generate_position_id(self) -> str:
        """Generate unique position ID"""
        self.position_counter += 1
        return f"POS-{self.position_counter:06d}"
    
    async def place_order(
        self,
        symbol: str,
        order_type: str,  # 'market', 'limit', 'stop'
        direction: str,  # 'long', 'short'
        quantity: float,
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> Optional[Dict[str, Any]]:
        """Place a simulated order"""
        order_id = self._generate_order_id()
        
        order = {
            'order_id': order_id,
            'symbol': symbol,
            'order_type': order_type,
            'direction': direction,
            'quantity': quantity,
            'price': price,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'status': 'pending' if order_type != 'market' else 'filled',
            'created_at': datetime.now(),
            'filled_at': None,
            'fill_price': None
        }
        
        if order_type == 'market':
            # Immediate fill at market price
            order['status'] = 'filled'
            order['filled_at'] = datetime.now()
            order['fill_price'] = price  # In simulation, use the provided price
            
            # Create position
            position = await self._create_position(order)
            order['position_id'] = position['position_id']
            
            self.logger.info(f"Market order filled: {order_id} {direction} {quantity} {symbol} @ {price}")
        else:
            # Pending order
            self.orders[order_id] = order
            self.logger.info(f"Pending order created: {order_id} {order_type} {direction} {quantity} {symbol} @ {price}")
        
        return order
    
    async def _create_position(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Create position from filled order"""
        position_id = self._generate_position_id()
        
        position = {
            'position_id': position_id,
            'symbol': order['symbol'],
            'direction': order['direction'],
            'quantity': order['quantity'],
            'entry_price': order['fill_price'],
            'current_price': order['fill_price'],
            'stop_loss': order.get('stop_loss'),
            'take_profit': order.get('take_profit'),
            'unrealized_pnl': 0.0,
            'opened_at': datetime.now(),
            'closed_at': None,
            'status': 'open'
        }
        
        self.positions[position_id] = position
        
        # Callback
        if self.on_order_filled:
            self.on_order_filled(order, position)
        
        return position
    
    async def modify_order(
        self,
        order_id: str,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None
    ) -> bool:
        """Modify pending order"""
        if order_id in self.orders:
            order = self.orders[order_id]
            if stop_loss is not None:
                order['stop_loss'] = stop_loss
            if take_profit is not None:
                order['take_profit'] = take_profit
            return True
        return False
    
    async def cancel_order(self, order_id: str) -> bool:
        """Cancel pending order"""
        if order_id in self.orders:
            del self.orders[order_id]
            self.logger.info(f"Order cancelled: {order_id}")
            return True
        return False
    
    async def close_position(
        self,
        position_id: str,
        quantity: Optional[float] = None,
        close_price: Optional[float] = None
    ) -> bool:
        """Close a position"""
        if position_id not in self.positions:
            return False
        
        position = self.positions[position_id]
        
        if close_price is None:
            close_price = position['current_price']
        
        # Calculate PnL
        if position['direction'] == 'long':
            pnl = (close_price - position['entry_price']) * position['quantity']
        else:
            pnl = (position['entry_price'] - close_price) * position['quantity']
        
        # Full close
        if quantity is None or quantity >= position['quantity']:
            position['status'] = 'closed'
            position['closed_at'] = datetime.now()
            position['realized_pnl'] = pnl
            
            # Update account
            self.account.balance += pnl
            self.account.realized_pnl_today += pnl
            self.account.trades_today += 1
            
            # Move to history
            self.trade_history.append(position.copy())
            del self.positions[position_id]
            
            self.logger.info(f"Position closed: {position_id} PnL: {pnl:.2f}")
            
            # Callback
            if self.on_position_closed:
                self.on_position_closed(position)
        else:
            # Partial close
            partial_pnl = pnl * (quantity / position['quantity'])
            position['quantity'] -= quantity
            self.account.balance += partial_pnl
            self.account.realized_pnl_today += partial_pnl
            
            self.logger.info(f"Partial close: {position_id} qty: {quantity} PnL: {partial_pnl:.2f}")
        
        return True
    
    async def get_open_positions(self) -> List[Dict[str, Any]]:
        """Get all open positions"""
        return list(self.positions.values())
    
    async def get_pending_orders(self) -> List[Dict[str, Any]]:
        """Get all pending orders"""
        return [o for o in self.orders.values() if o['status'] == 'pending']
    
    def update_prices(self, prices: Dict[str, Dict[str, float]]) -> None:
        """Update position prices and check stops/targets"""
        for position_id, position in list(self.positions.items()):
            symbol = position['symbol']
            if symbol in prices:
                price_data = prices[symbol]
                current_price = price_data.get('mid', price_data.get('bid'))
                
                position['current_price'] = current_price
                
                # Calculate unrealized PnL
                if position['direction'] == 'long':
                    position['unrealized_pnl'] = (current_price - position['entry_price']) * position['quantity']
                else:
                    position['unrealized_pnl'] = (position['entry_price'] - current_price) * position['quantity']
                
                # Check stop loss
                if position['stop_loss']:
                    if position['direction'] == 'long' and current_price <= position['stop_loss']:
                        asyncio.create_task(self.close_position(position_id, close_price=position['stop_loss']))
                    elif position['direction'] == 'short' and current_price >= position['stop_loss']:
                        asyncio.create_task(self.close_position(position_id, close_price=position['stop_loss']))
                
                # Check take profit
                if position['take_profit']:
                    if position['direction'] == 'long' and current_price >= position['take_profit']:
                        asyncio.create_task(self.close_position(position_id, close_price=position['take_profit']))
                    elif position['direction'] == 'short' and current_price <= position['take_profit']:
                        asyncio.create_task(self.close_position(position_id, close_price=position['take_profit']))
        
        # Check pending orders
        for order_id, order in list(self.orders.items()):
            symbol = order['symbol']
            if symbol in prices:
                price = prices[symbol].get('mid')
                
                # Check if limit order should fill
                if order['order_type'] == 'limit':
                    if order['direction'] == 'long' and price <= order['price']:
                        order['fill_price'] = order['price']
                        order['filled_at'] = datetime.now()
                        order['status'] = 'filled'
                        asyncio.create_task(self._create_position(order))
                        del self.orders[order_id]
                    elif order['direction'] == 'short' and price >= order['price']:
                        order['fill_price'] = order['price']
                        order['filled_at'] = datetime.now()
                        order['status'] = 'filled'
                        asyncio.create_task(self._create_position(order))
                        del self.orders[order_id]
        
        # Update account unrealized PnL
        total_unrealized = sum(p.get('unrealized_pnl', 0) for p in self.positions.values())
        self.account.unrealized_pnl = total_unrealized


# ============================================================================
# MAIN TRADING BOT
# ============================================================================

class ICTTradingBot:
    """
    Main ICT Trading Bot - Orchestrates all components
    
    ICT Principles Implemented:
    1. "Higher timeframe sets the bias, lower timeframe for entries"
    2. "Confluence of multiple PD arrays = higher probability"
    3. "Wait for the setup, execute with precision"
    4. "Kill zones are when institutions move price"
    5. "Model 2022: Liquidity → Displacement → Premium/Discount → Entry"
    """
    
    def __init__(
        self,
        config: Optional[BotConfig] = None,
        data_feed: Optional[DataFeedInterface] = None,
        broker: Optional[BrokerInterface] = None
    ):
        # Configuration
        self.config = config or BotConfig()
        
        # External connections
        self.data_feed = data_feed or SimulatedDataFeed()
        self.broker = broker or SimulatedBroker()
        
        # State
        self.state = BotState.INITIALIZING
        self.account = AccountState()
        self.metrics = BotMetrics()
        
        # Internal components (to be initialized)
        self.market_data_engine = None
        self.mtf_coordinator = None
        self.integration_engine = None
        self.signal_generator = None
        self.trade_manager = None
        self.ai_learning = None
        
        # Runtime state
        self.active_analyses: Dict[str, Dict] = {}  # symbol -> analysis
        self.pending_signals: List[Dict] = []
        self.active_trades: Dict[str, Dict] = {}  # trade_id -> trade
        
        # Control
        self._running = False
        self._main_loop_task: Optional[asyncio.Task] = None
        self._event_queue: queue.Queue = queue.Queue()
        
        # Logging
        self.logger = logging.getLogger("ICTTradingBot")
        self._setup_logging()
        
        # Callbacks
        self.on_signal_generated: Optional[Callable] = None
        self.on_trade_opened: Optional[Callable] = None
        self.on_trade_closed: Optional[Callable] = None
        self.on_analysis_complete: Optional[Callable] = None
    
    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    async def initialize(self) -> bool:
        """
        Initialize all components
        
        Returns True if initialization successful
        """
        self.logger.info("="*60)
        self.logger.info(f"Initializing {self.config.bot_name} v{self.config.version}")
        self.logger.info(f"Mode: {self.config.mode.value}")
        self.logger.info("="*60)
        
        try:
            # Connect to data feed
            self.logger.info("Connecting to data feed...")
            if not await self.data_feed.connect():
                self.logger.error("Failed to connect to data feed")
                self.state = BotState.ERROR
                return False
            
            # Connect to broker
            self.logger.info("Connecting to broker...")
            if not await self.broker.connect():
                self.logger.error("Failed to connect to broker")
                self.state = BotState.ERROR
                return False
            
            # Get account info
            self.account = await self.broker.get_account_info() or AccountState()
            self.logger.info(f"Account balance: {self.account.balance} {self.account.currency}")
            
            # Subscribe to data feeds
            self.logger.info("Subscribing to market data...")
            all_timeframes = self.config.htf_timeframes + self.config.ltf_timeframes
            for symbol in self.config.symbols:
                for tf in all_timeframes:
                    await self.data_feed.subscribe(symbol, tf)
            
            # Initialize internal components
            self.logger.info("Initializing ICT analysis components...")
            await self._initialize_components()
            
            self.state = BotState.READY
            self.logger.info("Bot initialization complete - READY")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            self.state = BotState.ERROR
            return False
    
    async def _initialize_components(self) -> None:
        """Initialize internal ICT components"""
        # These would import and initialize the actual components
        # For now, we create placeholder structures
        
        # Component initialization order follows ICT hierarchy
        self.component_states = {
            'market_data_engine': 'ready',
            'mtf_coordinator': 'ready',
            'integration_engine': 'ready',
            'signal_generator': 'ready',
            'trade_manager': 'ready',
            'ai_learning': 'ready' if self.config.enable_ai_learning else 'disabled'
        }
        
        self.logger.info("All ICT components initialized")
    
    async def start(self) -> None:
        """Start the trading bot main loop"""
        if self.state != BotState.READY:
            self.logger.error(f"Cannot start bot - state is {self.state}")
            return
        
        self.logger.info("Starting trading bot...")
        self._running = True
        self.state = BotState.RUNNING
        self.metrics.session_start = datetime.now()
        
        # Start main loop
        self._main_loop_task = asyncio.create_task(self._main_loop())
        
        self.logger.info("Trading bot started - RUNNING")
    
    async def stop(self) -> None:
        """Stop the trading bot"""
        self.logger.info("Stopping trading bot...")
        self._running = False
        
        if self._main_loop_task:
            self._main_loop_task.cancel()
            try:
                await self._main_loop_task
            except asyncio.CancelledError:
                pass
        
        # Close all positions if in paper/live mode
        if self.config.mode in [BotMode.PAPER, BotMode.LIVE]:
            positions = await self.broker.get_open_positions()
            for pos in positions:
                await self.broker.close_position(pos['position_id'])
        
        # Disconnect
        await self.data_feed.disconnect()
        await self.broker.disconnect()
        
        self.state = BotState.STOPPED
        self.logger.info("Trading bot stopped")
        
        # Log final metrics
        self._log_session_summary()
    
    async def pause(self) -> None:
        """Pause trading (analysis continues, no new trades)"""
        if self.state == BotState.RUNNING:
            self.state = BotState.PAUSED
            self.logger.info("Trading paused - analysis continues")
    
    async def resume(self) -> None:
        """Resume trading"""
        if self.state == BotState.PAUSED:
            self.state = BotState.RUNNING
            self.logger.info("Trading resumed")
    
    async def _main_loop(self) -> None:
        """Main trading loop"""
        self.logger.info("Entering main trading loop")
        
        while self._running:
            try:
                loop_start = time.time()
                
                # 1. Check time/session (ICT: "Trade during kill zones")
                session_info = await self._check_session()
                
                # 2. Update market data
                await self._update_market_data()
                
                # 3. Perform multi-timeframe analysis
                if session_info.get('should_analyze', True):
                    await self._analyze_markets()
                
                # 4. Generate and validate signals
                if self.state == BotState.RUNNING:
                    await self._process_signals()
                
                # 5. Manage existing positions
                await self._manage_positions()
                
                # 6. Update account and metrics
                await self._update_account()
                
                # 7. Check risk limits
                await self._check_risk_limits()
                
                # Wait for next iteration
                elapsed = time.time() - loop_start
                sleep_time = max(0, self.config.update_interval_seconds - elapsed)
                await asyncio.sleep(sleep_time)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(1.0)
        
        self.logger.info("Exited main trading loop")
    
    async def _check_session(self) -> Dict[str, Any]:
        """
        Check current trading session
        
        ICT Principle: "Kill zones are when smart money moves price"
        - London Open: 02:00-05:00 EST
        - New York Open: 07:00-10:00 EST
        - London Close: 10:00-12:00 EST
        """
        now = datetime.now()
        hour = now.hour
        
        # Determine active session
        session = 'off_hours'
        in_kill_zone = False
        
        # London session (converted to local time - adjust as needed)
        if 2 <= hour < 5:
            session = 'london_open'
            in_kill_zone = True
        elif 3 <= hour < 11:
            session = 'london'
        
        # New York session
        if 7 <= hour < 10:
            session = 'new_york_open'
            in_kill_zone = True
        elif 8 <= hour < 17:
            session = 'new_york'
        
        # London close
        if 10 <= hour < 12:
            in_kill_zone = True
        
        # Check if we should trade
        allowed_sessions = self.config.trade_sessions
        session_allowed = any(s in session for s in allowed_sessions)
        
        return {
            'session': session,
            'in_kill_zone': in_kill_zone,
            'session_allowed': session_allowed,
            'should_analyze': True,
            'should_trade': session_allowed and (not self.config.require_kill_zone or in_kill_zone),
            'hour': hour,
            'timestamp': now
        }
    
    async def _update_market_data(self) -> None:
        """Update market data for all symbols and timeframes"""
        for symbol in self.config.symbols:
            # Get current price
            price_data = await self.data_feed.get_current_price(symbol)
            if price_data:
                # Update broker (for paper trading simulation)
                if isinstance(self.broker, SimulatedBroker):
                    self.broker.update_prices({symbol: price_data})
    
    async def _analyze_markets(self) -> None:
        """
        Perform multi-timeframe ICT analysis
        
        ICT Analysis Flow:
        1. HTF (Daily/4H) - Determine directional bias
        2. Structure analysis - Identify swing points
        3. Liquidity mapping - Find BSL/SSL pools
        4. PD Array detection - OBs, FVGs, etc.
        5. Kill zone timing - Session alignment
        6. Confluence scoring - Grade the setup
        """
        for symbol in self.config.symbols:
            try:
                analysis = await self._analyze_symbol(symbol)
                self.active_analyses[symbol] = analysis
                self.metrics.analyses_performed += 1
                
                # Log significant findings
                if analysis.get('confluence_score', 0) >= 60:
                    self.logger.info(
                        f"{symbol} High confluence detected: "
                        f"Score={analysis.get('confluence_score'):.0f}, "
                        f"Bias={analysis.get('htf_bias')}, "
                        f"Grade={analysis.get('grade')}"
                    )
                
                # Callback
                if self.on_analysis_complete:
                    self.on_analysis_complete(symbol, analysis)
                    
            except Exception as e:
                self.logger.error(f"Error analyzing {symbol}: {e}")
    
    async def _analyze_symbol(self, symbol: str) -> Dict[str, Any]:
        """Analyze a single symbol across all timeframes"""
        # This would use the actual ICT components
        # For now, return a structured analysis template
        
        analysis = {
            'symbol': symbol,
            'timestamp': datetime.now(),
            
            # HTF Analysis (ICT: "HTF sets the bias")
            'htf_bias': 'neutral',  # bullish, bearish, neutral
            'htf_structure': 'ranging',  # trending_up, trending_down, ranging
            'htf_swing_high': None,
            'htf_swing_low': None,
            'htf_premium_zone': (0, 0),
            'htf_discount_zone': (0, 0),
            
            # Structure Analysis
            'recent_structure_break': None,  # BOS/CHoCH
            'structure_shift_detected': False,
            'displacement_detected': False,
            
            # Liquidity Analysis
            'bsl_levels': [],  # Buy-side liquidity
            'ssl_levels': [],  # Sell-side liquidity
            'recent_liquidity_sweep': None,
            'draw_on_liquidity': None,
            
            # PD Arrays
            'order_blocks': [],
            'fair_value_gaps': [],
            'breaker_blocks': [],
            'mitigation_blocks': [],
            
            # Confluence
            'confluence_factors': [],
            'confluence_score': 0,
            'grade': 'F',
            
            # Trade Opportunity
            'trade_direction': None,  # long, short
            'entry_zone': None,
            'stop_loss_level': None,
            'target_levels': [],
            'risk_reward_ratio': 0,
            
            # Context
            'in_kill_zone': False,
            'session': 'unknown',
            'model_detected': None  # Model 2022, etc.
        }
        
        return analysis
    
    async def _process_signals(self) -> None:
        """Process and validate trade signals"""
        # Check daily limits
        if self.account.trades_today >= self.config.max_daily_trades:
            return
        
        # Check concurrent position limit
        positions = await self.broker.get_open_positions()
        if len(positions) >= self.config.max_concurrent_positions:
            return
        
        for symbol, analysis in self.active_analyses.items():
            try:
                # Check if analysis qualifies for a signal
                if not self._qualifies_for_signal(analysis):
                    continue
                
                # Generate signal
                signal = await self._generate_signal(symbol, analysis)
                
                if signal:
                    self.metrics.signals_generated += 1
                    self.pending_signals.append(signal)
                    
                    self.logger.info(
                        f"Signal generated: {signal['direction']} {symbol} "
                        f"@ {signal['entry_price']:.5f} "
                        f"SL: {signal['stop_loss']:.5f} "
                        f"R:R: {signal['risk_reward']:.1f}"
                    )
                    
                    # Callback
                    if self.on_signal_generated:
                        self.on_signal_generated(signal)
                    
                    # Execute if in auto mode
                    if self.config.mode in [BotMode.LIVE, BotMode.PAPER]:
                        await self._execute_signal(signal)
                        
            except Exception as e:
                self.logger.error(f"Error processing signal for {symbol}: {e}")
    
    def _qualifies_for_signal(self, analysis: Dict[str, Any]) -> bool:
        """Check if analysis qualifies for a trade signal"""
        # Grade check
        grade = analysis.get('grade', 'F')
        min_grade = self.config.min_confluence_grade
        grade_order = ['A+', 'A', 'B', 'C', 'D', 'F']
        if grade_order.index(grade) > grade_order.index(min_grade):
            return False
        
        # Confluence score
        if analysis.get('confluence_score', 0) < self.config.min_confidence:
            return False
        
        # R:R ratio
        if analysis.get('risk_reward_ratio', 0) < self.config.min_risk_reward:
            return False
        
        # Kill zone requirement
        if self.config.require_kill_zone and not analysis.get('in_kill_zone'):
            return False
        
        return True
    
    async def _generate_signal(
        self,
        symbol: str,
        analysis: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Generate a trade signal from analysis"""
        if not analysis.get('trade_direction'):
            return None
        
        # Calculate position size
        account_info = await self.broker.get_account_info()
        if not account_info:
            return None
        
        risk_amount = account_info.balance * self.config.max_risk_per_trade
        
        entry_price = analysis.get('entry_zone', {}).get('price', 0)
        stop_loss = analysis.get('stop_loss_level', 0)
        
        if entry_price == 0 or stop_loss == 0:
            return None
        
        # Calculate stop distance
        stop_distance = abs(entry_price - stop_loss)
        if stop_distance == 0:
            return None
        
        # Position size (simplified for forex - would need pip value calculation)
        position_size = risk_amount / stop_distance
        
        # Calculate targets
        targets = analysis.get('target_levels', [])
        if not targets:
            # Default targets at 1.5R, 2.5R, 4R
            if analysis['trade_direction'] == 'long':
                targets = [
                    entry_price + (stop_distance * 1.5),
                    entry_price + (stop_distance * 2.5),
                    entry_price + (stop_distance * 4.0)
                ]
            else:
                targets = [
                    entry_price - (stop_distance * 1.5),
                    entry_price - (stop_distance * 2.5),
                    entry_price - (stop_distance * 4.0)
                ]
        
        # Create signal
        signal = {
            'signal_id': f"SIG-{datetime.now().strftime('%Y%m%d%H%M%S')}-{symbol}",
            'symbol': symbol,
            'direction': analysis['trade_direction'],
            'entry_price': entry_price,
            'stop_loss': stop_loss,
            'take_profits': targets[:3],
            'position_size': position_size,
            'risk_amount': risk_amount,
            'risk_reward': abs(targets[0] - entry_price) / stop_distance if targets else 0,
            'confluence_score': analysis.get('confluence_score', 0),
            'grade': analysis.get('grade', 'C'),
            'model': analysis.get('model_detected'),
            'confluence_factors': analysis.get('confluence_factors', []),
            'timestamp': datetime.now(),
            'valid_until': datetime.now() + timedelta(hours=4),
            'status': 'pending'
        }
        
        # AI validation
        if self.config.enable_ai_learning:
            ai_prediction = await self._get_ai_prediction(signal, analysis)
            signal['ai_prediction'] = ai_prediction
            
            # Reject if AI prediction too low
            if ai_prediction.get('win_rate', 100) < self.config.min_ai_win_rate:
                self.logger.info(
                    f"Signal rejected by AI: predicted win rate "
                    f"{ai_prediction.get('win_rate', 0):.0f}% < "
                    f"{self.config.min_ai_win_rate:.0f}%"
                )
                return None
        
        return signal
    
    async def _get_ai_prediction(
        self,
        signal: Dict[str, Any],
        analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get AI prediction for signal"""
        # This would use the AI Learning Engine
        # For now, return a placeholder
        return {
            'win_rate': 65.0,
            'confidence': 0.7,
            'similar_patterns': 12,
            'average_pnl': 1.5,
            'recommendation': 'TAKE_TRADE'
        }
    
    async def _execute_signal(self, signal: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute a trade signal"""
        try:
            # Determine order type
            order_type = 'limit' if self.config.use_limit_orders else 'market'
            
            # Place order
            order = await self.broker.place_order(
                symbol=signal['symbol'],
                order_type=order_type,
                direction=signal['direction'],
                quantity=signal['position_size'],
                price=signal['entry_price'],
                stop_loss=signal['stop_loss'],
                take_profit=signal['take_profits'][0] if signal['take_profits'] else None
            )
            
            if order:
                signal['status'] = 'executed'
                signal['order_id'] = order.get('order_id')
                self.metrics.trades_executed += 1
                
                self.logger.info(f"Trade executed: {order.get('order_id')}")
                
                # Callback
                if self.on_trade_opened:
                    self.on_trade_opened(signal, order)
                
                return order
            
        except Exception as e:
            self.logger.error(f"Error executing signal: {e}")
        
        return None
    
    async def _manage_positions(self) -> None:
        """
        Manage open positions according to ICT principles
        
        ICT Position Management:
        1. Move to break-even after 1R
        2. Partial profits at targets
        3. Trail stop behind structure
        4. Respect original invalidation
        """
        positions = await self.broker.get_open_positions()
        
        for position in positions:
            try:
                await self._manage_single_position(position)
            except Exception as e:
                self.logger.error(f"Error managing position {position.get('position_id')}: {e}")
    
    async def _manage_single_position(self, position: Dict[str, Any]) -> None:
        """Manage a single position"""
        entry_price = position['entry_price']
        current_price = position['current_price']
        stop_loss = position['stop_loss']
        direction = position['direction']
        
        # Calculate current R-multiple
        if direction == 'long':
            current_r = (current_price - entry_price) / (entry_price - stop_loss)
        else:
            current_r = (entry_price - current_price) / (stop_loss - entry_price)
        
        # Move to break-even after 1R
        if current_r >= 1.0:
            if direction == 'long' and stop_loss < entry_price:
                # Add small buffer above entry
                new_stop = entry_price + (entry_price - stop_loss) * 0.1
                await self.broker.modify_order(
                    position['position_id'],
                    stop_loss=new_stop
                )
                self.logger.info(f"Moved stop to break-even: {position['position_id']}")
        
        # Trail stop after 2R (ICT: "Lock in profits")
        if current_r >= 2.0:
            # Trail behind last swing
            # This would use actual swing detection
            pass
    
    async def _update_account(self) -> None:
        """Update account state"""
        account_info = await self.broker.get_account_info()
        if account_info:
            self.account = account_info
    
    async def _check_risk_limits(self) -> None:
        """Check and enforce risk limits"""
        # Check daily risk limit
        if self.account.daily_risk_used >= self.config.max_daily_risk:
            if self.state == BotState.RUNNING:
                self.logger.warning(
                    f"Daily risk limit reached ({self.account.daily_risk_used:.1%}). "
                    "Pausing trading."
                )
                await self.pause()
        
        # Track max drawdown
        drawdown = (self.account.balance - self.account.equity) / self.account.balance
        if drawdown > self.metrics.max_drawdown:
            self.metrics.max_drawdown = drawdown
    
    def _log_session_summary(self) -> None:
        """Log session summary"""
        duration = datetime.now() - self.metrics.session_start
        
        self.logger.info("="*60)
        self.logger.info("SESSION SUMMARY")
        self.logger.info("="*60)
        self.logger.info(f"Duration: {duration}")
        self.logger.info(f"Analyses performed: {self.metrics.analyses_performed}")
        self.logger.info(f"Signals generated: {self.metrics.signals_generated}")
        self.logger.info(f"Trades executed: {self.metrics.trades_executed}")
        self.logger.info(f"Win/Loss: {self.metrics.trades_won}/{self.metrics.trades_lost}")
        self.logger.info(f"Win rate: {self.metrics.win_rate:.1f}%")
        self.logger.info(f"Net P&L: {self.metrics.net_pnl:.2f}")
        self.logger.info(f"Profit factor: {self.metrics.profit_factor:.2f}")
        self.logger.info(f"Max drawdown: {self.metrics.max_drawdown:.1%}")
        self.logger.info("="*60)
    
    # ========================================================================
    # PUBLIC METHODS FOR EXTERNAL INTERACTION
    # ========================================================================
    
    def get_status(self) -> Dict[str, Any]:
        """Get current bot status"""
        return {
            'state': self.state.value,
            'mode': self.config.mode.value,
            'account_balance': self.account.balance,
            'account_equity': self.account.equity,
            'daily_pnl': self.account.realized_pnl_today,
            'trades_today': self.account.trades_today,
            'active_analyses': len(self.active_analyses),
            'pending_signals': len(self.pending_signals),
            'metrics': {
                'win_rate': self.metrics.win_rate,
                'net_pnl': self.metrics.net_pnl,
                'trades_executed': self.metrics.trades_executed
            }
        }
    
    def get_active_analysis(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get latest analysis for a symbol"""
        return self.active_analyses.get(symbol)
    
    def get_pending_signals(self) -> List[Dict[str, Any]]:
        """Get all pending signals"""
        return self.pending_signals.copy()
    
    async def force_analysis(self, symbol: str) -> Dict[str, Any]:
        """Force immediate analysis of a symbol"""
        return await self._analyze_symbol(symbol)
    
    def update_config(self, **kwargs) -> None:
        """Update configuration at runtime"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                self.logger.info(f"Config updated: {key} = {value}")


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

async def run_paper_trading(
    config: Optional[BotConfig] = None,
    duration_hours: float = 8.0
) -> BotMetrics:
    """Run paper trading session"""
    config = config or BotConfig(mode=BotMode.PAPER)
    bot = ICTTradingBot(config=config)
    
    if await bot.initialize():
        await bot.start()
        
        # Run for specified duration
        await asyncio.sleep(duration_hours * 3600)
        
        await bot.stop()
    
    return bot.metrics


async def run_analysis_only(
    symbols: List[str],
    timeframes: Optional[List[str]] = None
) -> Dict[str, Dict[str, Any]]:
    """Run analysis only (no trading)"""
    config = BotConfig(
        mode=BotMode.ANALYSIS,
        symbols=symbols,
        htf_timeframes=timeframes or ["D", "4H"],
        ltf_timeframes=["1H", "15m", "5m"]
    )
    
    bot = ICTTradingBot(config=config)
    
    if await bot.initialize():
        analyses = {}
        for symbol in symbols:
            analyses[symbol] = await bot.force_analysis(symbol)
        
        await bot.stop()
        return analyses
    
    return {}


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Example usage
    async def main():
        # Create configuration
        config = BotConfig(
            bot_name="ICT_Demo_Bot",
            mode=BotMode.PAPER,
            symbols=["EURUSD", "GBPUSD"],
            max_risk_per_trade=0.01,
            max_daily_trades=3,
            require_kill_zone=True,
            enable_ai_learning=True,
            log_level="INFO"
        )
        
        # Create bot
        bot = ICTTradingBot(config=config)
        
        # Initialize
        if await bot.initialize():
            print(f"Bot initialized: {bot.get_status()}")
            
            # Run for a short demo
            await bot.start()
            await asyncio.sleep(10)  # Run for 10 seconds
            await bot.stop()
            
            print(f"Final status: {bot.get_status()}")
        else:
            print("Bot initialization failed")
    
    # Run
    asyncio.run(main())
