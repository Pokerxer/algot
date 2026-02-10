"""
ICT Trade Executor - Order Management and Position Handling
============================================================

Handles trade execution, position management, and order lifecycle
with ICT-specific entry and exit strategies.

EXECUTION ARCHITECTURE:
======================

┌─────────────────────────────────────────────────────────────────────┐
│                      TRADE EXECUTOR                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                  ORDER GENERATION                             │    │
│  │                                                                │    │
│  │  Trade Setup → Order Parameters → Order Validation            │    │
│  │                                                                │    │
│  │  Entry Types:                                                  │    │
│  │   • Limit Order (at PD array level)                           │    │
│  │   • Market Order (immediate entry)                            │    │
│  │   • Stop Entry (break confirmation)                           │    │
│  │   • Scaled Entry (multiple positions)                         │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                        │
│                              ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                 RISK MANAGEMENT                               │    │
│  │                                                                │    │
│  │  ┌───────────────────────────────────────────────────────┐   │    │
│  │  │ Position Sizing:                                       │   │    │
│  │  │   • Fixed fractional (% of account)                    │   │    │
│  │  │   • Fixed dollar risk                                  │   │    │
│  │  │   • Volatility-adjusted                                │   │    │
│  │  │   • Kelly criterion (optional)                         │   │    │
│  │  └───────────────────────────────────────────────────────┘   │    │
│  │  ┌───────────────────────────────────────────────────────┐   │    │
│  │  │ Stop Loss Types (ICT):                                 │   │    │
│  │  │   • Below/Above OB                                     │   │    │
│  │  │   • Below/Above FVG                                    │   │    │
│  │  │   • Swing High/Low                                     │   │    │
│  │  │   • Consequent Encroachment based                      │   │    │
│  │  │   • Quadrant-based                                     │   │    │
│  │  └───────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                        │
│                              ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                 ORDER EXECUTION                               │    │
│  │                                                                │    │
│  │  Submit Order → Monitor Fill → Confirm Entry → Set Stops     │    │
│  │                                                                │    │
│  │  Execution Modes:                                              │    │
│  │   • Paper Trading (simulation)                                 │    │
│  │   • Live Trading (broker integration)                          │    │
│  │   • Hybrid (paper with alerts)                                 │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                        │
│                              ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                POSITION MANAGEMENT                            │    │
│  │                                                                │    │
│  │  ┌───────────────────────────────────────────────────────┐   │    │
│  │  │ Exit Strategies:                                       │   │    │
│  │  │   • Target 1: First FVG/OB target (partials)          │   │    │
│  │  │   • Target 2: Liquidity target (runner)               │   │    │
│  │  │   • Break-even: Move stop after target 1              │   │    │
│  │  │   • Trailing: Follow structure                        │   │    │
│  │  │   • Time-based: Close before session end              │   │    │
│  │  └───────────────────────────────────────────────────────┘   │    │
│  │  ┌───────────────────────────────────────────────────────┐   │    │
│  │  │ Position Updates:                                      │   │    │
│  │  │   • Scale out at targets                              │   │    │
│  │  │   • Move to break-even                                │   │    │
│  │  │   • Trail stops                                       │   │    │
│  │  │   • Add to winners (pyramiding)                       │   │    │
│  │  └───────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                        │
│                              ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                 TRADE TRACKING                                │    │
│  │                                                                │    │
│  │  • Real-time P&L                                              │    │
│  │  • MAE/MFE tracking                                           │    │
│  │  • Trade journal logging                                      │    │
│  │  • Outcome recording for learning                             │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘

ICT Execution Principles:
- Wait for price to come to you (limit orders)
- Enter at CE (consequent encroachment) for best fills
- Use time distortion (consolidation) for confirmation
- Scale out at logical targets
- Never widen stops, only tighten
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from datetime import datetime, timedelta
import logging
import json
import uuid
from abc import ABC, abstractmethod

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side"""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class PositionStatus(Enum):
    """Position status"""
    OPEN = "open"
    PARTIAL_CLOSED = "partial_closed"
    CLOSED = "closed"
    BREAK_EVEN = "break_even"


class ExecutionMode(Enum):
    """Execution mode"""
    PAPER = "paper"            # Simulated trading
    LIVE = "live"              # Real trading
    HYBRID = "hybrid"          # Paper with alerts


class StopType(Enum):
    """Stop loss placement type"""
    BELOW_OB = "below_ob"
    ABOVE_OB = "above_ob"
    BELOW_FVG = "below_fvg"
    ABOVE_FVG = "above_fvg"
    SWING_HIGH = "swing_high"
    SWING_LOW = "swing_low"
    FIXED_POINTS = "fixed_points"
    QUADRANT = "quadrant"
    CE_BASED = "ce_based"


class TargetType(Enum):
    """Take profit target type"""
    LIQUIDITY = "liquidity"
    FVG = "fvg"
    ORDER_BLOCK = "order_block"
    FIXED_RR = "fixed_rr"
    SWING = "swing"
    STANDARD_DEV = "standard_dev"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Order:
    """Order representation"""
    order_id: str
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None
    stop_price: Optional[float] = None
    
    # Status
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    
    # Timestamps
    created_at: datetime = field(default_factory=datetime.now)
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    
    # Metadata
    parent_order_id: Optional[str] = None
    is_stop_loss: bool = False
    is_take_profit: bool = False
    notes: str = ""
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'filled_price': self.filled_price,
            'created_at': self.created_at.isoformat(),
        }


@dataclass
class StopLoss:
    """Stop loss configuration"""
    price: float
    stop_type: StopType
    original_price: float           # Initial SL price
    
    # Trailing configuration
    is_trailing: bool = False
    trail_distance: float = 0.0     # Points
    trail_trigger: float = 0.0      # Price at which to start trailing
    
    # Break-even configuration
    move_to_be: bool = False
    be_trigger: float = 0.0         # Price at which to move to BE
    be_offset: float = 0.0          # Offset from entry for BE
    
    # Status
    is_at_break_even: bool = False
    times_adjusted: int = 0
    last_adjusted: Optional[datetime] = None


@dataclass
class TakeProfit:
    """Take profit configuration"""
    price: float
    target_type: TargetType
    
    # Partial configuration
    is_partial: bool = False
    partial_quantity_pct: float = 0.5  # 50% by default
    
    # Status
    is_hit: bool = False
    hit_at: Optional[datetime] = None


@dataclass
class Position:
    """Position representation"""
    position_id: str
    symbol: str
    direction: str                  # 'long' or 'short'
    
    # Entry
    entry_price: float
    entry_time: datetime
    quantity: float
    remaining_quantity: float
    
    # Stop/Target
    stop_loss: StopLoss
    take_profit_1: TakeProfit
    take_profit_2: Optional[TakeProfit] = None
    
    # Status
    status: PositionStatus = PositionStatus.OPEN
    
    # P&L
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    
    # Tracking
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0
    current_price: float = 0.0
    
    # Associated orders
    entry_order_id: str = ""
    stop_order_id: str = ""
    tp1_order_id: str = ""
    tp2_order_id: str = ""
    
    # Metadata
    setup_grade: str = ""
    model_type: str = ""
    confluence_score: float = 0.0
    notes: List[str] = field(default_factory=list)
    
    def update_pnl(self, current_price: float):
        """Update P&L based on current price"""
        self.current_price = current_price
        
        if self.direction == 'long':
            self.unrealized_pnl = (current_price - self.entry_price) * self.remaining_quantity
            excursion = current_price - self.entry_price
        else:
            self.unrealized_pnl = (self.entry_price - current_price) * self.remaining_quantity
            excursion = self.entry_price - current_price
        
        # Track MAE/MFE
        if excursion > 0:
            self.max_favorable_excursion = max(self.max_favorable_excursion, excursion)
        else:
            self.max_adverse_excursion = max(self.max_adverse_excursion, abs(excursion))
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'position_id': self.position_id,
            'symbol': self.symbol,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'quantity': self.quantity,
            'remaining_quantity': self.remaining_quantity,
            'stop_loss': self.stop_loss.price,
            'take_profit_1': self.take_profit_1.price,
            'status': self.status.value,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'mfe': self.max_favorable_excursion,
            'mae': self.max_adverse_excursion,
        }


@dataclass
class TradeExecution:
    """Complete trade execution record"""
    execution_id: str
    position: Position
    
    # Orders
    entry_order: Order
    stop_order: Optional[Order] = None
    tp_orders: List[Order] = field(default_factory=list)
    
    # Execution details
    slippage: float = 0.0
    commission: float = 0.0
    
    # Outcome
    outcome: Optional[str] = None     # 'win', 'loss', 'breakeven'
    exit_price: float = 0.0
    exit_time: Optional[datetime] = None
    exit_reason: str = ""
    
    # Analytics
    time_in_trade_minutes: int = 0
    hit_target_1: bool = False
    hit_target_2: bool = False


@dataclass
class ExecutorConfig:
    """Executor configuration"""
    # Execution mode
    mode: ExecutionMode = ExecutionMode.PAPER
    
    # Position sizing
    default_risk_percent: float = 1.0
    max_risk_percent: float = 2.0
    max_positions: int = 3
    
    # Entry settings
    use_limit_orders: bool = True
    limit_order_offset: float = 0.0   # Points from signal price
    order_timeout_minutes: int = 30
    
    # Stop/Target settings
    default_stop_buffer: float = 2.0  # Points buffer
    move_to_be_at_r: float = 1.0      # Move to BE at 1R profit
    trail_after_r: float = 2.0        # Start trailing after 2R
    
    # Partial exits
    enable_partials: bool = True
    partial_1_quantity: float = 0.5   # 50% at TP1
    partial_2_quantity: float = 0.3   # 30% at TP2
    
    # Session management
    close_before_session_end: bool = True
    session_end_buffer_minutes: int = 15


# =============================================================================
# BROKER INTERFACE (ABSTRACT)
# =============================================================================

class BrokerInterface(ABC):
    """Abstract broker interface"""
    
    @abstractmethod
    def submit_order(self, order: Order) -> Tuple[bool, str]:
        """Submit order to broker"""
        pass
    
    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        pass
    
    @abstractmethod
    def modify_order(self, order_id: str, new_price: float) -> bool:
        """Modify order price"""
        pass
    
    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get current position"""
        pass
    
    @abstractmethod
    def get_account_balance(self) -> float:
        """Get account balance"""
        pass


class PaperBroker(BrokerInterface):
    """Simulated paper trading broker"""
    
    def __init__(self, initial_balance: float = 100000):
        self.balance = initial_balance
        self.positions: Dict[str, Dict] = {}
        self.orders: Dict[str, Order] = {}
        self._next_fill_price: Optional[float] = None
    
    def submit_order(self, order: Order) -> Tuple[bool, str]:
        """Simulate order submission"""
        order.status = OrderStatus.SUBMITTED
        order.submitted_at = datetime.now()
        
        # Simulate immediate fill for market orders
        if order.order_type == OrderType.MARKET:
            order.status = OrderStatus.FILLED
            order.filled_at = datetime.now()
            order.filled_quantity = order.quantity
            order.filled_price = self._next_fill_price or order.price or 0
            return True, "Order filled"
        
        self.orders[order.order_id] = order
        return True, "Order submitted"
    
    def cancel_order(self, order_id: str) -> bool:
        """Cancel order"""
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELLED
            return True
        return False
    
    def modify_order(self, order_id: str, new_price: float) -> bool:
        """Modify order price"""
        if order_id in self.orders:
            self.orders[order_id].price = new_price
            return True
        return False
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get position"""
        return self.positions.get(symbol)
    
    def get_account_balance(self) -> float:
        """Get balance"""
        return self.balance
    
    def set_next_fill_price(self, price: float):
        """Set price for next fill (for simulation)"""
        self._next_fill_price = price
    
    def simulate_fill(self, order_id: str, price: float):
        """Simulate order fill"""
        if order_id in self.orders:
            order = self.orders[order_id]
            order.status = OrderStatus.FILLED
            order.filled_at = datetime.now()
            order.filled_quantity = order.quantity
            order.filled_price = price


# =============================================================================
# MAIN EXECUTOR CLASS
# =============================================================================

class TradeExecutor:
    """
    ICT Trade Executor
    
    Handles trade execution, position management, and order lifecycle
    with ICT-specific entry and exit strategies.
    
    Usage:
        executor = TradeExecutor(config)
        
        # Execute trade from setup
        execution = executor.execute_trade(setup)
        
        # Update positions with new price
        executor.update_positions(current_price)
        
        # Get open positions
        positions = executor.get_open_positions()
    """
    
    def __init__(self, config: Optional[ExecutorConfig] = None,
                 broker: Optional[BrokerInterface] = None):
        """
        Initialize the Trade Executor
        
        Args:
            config: Executor configuration
            broker: Broker interface (defaults to paper broker)
        """
        self.config = config or ExecutorConfig()
        self.broker = broker or PaperBroker()
        
        # Position tracking
        self._open_positions: Dict[str, Position] = {}
        self._closed_positions: List[Position] = []
        self._executions: Dict[str, TradeExecution] = {}
        
        # Order tracking
        self._pending_orders: Dict[str, Order] = {}
        self._filled_orders: Dict[str, Order] = {}
        
        # Account state
        self._account_balance = self.broker.get_account_balance()
        self._equity = self._account_balance
        self._open_pnl = 0.0
        
        # Statistics
        self._total_trades = 0
        self._winning_trades = 0
        self._losing_trades = 0
        
        logger.info(f"Trade Executor initialized in {self.config.mode.value} mode")
    
    # =========================================================================
    # TRADE EXECUTION
    # =========================================================================
    
    def execute_trade(self, setup: Dict[str, Any]) -> Optional[TradeExecution]:
        """
        Execute a trade from setup
        
        Args:
            setup: Trade setup dictionary with:
                - direction: 'long' or 'short'
                - symbol: Market symbol
                - entry_price: Entry price
                - stop_loss: Stop loss price
                - take_profit_1: First target
                - take_profit_2: Optional second target
                - grade: Setup grade
                - model_type: ICT model type
                - confluence_score: Confluence score
        
        Returns:
            TradeExecution or None if execution fails
        """
        # Validate setup
        if not self._validate_setup(setup):
            logger.warning("Setup validation failed")
            return None
        
        # Check position limits
        if len(self._open_positions) >= self.config.max_positions:
            logger.warning(f"Max positions ({self.config.max_positions}) reached")
            return None
        
        # Calculate position size
        position_size = self._calculate_position_size(
            entry=setup['entry_price'],
            stop=setup['stop_loss'],
            risk_percent=self.config.default_risk_percent
        )
        
        if position_size <= 0:
            logger.warning("Invalid position size calculated")
            return None
        
        # Generate order ID
        order_id = str(uuid.uuid4())[:8]
        position_id = str(uuid.uuid4())[:8]
        
        # Create entry order
        side = OrderSide.BUY if setup['direction'] == 'long' else OrderSide.SELL
        order_type = OrderType.LIMIT if self.config.use_limit_orders else OrderType.MARKET
        
        entry_order = Order(
            order_id=order_id,
            symbol=setup['symbol'],
            side=side,
            order_type=order_type,
            quantity=position_size,
            price=setup['entry_price'],
        )
        
        # Submit entry order
        success, message = self.broker.submit_order(entry_order)
        
        if not success:
            logger.error(f"Order submission failed: {message}")
            return None
        
        # Create stop loss
        stop_type = StopType.BELOW_OB if setup['direction'] == 'long' else StopType.ABOVE_OB
        stop_loss = StopLoss(
            price=setup['stop_loss'],
            stop_type=stop_type,
            original_price=setup['stop_loss'],
            move_to_be=True,
            be_trigger=setup['entry_price'] + (
                (setup['take_profit_1'] - setup['entry_price']) * self.config.move_to_be_at_r
                if setup['direction'] == 'long' else
                (setup['entry_price'] - setup['take_profit_1']) * self.config.move_to_be_at_r
            ) * (1 if setup['direction'] == 'long' else -1)
        )
        
        # Create take profit targets
        tp1 = TakeProfit(
            price=setup['take_profit_1'],
            target_type=TargetType.LIQUIDITY,
            is_partial=self.config.enable_partials,
            partial_quantity_pct=self.config.partial_1_quantity,
        )
        
        tp2 = None
        if setup.get('take_profit_2'):
            tp2 = TakeProfit(
                price=setup['take_profit_2'],
                target_type=TargetType.LIQUIDITY,
                is_partial=True,
                partial_quantity_pct=self.config.partial_2_quantity,
            )
        
        # Create position
        position = Position(
            position_id=position_id,
            symbol=setup['symbol'],
            direction=setup['direction'],
            entry_price=setup['entry_price'],
            entry_time=datetime.now(),
            quantity=position_size,
            remaining_quantity=position_size,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            entry_order_id=order_id,
            setup_grade=setup.get('grade', ''),
            model_type=setup.get('model_type', ''),
            confluence_score=setup.get('confluence_score', 0),
        )
        
        # Create execution record
        execution = TradeExecution(
            execution_id=position_id,
            position=position,
            entry_order=entry_order,
        )
        
        # Store
        self._pending_orders[order_id] = entry_order
        
        # For market orders, position is immediately open
        if order_type == OrderType.MARKET and entry_order.status == OrderStatus.FILLED:
            position.entry_price = entry_order.filled_price
            self._open_positions[position_id] = position
            self._executions[position_id] = execution
            self._total_trades += 1
            
            # Submit stop and TP orders
            self._submit_exit_orders(position)
            
            logger.info(f"Trade executed: {setup['direction'].upper()} {setup['symbol']} "
                       f"@ {entry_order.filled_price}, SL: {stop_loss.price}, "
                       f"TP1: {tp1.price}")
        
        return execution
    
    def _validate_setup(self, setup: Dict) -> bool:
        """Validate trade setup"""
        required = ['direction', 'symbol', 'entry_price', 'stop_loss', 'take_profit_1']
        for field in required:
            if field not in setup:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Validate direction
        if setup['direction'] not in ['long', 'short']:
            logger.error(f"Invalid direction: {setup['direction']}")
            return False
        
        # Validate prices
        if setup['direction'] == 'long':
            if setup['stop_loss'] >= setup['entry_price']:
                logger.error("Stop loss must be below entry for long")
                return False
            if setup['take_profit_1'] <= setup['entry_price']:
                logger.error("Take profit must be above entry for long")
                return False
        else:
            if setup['stop_loss'] <= setup['entry_price']:
                logger.error("Stop loss must be above entry for short")
                return False
            if setup['take_profit_1'] >= setup['entry_price']:
                logger.error("Take profit must be below entry for short")
                return False
        
        return True
    
    def _calculate_position_size(self, entry: float, stop: float,
                                risk_percent: float) -> float:
        """Calculate position size based on risk"""
        risk_amount = self._account_balance * (risk_percent / 100)
        risk_per_unit = abs(entry - stop)
        
        if risk_per_unit == 0:
            return 0
        
        size = risk_amount / risk_per_unit
        
        # Round to appropriate precision
        return round(size, 2)
    
    def _submit_exit_orders(self, position: Position):
        """Submit stop loss and take profit orders"""
        # Stop loss order
        stop_side = OrderSide.SELL if position.direction == 'long' else OrderSide.BUY
        stop_order = Order(
            order_id=str(uuid.uuid4())[:8],
            symbol=position.symbol,
            side=stop_side,
            order_type=OrderType.STOP,
            quantity=position.remaining_quantity,
            stop_price=position.stop_loss.price,
            parent_order_id=position.entry_order_id,
            is_stop_loss=True,
        )
        self.broker.submit_order(stop_order)
        position.stop_order_id = stop_order.order_id
        
        # Take profit 1 order
        tp_side = OrderSide.SELL if position.direction == 'long' else OrderSide.BUY
        tp1_quantity = (position.remaining_quantity * 
                       position.take_profit_1.partial_quantity_pct
                       if position.take_profit_1.is_partial else position.remaining_quantity)
        
        tp1_order = Order(
            order_id=str(uuid.uuid4())[:8],
            symbol=position.symbol,
            side=tp_side,
            order_type=OrderType.LIMIT,
            quantity=tp1_quantity,
            price=position.take_profit_1.price,
            parent_order_id=position.entry_order_id,
            is_take_profit=True,
        )
        self.broker.submit_order(tp1_order)
        position.tp1_order_id = tp1_order.order_id
    
    # =========================================================================
    # POSITION MANAGEMENT
    # =========================================================================
    
    def update_positions(self, symbol: str, current_price: float):
        """
        Update all positions with current price
        
        Args:
            symbol: Market symbol
            current_price: Current market price
        """
        for pos_id, position in list(self._open_positions.items()):
            if position.symbol != symbol:
                continue
            
            # Update P&L
            position.update_pnl(current_price)
            
            # Check stop loss hit
            if self._check_stop_hit(position, current_price):
                self._close_position(position, current_price, "Stop loss hit")
                continue
            
            # Check take profit 1
            if not position.take_profit_1.is_hit:
                if self._check_tp_hit(position, position.take_profit_1, current_price):
                    self._handle_partial_exit(position, position.take_profit_1, current_price)
            
            # Check take profit 2
            if position.take_profit_2 and not position.take_profit_2.is_hit:
                if self._check_tp_hit(position, position.take_profit_2, current_price):
                    self._handle_partial_exit(position, position.take_profit_2, current_price)
            
            # Check break-even trigger
            if not position.stop_loss.is_at_break_even:
                if self._check_be_trigger(position, current_price):
                    self._move_to_break_even(position)
        
        # Update equity
        self._update_equity()
    
    def _check_stop_hit(self, position: Position, price: float) -> bool:
        """Check if stop loss was hit"""
        if position.direction == 'long':
            return price <= position.stop_loss.price
        else:
            return price >= position.stop_loss.price
    
    def _check_tp_hit(self, position: Position, tp: TakeProfit, price: float) -> bool:
        """Check if take profit was hit"""
        if position.direction == 'long':
            return price >= tp.price
        else:
            return price <= tp.price
    
    def _check_be_trigger(self, position: Position, price: float) -> bool:
        """Check if break-even trigger was hit"""
        if not position.stop_loss.move_to_be:
            return False
        
        if position.direction == 'long':
            return price >= position.stop_loss.be_trigger
        else:
            return price <= position.stop_loss.be_trigger
    
    def _move_to_break_even(self, position: Position):
        """Move stop loss to break even"""
        position.stop_loss.price = position.entry_price + position.stop_loss.be_offset
        position.stop_loss.is_at_break_even = True
        position.stop_loss.times_adjusted += 1
        position.stop_loss.last_adjusted = datetime.now()
        position.status = PositionStatus.BREAK_EVEN
        
        # Update stop order with broker
        self.broker.modify_order(position.stop_order_id, position.stop_loss.price)
        
        position.notes.append(f"Moved to BE @ {position.stop_loss.price}")
        logger.info(f"Position {position.position_id} moved to break even")
    
    def _handle_partial_exit(self, position: Position, tp: TakeProfit, price: float):
        """Handle partial position exit"""
        tp.is_hit = True
        tp.hit_at = datetime.now()
        
        exit_quantity = position.remaining_quantity * tp.partial_quantity_pct
        position.remaining_quantity -= exit_quantity
        
        # Calculate realized P&L
        if position.direction == 'long':
            pnl = (price - position.entry_price) * exit_quantity
        else:
            pnl = (position.entry_price - price) * exit_quantity
        
        position.realized_pnl += pnl
        
        if position.remaining_quantity <= 0:
            self._close_position(position, price, "All targets hit")
        else:
            position.status = PositionStatus.PARTIAL_CLOSED
            position.notes.append(f"Partial exit: {exit_quantity} @ {price}")
        
        logger.info(f"Partial exit: {exit_quantity} @ {price}, P&L: {pnl:.2f}")
    
    def _close_position(self, position: Position, exit_price: float, reason: str):
        """Close a position"""
        # Calculate final P&L
        if position.direction == 'long':
            final_pnl = (exit_price - position.entry_price) * position.remaining_quantity
        else:
            final_pnl = (position.entry_price - exit_price) * position.remaining_quantity
        
        position.realized_pnl += final_pnl
        position.remaining_quantity = 0
        position.status = PositionStatus.CLOSED
        
        # Update execution record
        if position.position_id in self._executions:
            execution = self._executions[position.position_id]
            execution.exit_price = exit_price
            execution.exit_time = datetime.now()
            execution.exit_reason = reason
            execution.time_in_trade_minutes = int(
                (datetime.now() - position.entry_time).total_seconds() / 60
            )
            
            if position.realized_pnl > 0:
                execution.outcome = 'win'
                self._winning_trades += 1
            elif position.realized_pnl < 0:
                execution.outcome = 'loss'
                self._losing_trades += 1
            else:
                execution.outcome = 'breakeven'
        
        # Move to closed positions
        del self._open_positions[position.position_id]
        self._closed_positions.append(position)
        
        # Cancel pending orders
        self.broker.cancel_order(position.stop_order_id)
        self.broker.cancel_order(position.tp1_order_id)
        if position.tp2_order_id:
            self.broker.cancel_order(position.tp2_order_id)
        
        # Update balance
        self._account_balance += position.realized_pnl
        
        logger.info(f"Position closed: {position.symbol} {position.direction}, "
                   f"P&L: {position.realized_pnl:.2f}, Reason: {reason}")
    
    def _update_equity(self):
        """Update account equity"""
        self._open_pnl = sum(p.unrealized_pnl for p in self._open_positions.values())
        self._equity = self._account_balance + self._open_pnl
    
    # =========================================================================
    # POSITION ACCESS
    # =========================================================================
    
    def get_open_positions(self) -> List[Position]:
        """Get all open positions"""
        return list(self._open_positions.values())
    
    def get_position(self, position_id: str) -> Optional[Position]:
        """Get specific position"""
        return self._open_positions.get(position_id)
    
    def get_closed_positions(self, limit: int = 100) -> List[Position]:
        """Get closed positions"""
        return self._closed_positions[-limit:]
    
    def get_execution(self, execution_id: str) -> Optional[TradeExecution]:
        """Get execution record"""
        return self._executions.get(execution_id)
    
    # =========================================================================
    # MANUAL CONTROLS
    # =========================================================================
    
    def close_position_manual(self, position_id: str, price: float, 
                             reason: str = "Manual close") -> bool:
        """Manually close a position"""
        if position_id not in self._open_positions:
            return False
        
        position = self._open_positions[position_id]
        self._close_position(position, price, reason)
        return True
    
    def modify_stop_loss(self, position_id: str, new_stop: float) -> bool:
        """Modify stop loss for position"""
        if position_id not in self._open_positions:
            return False
        
        position = self._open_positions[position_id]
        
        # Never widen stops (ICT rule)
        if position.direction == 'long':
            if new_stop < position.stop_loss.price:
                logger.warning("Cannot widen stop loss")
                return False
        else:
            if new_stop > position.stop_loss.price:
                logger.warning("Cannot widen stop loss")
                return False
        
        position.stop_loss.price = new_stop
        position.stop_loss.times_adjusted += 1
        position.stop_loss.last_adjusted = datetime.now()
        
        self.broker.modify_order(position.stop_order_id, new_stop)
        position.notes.append(f"SL modified to {new_stop}")
        
        return True
    
    def close_all_positions(self, price: float, reason: str = "Close all"):
        """Close all open positions"""
        for pos_id in list(self._open_positions.keys()):
            self.close_position_manual(pos_id, price, reason)
    
    # =========================================================================
    # ACCOUNT & STATISTICS
    # =========================================================================
    
    def get_account_summary(self) -> Dict[str, Any]:
        """Get account summary"""
        return {
            'balance': self._account_balance,
            'equity': self._equity,
            'open_pnl': self._open_pnl,
            'open_positions': len(self._open_positions),
            'total_trades': self._total_trades,
            'winning_trades': self._winning_trades,
            'losing_trades': self._losing_trades,
            'win_rate': (self._winning_trades / self._total_trades * 100) 
                       if self._total_trades > 0 else 0,
        }
    
    def get_execution_statistics(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self._closed_positions:
            return {'error': 'No closed trades'}
        
        closed = self._closed_positions
        
        pnls = [p.realized_pnl for p in closed]
        winners = [p for p in pnls if p > 0]
        losers = [p for p in pnls if p < 0]
        
        return {
            'total_trades': len(closed),
            'total_pnl': sum(pnls),
            'average_pnl': sum(pnls) / len(pnls),
            'win_rate': len(winners) / len(closed) * 100,
            'average_winner': sum(winners) / len(winners) if winners else 0,
            'average_loser': sum(losers) / len(losers) if losers else 0,
            'profit_factor': abs(sum(winners) / sum(losers)) if losers else 0,
            'average_mae': sum(p.max_adverse_excursion for p in closed) / len(closed),
            'average_mfe': sum(p.max_favorable_excursion for p in closed) / len(closed),
        }
    
    # =========================================================================
    # REPORTING
    # =========================================================================
    
    def generate_report(self) -> str:
        """Generate execution report"""
        lines = []
        lines.append("=" * 60)
        lines.append("TRADE EXECUTOR REPORT")
        lines.append("=" * 60)
        lines.append(f"Mode: {self.config.mode.value}")
        lines.append("")
        
        # Account summary
        summary = self.get_account_summary()
        lines.append("ACCOUNT SUMMARY:")
        lines.append(f"  Balance: ${summary['balance']:,.2f}")
        lines.append(f"  Equity: ${summary['equity']:,.2f}")
        lines.append(f"  Open P&L: ${summary['open_pnl']:,.2f}")
        lines.append(f"  Open Positions: {summary['open_positions']}")
        lines.append("")
        
        # Trading stats
        lines.append("TRADING STATISTICS:")
        lines.append(f"  Total Trades: {summary['total_trades']}")
        lines.append(f"  Winners: {summary['winning_trades']}")
        lines.append(f"  Losers: {summary['losing_trades']}")
        lines.append(f"  Win Rate: {summary['win_rate']:.1f}%")
        lines.append("")
        
        # Execution stats
        if self._closed_positions:
            stats = self.get_execution_statistics()
            lines.append("EXECUTION METRICS:")
            lines.append(f"  Total P&L: ${stats['total_pnl']:,.2f}")
            lines.append(f"  Avg P&L: ${stats['average_pnl']:,.2f}")
            lines.append(f"  Profit Factor: {stats['profit_factor']:.2f}")
            lines.append(f"  Avg MAE: {stats['average_mae']:.2f} pts")
            lines.append(f"  Avg MFE: {stats['average_mfe']:.2f} pts")
        lines.append("")
        
        # Open positions
        if self._open_positions:
            lines.append("OPEN POSITIONS:")
            for pos in self._open_positions.values():
                lines.append(f"  {pos.symbol} {pos.direction.upper()}")
                lines.append(f"    Entry: {pos.entry_price}")
                lines.append(f"    Current: {pos.current_price}")
                lines.append(f"    P&L: ${pos.unrealized_pnl:,.2f}")
                lines.append(f"    Stop: {pos.stop_loss.price}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("ICT Trade Executor")
    print("=" * 60)
    
    # Create executor
    config = ExecutorConfig(
        mode=ExecutionMode.PAPER,
        default_risk_percent=1.0,
        max_positions=3,
        enable_partials=True,
    )
    
    executor = TradeExecutor(config)
    
    # Simulate a trade setup
    setup = {
        'direction': 'long',
        'symbol': 'NQ',
        'entry_price': 21500.00,
        'stop_loss': 21450.00,
        'take_profit_1': 21600.00,
        'take_profit_2': 21700.00,
        'grade': 'A',
        'model_type': '2022_model',
        'confluence_score': 85,
    }
    
    print("\nExecuting trade setup...")
    execution = executor.execute_trade(setup)
    
    if execution:
        print(f"\nTrade executed successfully!")
        print(f"Position ID: {execution.position.position_id}")
        print(f"Entry: {execution.position.entry_price}")
        print(f"Stop: {execution.position.stop_loss.price}")
        print(f"TP1: {execution.position.take_profit_1.price}")
        
        # Simulate price movement
        print("\nSimulating price movement...")
        
        # Price moves to TP1
        executor.update_positions('NQ', 21600.00)
        print(f"Price @ 21600 - Position status: {execution.position.status.value}")
        
        # Check if partial hit
        if execution.position.take_profit_1.is_hit:
            print(f"TP1 hit! Realized P&L: ${execution.position.realized_pnl:,.2f}")
        
        # Continue to TP2
        executor.update_positions('NQ', 21700.00)
        
    # Print report
    print("\n" + executor.generate_report())
