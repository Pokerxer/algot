"""
ICT Signal Generator and Trade Manager
========================================

Generates trading signals and manages trade lifecycle following ICT principles.

SIGNAL GENERATION RULES:
1. HTF bias must be established
2. LTF must show structure shift in direction of HTF
3. Entry at PD array (FVG, OB, Breaker)
4. Time window must be valid (Kill Zone preferred)
5. Liquidity sweep adds confluence

TRADE MANAGEMENT RULES (ICT):
- "Scale out of winners"
- "Let runners run"
- "Don't fight the draw on liquidity"
- "Move stop to break-even after partial profit"

Author: ICT Signal Engine
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Callable
from enum import Enum
from datetime import datetime, time, timedelta
from collections import defaultdict
import pandas as pd
import numpy as np
import logging
import uuid

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class SignalType(Enum):
    """Type of trading signal"""
    ENTRY_LONG = "entry_long"
    ENTRY_SHORT = "entry_short"
    EXIT_LONG = "exit_long"
    EXIT_SHORT = "exit_short"
    SCALE_IN = "scale_in"
    SCALE_OUT = "scale_out"
    STOP_MOVE = "stop_move"
    ALERT = "alert"


class SignalPriority(Enum):
    """Signal priority level"""
    CRITICAL = 1       # Immediate action required
    HIGH = 2           # Action within 1 minute
    MEDIUM = 3         # Action within 5 minutes
    LOW = 4            # Information only
    

class TradePhase(Enum):
    """Current phase of trade"""
    PENDING = "pending"            # Signal generated, not filled
    ACTIVE = "active"              # Position open
    PARTIAL = "partial"            # Partially closed
    BREAK_EVEN = "break_even"      # Stop at break-even
    TRAILING = "trailing"          # Trailing stop active
    CLOSED = "closed"              # Position closed


class TradeOutcome(Enum):
    """Trade outcome"""
    WIN = "win"
    LOSS = "loss"
    BREAK_EVEN = "break_even"
    CANCELLED = "cancelled"


class OrderType(Enum):
    """Order types"""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderStatus(Enum):
    """Order status"""
    PENDING = "pending"
    SUBMITTED = "submitted"
    FILLED = "filled"
    PARTIALLY_FILLED = "partially_filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TradingSignal:
    """Complete trading signal"""
    signal_id: str
    timestamp: datetime
    symbol: str
    signal_type: SignalType
    priority: SignalPriority
    
    # Direction and price
    direction: str                 # long, short
    entry_price: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: Optional[float]
    
    # Context
    model: str                     # ICT model name
    grade: str                     # A+, A, B, C
    confidence: float              # 0-100
    
    # Entry details
    entry_type: OrderType
    entry_zone: Tuple[float, float]  # Price range
    pd_array_type: str             # FVG, OB, Breaker, etc.
    
    # Risk management
    risk_amount: float
    position_size: float
    risk_reward_ratio: float
    
    # Time constraints
    valid_until: datetime
    kill_zone_required: bool
    
    # Supporting data
    confluence_factors: List[str]
    htf_bias: str
    ltf_confirmation: str
    
    # Execution notes
    notes: List[str]
    
    # State
    is_active: bool = True
    triggered: bool = False
    trigger_time: Optional[datetime] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'signal_id': self.signal_id,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'type': self.signal_type.value,
            'direction': self.direction,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit_1': self.take_profit_1,
            'confidence': self.confidence,
            'model': self.model,
            'grade': self.grade
        }


@dataclass
class Order:
    """Trading order"""
    order_id: str
    signal_id: str
    timestamp: datetime
    symbol: str
    
    # Order details
    order_type: OrderType
    direction: str
    price: float
    quantity: float
    
    # Status
    status: OrderStatus
    filled_quantity: float = 0.0
    filled_price: float = 0.0
    
    # Lifecycle
    submitted_time: Optional[datetime] = None
    filled_time: Optional[datetime] = None
    cancelled_time: Optional[datetime] = None
    
    # Notes
    notes: str = ""


@dataclass
class Position:
    """Trading position"""
    position_id: str
    signal_id: str
    symbol: str
    direction: str
    
    # Entry
    entry_time: datetime
    entry_price: float
    quantity: float
    
    # Current state
    current_quantity: float
    average_exit_price: float
    realized_pnl: float
    unrealized_pnl: float
    
    # Risk management
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    take_profit_3: Optional[float]
    
    # Phase tracking
    phase: TradePhase
    tp1_hit: bool = False
    tp2_hit: bool = False
    tp3_hit: bool = False
    
    # Stop management
    initial_stop: float = 0.0
    current_stop: float = 0.0
    highest_profit: float = 0.0
    lowest_profit: float = 0.0
    
    # Performance tracking
    max_favorable_excursion: float = 0.0
    max_adverse_excursion: float = 0.0


@dataclass
class TradeRecord:
    """Complete trade record for analysis"""
    trade_id: str
    signal_id: str
    symbol: str
    direction: str
    model: str
    grade: str
    
    # Entry
    entry_time: datetime
    entry_price: float
    entry_quantity: float
    
    # Exit
    exit_time: datetime
    exit_price: float
    exit_quantity: float
    
    # Performance
    pnl: float
    pnl_percent: float
    outcome: TradeOutcome
    
    # Metrics
    max_favorable_excursion: float
    max_adverse_excursion: float
    time_in_trade: float           # minutes
    
    # Context
    confluence_factors: List[str]
    confidence: float
    
    # Execution quality
    entry_timing_score: float      # 0-100
    exit_timing_score: float       # 0-100
    execution_score: float         # 0-100


@dataclass
class SignalGeneratorConfig:
    """Configuration for signal generator"""
    # Signal thresholds
    min_confidence: float = 60.0
    min_risk_reward: float = 1.5
    max_risk_percent: float = 0.02
    
    # Time filters
    require_kill_zone: bool = True
    allowed_sessions: List[str] = field(default_factory=lambda: ['new_york', 'london'])
    
    # Model filters
    allowed_models: List[str] = field(default_factory=lambda: [
        'ICT 2022 Model', 'Silver Bullet', 'Venom', 
        'Turtle Soup', 'Smart Money Reversal'
    ])
    min_grade: str = 'C'
    
    # Position limits
    max_concurrent_signals: int = 3
    max_daily_signals: int = 5
    
    # Risk management
    tp1_percent: float = 0.33      # 33% at TP1
    tp2_percent: float = 0.33      # 33% at TP2
    tp3_percent: float = 0.34      # 34% runner
    move_to_be_after_tp1: bool = True
    trail_after_tp2: bool = True


# =============================================================================
# SIGNAL GENERATOR
# =============================================================================

class ICTSignalGenerator:
    """
    Generates trading signals based on ICT methodology.
    
    ICT: "Wait for the setup. Let the setup come to you.
    Don't force trades - let the market tell you what to do."
    """
    
    def __init__(self, config: SignalGeneratorConfig = None):
        """
        Initialize signal generator.
        
        Args:
            config: Generator configuration
        """
        self.config = config or SignalGeneratorConfig()
        
        # Signal tracking
        self.active_signals: Dict[str, TradingSignal] = {}
        self.signal_history: List[TradingSignal] = []
        self.daily_signal_count = 0
        self.last_reset_date = datetime.now().date()
        
        # Callbacks
        self.signal_callbacks: List[Callable] = []
        
        logger.info("ICT Signal Generator initialized")
    
    def generate_signal(
        self,
        setup: Dict,
        current_price: float,
        account_size: float
    ) -> Optional[TradingSignal]:
        """
        Generate a trading signal from a setup.
        
        Args:
            setup: Trade setup dictionary
            current_price: Current market price
            account_size: Account size for position sizing
            
        Returns:
            TradingSignal or None
        """
        # Reset daily count if new day
        self._check_daily_reset()
        
        # Validate setup
        if not self._validate_setup(setup):
            return None
            
        # Check limits
        if not self._check_limits():
            return None
            
        # Extract setup details
        direction = setup.get('direction', 'long')
        entry_price = setup.get('entry_price', current_price)
        stop_loss = setup.get('stop_loss', 0)
        
        # Validate stop loss
        if stop_loss == 0:
            logger.warning("Invalid stop loss")
            return None
            
        # Calculate risk
        risk_ticks = abs(entry_price - stop_loss)
        risk_amount = min(
            account_size * self.config.max_risk_percent,
            account_size * 0.01  # Max 1% hard limit
        )
        
        # Calculate position size
        position_size = risk_amount / risk_ticks if risk_ticks > 0 else 0
        
        # Calculate take profits
        tp1 = self._calculate_take_profit(entry_price, stop_loss, direction, 1.5)
        tp2 = self._calculate_take_profit(entry_price, stop_loss, direction, 2.5)
        tp3 = self._calculate_take_profit(entry_price, stop_loss, direction, 4.0)
        
        # Calculate risk/reward
        rr_ratio = abs(tp1 - entry_price) / risk_ticks if risk_ticks > 0 else 0
        
        # Validate R:R
        if rr_ratio < self.config.min_risk_reward:
            logger.info(f"R:R ratio {rr_ratio:.2f} below minimum {self.config.min_risk_reward}")
            return None
            
        # Create signal
        signal_id = str(uuid.uuid4())[:8]
        
        signal = TradingSignal(
            signal_id=signal_id,
            timestamp=datetime.now(),
            symbol=setup.get('symbol', 'UNKNOWN'),
            signal_type=SignalType.ENTRY_LONG if direction == 'long' else SignalType.ENTRY_SHORT,
            priority=self._determine_priority(setup),
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit_1=tp1,
            take_profit_2=tp2,
            take_profit_3=tp3,
            model=setup.get('model', 'Unknown'),
            grade=setup.get('grade', 'C'),
            confidence=setup.get('confidence', 50.0),
            entry_type=OrderType.LIMIT,
            entry_zone=(setup.get('entry_zone_low', entry_price), setup.get('entry_zone_high', entry_price)),
            pd_array_type=setup.get('pd_array_type', 'unknown'),
            risk_amount=risk_amount,
            position_size=position_size,
            risk_reward_ratio=rr_ratio,
            valid_until=datetime.now() + timedelta(hours=4),
            kill_zone_required=setup.get('kill_zone_required', True),
            confluence_factors=setup.get('confluence_factors', []),
            htf_bias=setup.get('htf_bias', 'neutral'),
            ltf_confirmation=setup.get('ltf_confirmation', 'none'),
            notes=self._build_signal_notes(setup)
        )
        
        # Track signal
        self.active_signals[signal_id] = signal
        self.daily_signal_count += 1
        
        # Notify callbacks
        self._notify_signal(signal)
        
        logger.info(f"Signal generated: {signal_id} - {direction.upper()} @ {entry_price}")
        
        return signal
    
    def check_signal_trigger(
        self,
        signal_id: str,
        current_price: float
    ) -> bool:
        """
        Check if signal entry conditions are met.
        
        Args:
            signal_id: Signal to check
            current_price: Current market price
            
        Returns:
            True if signal should trigger
        """
        signal = self.active_signals.get(signal_id)
        if not signal or not signal.is_active:
            return False
            
        # Check validity
        if datetime.now() > signal.valid_until:
            signal.is_active = False
            return False
            
        # Check if price in entry zone
        entry_low, entry_high = signal.entry_zone
        
        if signal.direction == 'long':
            # For long, we want price to touch/enter our buy zone
            if entry_low <= current_price <= entry_high:
                return True
            # Or if price wicked through but now above
            if current_price > entry_high and not signal.triggered:
                return False  # Missed entry
        else:
            # For short, we want price to touch/enter our sell zone
            if entry_low <= current_price <= entry_high:
                return True
            # Or if price wicked through but now below
            if current_price < entry_low and not signal.triggered:
                return False  # Missed entry
                
        return False
    
    def trigger_signal(self, signal_id: str):
        """
        Mark signal as triggered.
        
        Args:
            signal_id: Signal to trigger
        """
        signal = self.active_signals.get(signal_id)
        if signal:
            signal.triggered = True
            signal.trigger_time = datetime.now()
            logger.info(f"Signal triggered: {signal_id}")
    
    def cancel_signal(self, signal_id: str, reason: str = ""):
        """
        Cancel an active signal.
        
        Args:
            signal_id: Signal to cancel
            reason: Cancellation reason
        """
        signal = self.active_signals.get(signal_id)
        if signal:
            signal.is_active = False
            signal.notes.append(f"Cancelled: {reason}")
            self.signal_history.append(signal)
            del self.active_signals[signal_id]
            logger.info(f"Signal cancelled: {signal_id} - {reason}")
    
    def get_active_signals(self) -> List[TradingSignal]:
        """Get all active signals"""
        return list(self.active_signals.values())
    
    def register_callback(self, callback: Callable):
        """
        Register callback for new signals.
        
        Args:
            callback: Function to call with new signals
        """
        self.signal_callbacks.append(callback)
    
    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================
    
    def _validate_setup(self, setup: Dict) -> bool:
        """Validate setup meets requirements"""
        # Check confidence
        confidence = setup.get('confidence', 0)
        if confidence < self.config.min_confidence:
            return False
            
        # Check grade
        grade_order = {'A+': 5, 'A': 4, 'B': 3, 'C': 2, 'X': 0}
        min_grade_value = grade_order.get(self.config.min_grade, 0)
        setup_grade_value = grade_order.get(setup.get('grade', 'X'), 0)
        
        if setup_grade_value < min_grade_value:
            return False
            
        # Check model
        model = setup.get('model', '')
        if self.config.allowed_models and model not in self.config.allowed_models:
            return False
            
        return True
    
    def _check_limits(self) -> bool:
        """Check if within signal limits"""
        # Check concurrent
        if len(self.active_signals) >= self.config.max_concurrent_signals:
            return False
            
        # Check daily
        if self.daily_signal_count >= self.config.max_daily_signals:
            return False
            
        return True
    
    def _check_daily_reset(self):
        """Reset daily count if new day"""
        today = datetime.now().date()
        if today != self.last_reset_date:
            self.daily_signal_count = 0
            self.last_reset_date = today
    
    def _calculate_take_profit(
        self,
        entry: float,
        stop: float,
        direction: str,
        rr_multiple: float
    ) -> float:
        """Calculate take profit level"""
        risk = abs(entry - stop)
        
        if direction == 'long':
            return entry + (risk * rr_multiple)
        else:
            return entry - (risk * rr_multiple)
    
    def _determine_priority(self, setup: Dict) -> SignalPriority:
        """Determine signal priority"""
        grade = setup.get('grade', 'C')
        confidence = setup.get('confidence', 50)
        
        if grade == 'A+' and confidence >= 85:
            return SignalPriority.CRITICAL
        elif grade in ['A+', 'A'] and confidence >= 70:
            return SignalPriority.HIGH
        elif confidence >= 60:
            return SignalPriority.MEDIUM
        else:
            return SignalPriority.LOW
    
    def _build_signal_notes(self, setup: Dict) -> List[str]:
        """Build execution notes for signal"""
        notes = []
        
        # Direction note
        direction = setup.get('direction', 'long')
        notes.append(f"Direction: {direction.upper()}")
        
        # Model note
        model = setup.get('model', 'Unknown')
        notes.append(f"Model: {model}")
        
        # Confluence notes
        factors = setup.get('confluence_factors', [])
        if factors:
            notes.append(f"Confluence: {', '.join(factors[:3])}")
            
        # Kill zone note
        if setup.get('kill_zone_required', True):
            notes.append("Kill zone entry preferred")
            
        return notes
    
    def _notify_signal(self, signal: TradingSignal):
        """Notify all registered callbacks"""
        for callback in self.signal_callbacks:
            try:
                callback(signal)
            except Exception as e:
                logger.error(f"Callback error: {e}")


# =============================================================================
# TRADE MANAGER
# =============================================================================

class ICTTradeManager:
    """
    Manages active trades following ICT principles.
    
    ICT: "Scale out at targets. Let a portion run.
    Move stop to break-even to protect profits."
    """
    
    def __init__(self, config: SignalGeneratorConfig = None):
        """
        Initialize trade manager.
        
        Args:
            config: Configuration
        """
        self.config = config or SignalGeneratorConfig()
        
        # Active positions
        self.positions: Dict[str, Position] = {}
        
        # Order tracking
        self.orders: Dict[str, Order] = {}
        
        # Trade history
        self.trade_records: List[TradeRecord] = []
        
        # Performance metrics
        self.total_trades = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_pnl = 0.0
        
        logger.info("ICT Trade Manager initialized")
    
    def open_position(
        self,
        signal: TradingSignal,
        fill_price: float,
        fill_quantity: float
    ) -> Position:
        """
        Open a new position from a signal.
        
        Args:
            signal: Triggering signal
            fill_price: Actual fill price
            fill_quantity: Filled quantity
            
        Returns:
            New Position
        """
        position_id = str(uuid.uuid4())[:8]
        
        position = Position(
            position_id=position_id,
            signal_id=signal.signal_id,
            symbol=signal.symbol,
            direction=signal.direction,
            entry_time=datetime.now(),
            entry_price=fill_price,
            quantity=fill_quantity,
            current_quantity=fill_quantity,
            average_exit_price=0.0,
            realized_pnl=0.0,
            unrealized_pnl=0.0,
            stop_loss=signal.stop_loss,
            take_profit_1=signal.take_profit_1,
            take_profit_2=signal.take_profit_2,
            take_profit_3=signal.take_profit_3,
            phase=TradePhase.ACTIVE,
            initial_stop=signal.stop_loss,
            current_stop=signal.stop_loss
        )
        
        self.positions[position_id] = position
        
        logger.info(f"Position opened: {position_id} - {signal.direction} @ {fill_price}")
        
        return position
    
    def update_position(
        self,
        position_id: str,
        current_price: float
    ) -> List[Dict]:
        """
        Update position with current price and check targets.
        
        Args:
            position_id: Position to update
            current_price: Current market price
            
        Returns:
            List of actions to take
        """
        position = self.positions.get(position_id)
        if not position:
            return []
            
        actions = []
        
        # Calculate unrealized PnL
        if position.direction == 'long':
            pnl_per_unit = current_price - position.entry_price
        else:
            pnl_per_unit = position.entry_price - current_price
            
        position.unrealized_pnl = pnl_per_unit * position.current_quantity
        
        # Track MFE/MAE
        if position.unrealized_pnl > position.highest_profit:
            position.highest_profit = position.unrealized_pnl
            position.max_favorable_excursion = pnl_per_unit
            
        if position.unrealized_pnl < position.lowest_profit:
            position.lowest_profit = position.unrealized_pnl
            position.max_adverse_excursion = abs(pnl_per_unit)
        
        # Check stop loss
        if self._check_stop_hit(position, current_price):
            actions.append({
                'action': 'close_all',
                'reason': 'stop_loss',
                'price': current_price
            })
            return actions
            
        # Check take profit levels
        if not position.tp1_hit:
            if self._check_tp_hit(position, current_price, 1):
                position.tp1_hit = True
                tp1_qty = position.quantity * self.config.tp1_percent
                actions.append({
                    'action': 'partial_close',
                    'quantity': tp1_qty,
                    'reason': 'tp1',
                    'price': current_price
                })
                
                # Move stop to break-even
                if self.config.move_to_be_after_tp1:
                    actions.append({
                        'action': 'move_stop',
                        'new_stop': position.entry_price,
                        'reason': 'break_even'
                    })
                    position.current_stop = position.entry_price
                    position.phase = TradePhase.BREAK_EVEN
                    
        elif not position.tp2_hit:
            if self._check_tp_hit(position, current_price, 2):
                position.tp2_hit = True
                tp2_qty = position.quantity * self.config.tp2_percent
                actions.append({
                    'action': 'partial_close',
                    'quantity': tp2_qty,
                    'reason': 'tp2',
                    'price': current_price
                })
                
                # Start trailing
                if self.config.trail_after_tp2:
                    position.phase = TradePhase.TRAILING
                    
        elif not position.tp3_hit and position.take_profit_3:
            if self._check_tp_hit(position, current_price, 3):
                position.tp3_hit = True
                actions.append({
                    'action': 'close_all',
                    'reason': 'tp3',
                    'price': current_price
                })
                
        # Update trailing stop if in trailing phase
        if position.phase == TradePhase.TRAILING:
            new_stop = self._calculate_trailing_stop(position, current_price)
            if new_stop != position.current_stop:
                actions.append({
                    'action': 'move_stop',
                    'new_stop': new_stop,
                    'reason': 'trailing'
                })
                position.current_stop = new_stop
                
        return actions
    
    def close_position(
        self,
        position_id: str,
        exit_price: float,
        exit_quantity: float = None,
        reason: str = ""
    ) -> Optional[TradeRecord]:
        """
        Close a position (fully or partially).
        
        Args:
            position_id: Position to close
            exit_price: Exit price
            exit_quantity: Quantity to close (None = all)
            reason: Close reason
            
        Returns:
            TradeRecord if fully closed
        """
        position = self.positions.get(position_id)
        if not position:
            return None
            
        if exit_quantity is None:
            exit_quantity = position.current_quantity
            
        # Calculate PnL for this exit
        if position.direction == 'long':
            pnl = (exit_price - position.entry_price) * exit_quantity
        else:
            pnl = (position.entry_price - exit_price) * exit_quantity
            
        # Update position
        position.realized_pnl += pnl
        position.current_quantity -= exit_quantity
        
        # Update average exit price
        total_exits = position.quantity - position.current_quantity
        if total_exits > 0:
            position.average_exit_price = (
                (position.average_exit_price * (total_exits - exit_quantity) + exit_price * exit_quantity)
                / total_exits
            )
            
        logger.info(f"Position closed (partial): {position_id} - {exit_quantity} @ {exit_price}, PnL: {pnl:.2f}")
        
        # If fully closed, create trade record
        if position.current_quantity <= 0:
            record = self._create_trade_record(position, exit_price, reason)
            
            # Update stats
            self.total_trades += 1
            self.total_pnl += position.realized_pnl
            
            if position.realized_pnl > 0:
                self.winning_trades += 1
            elif position.realized_pnl < 0:
                self.losing_trades += 1
                
            position.phase = TradePhase.CLOSED
            del self.positions[position_id]
            
            self.trade_records.append(record)
            
            return record
            
        else:
            position.phase = TradePhase.PARTIAL
            
        return None
    
    def get_position(self, position_id: str) -> Optional[Position]:
        """Get position by ID"""
        return self.positions.get(position_id)
    
    def get_all_positions(self) -> List[Position]:
        """Get all active positions"""
        return list(self.positions.values())
    
    def get_performance_summary(self) -> Dict:
        """Get performance summary"""
        win_rate = (self.winning_trades / self.total_trades * 100) if self.total_trades > 0 else 0
        
        return {
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': win_rate,
            'total_pnl': self.total_pnl,
            'average_pnl': self.total_pnl / self.total_trades if self.total_trades > 0 else 0,
            'active_positions': len(self.positions)
        }
    
    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================
    
    def _check_stop_hit(self, position: Position, current_price: float) -> bool:
        """Check if stop loss is hit"""
        if position.direction == 'long':
            return current_price <= position.current_stop
        else:
            return current_price >= position.current_stop
    
    def _check_tp_hit(self, position: Position, current_price: float, tp_level: int) -> bool:
        """Check if take profit level is hit"""
        tp_price = {
            1: position.take_profit_1,
            2: position.take_profit_2,
            3: position.take_profit_3
        }.get(tp_level, 0)
        
        if tp_price == 0:
            return False
            
        if position.direction == 'long':
            return current_price >= tp_price
        else:
            return current_price <= tp_price
    
    def _calculate_trailing_stop(self, position: Position, current_price: float) -> float:
        """Calculate new trailing stop level"""
        # Trail at 50% of profit from entry
        if position.direction == 'long':
            profit = current_price - position.entry_price
            if profit > 0:
                new_stop = position.entry_price + (profit * 0.5)
                return max(position.current_stop, new_stop)
        else:
            profit = position.entry_price - current_price
            if profit > 0:
                new_stop = position.entry_price - (profit * 0.5)
                return min(position.current_stop, new_stop)
                
        return position.current_stop
    
    def _create_trade_record(
        self,
        position: Position,
        exit_price: float,
        reason: str
    ) -> TradeRecord:
        """Create trade record from closed position"""
        # Determine outcome
        if position.realized_pnl > 0:
            outcome = TradeOutcome.WIN
        elif position.realized_pnl < 0:
            outcome = TradeOutcome.LOSS
        else:
            outcome = TradeOutcome.BREAK_EVEN
            
        # Calculate time in trade
        time_in_trade = (datetime.now() - position.entry_time).total_seconds() / 60
        
        # Calculate execution scores (simplified)
        entry_timing = 70.0  # Default
        exit_timing = 70.0
        
        if position.tp1_hit:
            exit_timing += 10
        if position.tp2_hit:
            exit_timing += 10
        if position.tp3_hit:
            exit_timing += 10
            
        return TradeRecord(
            trade_id=str(uuid.uuid4())[:8],
            signal_id=position.signal_id,
            symbol=position.symbol,
            direction=position.direction,
            model="",  # Would need to look up from signal
            grade="",
            entry_time=position.entry_time,
            entry_price=position.entry_price,
            entry_quantity=position.quantity,
            exit_time=datetime.now(),
            exit_price=exit_price,
            exit_quantity=position.quantity,
            pnl=position.realized_pnl,
            pnl_percent=(position.realized_pnl / position.entry_price) * 100,
            outcome=outcome,
            max_favorable_excursion=position.max_favorable_excursion,
            max_adverse_excursion=position.max_adverse_excursion,
            time_in_trade=time_in_trade,
            confluence_factors=[],
            confidence=0,
            entry_timing_score=entry_timing,
            exit_timing_score=exit_timing,
            execution_score=(entry_timing + exit_timing) / 2
        )


# =============================================================================
# SIGNAL PROCESSOR
# =============================================================================

class SignalProcessor:
    """
    Processes signals from generation to execution.
    
    Coordinates between signal generator and trade manager.
    """
    
    def __init__(
        self,
        generator: ICTSignalGenerator,
        manager: ICTTradeManager
    ):
        """
        Initialize processor.
        
        Args:
            generator: Signal generator
            manager: Trade manager
        """
        self.generator = generator
        self.manager = manager
        
        # Signal-to-position mapping
        self.signal_positions: Dict[str, str] = {}  # signal_id -> position_id
        
        # Processing queue
        self.pending_fills: List[Dict] = []
        
    def process_tick(
        self,
        symbol: str,
        current_price: float,
        timestamp: datetime = None
    ) -> Dict[str, Any]:
        """
        Process a price tick.
        
        Args:
            symbol: Trading symbol
            current_price: Current price
            timestamp: Tick timestamp
            
        Returns:
            Processing results
        """
        results = {
            'signals_triggered': [],
            'position_actions': [],
            'positions_closed': []
        }
        
        # Check active signals for triggers
        for signal in self.generator.get_active_signals():
            if signal.symbol == symbol:
                if self.generator.check_signal_trigger(signal.signal_id, current_price):
                    self.generator.trigger_signal(signal.signal_id)
                    
                    # Open position
                    position = self.manager.open_position(
                        signal,
                        current_price,
                        signal.position_size
                    )
                    
                    self.signal_positions[signal.signal_id] = position.position_id
                    results['signals_triggered'].append(signal.signal_id)
                    
        # Update active positions
        for position in self.manager.get_all_positions():
            if position.symbol == symbol:
                actions = self.manager.update_position(
                    position.position_id,
                    current_price
                )
                
                for action in actions:
                    action['position_id'] = position.position_id
                    results['position_actions'].append(action)
                    
                    # Execute actions
                    if action['action'] == 'close_all':
                        record = self.manager.close_position(
                            position.position_id,
                            action['price'],
                            reason=action['reason']
                        )
                        if record:
                            results['positions_closed'].append(record.trade_id)
                            
                    elif action['action'] == 'partial_close':
                        self.manager.close_position(
                            position.position_id,
                            action['price'],
                            action['quantity'],
                            action['reason']
                        )
                        
        return results
    
    def get_status(self) -> Dict[str, Any]:
        """Get processor status"""
        return {
            'active_signals': len(self.generator.active_signals),
            'active_positions': len(self.manager.positions),
            'daily_signals': self.generator.daily_signal_count,
            'performance': self.manager.get_performance_summary()
        }


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Create components
    config = SignalGeneratorConfig(
        min_confidence=60,
        min_risk_reward=1.5,
        max_risk_percent=0.01,
        require_kill_zone=True
    )
    
    generator = ICTSignalGenerator(config)
    manager = ICTTradeManager(config)
    processor = SignalProcessor(generator, manager)
    
    # Example setup
    setup = {
        'symbol': 'EURUSD',
        'direction': 'long',
        'entry_price': 1.0850,
        'stop_loss': 1.0820,
        'model': 'ICT 2022 Model',
        'grade': 'A',
        'confidence': 75.0,
        'entry_zone_low': 1.0845,
        'entry_zone_high': 1.0855,
        'pd_array_type': 'FVG',
        'confluence_factors': ['liquidity_swept', 'structure_break', 'fvg_present'],
        'htf_bias': 'bullish',
        'ltf_confirmation': 'structure_shift',
        'kill_zone_required': True
    }
    
    # Generate signal
    signal = generator.generate_signal(
        setup,
        current_price=1.0852,
        account_size=100000
    )
    
    if signal:
        print("Signal Generated:")
        print(f"  ID: {signal.signal_id}")
        print(f"  Direction: {signal.direction}")
        print(f"  Entry: {signal.entry_price}")
        print(f"  Stop: {signal.stop_loss}")
        print(f"  TP1: {signal.take_profit_1}")
        print(f"  TP2: {signal.take_profit_2}")
        print(f"  R:R: {signal.risk_reward_ratio:.2f}")
        print(f"  Size: {signal.position_size:.2f}")
        print(f"  Priority: {signal.priority.name}")
        
        # Simulate price tick triggering entry
        results = processor.process_tick('EURUSD', 1.0850)
        print(f"\nTriggered: {results['signals_triggered']}")
        
        # Simulate price moving to TP1
        results = processor.process_tick('EURUSD', 1.0895)
        print(f"Actions: {results['position_actions']}")
        
    print("\nProcessor Status:")
    print(processor.get_status())
