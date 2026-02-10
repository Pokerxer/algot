"""
ICT Risk Management System
==========================

Comprehensive risk management for ICT algorithmic trading including:
- Position sizing based on account % risk
- Dynamic stop loss (ATR-based, swing point, structure-based)
- Drawdown protection (daily/weekly/monthly limits)
- Trailing stop logic (breakeven, partial profits, dynamic trails)
- Session guards (ICT Kill Zones, macro times)

RISK MANAGEMENT ARCHITECTURE:
============================

┌─────────────────────────────────────────────────────────────────────────────┐
│                        ICT RISK MANAGEMENT SYSTEM                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      POSITION SIZING ENGINE                          │    │
│  │                                                                       │    │
│  │  Account Balance → Risk % → Stop Distance → Position Size            │    │
│  │                                                                       │    │
│  │  Methods:                                                             │    │
│  │  • Fixed Percentage Risk (1-2% per trade)                            │    │
│  │  • Kelly Criterion (optimal sizing based on win rate)                │    │
│  │  • Volatility Adjusted (ATR-based scaling)                           │    │
│  │  • Confidence Scaled (higher confidence = larger size)               │    │
│  │  • Martingale/Anti-Martingale (optional)                             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      DYNAMIC STOP LOSS ENGINE                        │    │
│  │                                                                       │    │
│  │  Stop Types:                                                          │    │
│  │  • ATR-Based: Entry ± (ATR × multiplier)                             │    │
│  │  • Swing Point: Below/above recent swing low/high                    │    │
│  │  • Structure: Below BOS/CHoCH level                                  │    │
│  │  • Order Block: Below/above OB boundary                              │    │
│  │  • FVG: Below/above FVG boundary                                     │    │
│  │  • Fixed Pips: Simple pip-based stop                                 │    │
│  │  • Time-Based: Widen/tighten based on session                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      DRAWDOWN PROTECTION                              │    │
│  │                                                                       │    │
│  │  Limits:                         Actions:                             │    │
│  │  • Daily: -3% max               • Reduce position size               │    │
│  │  • Weekly: -6% max              • Pause trading                      │    │
│  │  • Monthly: -10% max            • Close all positions                │    │
│  │  • Consecutive losses: 3-5      • Alert/notification                 │    │
│  │                                                                       │    │
│  │  Recovery Mode:                                                       │    │
│  │  • Reduced size after drawdown                                       │    │
│  │  • Gradual size increase on wins                                     │    │
│  │  • Stricter entry requirements                                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      TRAILING STOP SYSTEM                             │    │
│  │                                                                       │    │
│  │  Stages:                                                              │    │
│  │  1. Initial Stop: At calculated stop loss                            │    │
│  │  2. Breakeven: Move to entry when +1R                                │    │
│  │  3. Lock Profit: Trail at 50% of max favorable                       │    │
│  │  4. Target Trail: Tighten near take profit                           │    │
│  │                                                                       │    │
│  │  Trail Methods:                                                       │    │
│  │  • Fixed Distance: Trail by X pips                                   │    │
│  │  • ATR Trail: Trail by ATR × multiplier                              │    │
│  │  • Swing Trail: Trail to swing points                                │    │
│  │  • Chandelier: Trail from highest high                               │    │
│  │  • Parabolic: Accelerating trail                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      SESSION GUARDS                                   │    │
│  │                                                                       │    │
│  │  ICT Kill Zones:                                                      │    │
│  │  • London Open: 02:00-05:00 EST                                      │    │
│  │  • NY AM: 08:30-11:00 EST                                            │    │
│  │  • NY Lunch: 12:00-13:00 EST (avoid)                                 │    │
│  │  • NY PM: 13:30-16:00 EST                                            │    │
│  │  • Asian: 20:00-00:00 EST                                            │    │
│  │                                                                       │    │
│  │  Macro Times:                                                         │    │
│  │  • :50-:10 of each hour (ICT macro)                                  │    │
│  │  • News events (block trading)                                       │    │
│  │  • Session opens/closes                                              │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘

ICT RISK PRINCIPLES:
===================
1. "Risk no more than 1-2% per trade" - Position sizing rule
2. "Stop loss below/above the order block" - Structure-based stops
3. "Trade only during kill zones" - Session management
4. "Let winners run, cut losers quick" - Trail management
5. "Three consecutive losses = step back" - Drawdown protection

Author: Claude (Anthropic)
Version: 1.0.0
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Callable
from abc import ABC, abstractmethod
import math

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class RiskLevel(Enum):
    """Risk level classifications"""
    CONSERVATIVE = auto()  # 0.5% risk
    MODERATE = auto()       # 1% risk
    STANDARD = auto()       # 1.5% risk
    AGGRESSIVE = auto()     # 2% risk
    VERY_AGGRESSIVE = auto()  # 3%+ risk


class PositionSizingMethod(Enum):
    """Position sizing methodologies"""
    FIXED_PERCENTAGE = auto()      # Fixed % of account
    FIXED_DOLLAR = auto()          # Fixed dollar amount
    KELLY_CRITERION = auto()       # Optimal Kelly sizing
    VOLATILITY_ADJUSTED = auto()   # ATR-based sizing
    CONFIDENCE_SCALED = auto()     # Scale by signal confidence
    ANTI_MARTINGALE = auto()       # Increase after wins
    MARTINGALE = auto()            # Increase after losses (dangerous)
    OPTIMAL_F = auto()             # Ralph Vince optimal f


class StopLossType(Enum):
    """Stop loss calculation methods"""
    FIXED_PIPS = auto()            # Simple pip-based
    ATR_BASED = auto()             # ATR multiplier
    SWING_POINT = auto()           # Recent swing high/low
    STRUCTURE_BASED = auto()       # BOS/CHoCH level
    ORDER_BLOCK = auto()           # OB boundary
    FVG_BASED = auto()             # FVG boundary
    PERCENTAGE = auto()            # % from entry
    VOLATILITY_BAND = auto()       # Bollinger/Keltner
    TIME_BASED = auto()            # Session-adjusted
    COMPOSITE = auto()             # Combination of methods


class TrailingStopMethod(Enum):
    """Trailing stop methodologies"""
    NONE = auto()                  # No trailing
    FIXED_DISTANCE = auto()        # Trail by fixed pips
    ATR_TRAIL = auto()             # Trail by ATR
    SWING_TRAIL = auto()           # Trail to swing points
    CHANDELIER = auto()            # Trail from highest high
    PARABOLIC = auto()             # Accelerating trail (SAR)
    BREAKEVEN_ONLY = auto()        # Only move to breakeven
    STEP_TRAIL = auto()            # Trail in steps (levels)
    PERCENTAGE_TRAIL = auto()      # Trail by % of profit


class TradingSession(Enum):
    """Trading sessions"""
    ASIAN = auto()
    LONDON = auto()
    NEW_YORK_AM = auto()
    NEW_YORK_LUNCH = auto()
    NEW_YORK_PM = auto()
    AFTER_HOURS = auto()


class DrawdownAction(Enum):
    """Actions when drawdown limits hit"""
    REDUCE_SIZE = auto()           # Reduce position size
    PAUSE_TRADING = auto()         # Pause new trades
    CLOSE_ALL = auto()             # Close all positions
    TIGHTEN_STOPS = auto()         # Move stops closer
    HEDGE = auto()                 # Open hedge position
    ALERT_ONLY = auto()            # Just send alert


class RecoveryMode(Enum):
    """Recovery mode after drawdown"""
    NORMAL = auto()                # Normal trading
    CAUTIOUS = auto()              # Reduced size, stricter criteria
    RECOVERY = auto()              # Minimal size, highest quality only
    PAUSED = auto()                # No trading allowed


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class AccountState:
    """Current account state"""
    balance: float
    equity: float
    margin_used: float
    margin_available: float
    unrealized_pnl: float
    daily_pnl: float
    weekly_pnl: float
    monthly_pnl: float
    peak_balance: float
    current_drawdown: float
    max_drawdown: float
    open_positions: int
    pending_orders: int
    consecutive_losses: int
    consecutive_wins: int
    total_trades: int
    win_rate: float
    avg_win: float
    avg_loss: float
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class RiskParameters:
    """Risk management parameters"""
    # Position sizing
    risk_per_trade_percent: float = 1.0  # 1% default
    max_risk_per_trade_percent: float = 2.0
    min_risk_per_trade_percent: float = 0.25
    max_position_size_percent: float = 10.0  # Max 10% of account in one position
    max_total_exposure_percent: float = 30.0  # Max 30% total exposure
    
    # Drawdown limits
    max_daily_loss_percent: float = 3.0
    max_weekly_loss_percent: float = 6.0
    max_monthly_loss_percent: float = 10.0
    max_consecutive_losses: int = 3
    
    # Recovery settings
    recovery_size_multiplier: float = 0.5  # 50% size in recovery
    cautious_size_multiplier: float = 0.75  # 75% size when cautious
    wins_to_exit_recovery: int = 3
    
    # Trailing stop settings
    breakeven_r_multiple: float = 1.0  # Move to BE at 1R
    trail_start_r_multiple: float = 1.5  # Start trailing at 1.5R
    trail_distance_atr: float = 1.5  # Trail by 1.5 ATR
    
    # Session settings
    allowed_sessions: List[TradingSession] = field(default_factory=lambda: [
        TradingSession.LONDON,
        TradingSession.NEW_YORK_AM,
        TradingSession.NEW_YORK_PM
    ])
    block_during_news: bool = True
    news_block_minutes_before: int = 30
    news_block_minutes_after: int = 15


@dataclass
class StopLossResult:
    """Stop loss calculation result"""
    stop_price: float
    stop_type: StopLossType
    distance_pips: float
    distance_percent: float
    risk_amount: float
    reasoning: str
    alternative_stops: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 1.0


@dataclass
class PositionSizeResult:
    """Position sizing calculation result"""
    position_size: float  # In lots or units
    risk_amount: float
    risk_percent: float
    method_used: PositionSizingMethod
    adjustments_applied: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    max_allowed: float = 0.0
    confidence_factor: float = 1.0


@dataclass
class TrailingStopUpdate:
    """Trailing stop update result"""
    new_stop: float
    old_stop: float
    should_update: bool
    trail_method: TrailingStopMethod
    profit_locked: float
    r_multiple_locked: float
    reasoning: str


@dataclass 
class SessionStatus:
    """Current session status"""
    current_session: TradingSession
    is_kill_zone: bool
    kill_zone_name: Optional[str]
    is_macro_time: bool
    minutes_to_macro: int
    can_trade: bool
    block_reason: Optional[str]
    session_start: Optional[datetime]
    session_end: Optional[datetime]
    next_kill_zone: Optional[str]
    minutes_to_next_kz: int


@dataclass
class DrawdownStatus:
    """Drawdown status and limits"""
    current_drawdown_percent: float
    daily_loss_percent: float
    weekly_loss_percent: float
    monthly_loss_percent: float
    consecutive_losses: int
    recovery_mode: RecoveryMode
    can_trade: bool
    size_multiplier: float
    limits_hit: List[str] = field(default_factory=list)
    actions_triggered: List[DrawdownAction] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


@dataclass
class RiskAssessment:
    """Complete risk assessment for a trade"""
    can_take_trade: bool
    position_size: PositionSizeResult
    stop_loss: StopLossResult
    session_status: SessionStatus
    drawdown_status: DrawdownStatus
    risk_score: float  # 0-100, higher = riskier
    risk_reward_ratio: float
    expected_value: float
    warnings: List[str] = field(default_factory=list)
    blocks: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


# =============================================================================
# POSITION SIZING ENGINE
# =============================================================================

class PositionSizingEngine:
    """
    Calculates optimal position size based on risk parameters.
    
    ICT Principle: "Risk no more than 1-2% per trade"
    """
    
    def __init__(self, params: RiskParameters):
        self.params = params
        self.history: List[Dict] = []
    
    def calculate_position_size(
        self,
        account: AccountState,
        entry_price: float,
        stop_loss: float,
        pip_value: float = 10.0,  # Value per pip per lot
        method: PositionSizingMethod = PositionSizingMethod.FIXED_PERCENTAGE,
        signal_confidence: float = 1.0,
        atr: Optional[float] = None,
        recovery_mode: RecoveryMode = RecoveryMode.NORMAL
    ) -> PositionSizeResult:
        """
        Calculate position size based on risk parameters.
        
        Args:
            account: Current account state
            entry_price: Planned entry price
            stop_loss: Stop loss price
            pip_value: Value per pip per standard lot
            method: Position sizing method
            signal_confidence: Signal confidence (0-1)
            atr: Average True Range (for volatility methods)
            recovery_mode: Current recovery mode
            
        Returns:
            PositionSizeResult with calculated size and details
        """
        adjustments = []
        warnings = []
        
        # Calculate stop distance
        stop_distance = abs(entry_price - stop_loss)
        stop_distance_pips = stop_distance / 0.0001 if entry_price > 10 else stop_distance / 0.01
        
        if stop_distance_pips <= 0:
            return PositionSizeResult(
                position_size=0,
                risk_amount=0,
                risk_percent=0,
                method_used=method,
                adjustments_applied=["Invalid stop distance"],
                warnings=["Stop distance must be positive"]
            )
        
        # Base risk calculation
        base_risk_percent = self.params.risk_per_trade_percent
        
        # Apply recovery mode adjustments
        if recovery_mode == RecoveryMode.RECOVERY:
            base_risk_percent *= self.params.recovery_size_multiplier
            adjustments.append(f"Recovery mode: {self.params.recovery_size_multiplier:.0%} size")
        elif recovery_mode == RecoveryMode.CAUTIOUS:
            base_risk_percent *= self.params.cautious_size_multiplier
            adjustments.append(f"Cautious mode: {self.params.cautious_size_multiplier:.0%} size")
        elif recovery_mode == RecoveryMode.PAUSED:
            return PositionSizeResult(
                position_size=0,
                risk_amount=0,
                risk_percent=0,
                method_used=method,
                adjustments_applied=["Trading paused"],
                warnings=["Trading is paused due to drawdown"]
            )
        
        # Calculate position size based on method
        if method == PositionSizingMethod.FIXED_PERCENTAGE:
            position_size, risk_amount = self._fixed_percentage(
                account.balance, base_risk_percent, stop_distance_pips, pip_value
            )
            
        elif method == PositionSizingMethod.FIXED_DOLLAR:
            risk_amount = account.balance * (base_risk_percent / 100)
            position_size = risk_amount / (stop_distance_pips * pip_value)
            
        elif method == PositionSizingMethod.KELLY_CRITERION:
            position_size, risk_amount = self._kelly_criterion(
                account, base_risk_percent, stop_distance_pips, pip_value
            )
            adjustments.append("Kelly criterion applied")
            
        elif method == PositionSizingMethod.VOLATILITY_ADJUSTED:
            if atr is None:
                warnings.append("ATR not provided, using fixed percentage")
                position_size, risk_amount = self._fixed_percentage(
                    account.balance, base_risk_percent, stop_distance_pips, pip_value
                )
            else:
                position_size, risk_amount = self._volatility_adjusted(
                    account.balance, base_risk_percent, stop_distance_pips, 
                    pip_value, atr, entry_price
                )
                adjustments.append("Volatility adjusted")
                
        elif method == PositionSizingMethod.CONFIDENCE_SCALED:
            base_size, risk_amount = self._fixed_percentage(
                account.balance, base_risk_percent, stop_distance_pips, pip_value
            )
            # Scale by confidence (0.5 to 1.5x based on confidence)
            confidence_factor = 0.5 + signal_confidence
            position_size = base_size * confidence_factor
            risk_amount *= confidence_factor
            adjustments.append(f"Confidence scaled: {confidence_factor:.2f}x")
            
        elif method == PositionSizingMethod.ANTI_MARTINGALE:
            position_size, risk_amount = self._anti_martingale(
                account, base_risk_percent, stop_distance_pips, pip_value
            )
            adjustments.append("Anti-martingale applied")
            
        elif method == PositionSizingMethod.OPTIMAL_F:
            position_size, risk_amount = self._optimal_f(
                account, base_risk_percent, stop_distance_pips, pip_value
            )
            adjustments.append("Optimal f applied")
            
        else:
            # Default to fixed percentage
            position_size, risk_amount = self._fixed_percentage(
                account.balance, base_risk_percent, stop_distance_pips, pip_value
            )
        
        # Apply confidence scaling (for non-confidence methods)
        if method != PositionSizingMethod.CONFIDENCE_SCALED and signal_confidence < 1.0:
            old_size = position_size
            position_size *= (0.7 + 0.3 * signal_confidence)  # Scale 70-100%
            if position_size != old_size:
                adjustments.append(f"Confidence factor: {signal_confidence:.2f}")
        
        # Calculate max allowed
        max_position_value = account.balance * (self.params.max_position_size_percent / 100)
        max_position_size = max_position_value / entry_price
        
        # Check position size limits
        if position_size > max_position_size:
            position_size = max_position_size
            risk_amount = position_size * stop_distance_pips * pip_value
            warnings.append(f"Position capped at {self.params.max_position_size_percent}% of account")
        
        # Check total exposure
        current_exposure = account.margin_used
        new_exposure = position_size * entry_price
        total_exposure = current_exposure + new_exposure
        max_exposure = account.balance * (self.params.max_total_exposure_percent / 100)
        
        if total_exposure > max_exposure:
            allowed_new = max_exposure - current_exposure
            if allowed_new > 0:
                position_size = allowed_new / entry_price
                risk_amount = position_size * stop_distance_pips * pip_value
                warnings.append("Position reduced due to total exposure limit")
            else:
                warnings.append("Max total exposure reached")
                position_size = 0
                risk_amount = 0
        
        # Minimum size check
        min_lot_size = 0.01  # Micro lot
        if position_size > 0 and position_size < min_lot_size:
            warnings.append(f"Position too small (min: {min_lot_size})")
            position_size = 0
            risk_amount = 0
        
        # Round to valid lot size
        position_size = round(position_size, 2)
        
        # Calculate actual risk percent
        actual_risk_percent = (risk_amount / account.balance) * 100 if account.balance > 0 else 0
        
        return PositionSizeResult(
            position_size=position_size,
            risk_amount=risk_amount,
            risk_percent=actual_risk_percent,
            method_used=method,
            adjustments_applied=adjustments,
            warnings=warnings,
            max_allowed=max_position_size,
            confidence_factor=signal_confidence
        )
    
    def _fixed_percentage(
        self, balance: float, risk_percent: float, 
        stop_pips: float, pip_value: float
    ) -> Tuple[float, float]:
        """Fixed percentage position sizing"""
        risk_amount = balance * (risk_percent / 100)
        position_size = risk_amount / (stop_pips * pip_value)
        return position_size, risk_amount
    
    def _kelly_criterion(
        self, account: AccountState, base_risk: float,
        stop_pips: float, pip_value: float
    ) -> Tuple[float, float]:
        """
        Kelly Criterion: f* = (bp - q) / b
        where b = avg_win/avg_loss, p = win_rate, q = 1-p
        """
        if account.avg_loss == 0 or account.win_rate == 0:
            return self._fixed_percentage(account.balance, base_risk, stop_pips, pip_value)
        
        b = account.avg_win / account.avg_loss
        p = account.win_rate
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Use fractional Kelly (half Kelly is common)
        kelly_fraction *= 0.5
        
        # Cap Kelly at max risk
        kelly_fraction = max(0, min(kelly_fraction, self.params.max_risk_per_trade_percent / 100))
        
        risk_amount = account.balance * kelly_fraction
        position_size = risk_amount / (stop_pips * pip_value)
        
        return position_size, risk_amount
    
    def _volatility_adjusted(
        self, balance: float, base_risk: float,
        stop_pips: float, pip_value: float,
        atr: float, entry_price: float
    ) -> Tuple[float, float]:
        """
        Volatility-adjusted sizing: reduce size in high volatility
        """
        # Normalize ATR as percentage
        atr_percent = (atr / entry_price) * 100
        
        # Base volatility assumption (adjust based on instrument)
        base_volatility = 0.5  # 0.5% daily range typical for forex
        
        # Volatility ratio
        vol_ratio = atr_percent / base_volatility if base_volatility > 0 else 1.0
        
        # Adjust risk inversely to volatility
        adjusted_risk = base_risk / vol_ratio if vol_ratio > 0 else base_risk
        
        # Cap adjustments
        adjusted_risk = max(
            self.params.min_risk_per_trade_percent,
            min(adjusted_risk, self.params.max_risk_per_trade_percent)
        )
        
        return self._fixed_percentage(balance, adjusted_risk, stop_pips, pip_value)
    
    def _anti_martingale(
        self, account: AccountState, base_risk: float,
        stop_pips: float, pip_value: float
    ) -> Tuple[float, float]:
        """
        Anti-martingale: increase size after wins, decrease after losses
        """
        # Adjust based on consecutive wins/losses
        if account.consecutive_wins > 0:
            # Increase by 25% per win, max 50% increase
            multiplier = min(1.5, 1.0 + (0.25 * account.consecutive_wins))
        elif account.consecutive_losses > 0:
            # Decrease by 25% per loss, min 25% of base
            multiplier = max(0.25, 1.0 - (0.25 * account.consecutive_losses))
        else:
            multiplier = 1.0
        
        adjusted_risk = base_risk * multiplier
        adjusted_risk = max(
            self.params.min_risk_per_trade_percent,
            min(adjusted_risk, self.params.max_risk_per_trade_percent)
        )
        
        return self._fixed_percentage(account.balance, adjusted_risk, stop_pips, pip_value)
    
    def _optimal_f(
        self, account: AccountState, base_risk: float,
        stop_pips: float, pip_value: float
    ) -> Tuple[float, float]:
        """
        Ralph Vince Optimal f: maximize geometric growth
        Simplified version using win rate and payoff ratio
        """
        if account.avg_loss == 0 or account.total_trades < 20:
            return self._fixed_percentage(account.balance, base_risk, stop_pips, pip_value)
        
        # Calculate optimal f
        win_rate = account.win_rate
        payoff = account.avg_win / account.avg_loss
        
        # Optimal f formula (simplified)
        optimal_f = win_rate - ((1 - win_rate) / payoff)
        
        # Use fraction of optimal f for safety
        safe_f = optimal_f * 0.3  # 30% of optimal
        
        safe_f = max(0, min(safe_f, self.params.max_risk_per_trade_percent / 100))
        
        risk_amount = account.balance * safe_f
        position_size = risk_amount / (stop_pips * pip_value)
        
        return position_size, risk_amount


# =============================================================================
# DYNAMIC STOP LOSS ENGINE
# =============================================================================

class DynamicStopLossEngine:
    """
    Calculates optimal stop loss based on market structure and ICT concepts.
    
    ICT Principle: "Stop loss below/above the order block"
    """
    
    def __init__(self, params: RiskParameters):
        self.params = params
    
    def calculate_stop_loss(
        self,
        entry_price: float,
        direction: str,  # 'long' or 'short'
        stop_type: StopLossType = StopLossType.ATR_BASED,
        atr: Optional[float] = None,
        atr_multiplier: float = 1.5,
        swing_high: Optional[float] = None,
        swing_low: Optional[float] = None,
        order_block: Optional[Dict] = None,
        fvg: Optional[Dict] = None,
        structure_level: Optional[float] = None,
        fixed_pips: float = 20.0,
        buffer_pips: float = 2.0,
        pip_size: float = 0.0001
    ) -> StopLossResult:
        """
        Calculate dynamic stop loss based on specified method.
        
        Args:
            entry_price: Planned entry price
            direction: Trade direction ('long' or 'short')
            stop_type: Stop loss calculation method
            atr: Average True Range
            atr_multiplier: Multiplier for ATR-based stops
            swing_high: Recent swing high
            swing_low: Recent swing low
            order_block: Order block data
            fvg: Fair Value Gap data
            structure_level: Key structure level (BOS/CHoCH)
            fixed_pips: Fixed pip distance
            buffer_pips: Extra buffer beyond calculated stop
            pip_size: Pip size for the instrument
            
        Returns:
            StopLossResult with stop price and details
        """
        alternatives = []
        is_long = direction.lower() == 'long'
        
        # Calculate stops using different methods
        stops_calculated = {}
        
        # 1. Fixed pips stop
        fixed_stop = (entry_price - (fixed_pips * pip_size) if is_long 
                     else entry_price + (fixed_pips * pip_size))
        stops_calculated['fixed'] = fixed_stop
        alternatives.append({
            'type': 'fixed_pips',
            'price': fixed_stop,
            'distance': fixed_pips
        })
        
        # 2. ATR-based stop
        if atr is not None:
            atr_distance = atr * atr_multiplier
            atr_stop = (entry_price - atr_distance if is_long 
                       else entry_price + atr_distance)
            stops_calculated['atr'] = atr_stop
            alternatives.append({
                'type': 'atr_based',
                'price': atr_stop,
                'distance': atr_distance / pip_size,
                'atr': atr,
                'multiplier': atr_multiplier
            })
        
        # 3. Swing point stop
        if swing_low is not None and is_long:
            swing_stop = swing_low - (buffer_pips * pip_size)
            stops_calculated['swing'] = swing_stop
            alternatives.append({
                'type': 'swing_point',
                'price': swing_stop,
                'distance': (entry_price - swing_stop) / pip_size,
                'swing_level': swing_low
            })
        elif swing_high is not None and not is_long:
            swing_stop = swing_high + (buffer_pips * pip_size)
            stops_calculated['swing'] = swing_stop
            alternatives.append({
                'type': 'swing_point',
                'price': swing_stop,
                'distance': (swing_stop - entry_price) / pip_size,
                'swing_level': swing_high
            })
        
        # 4. Order block stop
        if order_block is not None:
            ob_low = order_block.get('low', order_block.get('bottom'))
            ob_high = order_block.get('high', order_block.get('top'))
            
            if is_long and ob_low is not None:
                ob_stop = ob_low - (buffer_pips * pip_size)
                stops_calculated['ob'] = ob_stop
                alternatives.append({
                    'type': 'order_block',
                    'price': ob_stop,
                    'distance': (entry_price - ob_stop) / pip_size,
                    'ob_level': ob_low
                })
            elif not is_long and ob_high is not None:
                ob_stop = ob_high + (buffer_pips * pip_size)
                stops_calculated['ob'] = ob_stop
                alternatives.append({
                    'type': 'order_block',
                    'price': ob_stop,
                    'distance': (ob_stop - entry_price) / pip_size,
                    'ob_level': ob_high
                })
        
        # 5. FVG stop
        if fvg is not None:
            fvg_low = fvg.get('low', fvg.get('bottom'))
            fvg_high = fvg.get('high', fvg.get('top'))
            
            if is_long and fvg_low is not None:
                fvg_stop = fvg_low - (buffer_pips * pip_size)
                stops_calculated['fvg'] = fvg_stop
                alternatives.append({
                    'type': 'fvg_based',
                    'price': fvg_stop,
                    'distance': (entry_price - fvg_stop) / pip_size,
                    'fvg_level': fvg_low
                })
            elif not is_long and fvg_high is not None:
                fvg_stop = fvg_high + (buffer_pips * pip_size)
                stops_calculated['fvg'] = fvg_stop
                alternatives.append({
                    'type': 'fvg_based',
                    'price': fvg_stop,
                    'distance': (fvg_stop - entry_price) / pip_size,
                    'fvg_level': fvg_high
                })
        
        # 6. Structure-based stop
        if structure_level is not None:
            if is_long:
                struct_stop = structure_level - (buffer_pips * pip_size)
            else:
                struct_stop = structure_level + (buffer_pips * pip_size)
            stops_calculated['structure'] = struct_stop
            alternatives.append({
                'type': 'structure_based',
                'price': struct_stop,
                'distance': abs(entry_price - struct_stop) / pip_size,
                'structure_level': structure_level
            })
        
        # Select stop based on requested type
        selected_stop = None
        reasoning = ""
        
        if stop_type == StopLossType.FIXED_PIPS:
            selected_stop = stops_calculated.get('fixed', fixed_stop)
            reasoning = f"Fixed {fixed_pips} pips stop"
            
        elif stop_type == StopLossType.ATR_BASED:
            selected_stop = stops_calculated.get('atr')
            if selected_stop is None:
                selected_stop = stops_calculated.get('fixed', fixed_stop)
                reasoning = "ATR not available, using fixed pips"
            else:
                reasoning = f"ATR-based stop ({atr_multiplier}x ATR)"
                
        elif stop_type == StopLossType.SWING_POINT:
            selected_stop = stops_calculated.get('swing')
            if selected_stop is None:
                selected_stop = stops_calculated.get('atr', stops_calculated.get('fixed', fixed_stop))
                reasoning = "Swing point not available, using fallback"
            else:
                reasoning = "Stop below/above swing point"
                
        elif stop_type == StopLossType.ORDER_BLOCK:
            selected_stop = stops_calculated.get('ob')
            if selected_stop is None:
                selected_stop = stops_calculated.get('swing', stops_calculated.get('atr', fixed_stop))
                reasoning = "Order block not available, using fallback"
            else:
                reasoning = "Stop below/above order block (ICT method)"
                
        elif stop_type == StopLossType.FVG_BASED:
            selected_stop = stops_calculated.get('fvg')
            if selected_stop is None:
                selected_stop = stops_calculated.get('ob', stops_calculated.get('atr', fixed_stop))
                reasoning = "FVG not available, using fallback"
            else:
                reasoning = "Stop below/above Fair Value Gap"
                
        elif stop_type == StopLossType.STRUCTURE_BASED:
            selected_stop = stops_calculated.get('structure')
            if selected_stop is None:
                selected_stop = stops_calculated.get('swing', stops_calculated.get('atr', fixed_stop))
                reasoning = "Structure level not available, using fallback"
            else:
                reasoning = "Stop below/above structure level"
                
        elif stop_type == StopLossType.COMPOSITE:
            # Use the tightest reasonable stop
            valid_stops = [s for s in stops_calculated.values() if s is not None]
            if valid_stops:
                if is_long:
                    # For longs, highest stop that's below entry
                    valid_stops = [s for s in valid_stops if s < entry_price]
                    selected_stop = max(valid_stops) if valid_stops else fixed_stop
                else:
                    # For shorts, lowest stop that's above entry
                    valid_stops = [s for s in valid_stops if s > entry_price]
                    selected_stop = min(valid_stops) if valid_stops else fixed_stop
                reasoning = "Composite: tightest reasonable stop from multiple methods"
            else:
                selected_stop = fixed_stop
                reasoning = "Composite: no valid stops found, using fixed"
        else:
            selected_stop = stops_calculated.get('fixed', fixed_stop)
            reasoning = "Default fixed pips stop"
        
        # Calculate distance metrics
        distance = abs(entry_price - selected_stop)
        distance_pips = distance / pip_size
        distance_percent = (distance / entry_price) * 100
        
        # Confidence based on stop type priority
        confidence_map = {
            StopLossType.ORDER_BLOCK: 0.95,
            StopLossType.STRUCTURE_BASED: 0.90,
            StopLossType.FVG_BASED: 0.85,
            StopLossType.SWING_POINT: 0.80,
            StopLossType.ATR_BASED: 0.75,
            StopLossType.COMPOSITE: 0.85,
            StopLossType.FIXED_PIPS: 0.60
        }
        confidence = confidence_map.get(stop_type, 0.70)
        
        return StopLossResult(
            stop_price=selected_stop,
            stop_type=stop_type,
            distance_pips=distance_pips,
            distance_percent=distance_percent,
            risk_amount=0,  # Will be calculated with position size
            reasoning=reasoning,
            alternative_stops=alternatives,
            confidence=confidence
        )


# =============================================================================
# DRAWDOWN PROTECTION ENGINE
# =============================================================================

class DrawdownProtectionEngine:
    """
    Monitors and enforces drawdown limits.
    
    ICT Principle: "Three consecutive losses = step back and reassess"
    """
    
    def __init__(self, params: RiskParameters):
        self.params = params
        self.daily_trades: List[Dict] = []
        self.weekly_trades: List[Dict] = []
        self.current_recovery_mode = RecoveryMode.NORMAL
        self.recovery_wins = 0
        self.callbacks: Dict[DrawdownAction, List[Callable]] = {
            action: [] for action in DrawdownAction
        }
    
    def register_callback(self, action: DrawdownAction, callback: Callable):
        """Register callback for drawdown action"""
        self.callbacks[action].append(callback)
    
    def _trigger_action(self, action: DrawdownAction, context: Dict = None):
        """Trigger callbacks for action"""
        for callback in self.callbacks[action]:
            try:
                callback(context or {})
            except Exception as e:
                logger.error(f"Callback error for {action}: {e}")
    
    def check_drawdown_status(self, account: AccountState) -> DrawdownStatus:
        """
        Check current drawdown status and determine if trading is allowed.
        
        Args:
            account: Current account state
            
        Returns:
            DrawdownStatus with current limits and actions
        """
        limits_hit = []
        actions = []
        warnings = []
        can_trade = True
        size_multiplier = 1.0
        
        # Calculate drawdown percentages
        daily_loss_pct = abs(min(0, account.daily_pnl / account.balance * 100)) if account.balance > 0 else 0
        weekly_loss_pct = abs(min(0, account.weekly_pnl / account.balance * 100)) if account.balance > 0 else 0
        monthly_loss_pct = abs(min(0, account.monthly_pnl / account.balance * 100)) if account.balance > 0 else 0
        current_dd_pct = account.current_drawdown
        
        # Check daily limit
        if daily_loss_pct >= self.params.max_daily_loss_percent:
            limits_hit.append(f"Daily loss limit ({self.params.max_daily_loss_percent}%)")
            actions.append(DrawdownAction.PAUSE_TRADING)
            can_trade = False
            self._trigger_action(DrawdownAction.PAUSE_TRADING, {'reason': 'daily_limit'})
        elif daily_loss_pct >= self.params.max_daily_loss_percent * 0.75:
            warnings.append(f"Approaching daily loss limit ({daily_loss_pct:.1f}%/{self.params.max_daily_loss_percent}%)")
            actions.append(DrawdownAction.REDUCE_SIZE)
            size_multiplier = min(size_multiplier, 0.5)
        elif daily_loss_pct >= self.params.max_daily_loss_percent * 0.5:
            warnings.append(f"50% of daily loss limit used ({daily_loss_pct:.1f}%)")
        
        # Check weekly limit
        if weekly_loss_pct >= self.params.max_weekly_loss_percent:
            limits_hit.append(f"Weekly loss limit ({self.params.max_weekly_loss_percent}%)")
            actions.append(DrawdownAction.PAUSE_TRADING)
            can_trade = False
            self._trigger_action(DrawdownAction.PAUSE_TRADING, {'reason': 'weekly_limit'})
        elif weekly_loss_pct >= self.params.max_weekly_loss_percent * 0.75:
            warnings.append(f"Approaching weekly loss limit ({weekly_loss_pct:.1f}%)")
            size_multiplier = min(size_multiplier, 0.75)
        
        # Check monthly limit
        if monthly_loss_pct >= self.params.max_monthly_loss_percent:
            limits_hit.append(f"Monthly loss limit ({self.params.max_monthly_loss_percent}%)")
            actions.append(DrawdownAction.CLOSE_ALL)
            can_trade = False
            self._trigger_action(DrawdownAction.CLOSE_ALL, {'reason': 'monthly_limit'})
        
        # Check consecutive losses
        if account.consecutive_losses >= self.params.max_consecutive_losses:
            limits_hit.append(f"Consecutive losses ({account.consecutive_losses})")
            actions.append(DrawdownAction.REDUCE_SIZE)
            size_multiplier = min(size_multiplier, 0.5)
            
            if account.consecutive_losses >= self.params.max_consecutive_losses + 2:
                actions.append(DrawdownAction.PAUSE_TRADING)
                can_trade = False
        
        # Determine recovery mode
        if not can_trade:
            self.current_recovery_mode = RecoveryMode.PAUSED
        elif len(limits_hit) > 0 or size_multiplier < 1.0:
            if self.current_recovery_mode == RecoveryMode.NORMAL:
                self.current_recovery_mode = RecoveryMode.CAUTIOUS
                self.recovery_wins = 0
        
        # Check if we can exit recovery mode
        if self.current_recovery_mode in [RecoveryMode.CAUTIOUS, RecoveryMode.RECOVERY]:
            if account.consecutive_wins >= self.params.wins_to_exit_recovery:
                self.current_recovery_mode = RecoveryMode.NORMAL
                self.recovery_wins = 0
                warnings.append("Exiting recovery mode after consecutive wins")
        
        # Apply recovery mode size adjustments
        if self.current_recovery_mode == RecoveryMode.RECOVERY:
            size_multiplier = min(size_multiplier, self.params.recovery_size_multiplier)
        elif self.current_recovery_mode == RecoveryMode.CAUTIOUS:
            size_multiplier = min(size_multiplier, self.params.cautious_size_multiplier)
        
        return DrawdownStatus(
            current_drawdown_percent=current_dd_pct,
            daily_loss_percent=daily_loss_pct,
            weekly_loss_percent=weekly_loss_pct,
            monthly_loss_percent=monthly_loss_pct,
            consecutive_losses=account.consecutive_losses,
            recovery_mode=self.current_recovery_mode,
            can_trade=can_trade,
            size_multiplier=size_multiplier,
            limits_hit=limits_hit,
            actions_triggered=actions,
            warnings=warnings
        )
    
    def record_trade_result(self, pnl: float, account_balance: float):
        """Record trade result for tracking"""
        now = datetime.utcnow()
        trade_record = {
            'timestamp': now,
            'pnl': pnl,
            'pnl_percent': (pnl / account_balance) * 100 if account_balance > 0 else 0
        }
        
        self.daily_trades.append(trade_record)
        self.weekly_trades.append(trade_record)
        
        # Clean old records
        day_ago = now - timedelta(days=1)
        week_ago = now - timedelta(days=7)
        
        self.daily_trades = [t for t in self.daily_trades if t['timestamp'] > day_ago]
        self.weekly_trades = [t for t in self.weekly_trades if t['timestamp'] > week_ago]
        
        # Update recovery tracking
        if pnl > 0:
            self.recovery_wins += 1
        else:
            self.recovery_wins = 0
    
    def reset_daily(self):
        """Reset daily counters"""
        self.daily_trades = []
    
    def reset_weekly(self):
        """Reset weekly counters"""
        self.weekly_trades = []
        self.daily_trades = []


# =============================================================================
# TRAILING STOP ENGINE
# =============================================================================

class TrailingStopEngine:
    """
    Manages trailing stops to lock in profits.
    
    ICT Principle: "Let winners run, cut losers quick"
    """
    
    def __init__(self, params: RiskParameters):
        self.params = params
    
    def calculate_trailing_stop(
        self,
        entry_price: float,
        current_price: float,
        current_stop: float,
        direction: str,
        initial_stop: float,
        take_profit: float,
        method: TrailingStopMethod = TrailingStopMethod.ATR_TRAIL,
        atr: Optional[float] = None,
        highest_price: Optional[float] = None,
        lowest_price: Optional[float] = None,
        swing_levels: Optional[List[float]] = None,
        pip_size: float = 0.0001
    ) -> TrailingStopUpdate:
        """
        Calculate trailing stop update.
        
        Args:
            entry_price: Original entry price
            current_price: Current market price
            current_stop: Current stop loss level
            direction: Trade direction ('long' or 'short')
            initial_stop: Initial stop loss level
            take_profit: Take profit target
            method: Trailing stop method
            atr: Current ATR value
            highest_price: Highest price since entry (for longs)
            lowest_price: Lowest price since entry (for shorts)
            swing_levels: Recent swing levels for swing trail
            pip_size: Pip size for instrument
            
        Returns:
            TrailingStopUpdate with new stop details
        """
        is_long = direction.lower() == 'long'
        
        # Calculate R-multiple (how many R's in profit)
        initial_risk = abs(entry_price - initial_stop)
        current_profit = (current_price - entry_price) if is_long else (entry_price - current_price)
        r_multiple = current_profit / initial_risk if initial_risk > 0 else 0
        
        new_stop = current_stop
        should_update = False
        reasoning = ""
        
        if method == TrailingStopMethod.NONE:
            return TrailingStopUpdate(
                new_stop=current_stop,
                old_stop=current_stop,
                should_update=False,
                trail_method=method,
                profit_locked=0,
                r_multiple_locked=0,
                reasoning="Trailing disabled"
            )
        
        # Check if we should move to breakeven
        if r_multiple >= self.params.breakeven_r_multiple:
            breakeven_stop = entry_price + (pip_size * 2 if is_long else -pip_size * 2)
            
            if is_long and current_stop < breakeven_stop:
                new_stop = breakeven_stop
                should_update = True
                reasoning = f"Move to breakeven at {self.params.breakeven_r_multiple}R"
            elif not is_long and current_stop > breakeven_stop:
                new_stop = breakeven_stop
                should_update = True
                reasoning = f"Move to breakeven at {self.params.breakeven_r_multiple}R"
        
        # Check if we should start trailing
        if r_multiple >= self.params.trail_start_r_multiple:
            trail_stop = current_stop
            
            if method == TrailingStopMethod.BREAKEVEN_ONLY:
                # Only breakeven, no further trailing
                pass
                
            elif method == TrailingStopMethod.FIXED_DISTANCE:
                fixed_trail = 15 * pip_size  # 15 pips trail
                if is_long:
                    trail_stop = current_price - fixed_trail
                else:
                    trail_stop = current_price + fixed_trail
                reasoning = "Fixed distance trail"
                
            elif method == TrailingStopMethod.ATR_TRAIL:
                if atr is not None:
                    trail_distance = atr * self.params.trail_distance_atr
                    if is_long:
                        trail_stop = current_price - trail_distance
                    else:
                        trail_stop = current_price + trail_distance
                    reasoning = f"ATR trail ({self.params.trail_distance_atr}x ATR)"
                    
            elif method == TrailingStopMethod.CHANDELIER:
                if is_long and highest_price is not None:
                    chandelier_distance = atr * 3 if atr else 30 * pip_size
                    trail_stop = highest_price - chandelier_distance
                    reasoning = "Chandelier exit from highest high"
                elif not is_long and lowest_price is not None:
                    chandelier_distance = atr * 3 if atr else 30 * pip_size
                    trail_stop = lowest_price + chandelier_distance
                    reasoning = "Chandelier exit from lowest low"
                    
            elif method == TrailingStopMethod.SWING_TRAIL:
                if swing_levels:
                    if is_long:
                        # Trail to swing lows below current price
                        valid_swings = [s for s in swing_levels if s < current_price]
                        if valid_swings:
                            trail_stop = max(valid_swings) - (2 * pip_size)
                            reasoning = "Trail to swing low"
                    else:
                        # Trail to swing highs above current price
                        valid_swings = [s for s in swing_levels if s > current_price]
                        if valid_swings:
                            trail_stop = min(valid_swings) + (2 * pip_size)
                            reasoning = "Trail to swing high"
                            
            elif method == TrailingStopMethod.STEP_TRAIL:
                # Trail in 1R steps
                steps_to_lock = int(r_multiple)
                if steps_to_lock > 0:
                    locked_profit = initial_risk * (steps_to_lock - 0.5)  # Lock half R less than current
                    if is_long:
                        trail_stop = entry_price + locked_profit
                    else:
                        trail_stop = entry_price - locked_profit
                    reasoning = f"Step trail: {steps_to_lock-0.5}R locked"
                    
            elif method == TrailingStopMethod.PERCENTAGE_TRAIL:
                # Trail at 50% of max profit
                if is_long and highest_price is not None:
                    max_profit = highest_price - entry_price
                    trail_stop = entry_price + (max_profit * 0.5)
                    reasoning = "Trail at 50% of max profit"
                elif not is_long and lowest_price is not None:
                    max_profit = entry_price - lowest_price
                    trail_stop = entry_price - (max_profit * 0.5)
                    reasoning = "Trail at 50% of max profit"
                    
            elif method == TrailingStopMethod.PARABOLIC:
                # Accelerating trail (simplified SAR-like)
                acceleration = min(0.02 * (1 + r_multiple * 0.5), 0.2)
                if is_long and highest_price is not None:
                    trail_stop = current_stop + acceleration * (highest_price - current_stop)
                elif not is_long and lowest_price is not None:
                    trail_stop = current_stop - acceleration * (current_stop - lowest_price)
                reasoning = f"Parabolic trail (accel: {acceleration:.3f})"
            
            # Check if new trail stop is better than current
            if is_long and trail_stop > current_stop:
                new_stop = trail_stop
                should_update = True
            elif not is_long and trail_stop < current_stop:
                new_stop = trail_stop
                should_update = True
        
        # Near take profit - tighten trail
        distance_to_tp = abs(take_profit - current_price)
        total_target_distance = abs(take_profit - entry_price)
        
        if total_target_distance > 0:
            completion_pct = 1 - (distance_to_tp / total_target_distance)
            
            if completion_pct > 0.8:  # Within 20% of target
                tight_trail = current_price - (distance_to_tp * 0.3) if is_long else current_price + (distance_to_tp * 0.3)
                
                if is_long and tight_trail > new_stop:
                    new_stop = tight_trail
                    should_update = True
                    reasoning = "Tightened trail near target"
                elif not is_long and tight_trail < new_stop:
                    new_stop = tight_trail
                    should_update = True
                    reasoning = "Tightened trail near target"
        
        # Calculate profit locked
        if is_long:
            profit_locked = new_stop - entry_price if new_stop > entry_price else 0
        else:
            profit_locked = entry_price - new_stop if new_stop < entry_price else 0
        
        r_locked = profit_locked / initial_risk if initial_risk > 0 else 0
        
        return TrailingStopUpdate(
            new_stop=new_stop,
            old_stop=current_stop,
            should_update=should_update,
            trail_method=method,
            profit_locked=profit_locked,
            r_multiple_locked=r_locked,
            reasoning=reasoning
        )


# =============================================================================
# SESSION GUARDS ENGINE
# =============================================================================

class SessionGuardsEngine:
    """
    Manages trading session restrictions based on ICT Kill Zones.
    
    ICT Principle: "Trade only during kill zones for maximum probability"
    """
    
    # ICT Kill Zone times (in EST/New York time)
    KILL_ZONES = {
        'asian': {'start': time(20, 0), 'end': time(23, 59), 'name': 'Asian Session'},
        'asian_late': {'start': time(0, 0), 'end': time(2, 0), 'name': 'Asian Session Late'},
        'london_open': {'start': time(2, 0), 'end': time(5, 0), 'name': 'London Open Kill Zone'},
        'london': {'start': time(3, 0), 'end': time(4, 0), 'name': 'London Kill Zone'},
        'ny_open': {'start': time(8, 30), 'end': time(11, 0), 'name': 'NY Open Kill Zone'},
        'ny_am': {'start': time(9, 30), 'end': time(10, 30), 'name': 'NY AM Kill Zone'},
        'ny_lunch': {'start': time(12, 0), 'end': time(13, 0), 'name': 'NY Lunch (Avoid)'},
        'ny_pm': {'start': time(13, 30), 'end': time(16, 0), 'name': 'NY PM Kill Zone'},
        'silver_bullet_london': {'start': time(3, 0), 'end': time(4, 0), 'name': 'London Silver Bullet'},
        'silver_bullet_ny_am': {'start': time(10, 0), 'end': time(11, 0), 'name': 'NY AM Silver Bullet'},
        'silver_bullet_ny_pm': {'start': time(14, 0), 'end': time(15, 0), 'name': 'NY PM Silver Bullet'}
    }
    
    # Macro times (ICT: price runs at these times)
    MACRO_TIMES = [
        {'minute': 50, 'duration': 20, 'name': 'Hourly Macro (:50-:10)'},
        {'minute': 20, 'duration': 10, 'name': 'Mid-Hour Macro (:20-:30)'}
    ]
    
    # Sessions to avoid
    AVOID_SESSIONS = ['ny_lunch']
    
    def __init__(self, params: RiskParameters, timezone_offset: int = 0):
        """
        Initialize session guards.
        
        Args:
            params: Risk parameters
            timezone_offset: Hours offset from EST (e.g., -5 for EST to UTC)
        """
        self.params = params
        self.timezone_offset = timezone_offset
        self.news_events: List[Dict] = []
        self.blocked_times: List[Dict] = []
    
    def add_news_event(self, event_time: datetime, impact: str = 'high', 
                       description: str = '', currency: str = ''):
        """Add high-impact news event to block trading around"""
        self.news_events.append({
            'time': event_time,
            'impact': impact,
            'description': description,
            'currency': currency
        })
    
    def add_blocked_time(self, start: datetime, end: datetime, reason: str = ''):
        """Add custom blocked time period"""
        self.blocked_times.append({
            'start': start,
            'end': end,
            'reason': reason
        })
    
    def _get_est_time(self, utc_time: datetime) -> datetime:
        """Convert UTC to EST"""
        return utc_time + timedelta(hours=self.timezone_offset)
    
    def _time_in_range(self, check_time: time, start: time, end: time) -> bool:
        """Check if time is within range (handles overnight ranges)"""
        if start <= end:
            return start <= check_time <= end
        else:  # Overnight range
            return check_time >= start or check_time <= end
    
    def get_session_status(self, current_time: Optional[datetime] = None) -> SessionStatus:
        """
        Get current session status and trading permissions.
        
        Args:
            current_time: Time to check (UTC). Defaults to now.
            
        Returns:
            SessionStatus with current session info and permissions
        """
        if current_time is None:
            current_time = datetime.utcnow()
        
        est_time = self._get_est_time(current_time)
        current_time_only = est_time.time()
        current_minute = est_time.minute
        
        # Determine current session
        current_session = TradingSession.AFTER_HOURS
        current_kz = None
        current_kz_name = None
        session_start = None
        session_end = None
        
        for kz_key, kz_data in self.KILL_ZONES.items():
            if self._time_in_range(current_time_only, kz_data['start'], kz_data['end']):
                current_kz = kz_key
                current_kz_name = kz_data['name']
                session_start = datetime.combine(est_time.date(), kz_data['start'])
                session_end = datetime.combine(est_time.date(), kz_data['end'])
                
                # Map to session enum
                if 'asian' in kz_key:
                    current_session = TradingSession.ASIAN
                elif 'london' in kz_key:
                    current_session = TradingSession.LONDON
                elif 'ny_am' in kz_key or 'ny_open' in kz_key:
                    current_session = TradingSession.NEW_YORK_AM
                elif 'lunch' in kz_key:
                    current_session = TradingSession.NEW_YORK_LUNCH
                elif 'ny_pm' in kz_key:
                    current_session = TradingSession.NEW_YORK_PM
                break
        
        # Check if in kill zone
        is_kill_zone = current_kz is not None and current_kz not in self.AVOID_SESSIONS
        
        # Check macro time
        is_macro = False
        minutes_to_macro = 60
        
        for macro in self.MACRO_TIMES:
            macro_start = macro['minute']
            macro_end = (macro_start + macro['duration']) % 60
            
            if macro_start <= current_minute < macro_end or \
               (macro_start > macro_end and (current_minute >= macro_start or current_minute < macro_end)):
                is_macro = True
                minutes_to_macro = 0
                break
            else:
                # Calculate minutes to next macro
                if current_minute < macro_start:
                    mins_to = macro_start - current_minute
                else:
                    mins_to = (60 - current_minute) + macro_start
                minutes_to_macro = min(minutes_to_macro, mins_to)
        
        # Check if trading is allowed
        can_trade = True
        block_reason = None
        
        # Session check
        if current_session not in self.params.allowed_sessions:
            can_trade = False
            block_reason = f"Session {current_session.name} not in allowed sessions"
        
        # Kill zone requirement
        if self.params.allowed_sessions and not is_kill_zone:
            if TradingSession.AFTER_HOURS not in self.params.allowed_sessions:
                can_trade = False
                block_reason = "Outside kill zone"
        
        # Avoid lunch
        if current_kz in self.AVOID_SESSIONS:
            can_trade = False
            block_reason = "NY Lunch - low probability period"
        
        # Check news blocks
        if self.params.block_during_news:
            for event in self.news_events:
                event_time = event['time']
                block_start = event_time - timedelta(minutes=self.params.news_block_minutes_before)
                block_end = event_time + timedelta(minutes=self.params.news_block_minutes_after)
                
                if block_start <= current_time <= block_end:
                    can_trade = False
                    block_reason = f"News event: {event.get('description', 'High impact')}"
                    break
        
        # Check custom blocked times
        for blocked in self.blocked_times:
            if blocked['start'] <= current_time <= blocked['end']:
                can_trade = False
                block_reason = f"Blocked: {blocked.get('reason', 'Custom block')}"
                break
        
        # Find next kill zone
        next_kz = None
        minutes_to_next = 24 * 60
        
        for kz_key, kz_data in self.KILL_ZONES.items():
            if kz_key in self.AVOID_SESSIONS:
                continue
            
            kz_start = kz_data['start']
            if kz_start > current_time_only:
                # Same day
                kz_datetime = datetime.combine(est_time.date(), kz_start)
            else:
                # Next day
                kz_datetime = datetime.combine(est_time.date() + timedelta(days=1), kz_start)
            
            mins_to = (kz_datetime - est_time).total_seconds() / 60
            if 0 < mins_to < minutes_to_next:
                minutes_to_next = mins_to
                next_kz = kz_data['name']
        
        return SessionStatus(
            current_session=current_session,
            is_kill_zone=is_kill_zone,
            kill_zone_name=current_kz_name,
            is_macro_time=is_macro,
            minutes_to_macro=minutes_to_macro,
            can_trade=can_trade,
            block_reason=block_reason,
            session_start=session_start,
            session_end=session_end,
            next_kill_zone=next_kz,
            minutes_to_next_kz=int(minutes_to_next)
        )
    
    def get_optimal_trading_windows(
        self, 
        date: Optional[datetime] = None,
        instrument: str = 'forex'
    ) -> List[Dict]:
        """
        Get optimal trading windows for a given date.
        
        Args:
            date: Date to check (defaults to today)
            instrument: Instrument type for session relevance
            
        Returns:
            List of optimal trading windows with times and quality
        """
        if date is None:
            date = datetime.utcnow()
        
        windows = []
        
        # Primary windows (highest probability)
        primary_kzs = ['london_open', 'ny_open', 'silver_bullet_london', 
                       'silver_bullet_ny_am', 'silver_bullet_ny_pm']
        
        for kz_key in primary_kzs:
            if kz_key in self.KILL_ZONES:
                kz = self.KILL_ZONES[kz_key]
                windows.append({
                    'name': kz['name'],
                    'start': datetime.combine(date.date(), kz['start']),
                    'end': datetime.combine(date.date(), kz['end']),
                    'quality': 'A+' if 'silver_bullet' in kz_key else 'A',
                    'type': 'primary',
                    'notes': 'Highest probability ICT window'
                })
        
        # Secondary windows
        secondary_kzs = ['london', 'ny_am', 'ny_pm']
        for kz_key in secondary_kzs:
            if kz_key in self.KILL_ZONES and kz_key not in primary_kzs:
                kz = self.KILL_ZONES[kz_key]
                windows.append({
                    'name': kz['name'],
                    'start': datetime.combine(date.date(), kz['start']),
                    'end': datetime.combine(date.date(), kz['end']),
                    'quality': 'B+',
                    'type': 'secondary',
                    'notes': 'Good probability window'
                })
        
        # Sort by start time
        windows.sort(key=lambda x: x['start'])
        
        return windows


# =============================================================================
# MAIN RISK MANAGER
# =============================================================================

class ICTRiskManager:
    """
    Master risk management system integrating all components.
    """
    
    def __init__(self, params: Optional[RiskParameters] = None):
        """
        Initialize ICT Risk Manager.
        
        Args:
            params: Risk parameters (uses defaults if not provided)
        """
        self.params = params or RiskParameters()
        
        # Initialize component engines
        self.position_sizer = PositionSizingEngine(self.params)
        self.stop_loss_engine = DynamicStopLossEngine(self.params)
        self.drawdown_engine = DrawdownProtectionEngine(self.params)
        self.trailing_engine = TrailingStopEngine(self.params)
        self.session_engine = SessionGuardsEngine(self.params)
        
        # State tracking
        self.open_trades: Dict[str, Dict] = {}
        self.trade_history: List[Dict] = []
        
        logger.info("ICT Risk Manager initialized")
    
    def assess_trade_risk(
        self,
        account: AccountState,
        entry_price: float,
        direction: str,
        take_profit: float,
        pip_value: float = 10.0,
        signal_confidence: float = 1.0,
        sizing_method: PositionSizingMethod = PositionSizingMethod.FIXED_PERCENTAGE,
        stop_type: StopLossType = StopLossType.ATR_BASED,
        atr: Optional[float] = None,
        swing_high: Optional[float] = None,
        swing_low: Optional[float] = None,
        order_block: Optional[Dict] = None,
        fvg: Optional[Dict] = None,
        structure_level: Optional[float] = None,
        pip_size: float = 0.0001
    ) -> RiskAssessment:
        """
        Perform complete risk assessment for a potential trade.
        
        Args:
            account: Current account state
            entry_price: Planned entry price
            direction: Trade direction ('long' or 'short')
            take_profit: Take profit target
            pip_value: Value per pip per lot
            signal_confidence: Signal confidence (0-1)
            sizing_method: Position sizing method
            stop_type: Stop loss type
            atr: Average True Range
            swing_high/low: Recent swing levels
            order_block: Order block data
            fvg: FVG data
            structure_level: Structure level
            pip_size: Pip size
            
        Returns:
            Complete RiskAssessment
        """
        warnings = []
        blocks = []
        recommendations = []
        
        # 1. Check session status
        session_status = self.session_engine.get_session_status()
        
        if not session_status.can_trade:
            blocks.append(f"Session block: {session_status.block_reason}")
        
        if session_status.is_macro_time:
            recommendations.append("In macro time - good for entries")
        elif session_status.minutes_to_macro <= 5:
            recommendations.append(f"Macro time in {session_status.minutes_to_macro} mins")
        
        # 2. Check drawdown status
        drawdown_status = self.drawdown_engine.check_drawdown_status(account)
        
        if not drawdown_status.can_trade:
            blocks.append(f"Drawdown block: {', '.join(drawdown_status.limits_hit)}")
        
        warnings.extend(drawdown_status.warnings)
        
        # 3. Calculate stop loss
        stop_loss = self.stop_loss_engine.calculate_stop_loss(
            entry_price=entry_price,
            direction=direction,
            stop_type=stop_type,
            atr=atr,
            swing_high=swing_high,
            swing_low=swing_low,
            order_block=order_block,
            fvg=fvg,
            structure_level=structure_level,
            pip_size=pip_size
        )
        
        if stop_loss.confidence < 0.7:
            warnings.append(f"Low stop confidence: {stop_loss.confidence:.0%}")
            recommendations.append("Consider using structure-based stop instead")
        
        # 4. Calculate position size
        position_size = self.position_sizer.calculate_position_size(
            account=account,
            entry_price=entry_price,
            stop_loss=stop_loss.stop_price,
            pip_value=pip_value,
            method=sizing_method,
            signal_confidence=signal_confidence,
            atr=atr,
            recovery_mode=drawdown_status.recovery_mode
        )
        
        warnings.extend(position_size.warnings)
        
        if position_size.position_size <= 0:
            blocks.append("Position size zero - cannot take trade")
        
        # Update stop loss with risk amount
        stop_loss.risk_amount = position_size.risk_amount
        
        # 5. Calculate risk/reward
        is_long = direction.lower() == 'long'
        reward = (take_profit - entry_price) if is_long else (entry_price - take_profit)
        risk = abs(entry_price - stop_loss.stop_price)
        
        risk_reward = reward / risk if risk > 0 else 0
        
        if risk_reward < 1.5:
            warnings.append(f"Low R:R ratio: {risk_reward:.2f}")
            recommendations.append("Consider wider target or tighter stop")
        elif risk_reward < 2.0:
            warnings.append(f"Moderate R:R: {risk_reward:.2f}")
        
        # 6. Calculate expected value
        # Using historical win rate or default
        win_rate = account.win_rate if account.win_rate > 0 else 0.5
        expected_value = (win_rate * reward) - ((1 - win_rate) * risk)
        
        if expected_value <= 0:
            warnings.append("Negative expected value")
            recommendations.append("Avoid trade or improve entry")
        
        # 7. Calculate risk score (0-100, higher = riskier)
        risk_score = 0
        
        # Session risk
        if not session_status.is_kill_zone:
            risk_score += 20
        if session_status.current_session == TradingSession.NEW_YORK_LUNCH:
            risk_score += 15
        
        # Drawdown risk
        risk_score += drawdown_status.daily_loss_percent * 5
        risk_score += drawdown_status.consecutive_losses * 10
        
        # R:R risk
        if risk_reward < 1.5:
            risk_score += 20
        elif risk_reward < 2.0:
            risk_score += 10
        
        # Confidence risk
        risk_score += (1 - signal_confidence) * 20
        
        # Cap at 100
        risk_score = min(100, risk_score)
        
        # 8. Determine if trade can be taken
        can_take = len(blocks) == 0 and position_size.position_size > 0
        
        # Final recommendations
        if can_take and risk_score > 60:
            recommendations.append("High risk - consider reducing size")
        if can_take and session_status.minutes_to_next_kz < 15:
            recommendations.append(f"Better window in {session_status.minutes_to_next_kz} mins")
        
        return RiskAssessment(
            can_take_trade=can_take,
            position_size=position_size,
            stop_loss=stop_loss,
            session_status=session_status,
            drawdown_status=drawdown_status,
            risk_score=risk_score,
            risk_reward_ratio=risk_reward,
            expected_value=expected_value,
            warnings=warnings,
            blocks=blocks,
            recommendations=recommendations
        )
    
    def register_trade(
        self,
        trade_id: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        direction: str,
        position_size: float
    ):
        """Register a new trade for tracking"""
        self.open_trades[trade_id] = {
            'entry_price': entry_price,
            'initial_stop': stop_loss,
            'current_stop': stop_loss,
            'take_profit': take_profit,
            'direction': direction,
            'position_size': position_size,
            'highest_price': entry_price,
            'lowest_price': entry_price,
            'entry_time': datetime.utcnow()
        }
        logger.info(f"Registered trade {trade_id}")
    
    def update_trade(
        self,
        trade_id: str,
        current_price: float,
        trail_method: TrailingStopMethod = TrailingStopMethod.ATR_TRAIL,
        atr: Optional[float] = None,
        swing_levels: Optional[List[float]] = None
    ) -> Optional[TrailingStopUpdate]:
        """
        Update trade with current price and calculate trailing stop.
        
        Returns trailing stop update if stop should be moved.
        """
        if trade_id not in self.open_trades:
            logger.warning(f"Trade {trade_id} not found")
            return None
        
        trade = self.open_trades[trade_id]
        
        # Update high/low
        trade['highest_price'] = max(trade['highest_price'], current_price)
        trade['lowest_price'] = min(trade['lowest_price'], current_price)
        
        # Calculate trailing stop
        update = self.trailing_engine.calculate_trailing_stop(
            entry_price=trade['entry_price'],
            current_price=current_price,
            current_stop=trade['current_stop'],
            direction=trade['direction'],
            initial_stop=trade['initial_stop'],
            take_profit=trade['take_profit'],
            method=trail_method,
            atr=atr,
            highest_price=trade['highest_price'],
            lowest_price=trade['lowest_price'],
            swing_levels=swing_levels
        )
        
        if update.should_update:
            trade['current_stop'] = update.new_stop
            logger.info(f"Trade {trade_id} stop updated: {update.old_stop:.5f} -> {update.new_stop:.5f}")
        
        return update
    
    def close_trade(self, trade_id: str, exit_price: float, exit_reason: str = ''):
        """Close trade and record result"""
        if trade_id not in self.open_trades:
            return
        
        trade = self.open_trades[trade_id]
        
        # Calculate P/L
        is_long = trade['direction'].lower() == 'long'
        pnl = (exit_price - trade['entry_price']) if is_long else (trade['entry_price'] - exit_price)
        pnl *= trade['position_size']
        
        # Record
        trade_record = {
            **trade,
            'exit_price': exit_price,
            'exit_time': datetime.utcnow(),
            'pnl': pnl,
            'exit_reason': exit_reason
        }
        self.trade_history.append(trade_record)
        
        # Remove from open
        del self.open_trades[trade_id]
        
        logger.info(f"Closed trade {trade_id}: PnL = {pnl:.2f}")
    
    def get_risk_summary(self, account: AccountState) -> Dict:
        """Get summary of current risk status"""
        session = self.session_engine.get_session_status()
        drawdown = self.drawdown_engine.check_drawdown_status(account)
        
        return {
            'can_trade': session.can_trade and drawdown.can_trade,
            'session': session.current_session.name,
            'kill_zone': session.kill_zone_name,
            'is_macro': session.is_macro_time,
            'recovery_mode': drawdown.recovery_mode.name,
            'daily_loss': f"{drawdown.daily_loss_percent:.1f}%",
            'consecutive_losses': drawdown.consecutive_losses,
            'open_trades': len(self.open_trades),
            'size_multiplier': drawdown.size_multiplier,
            'warnings': drawdown.warnings + ([session.block_reason] if session.block_reason else [])
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_default_risk_manager() -> ICTRiskManager:
    """Create risk manager with default ICT-aligned parameters"""
    params = RiskParameters(
        risk_per_trade_percent=1.0,
        max_risk_per_trade_percent=2.0,
        max_daily_loss_percent=3.0,
        max_weekly_loss_percent=6.0,
        max_consecutive_losses=3,
        breakeven_r_multiple=1.0,
        trail_start_r_multiple=1.5,
        allowed_sessions=[
            TradingSession.LONDON,
            TradingSession.NEW_YORK_AM,
            TradingSession.NEW_YORK_PM
        ],
        block_during_news=True
    )
    return ICTRiskManager(params)


def create_conservative_risk_manager() -> ICTRiskManager:
    """Create conservative risk manager"""
    params = RiskParameters(
        risk_per_trade_percent=0.5,
        max_risk_per_trade_percent=1.0,
        max_daily_loss_percent=2.0,
        max_weekly_loss_percent=4.0,
        max_consecutive_losses=2,
        recovery_size_multiplier=0.25,
        allowed_sessions=[
            TradingSession.LONDON,
            TradingSession.NEW_YORK_AM
        ]
    )
    return ICTRiskManager(params)


def create_aggressive_risk_manager() -> ICTRiskManager:
    """Create aggressive risk manager (use with caution)"""
    params = RiskParameters(
        risk_per_trade_percent=2.0,
        max_risk_per_trade_percent=3.0,
        max_daily_loss_percent=5.0,
        max_weekly_loss_percent=10.0,
        max_consecutive_losses=5,
        allowed_sessions=[
            TradingSession.ASIAN,
            TradingSession.LONDON,
            TradingSession.NEW_YORK_AM,
            TradingSession.NEW_YORK_PM
        ]
    )
    return ICTRiskManager(params)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    # Create risk manager
    risk_manager = create_default_risk_manager()
    
    # Sample account state
    account = AccountState(
        balance=10000.0,
        equity=10000.0,
        margin_used=0.0,
        margin_available=10000.0,
        unrealized_pnl=0.0,
        daily_pnl=-50.0,
        weekly_pnl=-100.0,
        monthly_pnl=200.0,
        peak_balance=10200.0,
        current_drawdown=2.0,
        max_drawdown=5.0,
        open_positions=0,
        pending_orders=0,
        consecutive_losses=1,
        consecutive_wins=0,
        total_trades=50,
        win_rate=0.55,
        avg_win=150.0,
        avg_loss=100.0
    )
    
    # Assess a potential trade
    assessment = risk_manager.assess_trade_risk(
        account=account,
        entry_price=1.0850,
        direction='long',
        take_profit=1.0900,
        signal_confidence=0.75,
        stop_type=StopLossType.ATR_BASED,
        atr=0.0015,
        swing_low=1.0820,
        order_block={'low': 1.0830, 'high': 1.0840}
    )
    
    print("\n" + "="*60)
    print("RISK ASSESSMENT")
    print("="*60)
    print(f"Can Take Trade: {assessment.can_take_trade}")
    print(f"Position Size: {assessment.position_size.position_size:.2f} lots")
    print(f"Risk Amount: ${assessment.position_size.risk_amount:.2f}")
    print(f"Stop Loss: {assessment.stop_loss.stop_price:.5f}")
    print(f"Risk/Reward: {assessment.risk_reward_ratio:.2f}")
    print(f"Risk Score: {assessment.risk_score:.0f}/100")
    print(f"Session: {assessment.session_status.current_session.name}")
    print(f"Kill Zone: {assessment.session_status.kill_zone_name or 'None'}")
    print(f"Recovery Mode: {assessment.drawdown_status.recovery_mode.name}")
    
    if assessment.warnings:
        print("\nWarnings:")
        for w in assessment.warnings:
            print(f"  ⚠ {w}")
    
    if assessment.blocks:
        print("\nBlocks:")
        for b in assessment.blocks:
            print(f"  🚫 {b}")
    
    if assessment.recommendations:
        print("\nRecommendations:")
        for r in assessment.recommendations:
            print(f"  → {r}")
    
    print("\n" + "="*60)
    print("RISK SUMMARY")
    print("="*60)
    summary = risk_manager.get_risk_summary(account)
    for key, value in summary.items():
        print(f"{key}: {value}")
