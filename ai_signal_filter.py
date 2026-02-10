"""
ICT AI Signal Filter - Phase 3: Signal Validation Module
=========================================================

This module provides AI-powered validation and filtering of ICT trading
signals before execution. It combines multiple ML models to assess signal
quality, probability, and optimal execution timing.

CORE RESPONSIBILITIES:
1. Multi-model signal validation
2. Quality scoring and ranking
3. False signal detection
4. Optimal execution timing
5. Risk-adjusted filtering
6. Adaptive threshold management

FILTER CRITERIA:
- ML model confidence scores
- Historical pattern matching
- Market regime alignment
- Time/session validity
- Risk/reward assessment
- Confluence verification

Author: ICT AI Engine
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Callable
from enum import Enum
from datetime import datetime, time, timedelta
import pandas as pd
import numpy as np
import json
import logging
from collections import deque, defaultdict
from abc import ABC, abstractmethod
import warnings

warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing ML libraries
try:
    from sklearn.ensemble import IsolationForest
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


# =============================================================================
# ENUMERATIONS
# =============================================================================

class FilterDecision(Enum):
    """Signal filter decisions"""
    ACCEPT = "accept"           # Signal passes all filters
    REJECT = "reject"           # Signal fails critical filters
    HOLD = "hold"               # Signal needs more confirmation
    MODIFY = "modify"           # Accept with modifications


class FilterReason(Enum):
    """Reasons for filter decisions"""
    HIGH_CONFIDENCE = "high_confidence"
    LOW_CONFIDENCE = "low_confidence"
    PATTERN_MATCH = "pattern_match"
    NO_PATTERN_MATCH = "no_pattern_match"
    REGIME_ALIGNED = "regime_aligned"
    REGIME_MISMATCH = "regime_mismatch"
    TIME_VALID = "time_valid"
    TIME_INVALID = "time_invalid"
    RR_ACCEPTABLE = "rr_acceptable"
    RR_TOO_LOW = "rr_too_low"
    CONFLUENCE_HIGH = "confluence_high"
    CONFLUENCE_LOW = "confluence_low"
    ANOMALY_DETECTED = "anomaly_detected"
    DAILY_LIMIT = "daily_limit"
    POSITION_LIMIT = "position_limit"


class SignalPriority(Enum):
    """Filtered signal priority"""
    CRITICAL = 5        # Execute immediately
    HIGH = 4            # Execute soon
    MEDIUM = 3          # Standard priority
    LOW = 2             # Wait for better setup
    VERY_LOW = 1        # May skip


class MarketRegime(Enum):
    """Market regime classification"""
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    LOW_VOLATILITY = "low_volatility"
    TRANSITION = "transition"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FilterConfig:
    """Configuration for signal filtering"""
    # Confidence thresholds
    min_ml_confidence: float = 0.55       # Minimum ML model confidence
    min_ensemble_confidence: float = 0.60  # Minimum ensemble confidence
    min_pattern_similarity: float = 0.70   # Minimum pattern match
    
    # Risk parameters
    min_risk_reward: float = 1.5          # Minimum R:R ratio
    max_risk_percent: float = 0.02        # Maximum risk per trade
    max_correlation: float = 0.7          # Maximum correlation with existing
    
    # Time filters
    require_kill_zone: bool = True
    allowed_sessions: List[str] = field(default_factory=lambda: ['london', 'new_york'])
    blocked_hours: List[int] = field(default_factory=lambda: [22, 23, 0, 1, 2])
    
    # Confluence requirements
    min_confluence_factors: int = 3
    require_htf_alignment: bool = True
    require_structure_confirmation: bool = True
    
    # Daily limits
    max_daily_signals: int = 5
    max_concurrent_positions: int = 3
    max_daily_loss_percent: float = 0.05  # 5% daily loss limit
    
    # Regime filters
    trade_in_ranging: bool = True
    trade_in_volatile: bool = False
    
    # Anomaly detection
    use_anomaly_detection: bool = True
    anomaly_threshold: float = -0.5       # Isolation forest threshold
    
    # Adaptive thresholds
    adaptive_thresholds: bool = True
    lookback_trades: int = 50


@dataclass
class SignalInput:
    """Input signal to be filtered"""
    signal_id: str
    timestamp: datetime
    symbol: str
    
    # Signal details
    direction: str                     # 'long' or 'short'
    entry_price: float
    stop_loss: float
    take_profit: float
    
    # ICT context
    ict_model: str                     # Model 2022, Silver Bullet, etc.
    grade: str                         # A+, A, B, C
    confluence_factors: List[str]
    confluence_score: float
    
    # Market context
    market_session: str
    in_kill_zone: bool
    htf_bias: str
    ltf_bias: str
    market_regime: str
    
    # Risk metrics
    risk_reward_ratio: float
    position_size: float
    risk_amount: float
    
    # ML predictions (optional)
    ml_confidence: float               # Primary model confidence
    ensemble_confidence: Optional[float] = None
    lstm_prediction: Optional[str] = None
    lstm_confidence: Optional[float] = None
    
    # Pattern matching (optional)
    pattern_similarity: Optional[float] = None
    similar_patterns_win_rate: Optional[float] = None
    
    # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FilterResult:
    """Result of signal filtering"""
    signal_id: str
    timestamp: datetime
    
    # Decision
    decision: FilterDecision
    priority: SignalPriority
    
    # Scores
    overall_score: float               # 0-100
    confidence_score: float            # ML confidence composite
    pattern_score: float               # Pattern matching score
    timing_score: float                # Time/session score
    risk_score: float                  # Risk assessment score
    regime_score: float                # Market regime alignment
    
    # Reasons
    accept_reasons: List[FilterReason]
    reject_reasons: List[FilterReason]
    warnings: List[str]
    
    # Modifications
    modified_entry: Optional[float] = None
    modified_stop: Optional[float] = None
    modified_target: Optional[float] = None
    modified_size: Optional[float] = None
    
    # Recommendations
    wait_for_confirmation: bool = False
    confirmation_criteria: Optional[str] = None
    optimal_entry_time: Optional[datetime] = None
    
    # Execution guidance
    execution_type: str = "limit"       # limit, market, conditional
    max_slippage: float = 0.0
    valid_until: Optional[datetime] = None


@dataclass
class FilterState:
    """Current state of the filter system"""
    daily_signals_count: int = 0
    daily_accepted: int = 0
    daily_rejected: int = 0
    current_positions: int = 0
    daily_pnl_percent: float = 0.0
    
    # Recent history
    recent_signals: List[FilterResult] = field(default_factory=list)
    recent_trades: List[Dict] = field(default_factory=list)
    
    # Adaptive metrics
    rolling_win_rate: float = 0.5
    rolling_avg_rr: float = 1.5
    current_regime: MarketRegime = MarketRegime.RANGING
    
    # Session tracking
    session_start: Optional[datetime] = None
    signals_this_session: int = 0


# =============================================================================
# FILTER COMPONENTS
# =============================================================================

class BaseFilter(ABC):
    """Abstract base class for filters"""
    
    @abstractmethod
    def filter(
        self, 
        signal: SignalInput, 
        state: FilterState,
        config: FilterConfig
    ) -> Tuple[bool, float, List[FilterReason]]:
        """
        Apply filter to signal.
        
        Returns:
            (passes, score, reasons)
        """
        pass


class ConfidenceFilter(BaseFilter):
    """Filter based on ML confidence scores"""
    
    def filter(
        self, 
        signal: SignalInput, 
        state: FilterState,
        config: FilterConfig
    ) -> Tuple[bool, float, List[FilterReason]]:
        reasons = []
        
        # Primary ML confidence
        ml_conf = signal.ml_confidence
        
        # Ensemble confidence if available
        ensemble_conf = signal.ensemble_confidence or ml_conf
        
        # LSTM alignment bonus
        lstm_bonus = 0
        if signal.lstm_prediction and signal.lstm_confidence:
            if signal.lstm_prediction.lower() == signal.direction.lower():
                lstm_bonus = signal.lstm_confidence * 0.1
            else:
                lstm_bonus = -signal.lstm_confidence * 0.1
                
        # Calculate composite confidence
        composite = (ml_conf * 0.5 + ensemble_conf * 0.3 + lstm_bonus)
        
        # Grade bonus
        grade_bonus = {'A+': 0.15, 'A': 0.10, 'B': 0.05, 'C': 0}.get(signal.grade, 0)
        composite += grade_bonus
        
        # Normalize to 0-100
        score = min(100, max(0, composite * 100))
        
        # Check thresholds
        passes = True
        
        if ml_conf < config.min_ml_confidence:
            passes = False
            reasons.append(FilterReason.LOW_CONFIDENCE)
        else:
            reasons.append(FilterReason.HIGH_CONFIDENCE)
            
        if ensemble_conf < config.min_ensemble_confidence:
            passes = False
            
        return passes, score, reasons


class PatternFilter(BaseFilter):
    """Filter based on historical pattern matching"""
    
    def filter(
        self, 
        signal: SignalInput, 
        state: FilterState,
        config: FilterConfig
    ) -> Tuple[bool, float, List[FilterReason]]:
        reasons = []
        
        if signal.pattern_similarity is None:
            return True, 50.0, [FilterReason.PATTERN_MATCH]
            
        similarity = signal.pattern_similarity
        win_rate = signal.similar_patterns_win_rate or 0.5
        
        # Calculate pattern score
        score = (similarity * 50 + win_rate * 50)
        
        passes = True
        
        if similarity < config.min_pattern_similarity:
            passes = False
            reasons.append(FilterReason.NO_PATTERN_MATCH)
        else:
            reasons.append(FilterReason.PATTERN_MATCH)
            
        # Bonus for high win rate patterns
        if win_rate > 0.65:
            score += 10
            
        return passes, min(100, score), reasons


class TimingFilter(BaseFilter):
    """Filter based on time and session"""
    
    def filter(
        self, 
        signal: SignalInput, 
        state: FilterState,
        config: FilterConfig
    ) -> Tuple[bool, float, List[FilterReason]]:
        reasons = []
        score = 50.0
        passes = True
        
        # Check kill zone requirement
        if config.require_kill_zone and not signal.in_kill_zone:
            passes = False
            reasons.append(FilterReason.TIME_INVALID)
            score -= 20
        else:
            if signal.in_kill_zone:
                score += 15
                reasons.append(FilterReason.TIME_VALID)
                
        # Check session
        session = signal.market_session.lower()
        if session not in [s.lower() for s in config.allowed_sessions]:
            passes = False
            reasons.append(FilterReason.TIME_INVALID)
            score -= 15
        else:
            score += 10
            
        # Check blocked hours
        hour = signal.timestamp.hour
        if hour in config.blocked_hours:
            passes = False
            reasons.append(FilterReason.TIME_INVALID)
            score -= 25
            
        # Day of week adjustment
        dow = signal.timestamp.weekday()
        if dow == 0:  # Monday
            score -= 5  # Monday setups often less reliable
        elif dow == 4:  # Friday
            score -= 10  # Avoid Friday afternoon trades
            
        return passes, max(0, min(100, score)), reasons


class RiskFilter(BaseFilter):
    """Filter based on risk assessment"""
    
    def filter(
        self, 
        signal: SignalInput, 
        state: FilterState,
        config: FilterConfig
    ) -> Tuple[bool, float, List[FilterReason]]:
        reasons = []
        score = 50.0
        passes = True
        
        # Check risk/reward
        rr = signal.risk_reward_ratio
        if rr < config.min_risk_reward:
            passes = False
            reasons.append(FilterReason.RR_TOO_LOW)
            score -= 20
        else:
            reasons.append(FilterReason.RR_ACCEPTABLE)
            score += min(20, (rr - config.min_risk_reward) * 10)
            
        # Check position size
        if signal.risk_amount > config.max_risk_percent:
            passes = False
            score -= 15
            
        # Check daily loss limit
        if state.daily_pnl_percent < -config.max_daily_loss_percent:
            passes = False
            reasons.append(FilterReason.DAILY_LIMIT)
            score -= 30
            
        # Check position limits
        if state.current_positions >= config.max_concurrent_positions:
            passes = False
            reasons.append(FilterReason.POSITION_LIMIT)
            score -= 25
            
        # Adjust for current drawdown
        if state.daily_pnl_percent < -0.02:
            score -= 15  # Reduce aggression in drawdown
            
        return passes, max(0, min(100, score)), reasons


class ConfluenceFilter(BaseFilter):
    """Filter based on ICT confluence"""
    
    def filter(
        self, 
        signal: SignalInput, 
        state: FilterState,
        config: FilterConfig
    ) -> Tuple[bool, float, List[FilterReason]]:
        reasons = []
        passes = True
        
        # Count confluence factors
        factor_count = len(signal.confluence_factors)
        conf_score = signal.confluence_score
        
        # Calculate score
        score = min(100, conf_score)
        
        if factor_count < config.min_confluence_factors:
            passes = False
            reasons.append(FilterReason.CONFLUENCE_LOW)
        else:
            reasons.append(FilterReason.CONFLUENCE_HIGH)
            
        # HTF alignment check
        if config.require_htf_alignment:
            htf_aligned = signal.htf_bias.lower() in signal.direction.lower() or \
                         (signal.htf_bias.lower() == 'bullish' and signal.direction == 'long') or \
                         (signal.htf_bias.lower() == 'bearish' and signal.direction == 'short')
            if not htf_aligned:
                passes = False
                score -= 20
                
        # LTF confirmation
        ltf_aligned = signal.ltf_bias.lower() in signal.direction.lower() or \
                     (signal.ltf_bias.lower() == 'bullish' and signal.direction == 'long') or \
                     (signal.ltf_bias.lower() == 'bearish' and signal.direction == 'short')
        if ltf_aligned:
            score += 10
        else:
            score -= 10
            
        return passes, max(0, min(100, score)), reasons


class RegimeFilter(BaseFilter):
    """Filter based on market regime"""
    
    def filter(
        self, 
        signal: SignalInput, 
        state: FilterState,
        config: FilterConfig
    ) -> Tuple[bool, float, List[FilterReason]]:
        reasons = []
        passes = True
        score = 50.0
        
        regime = signal.market_regime.lower()
        
        # Regime-specific filtering
        if 'ranging' in regime:
            if not config.trade_in_ranging:
                passes = False
            score += 0  # Neutral for ranging
            
        elif 'volatile' in regime:
            if not config.trade_in_volatile:
                passes = False
                score -= 20
            else:
                score += 10  # Opportunity in volatility
                
        elif 'trending' in regime:
            # Check trend alignment
            if ('up' in regime and signal.direction == 'long') or \
               ('down' in regime and signal.direction == 'short'):
                score += 20
                reasons.append(FilterReason.REGIME_ALIGNED)
            else:
                score -= 15
                reasons.append(FilterReason.REGIME_MISMATCH)
                
        elif 'transition' in regime:
            score -= 10  # Be cautious in transitions
            
        return passes, max(0, min(100, score)), reasons


class AnomalyFilter(BaseFilter):
    """Filter using anomaly detection"""
    
    def __init__(self):
        self.isolation_forest = None
        self.feature_history = deque(maxlen=1000)
        self._is_fitted = False
        
    def filter(
        self, 
        signal: SignalInput, 
        state: FilterState,
        config: FilterConfig
    ) -> Tuple[bool, float, List[FilterReason]]:
        if not config.use_anomaly_detection or not SKLEARN_AVAILABLE:
            return True, 50.0, []
            
        reasons = []
        passes = True
        score = 50.0
        
        # Extract features for anomaly detection
        features = self._extract_features(signal)
        
        # Add to history
        self.feature_history.append(features)
        
        # Fit/refit model if enough data
        if len(self.feature_history) >= 50:
            if not self._is_fitted or len(self.feature_history) % 100 == 0:
                self._fit_model()
                
            # Predict anomaly score
            if self._is_fitted:
                anomaly_score = self.isolation_forest.decision_function([features])[0]
                
                if anomaly_score < config.anomaly_threshold:
                    passes = False
                    reasons.append(FilterReason.ANOMALY_DETECTED)
                    score -= 25
                else:
                    score += 5
                    
        return passes, max(0, min(100, score)), reasons
    
    def _extract_features(self, signal: SignalInput) -> List[float]:
        """Extract numerical features for anomaly detection"""
        return [
            signal.ml_confidence,
            signal.confluence_score,
            signal.risk_reward_ratio,
            signal.pattern_similarity or 0.5,
            1 if signal.in_kill_zone else 0,
            len(signal.confluence_factors),
            signal.entry_price,
            abs(signal.entry_price - signal.stop_loss)
        ]
    
    def _fit_model(self):
        """Fit isolation forest on historical data"""
        if len(self.feature_history) < 50:
            return
            
        X = np.array(list(self.feature_history))
        
        self.isolation_forest = IsolationForest(
            contamination=0.1,
            random_state=42
        )
        self.isolation_forest.fit(X)
        self._is_fitted = True


# =============================================================================
# MAIN SIGNAL FILTER
# =============================================================================

class AISignalFilter:
    """
    Main AI Signal Filter combining all filter components.
    
    FILTERING PROCESS:
    1. Apply individual filters
    2. Calculate weighted composite score
    3. Make accept/reject/hold decision
    4. Generate modifications if needed
    5. Update filter state
    """
    
    def __init__(self, config: Optional[FilterConfig] = None):
        self.config = config or FilterConfig()
        self.state = FilterState()
        
        # Initialize filters
        self.filters = {
            'confidence': ConfidenceFilter(),
            'pattern': PatternFilter(),
            'timing': TimingFilter(),
            'risk': RiskFilter(),
            'confluence': ConfluenceFilter(),
            'regime': RegimeFilter(),
            'anomaly': AnomalyFilter()
        }
        
        # Filter weights for composite score
        self.filter_weights = {
            'confidence': 0.25,
            'pattern': 0.15,
            'timing': 0.15,
            'risk': 0.20,
            'confluence': 0.15,
            'regime': 0.10
        }
        
        # Adaptive threshold tracking
        self.threshold_history = deque(maxlen=100)
        
        # Callbacks
        self.on_accept: Optional[Callable] = None
        self.on_reject: Optional[Callable] = None
        
    def filter_signal(self, signal: SignalInput) -> FilterResult:
        """
        Apply all filters to a signal.
        
        Args:
            signal: Signal to filter
            
        Returns:
            FilterResult with decision and scores
        """
        # Check daily limits first
        if self.state.daily_signals_count >= self.config.max_daily_signals:
            return self._create_reject_result(
                signal, 
                [FilterReason.DAILY_LIMIT],
                "Daily signal limit reached"
            )
            
        # Apply all filters
        filter_results = {}
        all_reasons_accept = []
        all_reasons_reject = []
        all_pass = True
        
        for name, filter_obj in self.filters.items():
            passes, score, reasons = filter_obj.filter(
                signal, self.state, self.config
            )
            filter_results[name] = {
                'passes': passes,
                'score': score,
                'reasons': reasons
            }
            
            if not passes:
                all_pass = False
                all_reasons_reject.extend([r for r in reasons if r not in all_reasons_reject])
            else:
                all_reasons_accept.extend([r for r in reasons if r not in all_reasons_accept])
                
        # Calculate composite scores
        confidence_score = filter_results['confidence']['score']
        pattern_score = filter_results['pattern']['score']
        timing_score = filter_results['timing']['score']
        risk_score = filter_results['risk']['score']
        regime_score = filter_results['regime']['score']
        
        # Weighted overall score
        overall_score = sum(
            filter_results[name]['score'] * weight
            for name, weight in self.filter_weights.items()
            if name in filter_results
        )
        
        # Determine decision
        decision = self._determine_decision(
            all_pass, overall_score, filter_results
        )
        
        # Determine priority
        priority = self._determine_priority(overall_score, signal.grade)
        
        # Check for modifications
        modified_entry, modified_stop, modified_target, modified_size = \
            self._calculate_modifications(signal, filter_results)
            
        # Generate warnings
        warnings = self._generate_warnings(signal, filter_results)
        
        # Create result
        result = FilterResult(
            signal_id=signal.signal_id,
            timestamp=datetime.now(),
            decision=decision,
            priority=priority,
            overall_score=overall_score,
            confidence_score=confidence_score,
            pattern_score=pattern_score,
            timing_score=timing_score,
            risk_score=risk_score,
            regime_score=regime_score,
            accept_reasons=all_reasons_accept,
            reject_reasons=all_reasons_reject,
            warnings=warnings,
            modified_entry=modified_entry,
            modified_stop=modified_stop,
            modified_target=modified_target,
            modified_size=modified_size,
            wait_for_confirmation=decision == FilterDecision.HOLD,
            confirmation_criteria=self._get_confirmation_criteria(filter_results) if decision == FilterDecision.HOLD else None,
            optimal_entry_time=self._calculate_optimal_entry_time(signal),
            execution_type=self._determine_execution_type(signal, overall_score),
            valid_until=signal.timestamp + timedelta(hours=4)
        )
        
        # Update state
        self._update_state(result)
        
        # Trigger callbacks
        if decision == FilterDecision.ACCEPT and self.on_accept:
            self.on_accept(result)
        elif decision == FilterDecision.REJECT and self.on_reject:
            self.on_reject(result)
            
        return result
    
    def _determine_decision(
        self,
        all_pass: bool,
        overall_score: float,
        filter_results: Dict
    ) -> FilterDecision:
        """Determine final filter decision"""
        # Hard reject conditions
        if not filter_results['risk']['passes']:
            return FilterDecision.REJECT
            
        if not filter_results['timing']['passes'] and self.config.require_kill_zone:
            return FilterDecision.REJECT
            
        if overall_score < 40:
            return FilterDecision.REJECT
            
        # Hold conditions
        if not all_pass and overall_score >= 50:
            return FilterDecision.HOLD
            
        if overall_score >= 50 and overall_score < 65:
            return FilterDecision.HOLD
            
        # Modify conditions
        if all_pass and overall_score >= 65 and overall_score < 75:
            return FilterDecision.MODIFY
            
        # Accept
        if all_pass and overall_score >= 65:
            return FilterDecision.ACCEPT
            
        return FilterDecision.REJECT
    
    def _determine_priority(self, score: float, grade: str) -> SignalPriority:
        """Determine signal priority based on score and grade"""
        # Grade bonus
        grade_priority = {'A+': 1, 'A': 0.5, 'B': 0, 'C': -0.5}.get(grade, 0)
        adjusted_score = score + grade_priority * 10
        
        if adjusted_score >= 85:
            return SignalPriority.CRITICAL
        elif adjusted_score >= 75:
            return SignalPriority.HIGH
        elif adjusted_score >= 65:
            return SignalPriority.MEDIUM
        elif adjusted_score >= 55:
            return SignalPriority.LOW
        else:
            return SignalPriority.VERY_LOW
    
    def _calculate_modifications(
        self,
        signal: SignalInput,
        filter_results: Dict
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Calculate any signal modifications"""
        modified_entry = None
        modified_stop = None
        modified_target = None
        modified_size = None
        
        # Reduce position size if confidence is borderline
        conf_score = filter_results['confidence']['score']
        if 50 <= conf_score < 65:
            modified_size = signal.position_size * 0.5  # Half size
        elif 65 <= conf_score < 75:
            modified_size = signal.position_size * 0.75
            
        # Tighten stop if regime is volatile
        regime_score = filter_results['regime']['score']
        if regime_score < 40:
            # Move stop closer
            stop_distance = abs(signal.entry_price - signal.stop_loss)
            if signal.direction == 'long':
                modified_stop = signal.entry_price - stop_distance * 0.75
            else:
                modified_stop = signal.entry_price + stop_distance * 0.75
                
        return modified_entry, modified_stop, modified_target, modified_size
    
    def _generate_warnings(
        self,
        signal: SignalInput,
        filter_results: Dict
    ) -> List[str]:
        """Generate warning messages"""
        warnings = []
        
        if filter_results['confidence']['score'] < 60:
            warnings.append("Low ML confidence - consider reduced position size")
            
        if filter_results['timing']['score'] < 60:
            warnings.append("Sub-optimal timing - may wait for better entry")
            
        if filter_results['regime']['score'] < 50:
            warnings.append("Market regime not ideal for this setup")
            
        if signal.risk_reward_ratio < 2.0:
            warnings.append("R:R below 2:1 - ensure tight risk management")
            
        if self.state.daily_pnl_percent < -0.01:
            warnings.append("In drawdown - exercise extra caution")
            
        if not signal.in_kill_zone and self.config.require_kill_zone:
            warnings.append("Outside kill zone - timing may not be optimal")
            
        return warnings
    
    def _get_confirmation_criteria(self, filter_results: Dict) -> str:
        """Get criteria needed to confirm a HOLD signal"""
        criteria = []
        
        if filter_results['confidence']['score'] < 65:
            criteria.append("Wait for higher ML confidence")
            
        if filter_results['timing']['score'] < 70:
            criteria.append("Wait for kill zone entry")
            
        if filter_results['confluence']['score'] < 60:
            criteria.append("Need additional confluence factors")
            
        return "; ".join(criteria) if criteria else "Wait for market structure confirmation"
    
    def _calculate_optimal_entry_time(
        self,
        signal: SignalInput
    ) -> Optional[datetime]:
        """Calculate optimal entry time if not currently in kill zone"""
        if signal.in_kill_zone:
            return signal.timestamp
            
        # Calculate next kill zone
        current_hour = signal.timestamp.hour
        
        # London kill zone: 8-11 AM UTC
        # NY kill zone: 14-17 PM UTC
        if current_hour < 8:
            next_kz = signal.timestamp.replace(hour=8, minute=0)
        elif current_hour < 14:
            next_kz = signal.timestamp.replace(hour=14, minute=0)
        else:
            next_kz = signal.timestamp.replace(hour=8, minute=0) + timedelta(days=1)
            
        return next_kz
    
    def _determine_execution_type(
        self,
        signal: SignalInput,
        score: float
    ) -> str:
        """Determine best execution type"""
        if score >= 85:
            return "market"  # High confidence, execute immediately
        elif signal.in_kill_zone:
            return "limit"   # In kill zone, use limit for better entry
        else:
            return "conditional"  # Wait for specific conditions
    
    def _create_reject_result(
        self,
        signal: SignalInput,
        reasons: List[FilterReason],
        message: str
    ) -> FilterResult:
        """Create a rejection result"""
        return FilterResult(
            signal_id=signal.signal_id,
            timestamp=datetime.now(),
            decision=FilterDecision.REJECT,
            priority=SignalPriority.VERY_LOW,
            overall_score=0,
            confidence_score=0,
            pattern_score=0,
            timing_score=0,
            risk_score=0,
            regime_score=0,
            accept_reasons=[],
            reject_reasons=reasons,
            warnings=[message]
        )
    
    def _update_state(self, result: FilterResult):
        """Update filter state after processing"""
        self.state.daily_signals_count += 1
        
        if result.decision == FilterDecision.ACCEPT:
            self.state.daily_accepted += 1
        elif result.decision == FilterDecision.REJECT:
            self.state.daily_rejected += 1
            
        self.state.recent_signals.append(result)
        if len(self.state.recent_signals) > 100:
            self.state.recent_signals.pop(0)
            
        # Update adaptive thresholds
        if self.config.adaptive_thresholds:
            self.threshold_history.append(result.overall_score)
    
    def record_trade_outcome(
        self,
        signal_id: str,
        outcome: str,
        pnl_percent: float
    ):
        """Record trade outcome for adaptive learning"""
        self.state.daily_pnl_percent += pnl_percent
        
        self.state.recent_trades.append({
            'signal_id': signal_id,
            'outcome': outcome,
            'pnl': pnl_percent
        })
        
        # Update rolling metrics
        if len(self.state.recent_trades) >= 10:
            recent = self.state.recent_trades[-self.config.lookback_trades:]
            wins = sum(1 for t in recent if t['outcome'] == 'win')
            self.state.rolling_win_rate = wins / len(recent)
            self.state.rolling_avg_rr = np.mean([t['pnl'] for t in recent if t['pnl'] > 0]) / \
                                        abs(np.mean([t['pnl'] for t in recent if t['pnl'] < 0]) or 1)
    
    def reset_daily_state(self):
        """Reset daily state (call at start of trading day)"""
        self.state.daily_signals_count = 0
        self.state.daily_accepted = 0
        self.state.daily_rejected = 0
        self.state.daily_pnl_percent = 0.0
        self.state.signals_this_session = 0
        self.state.session_start = datetime.now()
    
    def update_position_count(self, count: int):
        """Update current position count"""
        self.state.current_positions = count
    
    def update_market_regime(self, regime: MarketRegime):
        """Update current market regime"""
        self.state.current_regime = regime
    
    def get_filter_stats(self) -> Dict[str, Any]:
        """Get filter statistics"""
        return {
            'daily_signals': self.state.daily_signals_count,
            'daily_accepted': self.state.daily_accepted,
            'daily_rejected': self.state.daily_rejected,
            'acceptance_rate': self.state.daily_accepted / max(1, self.state.daily_signals_count),
            'current_positions': self.state.current_positions,
            'daily_pnl': self.state.daily_pnl_percent,
            'rolling_win_rate': self.state.rolling_win_rate,
            'current_regime': self.state.current_regime.value
        }
    
    def configure(self, **kwargs):
        """Update filter configuration"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)


# =============================================================================
# SIGNAL RANKER
# =============================================================================

class SignalRanker:
    """
    Rank multiple signals to determine execution priority.
    """
    
    def __init__(self, filter_instance: AISignalFilter):
        self.filter = filter_instance
        
    def rank_signals(
        self,
        signals: List[SignalInput]
    ) -> List[Tuple[SignalInput, FilterResult]]:
        """
        Filter and rank multiple signals.
        
        Args:
            signals: List of signals to rank
            
        Returns:
            List of (signal, result) tuples sorted by priority
        """
        results = []
        
        for signal in signals:
            result = self.filter.filter_signal(signal)
            results.append((signal, result))
            
        # Sort by decision (accept first) then by overall score
        decision_priority = {
            FilterDecision.ACCEPT: 0,
            FilterDecision.MODIFY: 1,
            FilterDecision.HOLD: 2,
            FilterDecision.REJECT: 3
        }
        
        results.sort(key=lambda x: (
            decision_priority[x[1].decision],
            -x[1].overall_score,
            -x[1].priority.value
        ))
        
        return results
    
    def get_best_signal(
        self,
        signals: List[SignalInput]
    ) -> Optional[Tuple[SignalInput, FilterResult]]:
        """Get the best signal from a list"""
        ranked = self.rank_signals(signals)
        
        # Return best acceptable signal
        for signal, result in ranked:
            if result.decision in [FilterDecision.ACCEPT, FilterDecision.MODIFY]:
                return signal, result
                
        return None


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("ICT AI Signal Filter - Phase 3 Signal Validation Module")
    print("=" * 60)
    
    # Create filter instance
    config = FilterConfig(
        min_ml_confidence=0.55,
        min_risk_reward=1.5,
        require_kill_zone=True,
        max_daily_signals=5
    )
    
    filter_system = AISignalFilter(config)
    
    # Create sample signal
    signal = SignalInput(
        signal_id="SIG_001",
        timestamp=datetime.now(),
        symbol="EURUSD",
        direction="long",
        entry_price=1.1050,
        stop_loss=1.1020,
        take_profit=1.1110,
        ict_model="model_2022",
        grade="A",
        confluence_factors=["order_block", "fvg", "liquidity_swept", "htf_aligned"],
        confluence_score=75.0,
        ml_confidence=0.72,
        ensemble_confidence=0.68,
        lstm_prediction="up",
        lstm_confidence=0.65,
        pattern_similarity=0.78,
        similar_patterns_win_rate=0.62,
        market_session="london",
        in_kill_zone=True,
        htf_bias="bullish",
        ltf_bias="bullish",
        market_regime="trending_up",
        risk_reward_ratio=2.0,
        position_size=0.1,
        risk_amount=0.01
    )
    
    print("\n📊 Sample Signal:")
    print(f"   Symbol: {signal.symbol}")
    print(f"   Direction: {signal.direction}")
    print(f"   ICT Model: {signal.ict_model}")
    print(f"   Grade: {signal.grade}")
    print(f"   ML Confidence: {signal.ml_confidence:.2%}")
    
    # Filter the signal
    result = filter_system.filter_signal(signal)
    
    print(f"\n🔍 Filter Result:")
    print(f"   Decision: {result.decision.value.upper()}")
    print(f"   Priority: {result.priority.name}")
    print(f"   Overall Score: {result.overall_score:.1f}/100")
    print(f"\n   Component Scores:")
    print(f"   - Confidence: {result.confidence_score:.1f}")
    print(f"   - Pattern: {result.pattern_score:.1f}")
    print(f"   - Timing: {result.timing_score:.1f}")
    print(f"   - Risk: {result.risk_score:.1f}")
    print(f"   - Regime: {result.regime_score:.1f}")
    
    if result.accept_reasons:
        print(f"\n   ✓ Accept Reasons:")
        for reason in result.accept_reasons:
            print(f"     - {reason.value}")
            
    if result.reject_reasons:
        print(f"\n   ✗ Reject Reasons:")
        for reason in result.reject_reasons:
            print(f"     - {reason.value}")
            
    if result.warnings:
        print(f"\n   ⚠️ Warnings:")
        for warning in result.warnings:
            print(f"     - {warning}")
    
    # Get filter stats
    stats = filter_system.get_filter_stats()
    print(f"\n📈 Filter Stats:")
    print(f"   Daily Signals: {stats['daily_signals']}")
    print(f"   Accepted: {stats['daily_accepted']}")
    print(f"   Rejected: {stats['daily_rejected']}")
    
    print("\n" + "=" * 60)
    print("Module ready for integration")
    print("=" * 60)
