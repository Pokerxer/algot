"""
ICT AI Learning Engine - Pattern Recognition and Adaptive Learning
===================================================================

Machine learning component that learns from historical trade data,
recognizes ICT patterns, and adapts trading strategies over time.

LEARNING ARCHITECTURE:
=====================

┌─────────────────────────────────────────────────────────────────────┐
│                    AI LEARNING ENGINE                                │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                  DATA COLLECTION                              │    │
│  │                                                                │    │
│  │  • Trade outcomes (win/loss/breakeven)                        │    │
│  │  • Setup characteristics (confluence, time, model)            │    │
│  │  • Market conditions at entry/exit                            │    │
│  │  • Pattern features (OB, FVG, liquidity levels)               │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                        │
│                              ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                FEATURE EXTRACTION                             │    │
│  │                                                                │    │
│  │  ┌───────────────────────────────────────────────────────┐   │    │
│  │  │ Pattern Features:                                      │   │    │
│  │  │   • Structure shift type (CHoCH, BOS)                  │   │    │
│  │  │   • OB characteristics (type, size, age)               │   │    │
│  │  │   • FVG characteristics (size, filled %)               │   │    │
│  │  │   • Liquidity sweep distance/velocity                  │   │    │
│  │  │   • Premium/Discount zone (depth)                      │   │    │
│  │  │   • OTE zone alignment                                 │   │    │
│  │  └───────────────────────────────────────────────────────┘   │    │
│  │  ┌───────────────────────────────────────────────────────┐   │    │
│  │  │ Time Features:                                         │   │    │
│  │  │   • Session (Asian, London, NY)                        │   │    │
│  │  │   • Kill zone presence                                 │   │    │
│  │  │   • Macro time alignment                               │   │    │
│  │  │   • Day of week                                        │   │    │
│  │  │   • Time since session open                            │   │    │
│  │  └───────────────────────────────────────────────────────┘   │    │
│  │  ┌───────────────────────────────────────────────────────┐   │    │
│  │  │ Market Features:                                       │   │    │
│  │  │   • Volatility state                                   │   │    │
│  │  │   • Trend strength                                     │   │    │
│  │  │   • MTF alignment score                                │   │    │
│  │  │   • Range expansion/contraction                        │   │    │
│  │  └───────────────────────────────────────────────────────┘   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                        │
│                              ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                   LEARNING MODELS                             │    │
│  │                                                                │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │    │
│  │  │ Pattern     │  │ Outcome     │  │ Parameter   │          │    │
│  │  │ Classifier  │  │ Predictor   │  │ Optimizer   │          │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘          │    │
│  │                                                                │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │    │
│  │  │ Regime      │  │ Risk        │  │ Timing      │          │    │
│  │  │ Detector    │  │ Assessor    │  │ Optimizer   │          │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘          │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                        │
│                              ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                  OUTPUT / ADAPTATION                          │    │
│  │                                                                │    │
│  │  • Win probability estimates                                  │    │
│  │  • Optimal parameter adjustments                              │    │
│  │  • Pattern quality scores                                     │    │
│  │  • Risk adjustment recommendations                            │    │
│  │  • Strategy performance metrics                               │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘

Learning Principles:
- Learn from outcomes, not predictions
- Weight recent trades more heavily
- Adapt to changing market conditions
- Track performance by pattern type
- Optimize parameters based on results
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from datetime import datetime, timedelta
import logging
import json
import math
from collections import defaultdict
import random

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class TradeOutcome(Enum):
    """Trade outcome classification"""
    WIN = "win"                    # Hit TP
    LOSS = "loss"                  # Hit SL
    BREAKEVEN = "breakeven"        # BE or small win/loss
    PARTIAL = "partial"            # Partial TP hit
    SCRATCH = "scratch"            # Closed early, minimal P/L


class PatternType(Enum):
    """ICT pattern types"""
    MODEL_2022 = "model_2022"
    SILVER_BULLET = "silver_bullet"
    VENOM = "venom"
    TURTLE_SOUP = "turtle_soup"
    POWER_OF_THREE = "power_of_three"
    UNICORN = "unicorn"
    OB_ENTRY = "ob_entry"
    FVG_ENTRY = "fvg_entry"
    BREAKER_ENTRY = "breaker_entry"
    LIQUIDITY_SWEEP = "liquidity_sweep"


class LearningMode(Enum):
    """Learning engine modes"""
    TRAINING = "training"          # Active learning
    INFERENCE = "inference"        # Using learned parameters
    ADAPTATION = "adaptation"      # Continuous learning


class RiskLevel(Enum):
    """Risk level classification"""
    CONSERVATIVE = "conservative"  # Lower position size
    NORMAL = "normal"              # Standard position size
    AGGRESSIVE = "aggressive"      # Higher position size


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TradeFeatures:
    """Features extracted from a trade setup"""
    # Pattern features
    pattern_type: PatternType
    has_structure_shift: bool = False
    structure_shift_type: str = ""
    has_order_block: bool = False
    ob_size: float = 0.0           # As % of range
    ob_age: int = 0                # Candles since formation
    has_fvg: bool = False
    fvg_size: float = 0.0
    fvg_filled_pct: float = 0.0
    has_breaker: bool = False
    
    # Zone features
    in_premium: bool = False
    in_discount: bool = False
    zone_depth: float = 0.0        # How deep in zone (0-1)
    in_ote: bool = False
    ote_level: float = 0.0         # 0.62-0.79
    
    # Liquidity features
    liquidity_swept: bool = False
    sweep_distance: float = 0.0    # In points
    sweep_velocity: float = 0.0    # Points per candle
    
    # Time features
    session: str = ""
    kill_zone: str = ""
    is_macro_time: bool = False
    day_of_week: int = 0
    minutes_since_open: int = 0
    
    # Market features
    volatility_state: str = ""
    trend_strength: float = 0.0
    mtf_alignment: float = 0.0
    range_expansion: bool = False
    
    # Confluence
    confluence_score: float = 0.0
    confluence_factors: int = 0
    
    def to_vector(self) -> List[float]:
        """Convert to feature vector for ML"""
        return [
            float(self.has_structure_shift),
            float(self.has_order_block),
            self.ob_size,
            min(self.ob_age / 100, 1.0),  # Normalize
            float(self.has_fvg),
            self.fvg_size,
            self.fvg_filled_pct,
            float(self.has_breaker),
            float(self.in_premium),
            float(self.in_discount),
            self.zone_depth,
            float(self.in_ote),
            self.ote_level,
            float(self.liquidity_swept),
            min(self.sweep_distance / 50, 1.0),  # Normalize
            min(self.sweep_velocity / 10, 1.0),  # Normalize
            float(self.is_macro_time),
            self.day_of_week / 4,  # Mon=0, Fri=4
            min(self.minutes_since_open / 300, 1.0),  # Normalize
            self.trend_strength,
            self.mtf_alignment / 100,
            float(self.range_expansion),
            self.confluence_score / 100,
            min(self.confluence_factors / 10, 1.0),
        ]


@dataclass
class TradeRecord:
    """Complete record of a trade"""
    # Identification
    trade_id: str
    timestamp: datetime
    symbol: str
    
    # Setup
    direction: str                 # 'long' or 'short'
    features: TradeFeatures
    entry_price: float
    stop_loss: float
    take_profit: float
    
    # Execution
    actual_entry: Optional[float] = None
    actual_exit: Optional[float] = None
    exit_time: Optional[datetime] = None
    
    # Outcome
    outcome: Optional[TradeOutcome] = None
    pnl_points: float = 0.0
    pnl_dollars: float = 0.0
    risk_reward_actual: float = 0.0
    
    # Analysis
    max_adverse_excursion: float = 0.0  # MAE in points
    max_favorable_excursion: float = 0.0  # MFE in points
    time_in_trade: int = 0              # Minutes
    
    # Learning
    prediction_correct: bool = False
    predicted_probability: float = 0.0
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for storage"""
        return {
            'trade_id': self.trade_id,
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'direction': self.direction,
            'pattern_type': self.features.pattern_type.value,
            'entry_price': self.entry_price,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit,
            'outcome': self.outcome.value if self.outcome else None,
            'pnl_points': self.pnl_points,
            'risk_reward_actual': self.risk_reward_actual,
            'confluence_score': self.features.confluence_score,
            'features': self.features.to_vector(),
        }


@dataclass
class PatternStats:
    """Statistics for a pattern type"""
    pattern_type: PatternType
    total_trades: int = 0
    wins: int = 0
    losses: int = 0
    breakevens: int = 0
    
    win_rate: float = 0.0
    avg_winner: float = 0.0        # In R
    avg_loser: float = 0.0         # In R
    profit_factor: float = 0.0
    expectancy: float = 0.0        # Per trade in R
    
    # Performance by conditions
    win_rate_by_session: Dict[str, float] = field(default_factory=dict)
    win_rate_by_day: Dict[int, float] = field(default_factory=dict)
    win_rate_by_volatility: Dict[str, float] = field(default_factory=dict)
    
    def update(self, record: TradeRecord):
        """Update stats with new trade record"""
        self.total_trades += 1
        
        if record.outcome == TradeOutcome.WIN:
            self.wins += 1
        elif record.outcome == TradeOutcome.LOSS:
            self.losses += 1
        else:
            self.breakevens += 1
        
        if self.total_trades > 0:
            self.win_rate = self.wins / self.total_trades * 100
        
        # Update session stats
        session = record.features.session
        if session:
            wins_in_session = sum(1 for r in [record] if r.outcome == TradeOutcome.WIN)
            if session not in self.win_rate_by_session:
                self.win_rate_by_session[session] = 0
            # Running average
            self.win_rate_by_session[session] = (
                self.win_rate_by_session[session] * 0.9 + 
                (100 if record.outcome == TradeOutcome.WIN else 0) * 0.1
            )


@dataclass
class LearningState:
    """Current state of the learning engine"""
    mode: LearningMode
    total_trades_analyzed: int = 0
    last_learning_update: Optional[datetime] = None
    
    # Performance metrics
    overall_win_rate: float = 0.0
    overall_profit_factor: float = 0.0
    overall_expectancy: float = 0.0
    
    # Model confidence
    pattern_classifier_accuracy: float = 0.0
    outcome_predictor_accuracy: float = 0.0
    
    # Adaptation state
    parameter_adjustments: Dict[str, float] = field(default_factory=dict)


# =============================================================================
# SIMPLE ML MODELS
# =============================================================================

class SimpleClassifier:
    """Simple pattern classifier using weighted features"""
    
    def __init__(self, n_features: int):
        """Initialize with random weights"""
        self.weights = [random.uniform(-0.5, 0.5) for _ in range(n_features)]
        self.bias = 0.0
        self.learning_rate = 0.01
        
    def predict(self, features: List[float]) -> float:
        """Predict probability (0-1)"""
        if len(features) != len(self.weights):
            return 0.5
        
        score = self.bias + sum(w * f for w, f in zip(self.weights, features))
        # Sigmoid activation
        return 1 / (1 + math.exp(-max(-10, min(10, score))))
    
    def train(self, features: List[float], target: float):
        """Train on single example"""
        prediction = self.predict(features)
        error = target - prediction
        
        # Update weights
        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * error * features[i]
        self.bias += self.learning_rate * error


class RunningStats:
    """Running statistics calculator"""
    
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.min_val = float('inf')
        self.max_val = float('-inf')
    
    def update(self, value: float):
        """Update with new value"""
        self.n += 1
        delta = value - self.mean
        self.mean += delta / self.n
        delta2 = value - self.mean
        self.M2 += delta * delta2
        self.min_val = min(self.min_val, value)
        self.max_val = max(self.max_val, value)
    
    @property
    def variance(self) -> float:
        if self.n < 2:
            return 0.0
        return self.M2 / (self.n - 1)
    
    @property
    def std(self) -> float:
        return math.sqrt(self.variance)


# =============================================================================
# MAIN LEARNING ENGINE CLASS
# =============================================================================

class AILearningEngine:
    """
    AI Learning Engine for ICT Trading
    
    Learns from historical trade data, recognizes patterns,
    and adapts trading strategies over time.
    
    Usage:
        engine = AILearningEngine()
        
        # Record trade outcome
        engine.record_trade(trade_record)
        
        # Get prediction for new setup
        probability = engine.predict_outcome(features)
        
        # Get optimized parameters
        params = engine.get_optimized_parameters()
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the AI Learning Engine
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or self._default_config()
        
        # State
        self.state = LearningState(mode=LearningMode.TRAINING)
        
        # Trade records storage
        self._trade_records: List[TradeRecord] = []
        self._max_records = 10000
        
        # Pattern statistics
        self._pattern_stats: Dict[PatternType, PatternStats] = {
            pt: PatternStats(pattern_type=pt) for pt in PatternType
        }
        
        # ML Models
        self._outcome_predictor = SimpleClassifier(n_features=24)
        self._pattern_classifier = SimpleClassifier(n_features=24)
        
        # Running statistics
        self._pnl_stats = RunningStats()
        self._win_rate_stats = RunningStats()
        
        # Parameter optimization
        self._optimized_params: Dict[str, float] = {}
        self._param_history: List[Dict] = []
        
        # Session performance tracking
        self._session_performance: Dict[str, Dict] = defaultdict(lambda: {
            'trades': 0, 'wins': 0, 'pnl': 0.0
        })
        
        # Day of week performance
        self._dow_performance: Dict[int, Dict] = defaultdict(lambda: {
            'trades': 0, 'wins': 0, 'pnl': 0.0
        })
        
        logger.info("AI Learning Engine initialized")
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'learning_rate': 0.01,
            'min_trades_for_prediction': 30,
            'recency_weight': 0.1,       # Weight for recent trades
            'adaptation_threshold': 0.05, # Change threshold for adaptation
            'confidence_threshold': 0.65, # Minimum prediction confidence
        }
    
    # =========================================================================
    # TRADE RECORDING
    # =========================================================================
    
    def record_trade(self, record: TradeRecord):
        """
        Record a completed trade for learning
        
        Args:
            record: Complete TradeRecord with outcome
        """
        if record.outcome is None:
            logger.warning("Trade record has no outcome, skipping")
            return
        
        # Store record
        self._trade_records.append(record)
        if len(self._trade_records) > self._max_records:
            self._trade_records.pop(0)
        
        # Update pattern stats
        pattern_type = record.features.pattern_type
        self._pattern_stats[pattern_type].update(record)
        
        # Update running stats
        self._pnl_stats.update(record.pnl_points)
        
        # Train ML models
        self._train_models(record)
        
        # Update session performance
        session = record.features.session
        if session:
            self._session_performance[session]['trades'] += 1
            if record.outcome == TradeOutcome.WIN:
                self._session_performance[session]['wins'] += 1
            self._session_performance[session]['pnl'] += record.pnl_points
        
        # Update day of week performance
        dow = record.features.day_of_week
        self._dow_performance[dow]['trades'] += 1
        if record.outcome == TradeOutcome.WIN:
            self._dow_performance[dow]['wins'] += 1
        self._dow_performance[dow]['pnl'] += record.pnl_points
        
        # Update state
        self.state.total_trades_analyzed += 1
        self.state.last_learning_update = datetime.now()
        
        # Recalculate overall metrics
        self._update_overall_metrics()
        
        logger.info(f"Recorded trade: {record.trade_id} ({record.outcome.value})")
    
    def _train_models(self, record: TradeRecord):
        """Train ML models on new trade"""
        features = record.features.to_vector()
        
        # Target: 1 for win, 0 for loss
        target = 1.0 if record.outcome == TradeOutcome.WIN else 0.0
        
        # Train outcome predictor
        self._outcome_predictor.train(features, target)
        
        # Track prediction accuracy
        prediction = self._outcome_predictor.predict(features)
        predicted_win = prediction > 0.5
        actual_win = target > 0.5
        record.prediction_correct = predicted_win == actual_win
        record.predicted_probability = prediction
    
    def _update_overall_metrics(self):
        """Update overall performance metrics"""
        if not self._trade_records:
            return
        
        # Calculate win rate
        wins = sum(1 for r in self._trade_records if r.outcome == TradeOutcome.WIN)
        self.state.overall_win_rate = wins / len(self._trade_records) * 100
        
        # Calculate profit factor
        gross_profit = sum(r.pnl_points for r in self._trade_records if r.pnl_points > 0)
        gross_loss = abs(sum(r.pnl_points for r in self._trade_records if r.pnl_points < 0))
        self.state.overall_profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Calculate expectancy
        total_pnl = sum(r.pnl_points for r in self._trade_records)
        self.state.overall_expectancy = total_pnl / len(self._trade_records)
        
        # Calculate predictor accuracy
        correct = sum(1 for r in self._trade_records if r.prediction_correct)
        self.state.outcome_predictor_accuracy = correct / len(self._trade_records) * 100
    
    # =========================================================================
    # PREDICTION
    # =========================================================================
    
    def predict_outcome(self, features: TradeFeatures) -> Dict[str, Any]:
        """
        Predict trade outcome probability
        
        Args:
            features: TradeFeatures for the setup
            
        Returns:
            Dictionary with prediction and analysis
        """
        feature_vector = features.to_vector()
        
        # Get prediction from model
        win_probability = self._outcome_predictor.predict(feature_vector)
        
        # Get pattern historical stats
        pattern_stats = self._pattern_stats.get(features.pattern_type)
        historical_win_rate = pattern_stats.win_rate if pattern_stats else 50.0
        
        # Blend model prediction with historical data
        if self.state.total_trades_analyzed < self.config['min_trades_for_prediction']:
            # Use historical average more when data is limited
            blended_probability = historical_win_rate / 100
        else:
            # Weight model prediction more as we get more data
            model_weight = min(0.7, self.state.total_trades_analyzed / 200)
            blended_probability = (
                win_probability * model_weight +
                (historical_win_rate / 100) * (1 - model_weight)
            )
        
        # Confidence level
        confidence = 'high' if blended_probability > 0.7 or blended_probability < 0.3 else 'medium'
        
        # Risk recommendation
        if blended_probability >= 0.65:
            risk_level = RiskLevel.AGGRESSIVE
        elif blended_probability >= 0.55:
            risk_level = RiskLevel.NORMAL
        else:
            risk_level = RiskLevel.CONSERVATIVE
        
        return {
            'win_probability': blended_probability,
            'model_prediction': win_probability,
            'historical_win_rate': historical_win_rate,
            'confidence': confidence,
            'risk_recommendation': risk_level,
            'pattern_type': features.pattern_type.value,
            'sample_size': pattern_stats.total_trades if pattern_stats else 0,
        }
    
    def get_pattern_quality_score(self, features: TradeFeatures) -> float:
        """
        Get quality score for a pattern
        
        Args:
            features: TradeFeatures for the setup
            
        Returns:
            Quality score 0-100
        """
        prediction = self.predict_outcome(features)
        
        # Base score from win probability
        base_score = prediction['win_probability'] * 100
        
        # Adjust for confluence
        confluence_bonus = min(10, features.confluence_factors * 2)
        
        # Adjust for time (kill zone bonus)
        time_bonus = 5 if features.is_macro_time else 0
        time_bonus += 10 if features.kill_zone else 0
        
        # Adjust for MTF alignment
        alignment_bonus = features.mtf_alignment * 0.1
        
        # Penalty for low sample size
        sample_penalty = 0
        if prediction['sample_size'] < 10:
            sample_penalty = 10
        elif prediction['sample_size'] < 30:
            sample_penalty = 5
        
        total_score = base_score + confluence_bonus + time_bonus + alignment_bonus - sample_penalty
        
        return min(100, max(0, total_score))
    
    # =========================================================================
    # PARAMETER OPTIMIZATION
    # =========================================================================
    
    def get_optimized_parameters(self) -> Dict[str, Any]:
        """
        Get optimized trading parameters based on learning
        
        Returns:
            Dictionary of optimized parameters
        """
        if self.state.total_trades_analyzed < 20:
            return self._default_parameters()
        
        params = {}
        
        # Optimize confluence threshold based on performance
        high_conf_trades = [r for r in self._trade_records 
                          if r.features.confluence_score >= 70]
        if len(high_conf_trades) >= 10:
            high_conf_wr = sum(1 for r in high_conf_trades 
                              if r.outcome == TradeOutcome.WIN) / len(high_conf_trades)
            if high_conf_wr > 0.6:
                params['min_confluence_score'] = 70
            else:
                params['min_confluence_score'] = 60
        
        # Optimize by session
        best_session = None
        best_wr = 0
        for session, perf in self._session_performance.items():
            if perf['trades'] >= 10:
                wr = perf['wins'] / perf['trades']
                if wr > best_wr:
                    best_wr = wr
                    best_session = session
        params['best_session'] = best_session
        params['best_session_win_rate'] = best_wr * 100
        
        # Optimize by day of week
        best_day = None
        best_day_wr = 0
        for dow, perf in self._dow_performance.items():
            if perf['trades'] >= 5:
                wr = perf['wins'] / perf['trades']
                if wr > best_day_wr:
                    best_day_wr = wr
                    best_day = dow
        params['best_day'] = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday'][best_day] if best_day is not None else None
        
        # Optimal R:R based on win rate
        if self.state.overall_win_rate >= 60:
            params['recommended_rr'] = 2.0
        elif self.state.overall_win_rate >= 50:
            params['recommended_rr'] = 2.5
        else:
            params['recommended_rr'] = 3.0
        
        # Pattern recommendations
        pattern_rankings = []
        for pt, stats in self._pattern_stats.items():
            if stats.total_trades >= 5:
                pattern_rankings.append({
                    'pattern': pt.value,
                    'win_rate': stats.win_rate,
                    'trades': stats.total_trades,
                })
        pattern_rankings.sort(key=lambda x: x['win_rate'], reverse=True)
        params['pattern_rankings'] = pattern_rankings[:5]
        
        self._optimized_params = params
        return params
    
    def _default_parameters(self) -> Dict[str, Any]:
        """Default parameters when insufficient data"""
        return {
            'min_confluence_score': 60,
            'best_session': 'ny_am',
            'best_day': 'Tuesday',
            'recommended_rr': 2.5,
            'pattern_rankings': [],
            'note': 'Using defaults - insufficient trade data',
        }
    
    # =========================================================================
    # RISK ASSESSMENT
    # =========================================================================
    
    def assess_risk(self, features: TradeFeatures) -> Dict[str, Any]:
        """
        Assess risk for a trade setup
        
        Args:
            features: TradeFeatures for the setup
            
        Returns:
            Risk assessment dictionary
        """
        prediction = self.predict_outcome(features)
        
        # Base risk from prediction
        base_risk = 1.0 - prediction['win_probability']
        
        # Adjust for recent performance
        recent_trades = self._trade_records[-20:] if len(self._trade_records) >= 20 else self._trade_records
        if recent_trades:
            recent_losses = sum(1 for r in recent_trades if r.outcome == TradeOutcome.LOSS)
            losing_streak = 0
            for r in reversed(recent_trades):
                if r.outcome == TradeOutcome.LOSS:
                    losing_streak += 1
                else:
                    break
            
            # Increase risk during drawdown
            if losing_streak >= 3:
                base_risk *= 1.2
            if recent_losses / len(recent_trades) > 0.5:
                base_risk *= 1.1
        
        # Position size recommendation
        if base_risk < 0.3:
            position_multiplier = 1.2
        elif base_risk < 0.5:
            position_multiplier = 1.0
        else:
            position_multiplier = 0.7
        
        return {
            'risk_score': base_risk,
            'risk_level': 'low' if base_risk < 0.3 else 'medium' if base_risk < 0.6 else 'high',
            'position_multiplier': position_multiplier,
            'recent_losing_streak': losing_streak if 'losing_streak' in dir() else 0,
            'should_trade': base_risk < 0.6,
            'notes': self._generate_risk_notes(base_risk, features),
        }
    
    def _generate_risk_notes(self, risk: float, features: TradeFeatures) -> List[str]:
        """Generate risk notes"""
        notes = []
        
        if risk > 0.5:
            notes.append("Higher than average risk - consider smaller position")
        
        if not features.has_structure_shift:
            notes.append("Missing structure shift - key confirmation missing")
        
        if not features.liquidity_swept:
            notes.append("No liquidity sweep - may see stop hunt before move")
        
        if not features.is_macro_time and not features.kill_zone:
            notes.append("Outside key time windows")
        
        if features.confluence_factors < 4:
            notes.append(f"Low confluence ({features.confluence_factors} factors)")
        
        return notes
    
    # =========================================================================
    # ANALYTICS
    # =========================================================================
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        if not self._trade_records:
            return {'error': 'No trades recorded'}
        
        records = self._trade_records
        
        # Basic metrics
        total = len(records)
        wins = sum(1 for r in records if r.outcome == TradeOutcome.WIN)
        losses = sum(1 for r in records if r.outcome == TradeOutcome.LOSS)
        
        # P&L metrics
        total_pnl = sum(r.pnl_points for r in records)
        winners_pnl = [r.pnl_points for r in records if r.pnl_points > 0]
        losers_pnl = [r.pnl_points for r in records if r.pnl_points < 0]
        
        avg_win = sum(winners_pnl) / len(winners_pnl) if winners_pnl else 0
        avg_loss = sum(losers_pnl) / len(losers_pnl) if losers_pnl else 0
        
        # Win rate
        win_rate = wins / total * 100 if total > 0 else 0
        
        # Profit factor
        gross_profit = sum(winners_pnl)
        gross_loss = abs(sum(losers_pnl))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
        
        # Expectancy
        expectancy = total_pnl / total if total > 0 else 0
        
        # MAE/MFE analysis
        avg_mae = sum(r.max_adverse_excursion for r in records) / total
        avg_mfe = sum(r.max_favorable_excursion for r in records) / total
        
        # Streaks
        max_win_streak = 0
        max_loss_streak = 0
        current_win_streak = 0
        current_loss_streak = 0
        
        for r in records:
            if r.outcome == TradeOutcome.WIN:
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            elif r.outcome == TradeOutcome.LOSS:
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)
        
        return {
            'total_trades': total,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'expectancy_points': expectancy,
            'total_pnl_points': total_pnl,
            'avg_winner': avg_win,
            'avg_loser': avg_loss,
            'largest_winner': max(winners_pnl) if winners_pnl else 0,
            'largest_loser': min(losers_pnl) if losers_pnl else 0,
            'avg_mae': avg_mae,
            'avg_mfe': avg_mfe,
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak,
            'predictor_accuracy': self.state.outcome_predictor_accuracy,
        }
    
    def get_pattern_performance(self) -> Dict[str, Dict]:
        """Get performance breakdown by pattern type"""
        result = {}
        
        for pattern_type, stats in self._pattern_stats.items():
            if stats.total_trades > 0:
                result[pattern_type.value] = {
                    'total_trades': stats.total_trades,
                    'wins': stats.wins,
                    'losses': stats.losses,
                    'win_rate': stats.win_rate,
                    'session_breakdown': stats.win_rate_by_session,
                }
        
        return result
    
    def get_session_performance(self) -> Dict[str, Dict]:
        """Get performance breakdown by session"""
        result = {}
        
        for session, perf in self._session_performance.items():
            if perf['trades'] > 0:
                result[session] = {
                    'trades': perf['trades'],
                    'wins': perf['wins'],
                    'win_rate': perf['wins'] / perf['trades'] * 100,
                    'total_pnl': perf['pnl'],
                }
        
        return result
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def save_state(self, filepath: str):
        """Save learning state to file"""
        data = {
            'state': {
                'mode': self.state.mode.value,
                'total_trades_analyzed': self.state.total_trades_analyzed,
                'overall_win_rate': self.state.overall_win_rate,
                'overall_profit_factor': self.state.overall_profit_factor,
            },
            'trade_records': [r.to_dict() for r in self._trade_records[-1000:]],
            'optimized_params': self._optimized_params,
            'model_weights': self._outcome_predictor.weights,
            'model_bias': self._outcome_predictor.bias,
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"State saved to {filepath}")
    
    def load_state(self, filepath: str):
        """Load learning state from file"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        # Restore state
        self.state.mode = LearningMode(data['state']['mode'])
        self.state.total_trades_analyzed = data['state']['total_trades_analyzed']
        self.state.overall_win_rate = data['state']['overall_win_rate']
        self.state.overall_profit_factor = data['state']['overall_profit_factor']
        
        # Restore model
        if 'model_weights' in data:
            self._outcome_predictor.weights = data['model_weights']
            self._outcome_predictor.bias = data['model_bias']
        
        self._optimized_params = data.get('optimized_params', {})
        
        logger.info(f"State loaded from {filepath}")
    
    # =========================================================================
    # REPORTING
    # =========================================================================
    
    def generate_report(self) -> str:
        """Generate comprehensive learning report"""
        lines = []
        lines.append("=" * 60)
        lines.append("AI LEARNING ENGINE REPORT")
        lines.append("=" * 60)
        lines.append(f"Mode: {self.state.mode.value}")
        lines.append(f"Trades Analyzed: {self.state.total_trades_analyzed}")
        lines.append(f"Last Update: {self.state.last_learning_update}")
        lines.append("")
        
        # Performance metrics
        metrics = self.get_performance_metrics()
        if 'error' not in metrics:
            lines.append("PERFORMANCE METRICS:")
            lines.append(f"  Win Rate: {metrics['win_rate']:.1f}%")
            lines.append(f"  Profit Factor: {metrics['profit_factor']:.2f}")
            lines.append(f"  Expectancy: {metrics['expectancy_points']:.2f} pts/trade")
            lines.append(f"  Total P&L: {metrics['total_pnl_points']:.2f} pts")
            lines.append(f"  Predictor Accuracy: {metrics['predictor_accuracy']:.1f}%")
        lines.append("")
        
        # Pattern performance
        pattern_perf = self.get_pattern_performance()
        if pattern_perf:
            lines.append("PATTERN PERFORMANCE:")
            for pattern, perf in sorted(pattern_perf.items(), 
                                        key=lambda x: x[1]['win_rate'], reverse=True):
                lines.append(f"  {pattern}: {perf['win_rate']:.1f}% ({perf['total_trades']} trades)")
        lines.append("")
        
        # Session performance
        session_perf = self.get_session_performance()
        if session_perf:
            lines.append("SESSION PERFORMANCE:")
            for session, perf in sorted(session_perf.items(),
                                        key=lambda x: x[1]['win_rate'], reverse=True):
                lines.append(f"  {session}: {perf['win_rate']:.1f}% ({perf['trades']} trades)")
        lines.append("")
        
        # Optimized parameters
        params = self.get_optimized_parameters()
        lines.append("OPTIMIZED PARAMETERS:")
        for key, value in params.items():
            if key != 'pattern_rankings':
                lines.append(f"  {key}: {value}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("ICT AI Learning Engine")
    print("=" * 60)
    
    # Create engine
    engine = AILearningEngine()
    
    # Simulate some trades
    import uuid
    
    patterns = list(PatternType)
    sessions = ['asian', 'london', 'ny_am', 'ny_pm']
    outcomes = [TradeOutcome.WIN, TradeOutcome.WIN, TradeOutcome.WIN, 
                TradeOutcome.LOSS, TradeOutcome.LOSS, TradeOutcome.BREAKEVEN]
    
    print("\nSimulating 50 trades...")
    for i in range(50):
        features = TradeFeatures(
            pattern_type=random.choice(patterns),
            has_structure_shift=random.random() > 0.3,
            has_order_block=random.random() > 0.4,
            has_fvg=random.random() > 0.4,
            liquidity_swept=random.random() > 0.5,
            in_discount=random.random() > 0.5,
            is_macro_time=random.random() > 0.7,
            session=random.choice(sessions),
            kill_zone='ny_am' if random.random() > 0.5 else None,
            day_of_week=random.randint(0, 4),
            confluence_score=random.uniform(50, 90),
            confluence_factors=random.randint(3, 8),
            mtf_alignment=random.uniform(60, 100),
        )
        
        outcome = random.choice(outcomes)
        pnl = random.uniform(10, 50) if outcome == TradeOutcome.WIN else random.uniform(-30, -10)
        
        record = TradeRecord(
            trade_id=str(uuid.uuid4())[:8],
            timestamp=datetime.now() - timedelta(days=50-i),
            symbol='NQ',
            direction='long' if random.random() > 0.5 else 'short',
            features=features,
            entry_price=21500,
            stop_loss=21450,
            take_profit=21600,
            outcome=outcome,
            pnl_points=pnl,
        )
        
        engine.record_trade(record)
    
    # Get performance metrics
    print("\n" + engine.generate_report())
    
    # Test prediction
    test_features = TradeFeatures(
        pattern_type=PatternType.MODEL_2022,
        has_structure_shift=True,
        has_order_block=True,
        has_fvg=True,
        liquidity_swept=True,
        in_discount=True,
        is_macro_time=True,
        session='ny_am',
        kill_zone='ny_am',
        confluence_score=85,
        confluence_factors=7,
        mtf_alignment=90,
    )
    
    print("\n" + "=" * 60)
    print("PREDICTION FOR NEW SETUP:")
    prediction = engine.predict_outcome(test_features)
    print(f"  Win Probability: {prediction['win_probability']*100:.1f}%")
    print(f"  Confidence: {prediction['confidence']}")
    print(f"  Risk Level: {prediction['risk_recommendation'].value}")
    print(f"  Quality Score: {engine.get_pattern_quality_score(test_features):.0f}/100")
    
    print("\n" + "=" * 60)
    print("RISK ASSESSMENT:")
    risk = engine.assess_risk(test_features)
    print(f"  Risk Score: {risk['risk_score']:.2f}")
    print(f"  Risk Level: {risk['risk_level']}")
    print(f"  Should Trade: {risk['should_trade']}")
    print(f"  Position Multiplier: {risk['position_multiplier']}")
    
    print("\nAI Learning Engine ready!")
