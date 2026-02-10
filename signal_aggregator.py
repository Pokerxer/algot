"""
ICT Signal Aggregator - Confluence and Signal Ranking
======================================================

Aggregates signals from all ICT handlers, calculates confluence scores,
and ranks trade setups by quality using ICT-specific criteria.

SIGNAL AGGREGATION FLOW:
========================

┌─────────────────────────────────────────────────────────────────────┐
│                    SIGNAL AGGREGATOR                                 │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐│
│  │ Market      │  │ Order       │  │ FVG/Gap     │  │ Liquidity   ││
│  │ Structure   │  │ Block       │  │ Handler     │  │ Handler     ││
│  │ Signals     │  │ Signals     │  │ Signals     │  │ Signals     ││
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘│
│         │                │                │                │        │
│         └────────────────┴────────────────┴────────────────┘        │
│                                   │                                  │
│                                   ▼                                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  SIGNAL COLLECTION                            │   │
│  │            (Collect all signals by type)                      │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                   │                                  │
│                                   ▼                                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                 CONFLUENCE ANALYSIS                           │   │
│  │                                                                │   │
│  │  ┌───────────────────────────────────────────────────────┐   │   │
│  │  │ Factor Weights:                                        │   │   │
│  │  │   • Structure Shift (CHoCH/BOS):     15 pts            │   │   │
│  │  │   • Liquidity Sweep:                 12 pts            │   │   │
│  │  │   • Order Block Presence:            12 pts            │   │   │
│  │  │   • FVG Presence:                    10 pts            │   │   │
│  │  │   • Premium/Discount Zone:           10 pts            │   │   │
│  │  │   • Model Validation:                10 pts            │   │   │
│  │  │   • Breaker Block:                    8 pts            │   │   │
│  │  │   • Kill Zone Timing:                 8 pts            │   │   │
│  │  │   • Displacement:                     8 pts            │   │   │
│  │  │   • OTE Zone (62-79%):                7 pts            │   │   │
│  │  │   • Macro Time Window:                5 pts            │   │   │
│  │  │                                   ─────────            │   │   │
│  │  │   Maximum Score:                    105 pts            │   │   │
│  │  │   (Normalized to 0-100)                                │   │   │
│  │  └───────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                   │                                  │
│                                   ▼                                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                   SIGNAL RANKING                              │   │
│  │                                                                │   │
│  │  Ranking Criteria:                                            │   │
│  │   1. Confluence Score (40%)                                   │   │
│  │   2. MTF Alignment (20%)                                      │   │
│  │   3. Risk/Reward Ratio (15%)                                  │   │
│  │   4. Model Validation (15%)                                   │   │
│  │   5. Time Context (10%)                                       │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                   │                                  │
│                                   ▼                                  │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                  RANKED SIGNALS                               │   │
│  │              (Sorted by quality score)                        │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘

ICT Confluence Principles:
- Multiple confluences increase probability
- Structure shift is foundational
- Liquidity sweep confirms direction
- PD arrays provide entry zones
- Time alignment is critical
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
from datetime import datetime, time
import logging
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class SignalSource(Enum):
    """Source handler for a signal"""
    MARKET_STRUCTURE = "market_structure"
    ORDER_BLOCK = "order_block"
    FVG = "fvg"
    GAP = "gap"
    LIQUIDITY = "liquidity"
    PD_ARRAY = "pd_array"
    TRADING_MODEL = "trading_model"
    TIME_CONTEXT = "time_context"
    ENTRY_STOP = "entry_stop"


class SignalStrength(Enum):
    """Signal strength level"""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"


class ConfluenceLevel(Enum):
    """Confluence classification"""
    EXCEPTIONAL = "exceptional"    # 90-100
    HIGH = "high"                  # 75-89
    MODERATE = "moderate"          # 60-74
    LOW = "low"                    # 45-59
    INSUFFICIENT = "insufficient"  # <45


class RankingTier(Enum):
    """Signal ranking tier"""
    TIER_1 = "tier_1"  # Top priority - immediate execution
    TIER_2 = "tier_2"  # High quality - execute if conditions met
    TIER_3 = "tier_3"  # Acceptable - execute with caution
    TIER_4 = "tier_4"  # Low quality - consider passing
    TIER_5 = "tier_5"  # Avoid - do not trade


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class BaseSignal:
    """Base signal from any handler"""
    source: SignalSource
    signal_type: str              # 'bullish', 'bearish', 'neutral'
    strength: SignalStrength
    timestamp: datetime
    price_level: float
    description: str
    confidence: float = 0.0       # 0-100
    timeframe: str = ""
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'source': self.source.value,
            'signal_type': self.signal_type,
            'strength': self.strength.value,
            'timestamp': self.timestamp.isoformat(),
            'price_level': self.price_level,
            'description': self.description,
            'confidence': self.confidence,
            'timeframe': self.timeframe,
            'metadata': self.metadata,
        }


@dataclass
class StructureSignal(BaseSignal):
    """Market structure signal"""
    structure_type: str = ""      # CHoCH, BOS, SMS
    break_level: float = 0.0
    trend_direction: str = ""
    is_confirmed: bool = False


@dataclass
class OrderBlockSignal(BaseSignal):
    """Order block signal"""
    ob_type: str = ""             # bullish, bearish
    ob_high: float = 0.0
    ob_low: float = 0.0
    ob_mid: float = 0.0
    is_mitigated: bool = False
    is_reclaimed: bool = False
    quality_score: float = 0.0


@dataclass
class FVGSignal(BaseSignal):
    """Fair Value Gap signal"""
    fvg_type: str = ""            # bullish, bearish
    fvg_high: float = 0.0
    fvg_low: float = 0.0
    fvg_ce: float = 0.0           # Consequent Encroachment
    is_filled: bool = False
    is_inverted: bool = False
    in_premium: bool = False
    in_discount: bool = False


@dataclass
class LiquiditySignal(BaseSignal):
    """Liquidity signal"""
    liq_type: str = ""            # buy_side, sell_side
    liq_level: float = 0.0
    is_swept: bool = False
    sweep_candle_idx: int = -1
    is_target: bool = False


@dataclass
class ModelSignal(BaseSignal):
    """Trading model signal"""
    model_name: str = ""          # 2022_model, silver_bullet, etc.
    model_stage: int = 0
    stage_name: str = ""
    is_complete: bool = False
    entry_price: float = 0.0
    target_price: float = 0.0


@dataclass
class TimeSignal(BaseSignal):
    """Time-based signal"""
    session: str = ""
    kill_zone: Optional[str] = None
    is_macro_time: bool = False
    macro_window: str = ""
    is_optimal: bool = False


@dataclass
class ConfluenceResult:
    """Result of confluence analysis"""
    total_score: float            # 0-100 normalized
    raw_score: float              # Actual points
    level: ConfluenceLevel
    factor_breakdown: Dict[str, float] = field(default_factory=dict)
    
    # Individual flags
    has_structure_shift: bool = False
    has_liquidity_sweep: bool = False
    has_order_block: bool = False
    has_fvg: bool = False
    has_breaker: bool = False
    in_premium_discount: bool = False
    in_kill_zone: bool = False
    is_macro_time: bool = False
    model_validated: bool = False
    has_displacement: bool = False
    in_ote_zone: bool = False
    
    # Factor count
    active_factors: int = 0
    
    def get_summary(self) -> str:
        """Get confluence summary"""
        return (f"Confluence: {self.level.value.upper()} "
                f"({self.total_score:.0f}/100, {self.active_factors} factors)")


@dataclass
class RankedSignal:
    """Ranked trade signal"""
    rank: int
    tier: RankingTier
    direction: str                # 'long', 'short'
    
    # Scores
    overall_score: float          # 0-100
    confluence: ConfluenceResult
    
    # Components
    signals: List[BaseSignal] = field(default_factory=list)
    
    # Entry/Exit
    entry_price: float = 0.0
    stop_loss: float = 0.0
    target_1: float = 0.0
    target_2: float = 0.0
    risk_reward: float = 0.0
    
    # Context
    timeframe: str = ""
    model_name: str = ""
    pd_array_type: str = ""
    narrative: str = ""
    
    # Validation
    is_valid: bool = False
    validation_notes: List[str] = field(default_factory=list)


@dataclass
class AggregatorConfig:
    """Configuration for signal aggregator"""
    # Weights for ranking
    confluence_weight: float = 0.40
    mtf_alignment_weight: float = 0.20
    risk_reward_weight: float = 0.15
    model_weight: float = 0.15
    time_weight: float = 0.10
    
    # Thresholds
    min_confluence_score: float = 60.0
    min_risk_reward: float = 2.0
    min_overall_score: float = 65.0
    
    # Factor weights (for confluence calculation)
    factor_weights: Dict[str, float] = field(default_factory=lambda: {
        'structure_shift': 15.0,
        'liquidity_sweep': 12.0,
        'order_block': 12.0,
        'fvg': 10.0,
        'premium_discount': 10.0,
        'model_validated': 10.0,
        'breaker': 8.0,
        'kill_zone': 8.0,
        'displacement': 8.0,
        'ote_zone': 7.0,
        'macro_time': 5.0,
    })
    
    # Requirements
    require_structure_shift: bool = True
    require_kill_zone: bool = True
    require_liquidity_sweep: bool = False


# =============================================================================
# MAIN AGGREGATOR CLASS
# =============================================================================

class SignalAggregator:
    """
    ICT Signal Aggregator
    
    Collects signals from all handlers, calculates confluence,
    and ranks trade setups by quality.
    
    Usage:
        aggregator = SignalAggregator()
        
        # Add signals from handlers
        aggregator.add_signal(structure_signal)
        aggregator.add_signal(ob_signal)
        aggregator.add_signal(fvg_signal)
        
        # Calculate confluence
        confluence = aggregator.calculate_confluence()
        
        # Get ranked signals
        ranked = aggregator.get_ranked_signals()
    """
    
    def __init__(self, config: Optional[AggregatorConfig] = None):
        """
        Initialize the Signal Aggregator
        
        Args:
            config: Optional configuration
        """
        self.config = config or AggregatorConfig()
        
        # Signal storage by source
        self._signals: Dict[SignalSource, List[BaseSignal]] = defaultdict(list)
        
        # Aggregated results
        self._confluence_result: Optional[ConfluenceResult] = None
        self._ranked_signals: List[RankedSignal] = []
        
        # State
        self._last_aggregation: Optional[datetime] = None
        self._signal_count = 0
        
        logger.info("Signal Aggregator initialized")
    
    # =========================================================================
    # SIGNAL COLLECTION
    # =========================================================================
    
    def add_signal(self, signal: BaseSignal):
        """
        Add a signal to the aggregator
        
        Args:
            signal: Any BaseSignal subclass
        """
        self._signals[signal.source].append(signal)
        self._signal_count += 1
        logger.debug(f"Added {signal.source.value} signal: {signal.description}")
    
    def add_signals(self, signals: List[BaseSignal]):
        """Add multiple signals"""
        for signal in signals:
            self.add_signal(signal)
    
    def clear_signals(self):
        """Clear all collected signals"""
        self._signals.clear()
        self._confluence_result = None
        self._ranked_signals.clear()
        self._signal_count = 0
        logger.info("Signals cleared")
    
    def get_signals_by_source(self, source: SignalSource) -> List[BaseSignal]:
        """Get signals from a specific source"""
        return self._signals.get(source, [])
    
    def get_all_signals(self) -> Dict[SignalSource, List[BaseSignal]]:
        """Get all collected signals"""
        return dict(self._signals)
    
    def get_signal_count(self) -> int:
        """Get total signal count"""
        return self._signal_count
    
    # =========================================================================
    # CONFLUENCE CALCULATION
    # =========================================================================
    
    def calculate_confluence(self, direction: str = 'long') -> ConfluenceResult:
        """
        Calculate confluence score from collected signals
        
        Args:
            direction: 'long' or 'short'
            
        Returns:
            ConfluenceResult with scores and breakdown
        """
        weights = self.config.factor_weights
        
        # Initialize result
        result = ConfluenceResult(
            total_score=0,
            raw_score=0,
            level=ConfluenceLevel.INSUFFICIENT,
            factor_breakdown={},
        )
        
        raw_score = 0
        
        # Check each confluence factor
        
        # 1. Structure Shift (CHoCH/BOS)
        structure_signals = self._signals.get(SignalSource.MARKET_STRUCTURE, [])
        for sig in structure_signals:
            if isinstance(sig, StructureSignal):
                if sig.structure_type in ['CHoCH', 'BOS', 'SMS']:
                    if self._direction_matches(sig.signal_type, direction):
                        result.has_structure_shift = True
                        raw_score += weights['structure_shift']
                        result.factor_breakdown['structure_shift'] = weights['structure_shift']
                        break
        
        # 2. Liquidity Sweep
        liq_signals = self._signals.get(SignalSource.LIQUIDITY, [])
        for sig in liq_signals:
            if isinstance(sig, LiquiditySignal):
                if sig.is_swept:
                    # For longs, we want sell-side swept
                    # For shorts, we want buy-side swept
                    expected_sweep = 'sell_side' if direction == 'long' else 'buy_side'
                    if sig.liq_type == expected_sweep:
                        result.has_liquidity_sweep = True
                        raw_score += weights['liquidity_sweep']
                        result.factor_breakdown['liquidity_sweep'] = weights['liquidity_sweep']
                        break
        
        # 3. Order Block
        ob_signals = self._signals.get(SignalSource.ORDER_BLOCK, [])
        for sig in ob_signals:
            if isinstance(sig, OrderBlockSignal):
                if self._direction_matches(sig.ob_type, direction):
                    if not sig.is_mitigated:
                        result.has_order_block = True
                        raw_score += weights['order_block']
                        result.factor_breakdown['order_block'] = weights['order_block']
                        break
        
        # 4. FVG
        fvg_signals = self._signals.get(SignalSource.FVG, [])
        for sig in fvg_signals:
            if isinstance(sig, FVGSignal):
                if self._direction_matches(sig.fvg_type, direction):
                    if not sig.is_filled and not sig.is_inverted:
                        result.has_fvg = True
                        raw_score += weights['fvg']
                        result.factor_breakdown['fvg'] = weights['fvg']
                        
                        # Check premium/discount
                        if (direction == 'long' and sig.in_discount) or \
                           (direction == 'short' and sig.in_premium):
                            result.in_premium_discount = True
                            raw_score += weights['premium_discount']
                            result.factor_breakdown['premium_discount'] = weights['premium_discount']
                        break
        
        # 5. Model Validation
        model_signals = self._signals.get(SignalSource.TRADING_MODEL, [])
        for sig in model_signals:
            if isinstance(sig, ModelSignal):
                if self._direction_matches(sig.signal_type, direction):
                    if sig.is_complete or sig.model_stage >= 3:
                        result.model_validated = True
                        raw_score += weights['model_validated']
                        result.factor_breakdown['model_validated'] = weights['model_validated']
                        break
        
        # 6. Kill Zone / Time Context
        time_signals = self._signals.get(SignalSource.TIME_CONTEXT, [])
        for sig in time_signals:
            if isinstance(sig, TimeSignal):
                if sig.kill_zone:
                    result.in_kill_zone = True
                    raw_score += weights['kill_zone']
                    result.factor_breakdown['kill_zone'] = weights['kill_zone']
                
                if sig.is_macro_time:
                    result.is_macro_time = True
                    raw_score += weights['macro_time']
                    result.factor_breakdown['macro_time'] = weights['macro_time']
                break
        
        # 7. Check for breaker blocks (from metadata)
        for sig in ob_signals:
            if isinstance(sig, OrderBlockSignal):
                if sig.metadata.get('is_breaker', False):
                    result.has_breaker = True
                    raw_score += weights['breaker']
                    result.factor_breakdown['breaker'] = weights['breaker']
                    break
        
        # 8. Displacement (check FVG/Structure signals)
        for sig in structure_signals + fvg_signals:
            if sig.metadata.get('has_displacement', False):
                result.has_displacement = True
                raw_score += weights['displacement']
                result.factor_breakdown['displacement'] = weights['displacement']
                break
        
        # 9. OTE Zone (62-79%)
        for sig in self._signals.get(SignalSource.PD_ARRAY, []):
            if sig.metadata.get('in_ote_zone', False):
                result.in_ote_zone = True
                raw_score += weights['ote_zone']
                result.factor_breakdown['ote_zone'] = weights['ote_zone']
                break
        
        # Calculate totals
        result.raw_score = raw_score
        max_score = sum(weights.values())
        result.total_score = (raw_score / max_score) * 100 if max_score > 0 else 0
        
        # Count active factors
        result.active_factors = sum([
            result.has_structure_shift,
            result.has_liquidity_sweep,
            result.has_order_block,
            result.has_fvg,
            result.has_breaker,
            result.in_premium_discount,
            result.in_kill_zone,
            result.is_macro_time,
            result.model_validated,
            result.has_displacement,
            result.in_ote_zone,
        ])
        
        # Determine level
        if result.total_score >= 90:
            result.level = ConfluenceLevel.EXCEPTIONAL
        elif result.total_score >= 75:
            result.level = ConfluenceLevel.HIGH
        elif result.total_score >= 60:
            result.level = ConfluenceLevel.MODERATE
        elif result.total_score >= 45:
            result.level = ConfluenceLevel.LOW
        else:
            result.level = ConfluenceLevel.INSUFFICIENT
        
        self._confluence_result = result
        logger.info(f"Confluence calculated: {result.get_summary()}")
        
        return result
    
    def _direction_matches(self, signal_type: str, direction: str) -> bool:
        """Check if signal type matches direction"""
        bullish_types = ['bullish', 'long', 'buy']
        bearish_types = ['bearish', 'short', 'sell']
        
        if direction == 'long':
            return signal_type.lower() in bullish_types
        elif direction == 'short':
            return signal_type.lower() in bearish_types
        return False
    
    # =========================================================================
    # SIGNAL RANKING
    # =========================================================================
    
    def rank_signals(self, mtf_alignment: float = 100.0,
                    current_price: float = 0.0) -> List[RankedSignal]:
        """
        Rank all collected signals
        
        Args:
            mtf_alignment: Multi-timeframe alignment score (0-100)
            current_price: Current market price for R:R calculation
            
        Returns:
            List of RankedSignal sorted by quality
        """
        ranked_signals = []
        
        # Generate potential trade setups
        # For each direction (long/short), calculate confluence and create ranked signal
        
        for direction in ['long', 'short']:
            confluence = self.calculate_confluence(direction)
            
            # Skip if confluence too low
            if confluence.total_score < self.config.min_confluence_score:
                continue
            
            # Find entry/exit from signals
            entry, stop, target = self._find_entry_stop_target(direction)
            
            if entry == 0 or stop == 0:
                continue
            
            # Calculate risk/reward
            risk = abs(entry - stop)
            reward = abs(target - entry) if target else risk * 2
            rr_ratio = reward / risk if risk > 0 else 0
            
            # Skip if R:R too low
            if rr_ratio < self.config.min_risk_reward:
                continue
            
            # Calculate overall score
            overall_score = self._calculate_overall_score(
                confluence.total_score,
                mtf_alignment,
                rr_ratio,
                confluence.model_validated,
                confluence.in_kill_zone
            )
            
            # Determine tier
            tier = self._determine_tier(overall_score)
            
            # Get model name
            model_name = self._get_model_name()
            
            # Get PD array type
            pd_array_type = self._get_pd_array_type(direction)
            
            # Create ranked signal
            ranked = RankedSignal(
                rank=0,  # Will be set after sorting
                tier=tier,
                direction=direction,
                overall_score=overall_score,
                confluence=confluence,
                signals=self._get_direction_signals(direction),
                entry_price=entry,
                stop_loss=stop,
                target_1=target,
                target_2=target + (target - entry) if target else 0,
                risk_reward=rr_ratio,
                model_name=model_name,
                pd_array_type=pd_array_type,
                is_valid=True,
            )
            
            # Validate
            is_valid, notes = self._validate_signal(ranked)
            ranked.is_valid = is_valid
            ranked.validation_notes = notes
            
            # Generate narrative
            ranked.narrative = self._generate_narrative(ranked)
            
            ranked_signals.append(ranked)
        
        # Sort by overall score
        ranked_signals.sort(key=lambda x: x.overall_score, reverse=True)
        
        # Assign ranks
        for i, sig in enumerate(ranked_signals, 1):
            sig.rank = i
        
        self._ranked_signals = ranked_signals
        self._last_aggregation = datetime.now()
        
        logger.info(f"Ranked {len(ranked_signals)} signals")
        
        return ranked_signals
    
    def _calculate_overall_score(self, confluence: float,
                                mtf_alignment: float,
                                rr_ratio: float,
                                model_validated: bool,
                                in_kill_zone: bool) -> float:
        """Calculate overall signal score"""
        cfg = self.config
        
        # Normalize R:R (cap at 3:1 for scoring)
        rr_score = min(rr_ratio / 3.0 * 100, 100)
        
        # Model score
        model_score = 100 if model_validated else 0
        
        # Time score
        time_score = 100 if in_kill_zone else 50
        
        # Weighted calculation
        overall = (
            confluence * cfg.confluence_weight +
            mtf_alignment * cfg.mtf_alignment_weight +
            rr_score * cfg.risk_reward_weight +
            model_score * cfg.model_weight +
            time_score * cfg.time_weight
        )
        
        return min(overall, 100)
    
    def _determine_tier(self, score: float) -> RankingTier:
        """Determine ranking tier from score"""
        if score >= 85:
            return RankingTier.TIER_1
        elif score >= 75:
            return RankingTier.TIER_2
        elif score >= 65:
            return RankingTier.TIER_3
        elif score >= 55:
            return RankingTier.TIER_4
        else:
            return RankingTier.TIER_5
    
    def _find_entry_stop_target(self, direction: str) -> Tuple[float, float, float]:
        """Find entry, stop, and target from signals"""
        entry = 0.0
        stop = 0.0
        target = 0.0
        
        # Look for model signal with entry/target
        for sig in self._signals.get(SignalSource.TRADING_MODEL, []):
            if isinstance(sig, ModelSignal):
                if self._direction_matches(sig.signal_type, direction):
                    entry = sig.entry_price or 0
                    target = sig.target_price or 0
                    break
        
        # If no model entry, use OB or FVG level
        if entry == 0:
            ob_signals = self._signals.get(SignalSource.ORDER_BLOCK, [])
            for sig in ob_signals:
                if isinstance(sig, OrderBlockSignal):
                    if self._direction_matches(sig.ob_type, direction):
                        # Entry at OB midpoint
                        entry = sig.ob_mid
                        # Stop below/above OB
                        stop = sig.ob_low if direction == 'long' else sig.ob_high
                        break
        
        # If still no entry, use FVG CE
        if entry == 0:
            fvg_signals = self._signals.get(SignalSource.FVG, [])
            for sig in fvg_signals:
                if isinstance(sig, FVGSignal):
                    if self._direction_matches(sig.fvg_type, direction):
                        entry = sig.fvg_ce
                        stop = sig.fvg_low if direction == 'long' else sig.fvg_high
                        break
        
        # If no target, look for liquidity target
        if target == 0:
            liq_signals = self._signals.get(SignalSource.LIQUIDITY, [])
            for sig in liq_signals:
                if isinstance(sig, LiquiditySignal):
                    if sig.is_target:
                        target = sig.liq_level
                        break
        
        # Default target at 2R if not found
        if target == 0 and entry and stop:
            risk = abs(entry - stop)
            target = entry + (2 * risk) if direction == 'long' else entry - (2 * risk)
        
        return entry, stop, target
    
    def _get_model_name(self) -> str:
        """Get validated model name"""
        for sig in self._signals.get(SignalSource.TRADING_MODEL, []):
            if isinstance(sig, ModelSignal):
                return sig.model_name
        return "confluence"
    
    def _get_pd_array_type(self, direction: str) -> str:
        """Get primary PD array type"""
        # Check for OB first
        for sig in self._signals.get(SignalSource.ORDER_BLOCK, []):
            if isinstance(sig, OrderBlockSignal):
                if self._direction_matches(sig.ob_type, direction):
                    if sig.metadata.get('is_breaker'):
                        return 'breaker_block'
                    return 'order_block'
        
        # Then FVG
        for sig in self._signals.get(SignalSource.FVG, []):
            if isinstance(sig, FVGSignal):
                if self._direction_matches(sig.fvg_type, direction):
                    if sig.is_inverted:
                        return 'inversion_fvg'
                    return 'fvg'
        
        return 'confluence'
    
    def _get_direction_signals(self, direction: str) -> List[BaseSignal]:
        """Get all signals matching direction"""
        matching = []
        for signals in self._signals.values():
            for sig in signals:
                if self._direction_matches(sig.signal_type, direction):
                    matching.append(sig)
        return matching
    
    def _validate_signal(self, signal: RankedSignal) -> Tuple[bool, List[str]]:
        """Validate a ranked signal"""
        notes = []
        is_valid = True
        cfg = self.config
        
        # Structure shift requirement
        if cfg.require_structure_shift and not signal.confluence.has_structure_shift:
            is_valid = False
            notes.append("Missing structure shift")
        
        # Kill zone requirement
        if cfg.require_kill_zone and not signal.confluence.in_kill_zone:
            is_valid = False
            notes.append("Not in kill zone")
        
        # Minimum confluence
        if signal.confluence.total_score < cfg.min_confluence_score:
            is_valid = False
            notes.append(f"Confluence {signal.confluence.total_score:.0f} < {cfg.min_confluence_score}")
        
        # Minimum R:R
        if signal.risk_reward < cfg.min_risk_reward:
            is_valid = False
            notes.append(f"R:R {signal.risk_reward:.2f} < {cfg.min_risk_reward}")
        
        # Overall score
        if signal.overall_score < cfg.min_overall_score:
            is_valid = False
            notes.append(f"Score {signal.overall_score:.0f} < {cfg.min_overall_score}")
        
        if is_valid:
            notes.append("All validation checks passed")
        
        return is_valid, notes
    
    def _generate_narrative(self, signal: RankedSignal) -> str:
        """Generate trade narrative"""
        conf = signal.confluence
        
        parts = []
        parts.append(f"{signal.direction.upper()} trade setup")
        
        if conf.has_structure_shift:
            parts.append("confirmed by structure shift")
        
        if conf.has_liquidity_sweep:
            parts.append("with liquidity sweep")
        
        pd_desc = signal.pd_array_type.replace('_', ' ')
        parts.append(f"entry at {pd_desc}")
        
        if conf.in_premium_discount:
            zone = "discount" if signal.direction == 'long' else "premium"
            parts.append(f"in {zone} zone")
        
        if conf.in_kill_zone:
            parts.append("during kill zone")
        
        if signal.model_name:
            parts.append(f"validating {signal.model_name.replace('_', ' ')}")
        
        parts.append(f"with {signal.risk_reward:.1f}:1 R:R")
        
        return ", ".join(parts) + "."
    
    # =========================================================================
    # OUTPUT METHODS
    # =========================================================================
    
    def get_best_signal(self) -> Optional[RankedSignal]:
        """Get the highest ranked signal"""
        if not self._ranked_signals:
            return None
        return self._ranked_signals[0]
    
    def get_valid_signals(self) -> List[RankedSignal]:
        """Get all valid signals"""
        return [s for s in self._ranked_signals if s.is_valid]
    
    def get_tier_1_signals(self) -> List[RankedSignal]:
        """Get all Tier 1 signals"""
        return [s for s in self._ranked_signals if s.tier == RankingTier.TIER_1]
    
    def get_confluence_result(self) -> Optional[ConfluenceResult]:
        """Get last confluence result"""
        return self._confluence_result
    
    def generate_report(self) -> str:
        """Generate text report of aggregation results"""
        lines = []
        lines.append("=" * 60)
        lines.append("SIGNAL AGGREGATION REPORT")
        lines.append("=" * 60)
        lines.append(f"Signals Collected: {self._signal_count}")
        lines.append(f"Last Aggregation: {self._last_aggregation}")
        lines.append("")
        
        # Signal breakdown by source
        lines.append("SIGNALS BY SOURCE:")
        for source, signals in self._signals.items():
            lines.append(f"  {source.value}: {len(signals)}")
        lines.append("")
        
        # Confluence
        if self._confluence_result:
            lines.append("CONFLUENCE ANALYSIS:")
            lines.append(f"  {self._confluence_result.get_summary()}")
            lines.append("  Factor Breakdown:")
            for factor, score in self._confluence_result.factor_breakdown.items():
                lines.append(f"    • {factor}: {score:.0f} pts")
        lines.append("")
        
        # Ranked signals
        lines.append(f"RANKED SIGNALS: {len(self._ranked_signals)}")
        for sig in self._ranked_signals[:5]:  # Top 5
            lines.append(f"\n  #{sig.rank} [{sig.tier.value}] {sig.direction.upper()}")
            lines.append(f"     Score: {sig.overall_score:.0f}/100")
            lines.append(f"     R:R: {sig.risk_reward:.2f}")
            lines.append(f"     Valid: {'YES' if sig.is_valid else 'NO'}")
            lines.append(f"     {sig.narrative}")
        
        lines.append("")
        lines.append("=" * 60)
        
        return "\n".join(lines)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def create_structure_signal(signal_type: str, structure_type: str,
                           break_level: float, confidence: float = 80.0) -> StructureSignal:
    """Helper to create structure signal"""
    return StructureSignal(
        source=SignalSource.MARKET_STRUCTURE,
        signal_type=signal_type,
        strength=SignalStrength.STRONG if confidence >= 75 else SignalStrength.MODERATE,
        timestamp=datetime.now(),
        price_level=break_level,
        description=f"{structure_type} {signal_type} at {break_level}",
        confidence=confidence,
        structure_type=structure_type,
        break_level=break_level,
        trend_direction=signal_type,
        is_confirmed=True,
    )


def create_ob_signal(ob_type: str, high: float, low: float,
                    confidence: float = 75.0) -> OrderBlockSignal:
    """Helper to create order block signal"""
    mid = (high + low) / 2
    return OrderBlockSignal(
        source=SignalSource.ORDER_BLOCK,
        signal_type=ob_type,
        strength=SignalStrength.MODERATE,
        timestamp=datetime.now(),
        price_level=mid,
        description=f"{ob_type} OB {low}-{high}",
        confidence=confidence,
        ob_type=ob_type,
        ob_high=high,
        ob_low=low,
        ob_mid=mid,
    )


def create_fvg_signal(fvg_type: str, high: float, low: float,
                     in_premium: bool = False, in_discount: bool = False,
                     confidence: float = 70.0) -> FVGSignal:
    """Helper to create FVG signal"""
    ce = (high + low) / 2
    return FVGSignal(
        source=SignalSource.FVG,
        signal_type=fvg_type,
        strength=SignalStrength.MODERATE,
        timestamp=datetime.now(),
        price_level=ce,
        description=f"{fvg_type} FVG {low}-{high}",
        confidence=confidence,
        fvg_type=fvg_type,
        fvg_high=high,
        fvg_low=low,
        fvg_ce=ce,
        in_premium=in_premium,
        in_discount=in_discount,
    )


def create_liquidity_signal(liq_type: str, level: float,
                           is_swept: bool = False,
                           is_target: bool = False) -> LiquiditySignal:
    """Helper to create liquidity signal"""
    return LiquiditySignal(
        source=SignalSource.LIQUIDITY,
        signal_type='bullish' if liq_type == 'sell_side' else 'bearish',
        strength=SignalStrength.STRONG if is_swept else SignalStrength.MODERATE,
        timestamp=datetime.now(),
        price_level=level,
        description=f"{liq_type} liquidity at {level}",
        confidence=85 if is_swept else 60,
        liq_type=liq_type,
        liq_level=level,
        is_swept=is_swept,
        is_target=is_target,
    )


def create_time_signal(session: str, kill_zone: Optional[str] = None,
                      is_macro: bool = False) -> TimeSignal:
    """Helper to create time signal"""
    return TimeSignal(
        source=SignalSource.TIME_CONTEXT,
        signal_type='neutral',
        strength=SignalStrength.STRONG if kill_zone else SignalStrength.MODERATE,
        timestamp=datetime.now(),
        price_level=0,
        description=f"{session} session" + (f" ({kill_zone})" if kill_zone else ""),
        confidence=80 if kill_zone else 50,
        session=session,
        kill_zone=kill_zone,
        is_macro_time=is_macro,
        is_optimal=kill_zone in ['ny_open', 'ny_am', 'london_open'],
    )


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("ICT Signal Aggregator")
    print("=" * 60)
    
    # Create aggregator
    aggregator = SignalAggregator()
    
    # Add sample signals
    # Structure shift
    aggregator.add_signal(create_structure_signal(
        signal_type='bullish',
        structure_type='CHoCH',
        break_level=21450.00,
        confidence=85
    ))
    
    # Order block
    aggregator.add_signal(create_ob_signal(
        ob_type='bullish',
        high=21425.00,
        low=21400.00,
        confidence=75
    ))
    
    # FVG
    fvg = create_fvg_signal(
        fvg_type='bullish',
        high=21445.00,
        low=21430.00,
        in_discount=True,
        confidence=80
    )
    fvg.metadata['has_displacement'] = True
    aggregator.add_signal(fvg)
    
    # Liquidity sweep
    aggregator.add_signal(create_liquidity_signal(
        liq_type='sell_side',
        level=21395.00,
        is_swept=True
    ))
    
    # Liquidity target
    aggregator.add_signal(create_liquidity_signal(
        liq_type='buy_side',
        level=21550.00,
        is_target=True
    ))
    
    # Time context
    aggregator.add_signal(create_time_signal(
        session='ny_am',
        kill_zone='ny_am',
        is_macro=True
    ))
    
    # Model signal
    model = ModelSignal(
        source=SignalSource.TRADING_MODEL,
        signal_type='bullish',
        strength=SignalStrength.STRONG,
        timestamp=datetime.now(),
        price_level=21412.50,
        description="2022 Model Stage 4 Complete",
        confidence=90,
        model_name='2022_model',
        model_stage=4,
        stage_name='entry',
        is_complete=True,
        entry_price=21412.50,
        target_price=21550.00,
    )
    aggregator.add_signal(model)
    
    # Calculate confluence
    print("\nCalculating confluence for LONG direction...")
    confluence = aggregator.calculate_confluence('long')
    print(f"\n{confluence.get_summary()}")
    print("\nFactor Breakdown:")
    for factor, score in confluence.factor_breakdown.items():
        print(f"  • {factor}: {score:.0f} pts")
    
    # Rank signals
    print("\n" + "=" * 60)
    print("Ranking signals...")
    ranked = aggregator.rank_signals(mtf_alignment=85)
    
    # Show results
    print(f"\nFound {len(ranked)} ranked signals")
    for sig in ranked:
        print(f"\n#{sig.rank} [{sig.tier.value}] {sig.direction.upper()}")
        print(f"  Score: {sig.overall_score:.0f}/100")
        print(f"  Entry: {sig.entry_price}, Stop: {sig.stop_loss}, Target: {sig.target_1}")
        print(f"  R:R: {sig.risk_reward:.2f}")
        print(f"  Valid: {sig.is_valid}")
        print(f"  Narrative: {sig.narrative}")
    
    # Generate report
    print("\n" + "=" * 60)
    print(aggregator.generate_report())
