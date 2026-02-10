"""
ICT Core Integration Engine - Phase 2: Core Integration Layer
==============================================================

This module integrates ALL ICT trading handlers into a unified system:
- Order Block Handler
- Market Structure Handler
- Liquidity Handler
- FVG Handler
- Gap Handler
- PD Array Handler
- Timeframe Handler
- Trading Model Handler
- Market Data Engine
- Entry/Stop Management Handler
- Market Condition Handler

CORE RESPONSIBILITIES:
1. Multi-timeframe confluence detection
2. Handler coordination and state management
3. Trade setup generation with complete entry/exit rules
4. Confidence scoring based on ICT principles
5. AI/ML integration for pattern learning
6. Real-time signal generation

ICT INTEGRATION PRINCIPLES:
- "Higher timeframe sets the bias, lower timeframe for entries"
- "Confluence of PD arrays = higher probability"
- "Structure + Liquidity + Time = High probability trades"
- "Model 2022: Liquidity -> Premium/Discount -> PD Array Entry"

Author: ICT Signal Engine
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any, Union
from enum import Enum
from datetime import datetime, time, timedelta
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import json
import logging
from collections import defaultdict
import warnings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class IntegrationMode(Enum):
    """Operating mode for the integration engine"""
    ANALYSIS = "analysis"          # Analyze only, no signals
    SIGNAL = "signal"              # Generate signals
    BACKTEST = "backtest"          # Backtesting mode
    LIVE = "live"                  # Live trading mode
    PAPER = "paper"                # Paper trading


class ConfluenceLevel(Enum):
    """Confluence strength classification"""
    NONE = 0                       # No confluence
    WEAK = 1                       # 1-2 factors
    MODERATE = 2                   # 3-4 factors
    STRONG = 3                     # 5-6 factors
    EXTREME = 4                    # 7+ factors (A+ setup)


class SetupGrade(Enum):
    """ICT Setup grading system"""
    A_PLUS = "A+"                  # Perfect setup - all factors aligned
    A = "A"                        # Excellent setup - most factors aligned
    B = "B"                        # Good setup - sufficient factors
    C = "C"                        # Marginal setup - minimum factors
    INVALID = "X"                  # Invalid - missing critical factors


class TimeframeBias(Enum):
    """Bias from timeframe analysis"""
    STRONGLY_BULLISH = "strongly_bullish"
    BULLISH = "bullish"
    NEUTRAL_BULLISH = "neutral_bullish"
    NEUTRAL = "neutral"
    NEUTRAL_BEARISH = "neutral_bearish"
    BEARISH = "bearish"
    STRONGLY_BEARISH = "strongly_bearish"


class SignalStrength(Enum):
    """Trading signal strength"""
    EXTREME = 5                    # 90-100 confidence
    STRONG = 4                     # 75-89 confidence
    MODERATE = 3                   # 60-74 confidence
    WEAK = 2                       # 50-59 confidence
    VERY_WEAK = 1                  # Below 50


class TradingAction(Enum):
    """Recommended trading action"""
    STRONG_BUY = "strong_buy"
    BUY = "buy"
    SCALE_IN_BUY = "scale_in_buy"
    HOLD_LONG = "hold_long"
    NO_ACTION = "no_action"
    HOLD_SHORT = "hold_short"
    SCALE_IN_SELL = "scale_in_sell"
    SELL = "sell"
    STRONG_SELL = "strong_sell"
    EXIT = "exit"
    TAKE_PROFIT = "take_profit"
    STOP_OUT = "stop_out"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TimeframeAnalysis:
    """Complete analysis for a single timeframe"""
    timeframe: str
    timestamp: datetime
    
    # Structure Analysis
    trend: str                     # bullish, bearish, ranging
    structure_break: Optional[str] # BOS, CHoCH, MSS
    last_swing_high: float
    last_swing_low: float
    current_zone: str              # premium, discount, equilibrium
    
    # Order Blocks
    bullish_obs: List[Dict]
    bearish_obs: List[Dict]
    nearest_ob: Optional[Dict]
    
    # FVGs
    bullish_fvgs: List[Dict]
    bearish_fvgs: List[Dict]
    nearest_fvg: Optional[Dict]
    
    # Liquidity
    buy_side_liquidity: List[Dict]
    sell_side_liquidity: List[Dict]
    nearest_liquidity: Optional[Dict]
    liquidity_swept: bool
    
    # Gaps
    gaps: List[Dict]
    new_day_opening_gap: Optional[Dict]
    new_week_opening_gap: Optional[Dict]
    
    # PD Arrays
    pd_arrays_above: List[Dict]
    pd_arrays_below: List[Dict]
    
    # Session Info
    current_session: str
    kill_zone_active: bool
    macro_time_active: bool
    
    # Quality Metrics
    displacement_present: bool
    clean_price_action: bool
    volume_confirmation: bool


@dataclass
class ConfluenceResult:
    """Result of confluence analysis across timeframes"""
    timestamp: datetime
    symbol: str
    
    # Confluence Factors
    factors: List[str]
    factor_count: int
    confluence_level: ConfluenceLevel
    
    # Alignment Scores
    structure_alignment: float     # 0-100
    pd_array_alignment: float      # 0-100
    time_alignment: float          # 0-100
    liquidity_alignment: float     # 0-100
    
    # Overall Score
    total_score: float             # 0-100
    grade: SetupGrade
    
    # Directional Bias
    htf_bias: TimeframeBias
    ltf_bias: TimeframeBias
    bias_alignment: bool
    
    # Key Levels
    primary_draw: float            # Main target (liquidity/PDA)
    secondary_draw: float          # Secondary target
    invalidation_level: float      # Where setup fails
    
    # Details
    supporting_evidence: List[str]
    concerns: List[str]


@dataclass
class TradeSetup:
    """Complete trade setup with entry, stop, and targets"""
    setup_id: str
    timestamp: datetime
    symbol: str
    
    # Setup Classification
    model: str                     # 2022 Model, Silver Bullet, etc.
    grade: SetupGrade
    direction: str                 # long, short
    
    # Entry Details
    entry_type: str                # limit, market, stop_entry
    entry_price: float
    entry_zone: Tuple[float, float]  # Range for limit orders
    entry_pd_array: str            # Type of PD array for entry
    
    # Stop Loss
    stop_loss: float
    stop_type: str                 # candle, pd_array, structure
    stop_rationale: str
    
    # Targets
    target_1: float
    target_2: float
    target_3: Optional[float]
    ultimate_target: float         # Draw on liquidity
    
    # Risk Management
    risk_reward_1: float
    risk_reward_2: float
    risk_reward_3: Optional[float]
    risk_ticks: float
    recommended_position_size: float  # Based on risk %
    
    # Confluence
    confluence: ConfluenceResult
    confidence: float              # 0-100
    
    # Timeframe Context
    htf_context: Dict
    ltf_trigger: Dict
    
    # Time Constraints
    valid_until: datetime
    kill_zone_required: bool
    optimal_entry_window: Tuple[time, time]
    
    # Execution Notes
    execution_notes: List[str]
    management_rules: List[str]
    invalidation_criteria: List[str]


@dataclass
class IntegrationState:
    """Current state of the integration engine"""
    last_update: datetime
    mode: IntegrationMode
    
    # Active Analyses
    timeframe_analyses: Dict[str, TimeframeAnalysis]
    
    # Current Bias
    primary_bias: TimeframeBias
    bias_strength: float
    
    # Active Setups
    active_setups: List[TradeSetup]
    pending_signals: List[Dict]
    
    # Position Tracking
    current_position: Optional[Dict]
    
    # Performance Metrics
    signals_generated: int
    setups_triggered: int
    win_rate: float
    
    # Handler States
    handler_states: Dict[str, Dict]


@dataclass
class AILearningData:
    """Data structure for AI/ML learning"""
    setup_features: Dict[str, Any]
    market_context: Dict[str, Any]
    outcome: Optional[str]         # win, loss, breakeven
    pnl: Optional[float]
    max_adverse_excursion: Optional[float]
    max_favorable_excursion: Optional[float]
    time_in_trade: Optional[float]
    entry_timing_score: Optional[float]
    exit_timing_score: Optional[float]


# =============================================================================
# HANDLER INTERFACES (Abstract Base Classes)
# =============================================================================

class BaseHandler(ABC):
    """Base interface for all ICT handlers"""
    
    @abstractmethod
    def analyze(self, df: pd.DataFrame, **kwargs) -> Dict:
        """Run analysis on price data"""
        pass
    
    @abstractmethod
    def get_current_state(self) -> Dict:
        """Get current handler state"""
        pass
    
    @abstractmethod
    def reset(self):
        """Reset handler state"""
        pass


class OrderBlockHandlerInterface(BaseHandler):
    """Interface for Order Block Handler"""
    
    @abstractmethod
    def detect_order_blocks(self, df: pd.DataFrame) -> List[Dict]:
        pass
    
    @abstractmethod
    def get_nearest_ob(self, price: float, direction: str) -> Optional[Dict]:
        pass
    
    @abstractmethod
    def is_price_at_ob(self, price: float, tolerance: float) -> bool:
        pass


class MarketStructureHandlerInterface(BaseHandler):
    """Interface for Market Structure Handler"""
    
    @abstractmethod
    def analyze_structure(self, df: pd.DataFrame) -> Dict:
        pass
    
    @abstractmethod
    def detect_structure_break(self, df: pd.DataFrame) -> Optional[Dict]:
        pass
    
    @abstractmethod
    def get_current_zone(self, price: float) -> str:
        pass


class LiquidityHandlerInterface(BaseHandler):
    """Interface for Liquidity Handler"""
    
    @abstractmethod
    def detect_liquidity_pools(self, df: pd.DataFrame) -> Dict:
        pass
    
    @abstractmethod
    def check_liquidity_sweep(self, df: pd.DataFrame) -> Optional[Dict]:
        pass
    
    @abstractmethod
    def get_draw_on_liquidity(self, bias: str) -> Optional[Dict]:
        pass


class FVGHandlerInterface(BaseHandler):
    """Interface for FVG Handler"""
    
    @abstractmethod
    def detect_fvgs(self, df: pd.DataFrame) -> List[Dict]:
        pass
    
    @abstractmethod
    def get_nearest_fvg(self, price: float, direction: str) -> Optional[Dict]:
        pass
    
    @abstractmethod
    def check_fvg_fill(self, price: float, fvg: Dict) -> Dict:
        pass


# =============================================================================
# CONFLUENCE ANALYZER
# =============================================================================

class ConfluenceAnalyzer:
    """
    Analyzes confluence across multiple ICT factors.
    
    ICT: "When you have multiple PD arrays stacked in the same area,
    that's confluence, that's where you want to trade."
    """
    
    # Confluence factor weights
    FACTOR_WEIGHTS = {
        'htf_structure_aligned': 15,
        'ltf_structure_aligned': 10,
        'order_block_present': 12,
        'fvg_present': 12,
        'liquidity_swept': 15,
        'in_premium_discount': 10,
        'kill_zone_active': 8,
        'macro_time': 5,
        'displacement': 10,
        'breaker_block': 8,
        'mitigation_block': 5,
        'volume_confirmation': 5,
        'session_alignment': 5,
        'gap_support': 5,
        'pd_array_stacked': 10,
        'structure_break_recent': 10,
        'ote_zone': 8,
        'model_2022_complete': 15,
        'silver_bullet_window': 10,
        'venom_setup': 12,
        'turtle_soup': 10
    }
    
    # Minimum scores for grades
    GRADE_THRESHOLDS = {
        SetupGrade.A_PLUS: 85,
        SetupGrade.A: 70,
        SetupGrade.B: 55,
        SetupGrade.C: 40,
        SetupGrade.INVALID: 0
    }
    
    def __init__(self):
        self.active_factors = []
        self.current_score = 0.0
        
    def analyze_confluence(
        self,
        timeframe_analyses: Dict[str, TimeframeAnalysis],
        current_price: float,
        bias: str
    ) -> ConfluenceResult:
        """
        Analyze confluence across all timeframes and factors.
        
        Args:
            timeframe_analyses: Analysis from each timeframe
            current_price: Current market price
            bias: Expected direction (bullish/bearish)
            
        Returns:
            ConfluenceResult with complete confluence analysis
        """
        factors = []
        total_score = 0.0
        
        # Get HTF and LTF analyses
        htf_tf = self._get_highest_timeframe(timeframe_analyses)
        ltf_tf = self._get_lowest_timeframe(timeframe_analyses)
        
        htf = timeframe_analyses.get(htf_tf) if htf_tf else None
        ltf = timeframe_analyses.get(ltf_tf) if ltf_tf else None
        
        # 1. Structure Alignment
        structure_score, structure_factors = self._analyze_structure_alignment(
            timeframe_analyses, bias
        )
        factors.extend(structure_factors)
        total_score += structure_score
        
        # 2. PD Array Alignment
        pd_score, pd_factors = self._analyze_pd_array_alignment(
            timeframe_analyses, current_price, bias
        )
        factors.extend(pd_factors)
        total_score += pd_score
        
        # 3. Time Alignment
        time_score, time_factors = self._analyze_time_alignment(ltf)
        factors.extend(time_factors)
        total_score += time_score
        
        # 4. Liquidity Alignment
        liq_score, liq_factors = self._analyze_liquidity_alignment(
            timeframe_analyses, bias
        )
        factors.extend(liq_factors)
        total_score += liq_score
        
        # 5. Check for specific models
        model_score, model_factors = self._check_model_confluence(
            timeframe_analyses, current_price, bias
        )
        factors.extend(model_factors)
        total_score += model_score
        
        # Calculate confluence level
        factor_count = len(factors)
        confluence_level = self._get_confluence_level(factor_count)
        
        # Determine grade
        grade = self._calculate_grade(total_score)
        
        # Get bias from each timeframe level
        htf_bias = self._determine_bias(htf) if htf else TimeframeBias.NEUTRAL
        ltf_bias = self._determine_bias(ltf) if ltf else TimeframeBias.NEUTRAL
        
        # Check bias alignment
        bias_aligned = self._check_bias_alignment(htf_bias, ltf_bias, bias)
        
        # Get key levels
        primary_draw = self._get_primary_draw(timeframe_analyses, bias)
        secondary_draw = self._get_secondary_draw(timeframe_analyses, bias)
        invalidation = self._get_invalidation_level(timeframe_analyses, bias)
        
        # Build supporting evidence and concerns
        evidence = self._build_supporting_evidence(factors, timeframe_analyses)
        concerns = self._identify_concerns(timeframe_analyses, bias)
        
        return ConfluenceResult(
            timestamp=datetime.now(),
            symbol="",  # Set by caller
            factors=factors,
            factor_count=factor_count,
            confluence_level=confluence_level,
            structure_alignment=structure_score / 25 * 100,  # Normalize
            pd_array_alignment=pd_score / 30 * 100,
            time_alignment=time_score / 15 * 100,
            liquidity_alignment=liq_score / 15 * 100,
            total_score=min(total_score, 100),
            grade=grade,
            htf_bias=htf_bias,
            ltf_bias=ltf_bias,
            bias_alignment=bias_aligned,
            primary_draw=primary_draw,
            secondary_draw=secondary_draw,
            invalidation_level=invalidation,
            supporting_evidence=evidence,
            concerns=concerns
        )
    
    def _analyze_structure_alignment(
        self,
        analyses: Dict[str, TimeframeAnalysis],
        bias: str
    ) -> Tuple[float, List[str]]:
        """Analyze structure alignment across timeframes"""
        score = 0.0
        factors = []
        
        aligned_count = 0
        for tf, analysis in analyses.items():
            if analysis.trend == bias:
                aligned_count += 1
                
            if analysis.structure_break:
                if (bias == 'bullish' and 'bullish' in analysis.structure_break.lower()) or \
                   (bias == 'bearish' and 'bearish' in analysis.structure_break.lower()):
                    factors.append(f'structure_break_{tf}')
                    score += self.FACTOR_WEIGHTS['structure_break_recent'] * 0.5
        
        # HTF alignment is more important
        if len(analyses) > 0:
            alignment_pct = aligned_count / len(analyses)
            if alignment_pct >= 0.8:
                factors.append('htf_structure_aligned')
                score += self.FACTOR_WEIGHTS['htf_structure_aligned']
            if alignment_pct >= 0.6:
                factors.append('ltf_structure_aligned')
                score += self.FACTOR_WEIGHTS['ltf_structure_aligned']
                
        return score, factors
    
    def _analyze_pd_array_alignment(
        self,
        analyses: Dict[str, TimeframeAnalysis],
        current_price: float,
        bias: str
    ) -> Tuple[float, List[str]]:
        """Analyze PD array alignment"""
        score = 0.0
        factors = []
        
        # Check for stacked PD arrays
        pd_array_count = 0
        
        for tf, analysis in analyses.items():
            # Check Order Blocks
            if bias == 'bullish' and analysis.bullish_obs:
                for ob in analysis.bullish_obs:
                    if self._is_near_price(ob.get('low', 0), current_price):
                        pd_array_count += 1
                        factors.append(f'bullish_ob_{tf}')
                        break
            elif bias == 'bearish' and analysis.bearish_obs:
                for ob in analysis.bearish_obs:
                    if self._is_near_price(ob.get('high', 0), current_price):
                        pd_array_count += 1
                        factors.append(f'bearish_ob_{tf}')
                        break
                        
            # Check FVGs
            if bias == 'bullish' and analysis.bullish_fvgs:
                for fvg in analysis.bullish_fvgs:
                    if self._is_near_price(fvg.get('low', 0), current_price):
                        pd_array_count += 1
                        factors.append(f'bullish_fvg_{tf}')
                        break
            elif bias == 'bearish' and analysis.bearish_fvgs:
                for fvg in analysis.bearish_fvgs:
                    if self._is_near_price(fvg.get('high', 0), current_price):
                        pd_array_count += 1
                        factors.append(f'bearish_fvg_{tf}')
                        break
                        
        if pd_array_count >= 1:
            factors.append('order_block_present')
            score += self.FACTOR_WEIGHTS['order_block_present']
        if pd_array_count >= 2:
            factors.append('fvg_present')
            score += self.FACTOR_WEIGHTS['fvg_present']
        if pd_array_count >= 3:
            factors.append('pd_array_stacked')
            score += self.FACTOR_WEIGHTS['pd_array_stacked']
            
        return score, factors
    
    def _analyze_time_alignment(
        self,
        ltf: Optional[TimeframeAnalysis]
    ) -> Tuple[float, List[str]]:
        """Analyze time-based factors"""
        score = 0.0
        factors = []
        
        if ltf is None:
            return score, factors
            
        if ltf.kill_zone_active:
            factors.append('kill_zone_active')
            score += self.FACTOR_WEIGHTS['kill_zone_active']
            
        if ltf.macro_time_active:
            factors.append('macro_time')
            score += self.FACTOR_WEIGHTS['macro_time']
            
        # Session alignment
        if ltf.current_session in ['new_york', 'london']:
            factors.append('session_alignment')
            score += self.FACTOR_WEIGHTS['session_alignment']
            
        return score, factors
    
    def _analyze_liquidity_alignment(
        self,
        analyses: Dict[str, TimeframeAnalysis],
        bias: str
    ) -> Tuple[float, List[str]]:
        """Analyze liquidity factors"""
        score = 0.0
        factors = []
        
        for tf, analysis in analyses.items():
            if analysis.liquidity_swept:
                if (bias == 'bullish' and analysis.sell_side_liquidity) or \
                   (bias == 'bearish' and analysis.buy_side_liquidity):
                    factors.append(f'liquidity_swept_{tf}')
                    score += self.FACTOR_WEIGHTS['liquidity_swept'] * 0.5
                    
        if 'liquidity_swept' in ' '.join(factors):
            factors.append('liquidity_swept')
            score += self.FACTOR_WEIGHTS['liquidity_swept'] * 0.5
            
        return score, factors
    
    def _check_model_confluence(
        self,
        analyses: Dict[str, TimeframeAnalysis],
        current_price: float,
        bias: str
    ) -> Tuple[float, List[str]]:
        """Check for specific ICT model confluence"""
        score = 0.0
        factors = []
        
        # Check for displacement
        for tf, analysis in analyses.items():
            if analysis.displacement_present:
                factors.append('displacement')
                score += self.FACTOR_WEIGHTS['displacement']
                break
                
        # Check for premium/discount
        for tf, analysis in analyses.items():
            zone = analysis.current_zone
            if (bias == 'bullish' and zone == 'discount') or \
               (bias == 'bearish' and zone == 'premium'):
                factors.append('in_premium_discount')
                score += self.FACTOR_WEIGHTS['in_premium_discount']
                break
                
        return score, factors
    
    def _get_confluence_level(self, factor_count: int) -> ConfluenceLevel:
        """Determine confluence level from factor count"""
        if factor_count >= 7:
            return ConfluenceLevel.EXTREME
        elif factor_count >= 5:
            return ConfluenceLevel.STRONG
        elif factor_count >= 3:
            return ConfluenceLevel.MODERATE
        elif factor_count >= 1:
            return ConfluenceLevel.WEAK
        return ConfluenceLevel.NONE
    
    def _calculate_grade(self, score: float) -> SetupGrade:
        """Calculate setup grade from score"""
        if score >= self.GRADE_THRESHOLDS[SetupGrade.A_PLUS]:
            return SetupGrade.A_PLUS
        elif score >= self.GRADE_THRESHOLDS[SetupGrade.A]:
            return SetupGrade.A
        elif score >= self.GRADE_THRESHOLDS[SetupGrade.B]:
            return SetupGrade.B
        elif score >= self.GRADE_THRESHOLDS[SetupGrade.C]:
            return SetupGrade.C
        return SetupGrade.INVALID
    
    def _determine_bias(self, analysis: TimeframeAnalysis) -> TimeframeBias:
        """Determine bias from timeframe analysis"""
        if analysis.trend == 'bullish':
            if analysis.displacement_present:
                return TimeframeBias.STRONGLY_BULLISH
            return TimeframeBias.BULLISH
        elif analysis.trend == 'bearish':
            if analysis.displacement_present:
                return TimeframeBias.STRONGLY_BEARISH
            return TimeframeBias.BEARISH
        return TimeframeBias.NEUTRAL
    
    def _check_bias_alignment(
        self,
        htf_bias: TimeframeBias,
        ltf_bias: TimeframeBias,
        expected: str
    ) -> bool:
        """Check if HTF and LTF biases align with expected direction"""
        bullish_biases = [
            TimeframeBias.STRONGLY_BULLISH,
            TimeframeBias.BULLISH,
            TimeframeBias.NEUTRAL_BULLISH
        ]
        bearish_biases = [
            TimeframeBias.STRONGLY_BEARISH,
            TimeframeBias.BEARISH,
            TimeframeBias.NEUTRAL_BEARISH
        ]
        
        if expected == 'bullish':
            return htf_bias in bullish_biases and ltf_bias in bullish_biases
        elif expected == 'bearish':
            return htf_bias in bearish_biases and ltf_bias in bearish_biases
        return False
    
    def _is_near_price(
        self,
        level: float,
        current_price: float,
        tolerance_pct: float = 0.002
    ) -> bool:
        """Check if level is near current price"""
        if level == 0 or current_price == 0:
            return False
        return abs(level - current_price) / current_price <= tolerance_pct
    
    def _get_highest_timeframe(self, analyses: Dict[str, TimeframeAnalysis]) -> Optional[str]:
        """Get highest timeframe from analyses"""
        tf_order = ['1M', 'W', 'D', '4H', '1H', '30m', '15m', '5m', '1m']
        for tf in tf_order:
            if tf in analyses:
                return tf
        return list(analyses.keys())[0] if analyses else None
    
    def _get_lowest_timeframe(self, analyses: Dict[str, TimeframeAnalysis]) -> Optional[str]:
        """Get lowest timeframe from analyses"""
        tf_order = ['1m', '5m', '15m', '30m', '1H', '4H', 'D', 'W', '1M']
        for tf in tf_order:
            if tf in analyses:
                return tf
        return list(analyses.keys())[-1] if analyses else None
    
    def _get_primary_draw(
        self,
        analyses: Dict[str, TimeframeAnalysis],
        bias: str
    ) -> float:
        """Get primary draw on liquidity"""
        for tf, analysis in analyses.items():
            if bias == 'bullish' and analysis.buy_side_liquidity:
                return analysis.buy_side_liquidity[0].get('price', 0)
            elif bias == 'bearish' and analysis.sell_side_liquidity:
                return analysis.sell_side_liquidity[0].get('price', 0)
        return 0.0
    
    def _get_secondary_draw(
        self,
        analyses: Dict[str, TimeframeAnalysis],
        bias: str
    ) -> float:
        """Get secondary draw on liquidity"""
        for tf, analysis in analyses.items():
            if bias == 'bullish' and len(analysis.buy_side_liquidity) > 1:
                return analysis.buy_side_liquidity[1].get('price', 0)
            elif bias == 'bearish' and len(analysis.sell_side_liquidity) > 1:
                return analysis.sell_side_liquidity[1].get('price', 0)
        return 0.0
    
    def _get_invalidation_level(
        self,
        analyses: Dict[str, TimeframeAnalysis],
        bias: str
    ) -> float:
        """Get invalidation level for setup"""
        for tf, analysis in analyses.items():
            if bias == 'bullish':
                return analysis.last_swing_low
            elif bias == 'bearish':
                return analysis.last_swing_high
        return 0.0
    
    def _build_supporting_evidence(
        self,
        factors: List[str],
        analyses: Dict[str, TimeframeAnalysis]
    ) -> List[str]:
        """Build list of supporting evidence"""
        evidence = []
        
        if 'htf_structure_aligned' in factors:
            evidence.append("Higher timeframe structure aligned with bias")
        if 'liquidity_swept' in factors:
            evidence.append("Liquidity has been swept (stop hunt complete)")
        if 'pd_array_stacked' in factors:
            evidence.append("Multiple PD arrays stacked at entry zone")
        if 'kill_zone_active' in factors:
            evidence.append("Currently in active kill zone")
        if 'displacement' in factors:
            evidence.append("Strong displacement present")
            
        return evidence
    
    def _identify_concerns(
        self,
        analyses: Dict[str, TimeframeAnalysis],
        bias: str
    ) -> List[str]:
        """Identify potential concerns with the setup"""
        concerns = []
        
        for tf, analysis in analyses.items():
            if not analysis.clean_price_action:
                concerns.append(f"Choppy price action on {tf}")
                break
                
        # Check for opposing structure
        for tf, analysis in analyses.items():
            if analysis.trend != bias:
                concerns.append(f"{tf} trend not aligned with bias")
                
        return concerns


# =============================================================================
# TRADE SETUP GENERATOR
# =============================================================================

class TradeSetupGenerator:
    """
    Generates complete trade setups from confluence analysis.
    
    ICT: "Entry refinement comes from the lower timeframe PD arrays
    while the higher timeframe sets the narrative."
    """
    
    def __init__(self, risk_per_trade: float = 0.01):
        """
        Initialize setup generator.
        
        Args:
            risk_per_trade: Risk percentage per trade (default 1%)
        """
        self.risk_per_trade = risk_per_trade
        self.setup_counter = 0
        
    def generate_setup(
        self,
        confluence: ConfluenceResult,
        timeframe_analyses: Dict[str, TimeframeAnalysis],
        current_price: float,
        account_size: float,
        symbol: str
    ) -> Optional[TradeSetup]:
        """
        Generate a complete trade setup from confluence analysis.
        
        Args:
            confluence: Confluence analysis result
            timeframe_analyses: Analysis from each timeframe
            current_price: Current market price
            account_size: Trading account size
            symbol: Trading symbol
            
        Returns:
            TradeSetup or None if invalid
        """
        # Validate minimum grade
        if confluence.grade == SetupGrade.INVALID:
            logger.info("Setup rejected: Invalid grade")
            return None
            
        # Determine direction
        direction = self._determine_direction(confluence)
        if direction is None:
            logger.info("Setup rejected: Unable to determine direction")
            return None
            
        # Get entry details
        entry_info = self._calculate_entry(
            timeframe_analyses, current_price, direction
        )
        if entry_info is None:
            logger.info("Setup rejected: No valid entry found")
            return None
            
        # Calculate stop loss
        stop_info = self._calculate_stop_loss(
            timeframe_analyses, entry_info['price'], direction
        )
        
        # Calculate targets
        targets = self._calculate_targets(
            confluence, entry_info['price'], stop_info['price'], direction
        )
        
        # Calculate position size
        risk_amount = account_size * self.risk_per_trade
        risk_ticks = abs(entry_info['price'] - stop_info['price'])
        position_size = risk_amount / risk_ticks if risk_ticks > 0 else 0
        
        # Calculate risk/reward ratios
        rr1 = abs(targets['target_1'] - entry_info['price']) / risk_ticks if risk_ticks > 0 else 0
        rr2 = abs(targets['target_2'] - entry_info['price']) / risk_ticks if risk_ticks > 0 else 0
        rr3 = abs(targets.get('target_3', entry_info['price']) - entry_info['price']) / risk_ticks if risk_ticks > 0 and targets.get('target_3') else None
        
        # Determine model
        model = self._identify_model(confluence, timeframe_analyses)
        
        # Generate setup ID
        self.setup_counter += 1
        setup_id = f"{symbol}_{datetime.now().strftime('%Y%m%d%H%M%S')}_{self.setup_counter}"
        
        # Get HTF and LTF context
        htf_tf = self._get_highest_timeframe(timeframe_analyses)
        ltf_tf = self._get_lowest_timeframe(timeframe_analyses)
        
        htf_context = self._extract_context(timeframe_analyses.get(htf_tf))
        ltf_trigger = self._extract_context(timeframe_analyses.get(ltf_tf))
        
        # Calculate validity window
        valid_until = datetime.now() + timedelta(hours=4)
        
        # Get optimal entry window
        entry_window = self._get_entry_window(timeframe_analyses)
        
        # Build execution notes
        execution_notes = self._build_execution_notes(
            confluence, direction, entry_info
        )
        
        # Build management rules
        management_rules = self._build_management_rules(
            direction, entry_info['price'], stop_info['price'], targets
        )
        
        # Build invalidation criteria
        invalidation_criteria = self._build_invalidation_criteria(
            confluence, direction
        )
        
        return TradeSetup(
            setup_id=setup_id,
            timestamp=datetime.now(),
            symbol=symbol,
            model=model,
            grade=confluence.grade,
            direction=direction,
            entry_type=entry_info['type'],
            entry_price=entry_info['price'],
            entry_zone=entry_info['zone'],
            entry_pd_array=entry_info['pd_array'],
            stop_loss=stop_info['price'],
            stop_type=stop_info['type'],
            stop_rationale=stop_info['rationale'],
            target_1=targets['target_1'],
            target_2=targets['target_2'],
            target_3=targets.get('target_3'),
            ultimate_target=confluence.primary_draw,
            risk_reward_1=round(rr1, 2),
            risk_reward_2=round(rr2, 2),
            risk_reward_3=round(rr3, 2) if rr3 else None,
            risk_ticks=risk_ticks,
            recommended_position_size=position_size,
            confluence=confluence,
            confidence=confluence.total_score,
            htf_context=htf_context,
            ltf_trigger=ltf_trigger,
            valid_until=valid_until,
            kill_zone_required=confluence.grade != SetupGrade.A_PLUS,
            optimal_entry_window=entry_window,
            execution_notes=execution_notes,
            management_rules=management_rules,
            invalidation_criteria=invalidation_criteria
        )
    
    def _determine_direction(self, confluence: ConfluenceResult) -> Optional[str]:
        """Determine trade direction from confluence"""
        bullish_biases = [
            TimeframeBias.STRONGLY_BULLISH,
            TimeframeBias.BULLISH,
            TimeframeBias.NEUTRAL_BULLISH
        ]
        bearish_biases = [
            TimeframeBias.STRONGLY_BEARISH,
            TimeframeBias.BEARISH,
            TimeframeBias.NEUTRAL_BEARISH
        ]
        
        if confluence.htf_bias in bullish_biases and confluence.ltf_bias in bullish_biases:
            return 'long'
        elif confluence.htf_bias in bearish_biases and confluence.ltf_bias in bearish_biases:
            return 'short'
        return None
    
    def _calculate_entry(
        self,
        analyses: Dict[str, TimeframeAnalysis],
        current_price: float,
        direction: str
    ) -> Optional[Dict]:
        """Calculate entry price and type"""
        # Look for nearest PD array on LTF
        ltf_tf = self._get_lowest_timeframe(analyses)
        if ltf_tf is None:
            return None
            
        ltf = analyses[ltf_tf]
        
        if direction == 'long':
            # Look for bullish OB or FVG below price
            if ltf.bullish_obs:
                ob = ltf.bullish_obs[0]
                return {
                    'type': 'limit',
                    'price': ob.get('mean_threshold', ob.get('low', current_price)),
                    'zone': (ob.get('low', current_price), ob.get('high', current_price)),
                    'pd_array': 'order_block'
                }
            elif ltf.bullish_fvgs:
                fvg = ltf.bullish_fvgs[0]
                return {
                    'type': 'limit',
                    'price': fvg.get('ce', fvg.get('low', current_price)),
                    'zone': (fvg.get('low', current_price), fvg.get('high', current_price)),
                    'pd_array': 'fvg'
                }
        else:  # short
            if ltf.bearish_obs:
                ob = ltf.bearish_obs[0]
                return {
                    'type': 'limit',
                    'price': ob.get('mean_threshold', ob.get('high', current_price)),
                    'zone': (ob.get('low', current_price), ob.get('high', current_price)),
                    'pd_array': 'order_block'
                }
            elif ltf.bearish_fvgs:
                fvg = ltf.bearish_fvgs[0]
                return {
                    'type': 'limit',
                    'price': fvg.get('ce', fvg.get('high', current_price)),
                    'zone': (fvg.get('low', current_price), fvg.get('high', current_price)),
                    'pd_array': 'fvg'
                }
                
        # Fallback to market entry
        return {
            'type': 'market',
            'price': current_price,
            'zone': (current_price, current_price),
            'pd_array': 'none'
        }
    
    def _calculate_stop_loss(
        self,
        analyses: Dict[str, TimeframeAnalysis],
        entry_price: float,
        direction: str
    ) -> Dict:
        """Calculate stop loss placement"""
        ltf_tf = self._get_lowest_timeframe(analyses)
        if ltf_tf is None:
            # Default stop
            buffer = entry_price * 0.002
            if direction == 'long':
                return {
                    'price': entry_price - buffer,
                    'type': 'default',
                    'rationale': 'Default 0.2% stop'
                }
            else:
                return {
                    'price': entry_price + buffer,
                    'type': 'default',
                    'rationale': 'Default 0.2% stop'
                }
                
        ltf = analyses[ltf_tf]
        
        if direction == 'long':
            # Stop below swing low or OB low
            stop_price = ltf.last_swing_low
            if ltf.bullish_obs:
                ob_low = ltf.bullish_obs[0].get('low', stop_price)
                stop_price = min(stop_price, ob_low)
            buffer = (entry_price - stop_price) * 0.1
            return {
                'price': stop_price - buffer,
                'type': 'structure',
                'rationale': f'Below swing low + buffer'
            }
        else:
            stop_price = ltf.last_swing_high
            if ltf.bearish_obs:
                ob_high = ltf.bearish_obs[0].get('high', stop_price)
                stop_price = max(stop_price, ob_high)
            buffer = (stop_price - entry_price) * 0.1
            return {
                'price': stop_price + buffer,
                'type': 'structure',
                'rationale': f'Above swing high + buffer'
            }
    
    def _calculate_targets(
        self,
        confluence: ConfluenceResult,
        entry_price: float,
        stop_loss: float,
        direction: str
    ) -> Dict:
        """Calculate profit targets"""
        risk = abs(entry_price - stop_loss)
        
        if direction == 'long':
            target_1 = entry_price + (risk * 1.5)
            target_2 = entry_price + (risk * 2.5)
            target_3 = confluence.primary_draw if confluence.primary_draw > target_2 else target_2 + risk
        else:
            target_1 = entry_price - (risk * 1.5)
            target_2 = entry_price - (risk * 2.5)
            target_3 = confluence.primary_draw if confluence.primary_draw < target_2 else target_2 - risk
            
        return {
            'target_1': target_1,
            'target_2': target_2,
            'target_3': target_3
        }
    
    def _identify_model(
        self,
        confluence: ConfluenceResult,
        analyses: Dict[str, TimeframeAnalysis]
    ) -> str:
        """Identify which ICT model this setup represents"""
        factors = confluence.factors
        
        if 'model_2022_complete' in factors:
            return 'ICT 2022 Model'
        elif 'silver_bullet_window' in factors:
            return 'Silver Bullet'
        elif 'venom_setup' in factors:
            return 'Venom'
        elif 'turtle_soup' in factors:
            return 'Turtle Soup'
        elif 'liquidity_swept' in factors and 'displacement' in factors:
            return 'Smart Money Reversal'
        elif 'pd_array_stacked' in factors:
            return 'PD Array Confluence'
        return 'Standard ICT Setup'
    
    def _get_highest_timeframe(self, analyses: Dict[str, TimeframeAnalysis]) -> Optional[str]:
        """Get highest timeframe"""
        tf_order = ['1M', 'W', 'D', '4H', '1H', '30m', '15m', '5m', '1m']
        for tf in tf_order:
            if tf in analyses:
                return tf
        return list(analyses.keys())[0] if analyses else None
    
    def _get_lowest_timeframe(self, analyses: Dict[str, TimeframeAnalysis]) -> Optional[str]:
        """Get lowest timeframe"""
        tf_order = ['1m', '5m', '15m', '30m', '1H', '4H', 'D', 'W', '1M']
        for tf in tf_order:
            if tf in analyses:
                return tf
        return list(analyses.keys())[-1] if analyses else None
    
    def _extract_context(self, analysis: Optional[TimeframeAnalysis]) -> Dict:
        """Extract key context from analysis"""
        if analysis is None:
            return {}
        return {
            'trend': analysis.trend,
            'zone': analysis.current_zone,
            'structure_break': analysis.structure_break,
            'displacement': analysis.displacement_present,
            'liquidity_swept': analysis.liquidity_swept
        }
    
    def _get_entry_window(
        self,
        analyses: Dict[str, TimeframeAnalysis]
    ) -> Tuple[time, time]:
        """Get optimal entry time window"""
        # Default to NY Open kill zone
        return (time(7, 0), time(10, 0))
    
    def _build_execution_notes(
        self,
        confluence: ConfluenceResult,
        direction: str,
        entry_info: Dict
    ) -> List[str]:
        """Build execution notes for the setup"""
        notes = []
        
        notes.append(f"Direction: {direction.upper()}")
        notes.append(f"Entry Type: {entry_info['type'].upper()} at {entry_info['pd_array']}")
        
        if confluence.grade == SetupGrade.A_PLUS:
            notes.append("A+ SETUP: Execute with confidence")
        elif confluence.grade == SetupGrade.A:
            notes.append("A SETUP: Wait for confirmation candle")
        elif confluence.grade == SetupGrade.B:
            notes.append("B SETUP: Consider reduced position size")
        else:
            notes.append("C SETUP: Use tight stop, scale in")
            
        if not confluence.bias_alignment:
            notes.append("WARNING: HTF/LTF bias not fully aligned")
            
        return notes
    
    def _build_management_rules(
        self,
        direction: str,
        entry: float,
        stop: float,
        targets: Dict
    ) -> List[str]:
        """Build position management rules"""
        rules = []
        
        rules.append(f"Entry: {entry:.5f}")
        rules.append(f"Stop Loss: {stop:.5f}")
        rules.append(f"Take 33% at Target 1: {targets['target_1']:.5f}")
        rules.append(f"Take 33% at Target 2: {targets['target_2']:.5f}")
        rules.append(f"Let 34% run to Target 3: {targets['target_3']:.5f}")
        rules.append("Move stop to break-even after Target 1")
        rules.append("Trail stop below/above swing points after Target 2")
        
        return rules
    
    def _build_invalidation_criteria(
        self,
        confluence: ConfluenceResult,
        direction: str
    ) -> List[str]:
        """Build setup invalidation criteria"""
        criteria = []
        
        criteria.append(f"Stop loss hit at {confluence.invalidation_level:.5f}")
        criteria.append("Structure breaks against position")
        criteria.append("New opposing displacement forms")
        
        if direction == 'long':
            criteria.append("Lower low made on LTF")
        else:
            criteria.append("Higher high made on LTF")
            
        criteria.append("Entry window expires")
        
        return criteria


# =============================================================================
# AI/ML INTEGRATION LAYER
# =============================================================================

class AILearningEngine:
    """
    AI/ML integration for learning from trade outcomes.
    
    Tracks setup features and outcomes to improve future predictions.
    """
    
    def __init__(self):
        self.training_data: List[AILearningData] = []
        self.feature_importance: Dict[str, float] = {}
        self.model_accuracy: Dict[str, float] = {}
        self.pattern_memory: Dict[str, List[Dict]] = defaultdict(list)
        
    def record_setup(
        self,
        setup: TradeSetup,
        market_context: Dict[str, Any]
    ) -> str:
        """
        Record a trade setup for learning.
        
        Args:
            setup: Trade setup to record
            market_context: Additional market context
            
        Returns:
            Record ID
        """
        # Extract features from setup
        features = self._extract_features(setup)
        
        # Create learning record
        record = AILearningData(
            setup_features=features,
            market_context=market_context,
            outcome=None,
            pnl=None,
            max_adverse_excursion=None,
            max_favorable_excursion=None,
            time_in_trade=None,
            entry_timing_score=None,
            exit_timing_score=None
        )
        
        self.training_data.append(record)
        return f"record_{len(self.training_data)}"
    
    def record_outcome(
        self,
        record_id: str,
        outcome: str,
        pnl: float,
        mae: float,
        mfe: float,
        time_in_trade: float,
        entry_timing: float,
        exit_timing: float
    ):
        """
        Record the outcome of a trade.
        
        Args:
            record_id: ID of the record to update
            outcome: Trade outcome (win/loss/breakeven)
            pnl: Profit/loss in dollars
            mae: Maximum adverse excursion
            mfe: Maximum favorable excursion
            time_in_trade: Time in trade (minutes)
            entry_timing: Entry timing score (0-100)
            exit_timing: Exit timing score (0-100)
        """
        try:
            idx = int(record_id.split('_')[1]) - 1
            if 0 <= idx < len(self.training_data):
                record = self.training_data[idx]
                record.outcome = outcome
                record.pnl = pnl
                record.max_adverse_excursion = mae
                record.max_favorable_excursion = mfe
                record.time_in_trade = time_in_trade
                record.entry_timing_score = entry_timing
                record.exit_timing_score = exit_timing
                
                # Update pattern memory
                self._update_pattern_memory(record)
        except (IndexError, ValueError) as e:
            logger.error(f"Error recording outcome: {e}")
    
    def predict_outcome(
        self,
        setup: TradeSetup,
        market_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Predict outcome for a new setup based on historical data.
        
        Args:
            setup: Trade setup to predict
            market_context: Current market context
            
        Returns:
            Prediction with confidence and similar patterns
        """
        features = self._extract_features(setup)
        
        # Find similar patterns
        similar = self._find_similar_patterns(features)
        
        # Calculate win rate from similar patterns
        if similar:
            wins = sum(1 for p in similar if p.get('outcome') == 'win')
            win_rate = wins / len(similar) * 100
            avg_pnl = np.mean([p.get('pnl', 0) for p in similar])
            avg_mae = np.mean([p.get('mae', 0) for p in similar])
            avg_mfe = np.mean([p.get('mfe', 0) for p in similar])
        else:
            win_rate = 50.0
            avg_pnl = 0
            avg_mae = 0
            avg_mfe = 0
            
        # Adjust by feature importance
        adjusted_confidence = self._adjust_by_features(features, win_rate)
        
        return {
            'predicted_outcome': 'win' if adjusted_confidence > 50 else 'loss',
            'confidence': adjusted_confidence,
            'similar_patterns': len(similar),
            'historical_win_rate': win_rate,
            'average_pnl': avg_pnl,
            'average_mae': avg_mae,
            'average_mfe': avg_mfe,
            'recommendation': self._get_recommendation(adjusted_confidence)
        }
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get current feature importance rankings"""
        return dict(sorted(
            self.feature_importance.items(),
            key=lambda x: x[1],
            reverse=True
        ))
    
    def get_model_performance(self) -> Dict[str, Any]:
        """Get overall model performance metrics"""
        if not self.training_data:
            return {'status': 'No training data available'}
            
        completed = [d for d in self.training_data if d.outcome is not None]
        
        if not completed:
            return {'status': 'No completed trades'}
            
        wins = sum(1 for d in completed if d.outcome == 'win')
        losses = sum(1 for d in completed if d.outcome == 'loss')
        
        return {
            'total_trades': len(completed),
            'wins': wins,
            'losses': losses,
            'win_rate': wins / len(completed) * 100 if completed else 0,
            'total_pnl': sum(d.pnl for d in completed if d.pnl),
            'avg_pnl': np.mean([d.pnl for d in completed if d.pnl]),
            'avg_mae': np.mean([d.max_adverse_excursion for d in completed if d.max_adverse_excursion]),
            'avg_mfe': np.mean([d.max_favorable_excursion for d in completed if d.max_favorable_excursion])
        }
    
    def _extract_features(self, setup: TradeSetup) -> Dict[str, Any]:
        """Extract features from a trade setup"""
        return {
            'model': setup.model,
            'grade': setup.grade.value,
            'direction': setup.direction,
            'confluence_level': setup.confluence.confluence_level.value,
            'factor_count': setup.confluence.factor_count,
            'structure_alignment': setup.confluence.structure_alignment,
            'pd_array_alignment': setup.confluence.pd_array_alignment,
            'time_alignment': setup.confluence.time_alignment,
            'liquidity_alignment': setup.confluence.liquidity_alignment,
            'htf_bias': setup.confluence.htf_bias.value,
            'ltf_bias': setup.confluence.ltf_bias.value,
            'bias_aligned': setup.confluence.bias_alignment,
            'confidence': setup.confidence,
            'risk_reward_1': setup.risk_reward_1,
            'entry_pd_array': setup.entry_pd_array,
            'kill_zone_required': setup.kill_zone_required
        }
    
    def _find_similar_patterns(
        self,
        features: Dict[str, Any],
        min_similarity: float = 0.7
    ) -> List[Dict]:
        """Find similar patterns from historical data"""
        similar = []
        
        for record in self.training_data:
            if record.outcome is None:
                continue
                
            similarity = self._calculate_similarity(features, record.setup_features)
            if similarity >= min_similarity:
                similar.append({
                    'features': record.setup_features,
                    'outcome': record.outcome,
                    'pnl': record.pnl,
                    'mae': record.max_adverse_excursion,
                    'mfe': record.max_favorable_excursion,
                    'similarity': similarity
                })
                
        return sorted(similar, key=lambda x: x['similarity'], reverse=True)
    
    def _calculate_similarity(
        self,
        features1: Dict[str, Any],
        features2: Dict[str, Any]
    ) -> float:
        """Calculate similarity between two feature sets"""
        if not features1 or not features2:
            return 0.0
            
        matches = 0
        total = 0
        
        for key in features1:
            if key in features2:
                total += 1
                if features1[key] == features2[key]:
                    matches += 1
                elif isinstance(features1[key], (int, float)) and isinstance(features2[key], (int, float)):
                    # For numeric values, check if within 20%
                    if features2[key] != 0:
                        diff = abs(features1[key] - features2[key]) / abs(features2[key])
                        if diff <= 0.2:
                            matches += 0.5
                            
        return matches / total if total > 0 else 0.0
    
    def _adjust_by_features(
        self,
        features: Dict[str, Any],
        base_rate: float
    ) -> float:
        """Adjust prediction by feature importance"""
        adjustment = 0.0
        
        # Grade adjustment
        grade_adjustments = {'A+': 10, 'A': 5, 'B': 0, 'C': -5, 'X': -20}
        adjustment += grade_adjustments.get(features.get('grade', 'C'), 0)
        
        # Confluence adjustment
        conf_adjustments = {4: 10, 3: 5, 2: 0, 1: -5, 0: -10}
        adjustment += conf_adjustments.get(features.get('confluence_level', 0), 0)
        
        # Bias alignment adjustment
        if features.get('bias_aligned', False):
            adjustment += 5
        else:
            adjustment -= 5
            
        return max(0, min(100, base_rate + adjustment))
    
    def _update_pattern_memory(self, record: AILearningData):
        """Update pattern memory with completed trade"""
        model = record.setup_features.get('model', 'unknown')
        self.pattern_memory[model].append({
            'features': record.setup_features,
            'outcome': record.outcome,
            'pnl': record.pnl
        })
        
        # Update feature importance
        self._update_feature_importance()
    
    def _update_feature_importance(self):
        """Update feature importance based on outcomes"""
        if len(self.training_data) < 10:
            return
            
        completed = [d for d in self.training_data if d.outcome is not None]
        if len(completed) < 10:
            return
            
        # Simple feature importance calculation
        for record in completed:
            for feature, value in record.setup_features.items():
                if feature not in self.feature_importance:
                    self.feature_importance[feature] = 0.5
                    
                if record.outcome == 'win':
                    self.feature_importance[feature] = min(1.0, self.feature_importance[feature] + 0.01)
                else:
                    self.feature_importance[feature] = max(0.0, self.feature_importance[feature] - 0.01)
    
    def _get_recommendation(self, confidence: float) -> str:
        """Get trading recommendation based on confidence"""
        if confidence >= 80:
            return "STRONG TRADE: Execute with full position"
        elif confidence >= 65:
            return "GOOD TRADE: Execute with normal position"
        elif confidence >= 50:
            return "MARGINAL: Consider reduced position or skip"
        else:
            return "AVOID: Low probability setup"


# =============================================================================
# MAIN INTEGRATION ENGINE
# =============================================================================

class ICTIntegrationEngine:
    """
    Master Integration Engine for all ICT trading components.
    
    Coordinates all handlers and provides unified analysis and signal generation.
    
    ICT: "The algorithm knows where it's going. We just need to align
    with where the draw on liquidity is."
    """
    
    def __init__(
        self,
        mode: IntegrationMode = IntegrationMode.ANALYSIS,
        risk_per_trade: float = 0.01,
        account_size: float = 100000.0
    ):
        """
        Initialize the integration engine.
        
        Args:
            mode: Operating mode
            risk_per_trade: Risk percentage per trade
            account_size: Trading account size
        """
        self.mode = mode
        self.risk_per_trade = risk_per_trade
        self.account_size = account_size
        
        # Initialize components
        self.confluence_analyzer = ConfluenceAnalyzer()
        self.setup_generator = TradeSetupGenerator(risk_per_trade)
        self.ai_engine = AILearningEngine()
        
        # Handler registry (to be populated with actual handlers)
        self.handlers: Dict[str, Any] = {}
        
        # State
        self.state = IntegrationState(
            last_update=datetime.now(),
            mode=mode,
            timeframe_analyses={},
            primary_bias=TimeframeBias.NEUTRAL,
            bias_strength=0.0,
            active_setups=[],
            pending_signals=[],
            current_position=None,
            signals_generated=0,
            setups_triggered=0,
            win_rate=0.0,
            handler_states={}
        )
        
        # Configuration
        self.config = {
            'timeframes': ['D', '4H', '1H', '15m', '5m'],
            'min_grade_for_signal': SetupGrade.C,
            'require_kill_zone': True,
            'min_rr_ratio': 1.5,
            'max_daily_trades': 3,
            'session_filter': ['new_york', 'london']
        }
        
        logger.info(f"ICT Integration Engine initialized in {mode.value} mode")
    
    def register_handler(self, name: str, handler: Any):
        """
        Register a handler with the engine.
        
        Args:
            name: Handler name
            handler: Handler instance
        """
        self.handlers[name] = handler
        logger.info(f"Registered handler: {name}")
    
    def analyze_market(
        self,
        symbol: str,
        data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Perform complete market analysis across all timeframes.
        
        Args:
            symbol: Trading symbol
            data: Dict of DataFrames keyed by timeframe
            
        Returns:
            Complete analysis result
        """
        analyses = {}
        
        for tf, df in data.items():
            if df is None or df.empty:
                continue
                
            analysis = self._analyze_timeframe(tf, df)
            analyses[tf] = analysis
            
        self.state.timeframe_analyses = analyses
        self.state.last_update = datetime.now()
        
        # Determine primary bias
        self.state.primary_bias = self._determine_primary_bias(analyses)
        self.state.bias_strength = self._calculate_bias_strength(analyses)
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'analyses': analyses,
            'primary_bias': self.state.primary_bias.value,
            'bias_strength': self.state.bias_strength
        }
    
    def generate_signals(
        self,
        symbol: str,
        current_price: float
    ) -> List[TradeSetup]:
        """
        Generate trading signals based on current analysis.
        
        Args:
            symbol: Trading symbol
            current_price: Current market price
            
        Returns:
            List of trade setups
        """
        if not self.state.timeframe_analyses:
            logger.warning("No analysis available. Run analyze_market first.")
            return []
            
        setups = []
        
        # Check both directions
        for bias in ['bullish', 'bearish']:
            # Analyze confluence
            confluence = self.confluence_analyzer.analyze_confluence(
                self.state.timeframe_analyses,
                current_price,
                bias
            )
            confluence.symbol = symbol
            
            # Check minimum grade
            if self._grade_to_int(confluence.grade) < self._grade_to_int(self.config['min_grade_for_signal']):
                continue
                
            # Generate setup
            setup = self.setup_generator.generate_setup(
                confluence,
                self.state.timeframe_analyses,
                current_price,
                self.account_size,
                symbol
            )
            
            if setup and self._validate_setup(setup):
                setups.append(setup)
                self.state.signals_generated += 1
                
                # Record for AI learning
                if self.mode != IntegrationMode.BACKTEST:
                    self.ai_engine.record_setup(setup, {
                        'price': current_price,
                        'time': datetime.now()
                    })
                    
        self.state.active_setups = setups
        return setups
    
    def get_ai_prediction(
        self,
        setup: TradeSetup
    ) -> Dict[str, Any]:
        """
        Get AI prediction for a setup.
        
        Args:
            setup: Trade setup to predict
            
        Returns:
            AI prediction result
        """
        return self.ai_engine.predict_outcome(setup, {
            'time': datetime.now(),
            'bias': self.state.primary_bias.value
        })
    
    def record_trade_outcome(
        self,
        record_id: str,
        outcome: str,
        pnl: float,
        mae: float,
        mfe: float,
        time_in_trade: float,
        entry_timing: float = 50.0,
        exit_timing: float = 50.0
    ):
        """
        Record the outcome of a trade for AI learning.
        
        Args:
            record_id: Trade record ID
            outcome: Trade outcome (win/loss/breakeven)
            pnl: Profit/loss
            mae: Maximum adverse excursion
            mfe: Maximum favorable excursion
            time_in_trade: Time in trade (minutes)
            entry_timing: Entry timing score
            exit_timing: Exit timing score
        """
        self.ai_engine.record_outcome(
            record_id, outcome, pnl, mae, mfe,
            time_in_trade, entry_timing, exit_timing
        )
        
        # Update win rate
        performance = self.ai_engine.get_model_performance()
        self.state.win_rate = performance.get('win_rate', 0.0)
    
    def get_trading_rules(self) -> Dict[str, Any]:
        """Get current trading rules from all components"""
        return {
            'configuration': self.config,
            'confluence_weights': ConfluenceAnalyzer.FACTOR_WEIGHTS,
            'grade_thresholds': {k.value: v for k, v in ConfluenceAnalyzer.GRADE_THRESHOLDS.items()},
            'ai_performance': self.ai_engine.get_model_performance(),
            'feature_importance': self.ai_engine.get_feature_importance()
        }
    
    def get_state(self) -> Dict[str, Any]:
        """Get current engine state"""
        return {
            'mode': self.state.mode.value,
            'last_update': self.state.last_update.isoformat(),
            'primary_bias': self.state.primary_bias.value,
            'bias_strength': self.state.bias_strength,
            'active_setups': len(self.state.active_setups),
            'signals_generated': self.state.signals_generated,
            'setups_triggered': self.state.setups_triggered,
            'win_rate': self.state.win_rate,
            'registered_handlers': list(self.handlers.keys())
        }
    
    def configure(self, **kwargs):
        """
        Update configuration.
        
        Args:
            **kwargs: Configuration parameters to update
        """
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value
                logger.info(f"Configuration updated: {key} = {value}")
    
    def reset(self):
        """Reset engine state"""
        self.state = IntegrationState(
            last_update=datetime.now(),
            mode=self.mode,
            timeframe_analyses={},
            primary_bias=TimeframeBias.NEUTRAL,
            bias_strength=0.0,
            active_setups=[],
            pending_signals=[],
            current_position=None,
            signals_generated=0,
            setups_triggered=0,
            win_rate=0.0,
            handler_states={}
        )
        logger.info("Engine state reset")
    
    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================
    
    def _analyze_timeframe(
        self,
        timeframe: str,
        df: pd.DataFrame
    ) -> TimeframeAnalysis:
        """Analyze a single timeframe"""
        current_price = df['close'].iloc[-1] if len(df) > 0 else 0
        
        # Use handlers if available, otherwise create placeholder
        structure = self._get_structure_analysis(df)
        obs = self._get_order_blocks(df)
        fvgs = self._get_fvgs(df)
        liquidity = self._get_liquidity(df)
        gaps = self._get_gaps(df)
        
        # Calculate swings
        last_high = df['high'].rolling(20).max().iloc[-1] if len(df) >= 20 else df['high'].max()
        last_low = df['low'].rolling(20).min().iloc[-1] if len(df) >= 20 else df['low'].min()
        
        # Determine zone
        mid = (last_high + last_low) / 2
        if current_price > mid + (last_high - mid) * 0.2:
            zone = 'premium'
        elif current_price < mid - (mid - last_low) * 0.2:
            zone = 'discount'
        else:
            zone = 'equilibrium'
            
        # Check for displacement
        displacement = self._check_displacement(df)
        
        return TimeframeAnalysis(
            timeframe=timeframe,
            timestamp=datetime.now(),
            trend=structure.get('trend', 'ranging'),
            structure_break=structure.get('break_type'),
            last_swing_high=last_high,
            last_swing_low=last_low,
            current_zone=zone,
            bullish_obs=obs.get('bullish', []),
            bearish_obs=obs.get('bearish', []),
            nearest_ob=None,
            bullish_fvgs=fvgs.get('bullish', []),
            bearish_fvgs=fvgs.get('bearish', []),
            nearest_fvg=None,
            buy_side_liquidity=liquidity.get('buy_side', []),
            sell_side_liquidity=liquidity.get('sell_side', []),
            nearest_liquidity=None,
            liquidity_swept=liquidity.get('swept', False),
            gaps=gaps.get('all', []),
            new_day_opening_gap=gaps.get('ndog'),
            new_week_opening_gap=gaps.get('nwog'),
            pd_arrays_above=[],
            pd_arrays_below=[],
            current_session=self._get_current_session(),
            kill_zone_active=self._is_kill_zone_active(),
            macro_time_active=self._is_macro_time(),
            displacement_present=displacement,
            clean_price_action=True,
            volume_confirmation=True
        )
    
    def _get_structure_analysis(self, df: pd.DataFrame) -> Dict:
        """Get market structure analysis"""
        if 'market_structure' in self.handlers:
            try:
                return self.handlers['market_structure'].analyze_structure(df)
            except Exception as e:
                logger.error(f"Structure analysis error: {e}")
                
        # Fallback: simple trend detection
        if len(df) < 20:
            return {'trend': 'ranging', 'break_type': None}
            
        sma = df['close'].rolling(20).mean()
        current = df['close'].iloc[-1]
        
        if current > sma.iloc[-1]:
            return {'trend': 'bullish', 'break_type': None}
        elif current < sma.iloc[-1]:
            return {'trend': 'bearish', 'break_type': None}
        return {'trend': 'ranging', 'break_type': None}
    
    def _get_order_blocks(self, df: pd.DataFrame) -> Dict:
        """Get order blocks"""
        if 'order_block' in self.handlers:
            try:
                result = self.handlers['order_block'].detect_order_blocks(df)
                return {
                    'bullish': [ob for ob in result if ob.get('type') == 'bullish'],
                    'bearish': [ob for ob in result if ob.get('type') == 'bearish']
                }
            except Exception as e:
                logger.error(f"Order block detection error: {e}")
                
        return {'bullish': [], 'bearish': []}
    
    def _get_fvgs(self, df: pd.DataFrame) -> Dict:
        """Get fair value gaps"""
        if 'fvg' in self.handlers:
            try:
                result = self.handlers['fvg'].detect_fvgs(df)
                return {
                    'bullish': [fvg for fvg in result if fvg.get('type') == 'bullish'],
                    'bearish': [fvg for fvg in result if fvg.get('type') == 'bearish']
                }
            except Exception as e:
                logger.error(f"FVG detection error: {e}")
                
        # Fallback: simple FVG detection
        bullish_fvgs = []
        bearish_fvgs = []
        
        for i in range(2, len(df)):
            # Bullish FVG: Gap between candle[i-2] high and candle[i] low
            if df['low'].iloc[i] > df['high'].iloc[i-2]:
                bullish_fvgs.append({
                    'type': 'bullish',
                    'high': df['low'].iloc[i],
                    'low': df['high'].iloc[i-2],
                    'ce': (df['low'].iloc[i] + df['high'].iloc[i-2]) / 2
                })
            # Bearish FVG: Gap between candle[i] high and candle[i-2] low
            elif df['high'].iloc[i] < df['low'].iloc[i-2]:
                bearish_fvgs.append({
                    'type': 'bearish',
                    'high': df['low'].iloc[i-2],
                    'low': df['high'].iloc[i],
                    'ce': (df['low'].iloc[i-2] + df['high'].iloc[i]) / 2
                })
                
        return {'bullish': bullish_fvgs[-5:], 'bearish': bearish_fvgs[-5:]}
    
    def _get_liquidity(self, df: pd.DataFrame) -> Dict:
        """Get liquidity pools"""
        if 'liquidity' in self.handlers:
            try:
                result = self.handlers['liquidity'].detect_liquidity_pools(df)
                return result
            except Exception as e:
                logger.error(f"Liquidity detection error: {e}")
                
        # Fallback: simple high/low detection
        return {
            'buy_side': [{'price': df['high'].max()}],
            'sell_side': [{'price': df['low'].min()}],
            'swept': False
        }
    
    def _get_gaps(self, df: pd.DataFrame) -> Dict:
        """Get gap analysis"""
        if 'gap' in self.handlers:
            try:
                return self.handlers['gap'].analyze(df)
            except Exception as e:
                logger.error(f"Gap analysis error: {e}")
                
        return {'all': [], 'ndog': None, 'nwog': None}
    
    def _check_displacement(self, df: pd.DataFrame, lookback: int = 3) -> bool:
        """Check for displacement (large move)"""
        if len(df) < lookback + 1:
            return False
            
        # Calculate ATR for comparison
        ranges = df['high'] - df['low']
        avg_range = ranges.rolling(20).mean().iloc[-1] if len(df) >= 20 else ranges.mean()
        
        # Check recent candles for displacement
        for i in range(-lookback, 0):
            candle_range = df['high'].iloc[i] - df['low'].iloc[i]
            if candle_range > avg_range * 2:
                return True
                
        return False
    
    def _get_current_session(self) -> str:
        """Get current trading session"""
        now = datetime.now()
        hour = now.hour
        
        # Simplified session detection (EST)
        if 0 <= hour < 5:
            return 'asian'
        elif 5 <= hour < 12:
            return 'london'
        elif 12 <= hour < 17:
            return 'new_york'
        else:
            return 'asian'
    
    def _is_kill_zone_active(self) -> bool:
        """Check if in kill zone"""
        now = datetime.now()
        hour = now.hour
        
        # NY Open Kill Zone: 7:00-10:00 EST
        if 7 <= hour < 10:
            return True
        # London Kill Zone: 2:00-5:00 EST
        if 2 <= hour < 5:
            return True
            
        return False
    
    def _is_macro_time(self) -> bool:
        """Check if in macro time window"""
        now = datetime.now()
        minute = now.minute
        
        # Macro times around :00, :30, :50
        macro_minutes = [0, 30, 50]
        return any(abs(minute - m) <= 3 for m in macro_minutes)
    
    def _determine_primary_bias(
        self,
        analyses: Dict[str, TimeframeAnalysis]
    ) -> TimeframeBias:
        """Determine primary bias from all analyses"""
        if not analyses:
            return TimeframeBias.NEUTRAL
            
        bullish_count = sum(1 for a in analyses.values() if a.trend == 'bullish')
        bearish_count = sum(1 for a in analyses.values() if a.trend == 'bearish')
        
        total = len(analyses)
        
        if bullish_count / total >= 0.8:
            return TimeframeBias.STRONGLY_BULLISH
        elif bullish_count / total >= 0.6:
            return TimeframeBias.BULLISH
        elif bearish_count / total >= 0.8:
            return TimeframeBias.STRONGLY_BEARISH
        elif bearish_count / total >= 0.6:
            return TimeframeBias.BEARISH
        return TimeframeBias.NEUTRAL
    
    def _calculate_bias_strength(
        self,
        analyses: Dict[str, TimeframeAnalysis]
    ) -> float:
        """Calculate bias strength 0-100"""
        if not analyses:
            return 0.0
            
        aligned_count = 0
        total = len(analyses)
        
        primary = self.state.primary_bias
        bullish_biases = [TimeframeBias.STRONGLY_BULLISH, TimeframeBias.BULLISH]
        bearish_biases = [TimeframeBias.STRONGLY_BEARISH, TimeframeBias.BEARISH]
        
        for analysis in analyses.values():
            if primary in bullish_biases and analysis.trend == 'bullish':
                aligned_count += 1
            elif primary in bearish_biases and analysis.trend == 'bearish':
                aligned_count += 1
                
        return (aligned_count / total) * 100 if total > 0 else 0.0
    
    def _validate_setup(self, setup: TradeSetup) -> bool:
        """Validate a trade setup against configuration"""
        # Check minimum R:R
        if setup.risk_reward_1 < self.config['min_rr_ratio']:
            return False
            
        # Check kill zone requirement
        if self.config['require_kill_zone'] and not self._is_kill_zone_active():
            if setup.grade not in [SetupGrade.A_PLUS]:
                return False
                
        # Check session filter
        session = self._get_current_session()
        if session not in self.config['session_filter']:
            if setup.grade not in [SetupGrade.A_PLUS, SetupGrade.A]:
                return False
                
        return True
    
    def _grade_to_int(self, grade: SetupGrade) -> int:
        """Convert grade to integer for comparison"""
        mapping = {
            SetupGrade.A_PLUS: 5,
            SetupGrade.A: 4,
            SetupGrade.B: 3,
            SetupGrade.C: 2,
            SetupGrade.INVALID: 0
        }
        return mapping.get(grade, 0)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_integration_engine(
    mode: str = 'analysis',
    risk_per_trade: float = 0.01,
    account_size: float = 100000.0,
    handlers: Dict[str, Any] = None
) -> ICTIntegrationEngine:
    """
    Factory function to create an integration engine.
    
    Args:
        mode: Operating mode ('analysis', 'signal', 'backtest', 'live', 'paper')
        risk_per_trade: Risk percentage per trade
        account_size: Trading account size
        handlers: Dictionary of handlers to register
        
    Returns:
        Configured ICTIntegrationEngine instance
    """
    mode_map = {
        'analysis': IntegrationMode.ANALYSIS,
        'signal': IntegrationMode.SIGNAL,
        'backtest': IntegrationMode.BACKTEST,
        'live': IntegrationMode.LIVE,
        'paper': IntegrationMode.PAPER
    }
    
    engine = ICTIntegrationEngine(
        mode=mode_map.get(mode, IntegrationMode.ANALYSIS),
        risk_per_trade=risk_per_trade,
        account_size=account_size
    )
    
    if handlers:
        for name, handler in handlers.items():
            engine.register_handler(name, handler)
            
    return engine


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Create engine
    engine = create_integration_engine(
        mode='signal',
        risk_per_trade=0.01,
        account_size=100000
    )
    
    # Configure
    engine.configure(
        min_grade_for_signal=SetupGrade.B,
        require_kill_zone=True,
        min_rr_ratio=2.0
    )
    
    print("ICT Integration Engine Configuration:")
    print(json.dumps(engine.get_state(), indent=2, default=str))
    
    print("\nTrading Rules:")
    rules = engine.get_trading_rules()
    print(f"Confluence Weights: {len(rules['confluence_weights'])} factors")
    print(f"Grade Thresholds: {rules['grade_thresholds']}")
