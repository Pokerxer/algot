"""
ICT Multi-Timeframe Coordinator
================================

Manages multi-timeframe analysis following ICT principles:
- "Higher timeframe sets the bias"
- "Lower timeframe for entries"
- "Structure alignment across timeframes"
- "Premium/discount relative to each timeframe's dealing range"

TIMEFRAME HIERARCHY (ICT):
- Monthly/Weekly: Directional bias, major liquidity pools
- Daily: Trade idea formation, key levels
- 4H/1H: Intermediate structure confirmation
- 15m/5m: Entry refinement
- 1m: Precision entries (optional)

Author: ICT Signal Engine
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from enum import Enum
from datetime import datetime, timedelta
from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class TimeframePurpose(Enum):
    """Purpose of each timeframe in ICT methodology"""
    DIRECTIONAL = "directional"      # Monthly, Weekly - sets bias
    NARRATIVE = "narrative"          # Daily - forms trade idea
    CONFIRMATION = "confirmation"    # 4H, 1H - confirms structure
    ENTRY = "entry"                  # 15m, 5m - entry refinement
    PRECISION = "precision"          # 1m - ultra-precise entries


class TimeframeRelation(Enum):
    """Relationship between timeframes"""
    ALIGNED = "aligned"              # Same direction
    CONFLICTING = "conflicting"      # Opposite direction
    NEUTRAL = "neutral"              # No clear direction


class AnalysisStatus(Enum):
    """Status of timeframe analysis"""
    CURRENT = "current"              # Up to date
    STALE = "stale"                  # Needs refresh
    ERROR = "error"                  # Analysis failed
    PENDING = "pending"              # Analysis in progress


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TimeframeConfig:
    """Configuration for a timeframe"""
    name: str                        # e.g., '15m', '1H', 'D'
    minutes: int                     # Duration in minutes
    purpose: TimeframePurpose
    weight: float                    # Importance weight (0-1)
    lookback_bars: int               # How many bars to analyze
    refresh_interval: int            # Seconds between refreshes
    parent_tf: Optional[str]         # Higher timeframe
    child_tf: Optional[str]          # Lower timeframe
    
    # Analysis requirements
    requires_structure: bool = True
    requires_liquidity: bool = True
    requires_pd_arrays: bool = True
    requires_time_context: bool = False


@dataclass
class TimeframeState:
    """Current state of a timeframe"""
    config: TimeframeConfig
    last_analysis: Optional[datetime]
    status: AnalysisStatus
    current_data: Optional[pd.DataFrame]
    analysis_result: Optional[Dict]
    
    # Quick access fields
    trend: str = 'neutral'
    zone: str = 'equilibrium'
    has_structure_break: bool = False
    has_displacement: bool = False
    liquidity_swept: bool = False
    
    def is_stale(self) -> bool:
        """Check if analysis is stale"""
        if self.last_analysis is None:
            return True
        age = (datetime.now() - self.last_analysis).total_seconds()
        return age > self.config.refresh_interval


@dataclass
class MTFAnalysis:
    """Multi-timeframe analysis result"""
    timestamp: datetime
    symbol: str
    
    # Timeframe states
    timeframe_states: Dict[str, TimeframeState]
    
    # Alignment analysis
    overall_bias: str               # bullish, bearish, neutral
    bias_confidence: float          # 0-100
    alignment_score: float          # 0-100
    
    # Key levels from all timeframes
    htf_swing_high: float
    htf_swing_low: float
    htf_premium_zone: Tuple[float, float]
    htf_discount_zone: Tuple[float, float]
    
    # Draw on liquidity
    primary_draw: float
    secondary_draw: float
    
    # Conflicts
    conflicts: List[str]
    warnings: List[str]
    
    # Recommendations
    recommended_direction: Optional[str]
    wait_for_confirmation: bool
    invalidation_level: float


@dataclass
class TimeframeTransition:
    """Tracks transitions between timeframe analyses"""
    from_tf: str
    to_tf: str
    bias_change: bool
    structure_change: bool
    new_pd_arrays: int
    timestamp: datetime


# =============================================================================
# TIMEFRAME COORDINATOR
# =============================================================================

class MTFCoordinator:
    """
    Multi-Timeframe Coordinator
    
    ICT: "The monthly chart tells you where price is going.
    The weekly tells you how it's getting there.
    The daily gives you the trade idea.
    The 4H confirms.
    The 1H and below gives you entries."
    """
    
    # Default timeframe configurations
    DEFAULT_CONFIGS = {
        'W': TimeframeConfig(
            name='W',
            minutes=10080,
            purpose=TimeframePurpose.DIRECTIONAL,
            weight=1.0,
            lookback_bars=52,
            refresh_interval=86400,  # Daily refresh
            parent_tf=None,
            child_tf='D'
        ),
        'D': TimeframeConfig(
            name='D',
            minutes=1440,
            purpose=TimeframePurpose.NARRATIVE,
            weight=0.9,
            lookback_bars=90,
            refresh_interval=3600,   # Hourly refresh
            parent_tf='W',
            child_tf='4H'
        ),
        '4H': TimeframeConfig(
            name='4H',
            minutes=240,
            purpose=TimeframePurpose.CONFIRMATION,
            weight=0.8,
            lookback_bars=60,
            refresh_interval=900,    # 15min refresh
            parent_tf='D',
            child_tf='1H'
        ),
        '1H': TimeframeConfig(
            name='1H',
            minutes=60,
            purpose=TimeframePurpose.CONFIRMATION,
            weight=0.7,
            lookback_bars=100,
            refresh_interval=300,    # 5min refresh
            parent_tf='4H',
            child_tf='15m'
        ),
        '15m': TimeframeConfig(
            name='15m',
            minutes=15,
            purpose=TimeframePurpose.ENTRY,
            weight=0.6,
            lookback_bars=100,
            refresh_interval=60,     # 1min refresh
            parent_tf='1H',
            child_tf='5m',
            requires_time_context=True
        ),
        '5m': TimeframeConfig(
            name='5m',
            minutes=5,
            purpose=TimeframePurpose.ENTRY,
            weight=0.5,
            lookback_bars=100,
            refresh_interval=30,     # 30sec refresh
            parent_tf='15m',
            child_tf='1m',
            requires_time_context=True
        ),
        '1m': TimeframeConfig(
            name='1m',
            minutes=1,
            purpose=TimeframePurpose.PRECISION,
            weight=0.3,
            lookback_bars=200,
            refresh_interval=10,     # 10sec refresh
            parent_tf='5m',
            child_tf=None,
            requires_time_context=True
        )
    }
    
    def __init__(
        self,
        active_timeframes: List[str] = None,
        custom_configs: Dict[str, TimeframeConfig] = None
    ):
        """
        Initialize the coordinator.
        
        Args:
            active_timeframes: List of timeframes to use
            custom_configs: Custom timeframe configurations
        """
        # Merge configs
        self.configs = self.DEFAULT_CONFIGS.copy()
        if custom_configs:
            self.configs.update(custom_configs)
            
        # Set active timeframes
        if active_timeframes:
            self.active_timeframes = active_timeframes
        else:
            self.active_timeframes = ['D', '4H', '1H', '15m', '5m']
            
        # Initialize states
        self.states: Dict[str, TimeframeState] = {}
        for tf in self.active_timeframes:
            if tf in self.configs:
                self.states[tf] = TimeframeState(
                    config=self.configs[tf],
                    last_analysis=None,
                    status=AnalysisStatus.PENDING,
                    current_data=None,
                    analysis_result=None
                )
                
        # Transition history
        self.transitions: List[TimeframeTransition] = []
        
        # Analysis callbacks
        self.analysis_callbacks: Dict[str, Callable] = {}
        
        logger.info(f"MTF Coordinator initialized with timeframes: {self.active_timeframes}")
    
    def update_data(
        self,
        timeframe: str,
        data: pd.DataFrame
    ):
        """
        Update data for a timeframe.
        
        Args:
            timeframe: Timeframe to update
            data: New OHLCV data
        """
        if timeframe not in self.states:
            logger.warning(f"Timeframe {timeframe} not active")
            return
            
        self.states[timeframe].current_data = data
        self.states[timeframe].status = AnalysisStatus.PENDING
    
    def analyze_timeframe(
        self,
        timeframe: str,
        analyzer_func: Callable = None
    ) -> Optional[Dict]:
        """
        Analyze a single timeframe.
        
        Args:
            timeframe: Timeframe to analyze
            analyzer_func: Custom analysis function
            
        Returns:
            Analysis result dictionary
        """
        if timeframe not in self.states:
            return None
            
        state = self.states[timeframe]
        
        if state.current_data is None or state.current_data.empty:
            state.status = AnalysisStatus.ERROR
            return None
            
        try:
            # Use custom analyzer or default
            if analyzer_func:
                result = analyzer_func(state.current_data)
            elif timeframe in self.analysis_callbacks:
                result = self.analysis_callbacks[timeframe](state.current_data)
            else:
                result = self._default_analysis(state.current_data, state.config)
                
            # Update state
            state.analysis_result = result
            state.last_analysis = datetime.now()
            state.status = AnalysisStatus.CURRENT
            
            # Update quick access fields
            state.trend = result.get('trend', 'neutral')
            state.zone = result.get('zone', 'equilibrium')
            state.has_structure_break = result.get('structure_break', False)
            state.has_displacement = result.get('displacement', False)
            state.liquidity_swept = result.get('liquidity_swept', False)
            
            return result
            
        except Exception as e:
            logger.error(f"Analysis error for {timeframe}: {e}")
            state.status = AnalysisStatus.ERROR
            return None
    
    def analyze_all(
        self,
        data: Dict[str, pd.DataFrame],
        force: bool = False
    ) -> MTFAnalysis:
        """
        Analyze all active timeframes.
        
        Args:
            data: Dictionary of DataFrames by timeframe
            force: Force refresh even if not stale
            
        Returns:
            Complete MTF analysis
        """
        # Update data and analyze each timeframe
        for tf in self.active_timeframes:
            if tf in data:
                self.update_data(tf, data[tf])
                
            state = self.states.get(tf)
            if state and (force or state.is_stale()):
                self.analyze_timeframe(tf)
                
        # Build MTF analysis
        return self._build_mtf_analysis()
    
    def get_timeframe_chain(
        self,
        from_tf: str,
        to_tf: str
    ) -> List[str]:
        """
        Get the chain of timeframes between two levels.
        
        Args:
            from_tf: Starting timeframe (higher)
            to_tf: Ending timeframe (lower)
            
        Returns:
            List of timeframes in order
        """
        chain = []
        current = from_tf
        
        while current and current != to_tf:
            chain.append(current)
            config = self.configs.get(current)
            if config:
                current = config.child_tf
            else:
                break
                
        if current == to_tf:
            chain.append(to_tf)
            
        return chain
    
    def check_alignment(
        self,
        timeframes: List[str] = None
    ) -> Tuple[bool, List[str]]:
        """
        Check if timeframes are aligned in direction.
        
        Args:
            timeframes: Timeframes to check (default: all)
            
        Returns:
            Tuple of (is_aligned, list of conflicts)
        """
        if timeframes is None:
            timeframes = self.active_timeframes
            
        trends = []
        for tf in timeframes:
            state = self.states.get(tf)
            if state and state.trend != 'neutral':
                trends.append((tf, state.trend))
                
        if not trends:
            return True, []
            
        # Check if all same direction
        first_trend = trends[0][1]
        conflicts = []
        
        for tf, trend in trends[1:]:
            if trend != first_trend:
                conflicts.append(f"{tf} ({trend}) vs {trends[0][0]} ({first_trend})")
                
        return len(conflicts) == 0, conflicts
    
    def get_htf_bias(self) -> Tuple[str, float]:
        """
        Get bias from higher timeframes.
        
        Returns:
            Tuple of (bias direction, confidence)
        """
        htf_timeframes = [
            tf for tf in self.active_timeframes
            if self.configs.get(tf) and 
            self.configs[tf].purpose in [TimeframePurpose.DIRECTIONAL, TimeframePurpose.NARRATIVE]
        ]
        
        if not htf_timeframes:
            htf_timeframes = self.active_timeframes[:2] if len(self.active_timeframes) >= 2 else self.active_timeframes
            
        bullish_count = 0
        bearish_count = 0
        total_weight = 0
        
        for tf in htf_timeframes:
            state = self.states.get(tf)
            if state:
                weight = self.configs[tf].weight
                total_weight += weight
                
                if state.trend == 'bullish':
                    bullish_count += weight
                elif state.trend == 'bearish':
                    bearish_count += weight
                    
        if total_weight == 0:
            return 'neutral', 0.0
            
        if bullish_count > bearish_count:
            confidence = (bullish_count / total_weight) * 100
            return 'bullish', confidence
        elif bearish_count > bullish_count:
            confidence = (bearish_count / total_weight) * 100
            return 'bearish', confidence
        else:
            return 'neutral', 50.0
    
    def get_ltf_setup(
        self,
        bias: str
    ) -> Dict[str, Any]:
        """
        Get setup information from lower timeframes.
        
        Args:
            bias: Expected direction
            
        Returns:
            Setup information dictionary
        """
        ltf_timeframes = [
            tf for tf in self.active_timeframes
            if self.configs.get(tf) and 
            self.configs[tf].purpose in [TimeframePurpose.ENTRY, TimeframePurpose.PRECISION]
        ]
        
        setup = {
            'aligned': False,
            'structure_break': False,
            'displacement': False,
            'entry_tf': None,
            'pd_arrays': [],
            'liquidity_swept': False
        }
        
        for tf in ltf_timeframes:
            state = self.states.get(tf)
            if state:
                if state.trend == bias:
                    setup['aligned'] = True
                    
                if state.has_structure_break:
                    setup['structure_break'] = True
                    
                if state.has_displacement:
                    setup['displacement'] = True
                    
                if state.liquidity_swept:
                    setup['liquidity_swept'] = True
                    
                # Get PD arrays
                if state.analysis_result:
                    pd_arrays = state.analysis_result.get('pd_arrays', [])
                    for pda in pd_arrays:
                        pda['timeframe'] = tf
                        setup['pd_arrays'].append(pda)
                        
        # Determine entry timeframe
        if setup['aligned'] and setup['structure_break']:
            for tf in reversed(ltf_timeframes):
                state = self.states.get(tf)
                if state and state.trend == bias:
                    setup['entry_tf'] = tf
                    break
                    
        return setup
    
    def get_invalidation_level(self, bias: str) -> float:
        """
        Get invalidation level based on structure.
        
        Args:
            bias: Current bias direction
            
        Returns:
            Invalidation price level
        """
        for tf in self.active_timeframes:
            state = self.states.get(tf)
            if state and state.analysis_result:
                result = state.analysis_result
                
                if bias == 'bullish':
                    # Invalidation below recent swing low
                    swing_low = result.get('swing_low', 0)
                    if swing_low > 0:
                        return swing_low
                else:
                    # Invalidation above recent swing high
                    swing_high = result.get('swing_high', float('inf'))
                    if swing_high < float('inf'):
                        return swing_high
                        
        return 0.0
    
    def register_analyzer(
        self,
        timeframe: str,
        analyzer_func: Callable
    ):
        """
        Register a custom analyzer for a timeframe.
        
        Args:
            timeframe: Timeframe to register for
            analyzer_func: Analysis function
        """
        self.analysis_callbacks[timeframe] = analyzer_func
    
    def get_state_summary(self) -> Dict[str, Dict]:
        """Get summary of all timeframe states"""
        summary = {}
        
        for tf, state in self.states.items():
            summary[tf] = {
                'status': state.status.value,
                'trend': state.trend,
                'zone': state.zone,
                'structure_break': state.has_structure_break,
                'displacement': state.has_displacement,
                'liquidity_swept': state.liquidity_swept,
                'last_update': state.last_analysis.isoformat() if state.last_analysis else None,
                'is_stale': state.is_stale()
            }
            
        return summary
    
    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================
    
    def _default_analysis(
        self,
        df: pd.DataFrame,
        config: TimeframeConfig
    ) -> Dict:
        """Default timeframe analysis"""
        if df is None or df.empty:
            return {'trend': 'neutral', 'zone': 'equilibrium'}
            
        result = {}
        
        # Calculate trend
        if len(df) >= 20:
            sma = df['close'].rolling(20).mean().iloc[-1]
            current = df['close'].iloc[-1]
            
            if current > sma * 1.005:
                result['trend'] = 'bullish'
            elif current < sma * 0.995:
                result['trend'] = 'bearish'
            else:
                result['trend'] = 'neutral'
        else:
            result['trend'] = 'neutral'
            
        # Calculate swing levels
        lookback = min(20, len(df))
        result['swing_high'] = df['high'].tail(lookback).max()
        result['swing_low'] = df['low'].tail(lookback).min()
        
        # Calculate zone
        mid = (result['swing_high'] + result['swing_low']) / 2
        current = df['close'].iloc[-1]
        
        if current > mid + (result['swing_high'] - mid) * 0.3:
            result['zone'] = 'premium'
        elif current < mid - (mid - result['swing_low']) * 0.3:
            result['zone'] = 'discount'
        else:
            result['zone'] = 'equilibrium'
            
        # Check for structure break (simple version)
        if len(df) >= 3:
            recent_high = df['high'].tail(3).max()
            recent_low = df['low'].tail(3).min()
            prior_high = df['high'].iloc[-4:-1].max() if len(df) >= 4 else recent_high
            prior_low = df['low'].iloc[-4:-1].min() if len(df) >= 4 else recent_low
            
            result['structure_break'] = (recent_high > prior_high) or (recent_low < prior_low)
        else:
            result['structure_break'] = False
            
        # Check for displacement
        if len(df) >= 2:
            last_range = df['high'].iloc[-1] - df['low'].iloc[-1]
            avg_range = (df['high'] - df['low']).tail(20).mean() if len(df) >= 20 else last_range
            result['displacement'] = last_range > avg_range * 1.5
        else:
            result['displacement'] = False
            
        result['liquidity_swept'] = False  # Needs proper liquidity handler
        result['pd_arrays'] = []  # Needs proper PD array detection
        
        return result
    
    def _build_mtf_analysis(self) -> MTFAnalysis:
        """Build complete MTF analysis from current states"""
        # Get HTF bias
        bias, confidence = self.get_htf_bias()
        
        # Check alignment
        is_aligned, conflicts = self.check_alignment()
        
        # Calculate alignment score
        alignment_score = 100.0 if is_aligned else max(0, 100 - len(conflicts) * 20)
        
        # Get key levels from HTF
        htf_high = 0.0
        htf_low = float('inf')
        
        for tf in self.active_timeframes[:2]:  # First two are HTF
            state = self.states.get(tf)
            if state and state.analysis_result:
                htf_high = max(htf_high, state.analysis_result.get('swing_high', 0))
                htf_low = min(htf_low, state.analysis_result.get('swing_low', float('inf')))
                
        if htf_low == float('inf'):
            htf_low = 0.0
            
        # Calculate premium/discount zones
        mid = (htf_high + htf_low) / 2
        premium_zone = (mid + (htf_high - mid) * 0.3, htf_high)
        discount_zone = (htf_low, mid - (mid - htf_low) * 0.3)
        
        # Determine draws on liquidity
        primary_draw = htf_high if bias == 'bullish' else htf_low
        secondary_draw = htf_low if bias == 'bullish' else htf_high
        
        # Build warnings
        warnings = []
        if not is_aligned:
            warnings.append("Timeframes not aligned")
            
        for tf, state in self.states.items():
            if state.is_stale():
                warnings.append(f"{tf} analysis is stale")
                
        # Get invalidation level
        invalidation = self.get_invalidation_level(bias)
        
        # Determine if confirmation needed
        wait_for_confirmation = not is_aligned or confidence < 60
        
        return MTFAnalysis(
            timestamp=datetime.now(),
            symbol="",  # Set by caller
            timeframe_states=self.states,
            overall_bias=bias,
            bias_confidence=confidence,
            alignment_score=alignment_score,
            htf_swing_high=htf_high,
            htf_swing_low=htf_low,
            htf_premium_zone=premium_zone,
            htf_discount_zone=discount_zone,
            primary_draw=primary_draw,
            secondary_draw=secondary_draw,
            conflicts=conflicts,
            warnings=warnings,
            recommended_direction=bias if bias != 'neutral' else None,
            wait_for_confirmation=wait_for_confirmation,
            invalidation_level=invalidation
        )


# =============================================================================
# TIMEFRAME SYNCHRONIZER
# =============================================================================

class TimeframeSynchronizer:
    """
    Synchronizes data across timeframes.
    
    ICT: "You need to see the same story across timeframes.
    When HTF and LTF tell the same story, that's when you trade."
    """
    
    def __init__(self, coordinator: MTFCoordinator):
        self.coordinator = coordinator
        self.sync_history: List[Dict] = []
        
    def synchronize(
        self,
        data: Dict[str, pd.DataFrame],
        reference_tf: str = None
    ) -> Dict[str, pd.DataFrame]:
        """
        Synchronize data across timeframes.
        
        Args:
            data: Raw data by timeframe
            reference_tf: Reference timeframe for alignment
            
        Returns:
            Synchronized data
        """
        if not reference_tf:
            # Use lowest timeframe as reference
            for tf in reversed(self.coordinator.active_timeframes):
                if tf in data:
                    reference_tf = tf
                    break
                    
        if reference_tf not in data:
            return data
            
        ref_data = data[reference_tf]
        if ref_data.empty:
            return data
            
        # Get reference time range
        ref_end = ref_data.index[-1] if hasattr(ref_data.index, '__getitem__') else datetime.now()
        
        synchronized = {}
        
        for tf, df in data.items():
            if df is None or df.empty:
                synchronized[tf] = df
                continue
                
            # Filter to matching time range
            try:
                if hasattr(df.index, '__getitem__'):
                    synchronized[tf] = df[df.index <= ref_end]
                else:
                    synchronized[tf] = df
            except Exception:
                synchronized[tf] = df
                
        # Record sync
        self.sync_history.append({
            'timestamp': datetime.now(),
            'reference_tf': reference_tf,
            'timeframes': list(synchronized.keys())
        })
        
        return synchronized
    
    def get_candle_alignment(
        self,
        htf: str,
        ltf: str,
        ltf_data: pd.DataFrame
    ) -> Dict[str, List[int]]:
        """
        Get which LTF candles align with each HTF candle.
        
        Args:
            htf: Higher timeframe
            ltf: Lower timeframe
            ltf_data: Lower timeframe data
            
        Returns:
            Mapping of HTF indices to LTF indices
        """
        htf_config = self.coordinator.configs.get(htf)
        ltf_config = self.coordinator.configs.get(ltf)
        
        if not htf_config or not ltf_config:
            return {}
            
        # Calculate candles per HTF period
        ratio = htf_config.minutes // ltf_config.minutes
        
        alignment = {}
        total_candles = len(ltf_data)
        
        htf_idx = 0
        for i in range(0, total_candles, ratio):
            end_idx = min(i + ratio, total_candles)
            alignment[htf_idx] = list(range(i, end_idx))
            htf_idx += 1
            
        return alignment


# =============================================================================
# CONFLUENCE MATRIX
# =============================================================================

class ConfluenceMatrix:
    """
    Builds a confluence matrix across timeframes.
    
    Shows where multiple timeframes have PD arrays at the same levels.
    """
    
    def __init__(self, coordinator: MTFCoordinator):
        self.coordinator = coordinator
        
    def build_matrix(
        self,
        price_range: Tuple[float, float],
        divisions: int = 20
    ) -> pd.DataFrame:
        """
        Build confluence matrix.
        
        Args:
            price_range: (low, high) price range
            divisions: Number of price divisions
            
        Returns:
            DataFrame with confluence scores
        """
        low, high = price_range
        step = (high - low) / divisions
        
        # Create price levels
        levels = [low + step * i for i in range(divisions + 1)]
        
        # Initialize matrix
        matrix = pd.DataFrame(index=levels, columns=self.coordinator.active_timeframes)
        matrix = matrix.fillna(0)
        
        # Fill matrix with PD array presence
        for tf in self.coordinator.active_timeframes:
            state = self.coordinator.states.get(tf)
            if not state or not state.analysis_result:
                continue
                
            result = state.analysis_result
            pd_arrays = result.get('pd_arrays', [])
            
            for pda in pd_arrays:
                pda_high = pda.get('high', 0)
                pda_low = pda.get('low', 0)
                
                # Mark levels where PD array exists
                for level in levels:
                    if pda_low <= level <= pda_high:
                        matrix.loc[level, tf] += 1
                        
        # Add confluence score column
        matrix['confluence'] = matrix.sum(axis=1)
        
        return matrix
    
    def get_high_confluence_zones(
        self,
        matrix: pd.DataFrame,
        min_confluence: int = 2
    ) -> List[Tuple[float, float, int]]:
        """
        Get zones with high confluence.
        
        Args:
            matrix: Confluence matrix
            min_confluence: Minimum confluence score
            
        Returns:
            List of (low, high, score) tuples
        """
        zones = []
        
        # Find contiguous zones
        in_zone = False
        zone_start = 0
        zone_score = 0
        
        levels = matrix.index.tolist()
        
        for i, level in enumerate(levels):
            score = matrix.loc[level, 'confluence']
            
            if score >= min_confluence:
                if not in_zone:
                    in_zone = True
                    zone_start = level
                    zone_score = score
                else:
                    zone_score = max(zone_score, score)
            else:
                if in_zone:
                    zones.append((zone_start, levels[i-1], zone_score))
                    in_zone = False
                    
        # Handle zone at end
        if in_zone:
            zones.append((zone_start, levels[-1], zone_score))
            
        return sorted(zones, key=lambda x: x[2], reverse=True)


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    # Create coordinator
    coordinator = MTFCoordinator(
        active_timeframes=['D', '4H', '1H', '15m', '5m']
    )
    
    # Create sample data
    import random
    
    def generate_sample_data(periods: int = 100) -> pd.DataFrame:
        base_price = 1.1000
        prices = []
        current = base_price
        
        for _ in range(periods):
            change = random.uniform(-0.002, 0.002)
            current += change
            high = current + random.uniform(0, 0.001)
            low = current - random.uniform(0, 0.001)
            open_p = current + random.uniform(-0.0005, 0.0005)
            close = current
            prices.append({
                'open': open_p,
                'high': high,
                'low': low,
                'close': close,
                'volume': random.randint(1000, 10000)
            })
            
        return pd.DataFrame(prices)
    
    # Generate data for each timeframe
    data = {
        'D': generate_sample_data(90),
        '4H': generate_sample_data(60),
        '1H': generate_sample_data(100),
        '15m': generate_sample_data(100),
        '5m': generate_sample_data(100)
    }
    
    # Analyze all timeframes
    analysis = coordinator.analyze_all(data)
    
    print("MTF Analysis Result:")
    print(f"  Overall Bias: {analysis.overall_bias}")
    print(f"  Bias Confidence: {analysis.bias_confidence:.1f}%")
    print(f"  Alignment Score: {analysis.alignment_score:.1f}")
    print(f"  Primary Draw: {analysis.primary_draw:.5f}")
    print(f"  Conflicts: {analysis.conflicts}")
    print(f"  Warnings: {analysis.warnings}")
    print(f"  Wait for Confirmation: {analysis.wait_for_confirmation}")
    
    print("\nTimeframe States:")
    for tf, summary in coordinator.get_state_summary().items():
        print(f"  {tf}: {summary['trend']} ({summary['zone']})")
