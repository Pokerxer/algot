"""
Comprehensive ICT Liquidity Handler
Based on Inner Circle Trader methodology and transcript teachings

Key ICT Concepts Implemented:
- Buy-Side Liquidity: "Above equal highs, buy stops resting"
- Sell-Side Liquidity: "Below equal lows, sell stops resting"
- Equal Highs/Lows: "Too clean, too neat" - obvious liquidity targets
- Liquidity Sweep: Running stops before reversal
- Stop Hunt: "Turtle soup" - quick sweep and reversal
- Inducement: Liquidity used to trap traders on wrong side
- Session Liquidity: Asian/London/NY highs and lows
- Primary vs Secondary: "What's the draw on liquidity?"
- Liquidity Void: Gap in price delivery
- Low Resistance Liquidity Run: "Very little give back"
- High Resistance Liquidity Run: "Back and forth price action"

ICT Quotes from transcripts:
- "What's above equal highs? Buy stops. It's too clean, too neat."
- "Everything above these relative equal highs was retail getting trapped"
- "Run on liquidity - that's what we're looking for"
- "Low resistance liquidity run - down close candles support price"
- "They drop it so retail thinks it's going lower, then run up to gather buy stops"

Author: ICT Signal Engine
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from datetime import datetime, time
import pandas as pd
import numpy as np


# ==================== ENUMS ====================

class LiquiditySide(Enum):
    """Side of liquidity (where stops are resting)"""
    BUY_SIDE = "buy_side"       # Above price - sell stops become buy orders when hit
    SELL_SIDE = "sell_side"    # Below price - buy stops become sell orders when hit


class LiquidityType(Enum):
    """Classification of liquidity pool"""
    EQUAL_HIGHS = "equal_highs"           # ICT: "Too clean, too neat"
    EQUAL_LOWS = "equal_lows"
    SWING_HIGH = "swing_high"
    SWING_LOW = "swing_low"
    RELATIVE_EQUAL_HIGHS = "rel_eq_highs"  # ICT term for close-but-not-exact
    RELATIVE_EQUAL_LOWS = "rel_eq_lows"
    SESSION_HIGH = "session_high"
    SESSION_LOW = "session_low"
    ASIAN_HIGH = "asian_high"
    ASIAN_LOW = "asian_low"
    LONDON_HIGH = "london_high"
    LONDON_LOW = "london_low"
    NY_HIGH = "ny_high"
    NY_LOW = "ny_low"
    PRE_MARKET_HIGH = "pre_market_high"    # ICT: Key level
    PRE_MARKET_LOW = "pre_market_low"
    DAILY_HIGH = "daily_high"
    DAILY_LOW = "daily_low"
    WEEKLY_HIGH = "weekly_high"
    WEEKLY_LOW = "weekly_low"
    MONTHLY_HIGH = "monthly_high"
    MONTHLY_LOW = "monthly_low"
    PREVIOUS_DAY_HIGH = "pdh"
    PREVIOUS_DAY_LOW = "pdl"
    PREVIOUS_WEEK_HIGH = "pwh"
    PREVIOUS_WEEK_LOW = "pwl"
    OLD_HIGH = "old_high"
    OLD_LOW = "old_low"


class LiquidityStrength(Enum):
    """Strength/importance of liquidity pool"""
    MINOR = 1           # Single touch, small range
    MEDIUM = 2          # 2 touches, moderate range
    MAJOR = 3           # 3+ touches, significant range
    PRIMARY = 4         # HTF level, main draw on liquidity


class SweepType(Enum):
    """Type of liquidity sweep"""
    CLEAN_SWEEP = "clean_sweep"        # Full break and close beyond
    WICK_SWEEP = "wick_sweep"          # Wick through, body didn't close beyond
    PARTIAL_SWEEP = "partial_sweep"    # Some but not all equal levels taken
    STOP_HUNT = "stop_hunt"            # ICT "Turtle Soup" - quick sweep & reversal
    INDUCEMENT = "inducement"          # Used to trap traders
    FAILED_SWEEP = "failed_sweep"      # Approached but didn't take


class LiquidityRunType(Enum):
    """ICT Low/High Resistance Liquidity Run"""
    LOW_RESISTANCE = "low_resistance"   # ICT: "Very little give back, go go go"
    HIGH_RESISTANCE = "high_resistance"  # ICT: "Back and forth price action"
    NEUTRAL = "neutral"


# ==================== DATA CLASSES ====================

@dataclass
class LiquidityPool:
    """
    ICT Liquidity Pool
    
    ICT: "What's above equal highs? Buy stops."
    "It's too clean, too neat - price will gravitate there"
    """
    price: float
    side: LiquiditySide
    liquidity_type: LiquidityType
    strength: LiquidityStrength
    
    # Formation details
    timestamp: Optional[datetime] = None
    index: int = 0
    touched_count: int = 1              # Number of times price touched this level
    price_range: float = 0.0            # Range of the equal levels
    
    # Related levels (for equal highs/lows)
    related_indices: List[int] = field(default_factory=list)
    
    # Status
    is_swept: bool = False
    sweep_timestamp: Optional[datetime] = None
    sweep_index: Optional[int] = None
    sweep_type: Optional[SweepType] = None
    
    # Displacement after sweep
    displacement_after: float = 0.0
    displacement_pct: float = 0.0
    reversal_after_sweep: bool = False
    
    # ICT classification
    is_inducement: bool = False         # Used to trap traders
    is_primary: bool = False            # Primary draw on liquidity
    is_engineered: bool = False         # Deliberately created by smart money
    
    # Draw context
    is_draw_on_liquidity: bool = False  # Currently being targeted
    
    # Notes
    notes: List[str] = field(default_factory=list)
    
    def __str__(self):
        status = "SWEPT" if self.is_swept else "INTACT"
        primary = "[PRIMARY]" if self.is_primary else ""
        inducement = "[INDUCEMENT]" if self.is_inducement else ""
        return (f"{self.liquidity_type.value} @ {self.price:.5f} "
                f"({self.side.value}) {primary}{inducement} [{status}]")


@dataclass
class LiquidityVoid:
    """
    ICT Liquidity Void
    
    Gap in price delivery that needs to be filled.
    Similar to FVG but specifically about liquidity.
    """
    high: float
    low: float
    start_timestamp: Optional[datetime] = None
    end_timestamp: Optional[datetime] = None
    start_index: int = 0
    
    is_filled: bool = False
    fill_type: str = "none"     # "full", "partial", "ce" (consequent encroachment)
    fill_timestamp: Optional[datetime] = None
    
    @property
    def size(self) -> float:
        return self.high - self.low
    
    @property
    def midpoint(self) -> float:
        return (self.high + self.low) / 2
    
    def __str__(self):
        status = "FILLED" if self.is_filled else "OPEN"
        return f"Void {self.low:.5f} - {self.high:.5f} [{status}]"


@dataclass
class LiquiditySweepEvent:
    """
    ICT Liquidity Sweep Event
    
    When price runs through liquidity and potentially reverses.
    ICT: "Run on liquidity" followed by "displacement"
    """
    pool: LiquidityPool
    sweep_timestamp: Optional[datetime]
    sweep_index: int
    sweep_type: SweepType
    
    # Prices
    approach_price: float               # Price before sweep
    sweep_price: float                  # How far past liquidity
    post_sweep_price: float             # Price after sweep
    
    # Displacement
    displacement: float = 0.0
    displacement_pct: float = 0.0
    
    # Reversal tracking
    reversal_occurred: bool = False
    reversal_magnitude: float = 0.0
    candles_to_reversal: int = 0
    
    # FVG formation
    fvg_formed: bool = False            # ICT: Sweep + FVG = high probability
    
    def __str__(self):
        rev = "→ REVERSAL" if self.reversal_occurred else ""
        return f"{self.sweep_type.value} @ {self.sweep_price:.5f} {rev}"


@dataclass
class LiquidityRunAnalysis:
    """
    ICT Liquidity Run Type Analysis
    
    ICT: "Low resistance liquidity run - down close candles support price"
    "High resistance - back and forth, every FVG gets filled"
    """
    run_type: LiquidityRunType
    direction: str                      # 'bullish' or 'bearish'
    
    # Characteristics
    retracement_depth: float = 0.0      # How deep pullbacks are
    fvg_fill_rate: float = 0.0          # How many FVGs get filled
    candle_support_rate: float = 0.0    # How often same-direction candles support
    
    # Recommendations
    stop_management: str = ""           # ICT advice for this type
    position_sizing: str = ""


@dataclass
class LiquidityAnalysis:
    """Complete liquidity analysis result"""
    all_pools: List[LiquidityPool]
    buy_side_pools: List[LiquidityPool]
    sell_side_pools: List[LiquidityPool]
    intact_pools: List[LiquidityPool]
    swept_pools: List[LiquidityPool]
    
    primary_buy_side: Optional[LiquidityPool] = None
    primary_sell_side: Optional[LiquidityPool] = None
    
    sweep_events: List[LiquiditySweepEvent] = field(default_factory=list)
    stop_hunts: List[LiquiditySweepEvent] = field(default_factory=list)
    inducement_pools: List[LiquidityPool] = field(default_factory=list)
    
    liquidity_voids: List[LiquidityVoid] = field(default_factory=list)
    
    run_analysis: Optional[LiquidityRunAnalysis] = None
    
    # Draw on liquidity
    current_draw: Optional[LiquidityPool] = None


# ==================== MAIN HANDLER ====================

class LiquidityHandler:
    """
    Comprehensive ICT Liquidity Handler
    
    Implements all ICT liquidity concepts:
    - Equal highs/lows detection (relative equals)
    - Session liquidity (Asian/London/NY)
    - Swing liquidity
    - Liquidity sweep detection
    - Stop hunt (Turtle Soup) identification
    - Inducement tracking
    - Low/High resistance liquidity runs
    - Draw on liquidity targeting
    """
    
    def __init__(self,
                 equal_threshold_pips: float = 5.0,
                 min_touches: int = 2,
                 lookback_bars: int = 50,
                 displacement_threshold_pct: float = 0.3,
                 stop_hunt_reversal_pct: float = 0.5,
                 void_min_size_pips: float = 10.0):
        """
        Initialize Liquidity Handler
        
        Args:
            equal_threshold_pips: Tolerance for equal highs/lows
            min_touches: Min touches for valid liquidity pool
            lookback_bars: Bars to look back for liquidity
            displacement_threshold_pct: Min % move for strong displacement
            stop_hunt_reversal_pct: Min reversal % for stop hunt
            void_min_size_pips: Min void size to track
        """
        self.equal_threshold_pips = equal_threshold_pips
        self.min_touches = min_touches
        self.lookback_bars = lookback_bars
        self.displacement_threshold_pct = displacement_threshold_pct
        self.stop_hunt_reversal_pct = stop_hunt_reversal_pct
        self.void_min_size_pips = void_min_size_pips
        
        # Storage
        self.liquidity_pools: List[LiquidityPool] = []
        self.sweep_events: List[LiquiditySweepEvent] = []
        self.liquidity_voids: List[LiquidityVoid] = []
        
        # Pip value (auto-detected)
        self._pip_value: float = 0.0001
    
    # ==================== MAIN ANALYSIS ====================
    
    def analyze(self, df: pd.DataFrame) -> LiquidityAnalysis:
        """
        Complete liquidity analysis
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            LiquidityAnalysis with all results
        """
        # Reset
        self.liquidity_pools = []
        self.sweep_events = []
        self.liquidity_voids = []
        
        # Detect pip value
        self._pip_value = self._detect_pip_value(df)
        
        # 1. Identify all liquidity pools
        self._identify_equal_highs(df)
        self._identify_equal_lows(df)
        self._identify_swing_liquidity(df)
        self._identify_session_liquidity(df)
        
        # 2. Classify pool strength
        self._classify_strength()
        
        # 3. Identify primary liquidity (draw on liquidity)
        self._identify_primary_liquidity()
        
        # 4. Detect liquidity voids
        self._detect_voids(df)
        
        # 5. Detect sweeps
        self._detect_sweeps(df)
        
        # 6. Identify stop hunts
        self._identify_stop_hunts(df)
        
        # 7. Identify inducement
        self._identify_inducement()
        
        # 8. Analyze liquidity run type
        run_analysis = self._analyze_liquidity_run(df)
        
        # 9. Determine current draw
        current_draw = self._determine_draw(df)
        
        # Build result
        return self._build_analysis(run_analysis, current_draw)
    
    # ==================== EQUAL HIGHS/LOWS ====================
    
    def _identify_equal_highs(self, df: pd.DataFrame):
        """
        Identify equal highs (buy-side liquidity)
        
        ICT: "Too clean, too neat. What's above? Buy stops."
        "Relatively equal highs" - don't need to be exact
        """
        threshold = self.equal_threshold_pips * self._pip_value
        highs = df['high'].values
        
        for i in range(len(df) - 1):
            equal_indices = [i]
            base_high = highs[i]
            
            # Look forward for equal highs
            for j in range(i + 1, min(i + self.lookback_bars, len(df))):
                if abs(highs[j] - base_high) <= threshold:
                    equal_indices.append(j)
            
            # Need minimum touches
            if len(equal_indices) >= self.min_touches:
                # Check if not already captured
                if not self._pool_exists_near(base_high, LiquiditySide.BUY_SIDE):
                    # Use average of equal levels
                    avg_price = np.mean([highs[idx] for idx in equal_indices])
                    price_range = max(highs[idx] for idx in equal_indices) - min(highs[idx] for idx in equal_indices)
                    
                    timestamp = self._get_timestamp(df, equal_indices[0])
                    
                    pool = LiquidityPool(
                        price=avg_price,
                        side=LiquiditySide.BUY_SIDE,
                        liquidity_type=LiquidityType.EQUAL_HIGHS if price_range <= threshold else LiquidityType.RELATIVE_EQUAL_HIGHS,
                        strength=LiquidityStrength.MEDIUM,
                        timestamp=timestamp,
                        index=equal_indices[0],
                        touched_count=len(equal_indices),
                        price_range=price_range,
                        related_indices=equal_indices
                    )
                    
                    pool.notes.append(f"Equal highs: {len(equal_indices)} touches")
                    self.liquidity_pools.append(pool)
    
    def _identify_equal_lows(self, df: pd.DataFrame):
        """
        Identify equal lows (sell-side liquidity)
        
        ICT: "Sell stops below equal lows"
        """
        threshold = self.equal_threshold_pips * self._pip_value
        lows = df['low'].values
        
        for i in range(len(df) - 1):
            equal_indices = [i]
            base_low = lows[i]
            
            for j in range(i + 1, min(i + self.lookback_bars, len(df))):
                if abs(lows[j] - base_low) <= threshold:
                    equal_indices.append(j)
            
            if len(equal_indices) >= self.min_touches:
                if not self._pool_exists_near(base_low, LiquiditySide.SELL_SIDE):
                    avg_price = np.mean([lows[idx] for idx in equal_indices])
                    price_range = max(lows[idx] for idx in equal_indices) - min(lows[idx] for idx in equal_indices)
                    
                    timestamp = self._get_timestamp(df, equal_indices[0])
                    
                    pool = LiquidityPool(
                        price=avg_price,
                        side=LiquiditySide.SELL_SIDE,
                        liquidity_type=LiquidityType.EQUAL_LOWS if price_range <= threshold else LiquidityType.RELATIVE_EQUAL_LOWS,
                        strength=LiquidityStrength.MEDIUM,
                        timestamp=timestamp,
                        index=equal_indices[0],
                        touched_count=len(equal_indices),
                        price_range=price_range,
                        related_indices=equal_indices
                    )
                    
                    pool.notes.append(f"Equal lows: {len(equal_indices)} touches")
                    self.liquidity_pools.append(pool)
    
    # ==================== SWING LIQUIDITY ====================
    
    def _identify_swing_liquidity(self, df: pd.DataFrame, lookback: int = 5):
        """
        Identify swing high/low liquidity
        
        Single swing points that have liquidity above/below
        """
        for i in range(lookback, len(df) - lookback):
            # Swing high
            is_swing_high = True
            current_high = df.iloc[i]['high']
            for j in range(1, lookback + 1):
                if df.iloc[i - j]['high'] >= current_high or df.iloc[i + j]['high'] >= current_high:
                    is_swing_high = False
                    break
            
            if is_swing_high and not self._pool_exists_near(current_high, LiquiditySide.BUY_SIDE):
                pool = LiquidityPool(
                    price=current_high,
                    side=LiquiditySide.BUY_SIDE,
                    liquidity_type=LiquidityType.SWING_HIGH,
                    strength=LiquidityStrength.MINOR,
                    timestamp=self._get_timestamp(df, i),
                    index=i,
                    touched_count=1
                )
                self.liquidity_pools.append(pool)
            
            # Swing low
            is_swing_low = True
            current_low = df.iloc[i]['low']
            for j in range(1, lookback + 1):
                if df.iloc[i - j]['low'] <= current_low or df.iloc[i + j]['low'] <= current_low:
                    is_swing_low = False
                    break
            
            if is_swing_low and not self._pool_exists_near(current_low, LiquiditySide.SELL_SIDE):
                pool = LiquidityPool(
                    price=current_low,
                    side=LiquiditySide.SELL_SIDE,
                    liquidity_type=LiquidityType.SWING_LOW,
                    strength=LiquidityStrength.MINOR,
                    timestamp=self._get_timestamp(df, i),
                    index=i,
                    touched_count=1
                )
                self.liquidity_pools.append(pool)
    
    # ==================== SESSION LIQUIDITY ====================
    
    def _identify_session_liquidity(self, df: pd.DataFrame):
        """
        Identify session highs/lows
        
        ICT: Asian, London, NY session highs and lows are key liquidity
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            return
        
        # Group by date to find session extremes
        df_with_date = df.copy()
        df_with_date['date'] = df_with_date.index.date
        
        for date, group in df_with_date.groupby('date'):
            if len(group) < 5:
                continue
            
            # Daily high/low
            daily_high = group['high'].max()
            daily_low = group['low'].min()
            daily_high_idx = group['high'].idxmax()
            daily_low_idx = group['low'].idxmin()
            
            # Add daily high as buy-side liquidity
            if not self._pool_exists_near(daily_high, LiquiditySide.BUY_SIDE):
                pool = LiquidityPool(
                    price=daily_high,
                    side=LiquiditySide.BUY_SIDE,
                    liquidity_type=LiquidityType.DAILY_HIGH,
                    strength=LiquidityStrength.MAJOR,
                    timestamp=daily_high_idx,
                    index=df.index.get_loc(daily_high_idx) if daily_high_idx in df.index else 0
                )
                self.liquidity_pools.append(pool)
            
            # Add daily low as sell-side liquidity
            if not self._pool_exists_near(daily_low, LiquiditySide.SELL_SIDE):
                pool = LiquidityPool(
                    price=daily_low,
                    side=LiquiditySide.SELL_SIDE,
                    liquidity_type=LiquidityType.DAILY_LOW,
                    strength=LiquidityStrength.MAJOR,
                    timestamp=daily_low_idx,
                    index=df.index.get_loc(daily_low_idx) if daily_low_idx in df.index else 0
                )
                self.liquidity_pools.append(pool)
    
    # ==================== LIQUIDITY VOIDS ====================
    
    def _detect_voids(self, df: pd.DataFrame):
        """
        Detect liquidity voids (gaps in price delivery)
        """
        min_void = self.void_min_size_pips * self._pip_value
        
        for i in range(1, len(df)):
            prev_close = df.iloc[i - 1]['close']
            curr_open = df.iloc[i]['open']
            
            # Gap up
            if curr_open > prev_close:
                gap_size = curr_open - prev_close
                if gap_size >= min_void:
                    void = LiquidityVoid(
                        high=curr_open,
                        low=prev_close,
                        start_timestamp=self._get_timestamp(df, i - 1),
                        end_timestamp=self._get_timestamp(df, i),
                        start_index=i - 1
                    )
                    self.liquidity_voids.append(void)
            
            # Gap down
            elif curr_open < prev_close:
                gap_size = prev_close - curr_open
                if gap_size >= min_void:
                    void = LiquidityVoid(
                        high=prev_close,
                        low=curr_open,
                        start_timestamp=self._get_timestamp(df, i - 1),
                        end_timestamp=self._get_timestamp(df, i),
                        start_index=i - 1
                    )
                    self.liquidity_voids.append(void)
    
    # ==================== SWEEPS ====================
    
    def _detect_sweeps(self, df: pd.DataFrame):
        """
        Detect liquidity sweeps
        
        ICT: "Run on liquidity" - price takes out stops then reverses
        """
        for pool in self.liquidity_pools:
            if pool.is_swept:
                continue
            
            # Look for price taking out this pool
            for i in range(pool.index + 1, len(df)):
                candle = df.iloc[i]
                
                swept = False
                sweep_price = 0.0
                sweep_type = SweepType.WICK_SWEEP
                
                if pool.side == LiquiditySide.BUY_SIDE:
                    # Buy-side: check if high exceeded pool
                    if candle['high'] > pool.price:
                        swept = True
                        sweep_price = candle['high']
                        
                        # Check if body closed above (clean sweep) or just wick
                        body_high = max(candle['open'], candle['close'])
                        if body_high > pool.price:
                            sweep_type = SweepType.CLEAN_SWEEP
                        else:
                            sweep_type = SweepType.WICK_SWEEP
                
                else:  # SELL_SIDE
                    # Sell-side: check if low exceeded pool
                    if candle['low'] < pool.price:
                        swept = True
                        sweep_price = candle['low']
                        
                        body_low = min(candle['open'], candle['close'])
                        if body_low < pool.price:
                            sweep_type = SweepType.CLEAN_SWEEP
                        else:
                            sweep_type = SweepType.WICK_SWEEP
                
                if swept:
                    pool.is_swept = True
                    pool.sweep_index = i
                    pool.sweep_timestamp = self._get_timestamp(df, i)
                    pool.sweep_type = sweep_type
                    
                    # Calculate displacement after sweep
                    if i + 5 < len(df):
                        future = df.iloc[i:i + 10]
                        if pool.side == LiquiditySide.BUY_SIDE:
                            # After buy-side sweep, displacement is down
                            displacement = sweep_price - future['low'].min()
                        else:
                            # After sell-side sweep, displacement is up
                            displacement = future['high'].max() - sweep_price
                        
                        pool.displacement_after = displacement
                        pool.displacement_pct = (displacement / pool.price) * 100
                    
                    # Create sweep event
                    event = LiquiditySweepEvent(
                        pool=pool,
                        sweep_timestamp=pool.sweep_timestamp,
                        sweep_index=i,
                        sweep_type=sweep_type,
                        approach_price=df.iloc[i - 1]['close'] if i > 0 else pool.price,
                        sweep_price=sweep_price,
                        post_sweep_price=df.iloc[i]['close'],
                        displacement=pool.displacement_after,
                        displacement_pct=pool.displacement_pct
                    )
                    
                    self.sweep_events.append(event)
                    break
    
    # ==================== STOP HUNTS ====================
    
    def _identify_stop_hunts(self, df: pd.DataFrame):
        """
        Identify stop hunts (Turtle Soup)
        
        ICT: Quick sweep and immediate reversal
        "They run the stops then reverse"
        """
        for event in self.sweep_events:
            # Check for reversal after sweep
            if event.sweep_index + 5 >= len(df):
                continue
            
            future = df.iloc[event.sweep_index:event.sweep_index + 10]
            
            if event.pool.side == LiquiditySide.BUY_SIDE:
                # Buy-side sweep should reverse down for stop hunt
                reversal = event.sweep_price - future['low'].min()
                reversal_pct = (reversal / event.sweep_price) * 100
                
                if reversal_pct >= self.stop_hunt_reversal_pct:
                    event.sweep_type = SweepType.STOP_HUNT
                    event.reversal_occurred = True
                    event.reversal_magnitude = reversal
                    event.pool.sweep_type = SweepType.STOP_HUNT
                    event.pool.reversal_after_sweep = True
                    
                    # Find candles to reversal
                    reversal_idx = future['low'].idxmin()
                    event.candles_to_reversal = df.index.get_loc(reversal_idx) - event.sweep_index
            
            else:  # SELL_SIDE
                # Sell-side sweep should reverse up
                reversal = future['high'].max() - event.sweep_price
                reversal_pct = (reversal / event.sweep_price) * 100
                
                if reversal_pct >= self.stop_hunt_reversal_pct:
                    event.sweep_type = SweepType.STOP_HUNT
                    event.reversal_occurred = True
                    event.reversal_magnitude = reversal
                    event.pool.sweep_type = SweepType.STOP_HUNT
                    event.pool.reversal_after_sweep = True
                    
                    reversal_idx = future['high'].idxmax()
                    event.candles_to_reversal = df.index.get_loc(reversal_idx) - event.sweep_index
    
    # ==================== INDUCEMENT ====================
    
    def _identify_inducement(self):
        """
        Identify inducement liquidity
        
        ICT: Liquidity that's used to trap traders on wrong side
        before the real move
        """
        # Stop hunts are inducement by definition
        for event in self.sweep_events:
            if event.sweep_type == SweepType.STOP_HUNT:
                event.pool.is_inducement = True
                event.pool.notes.append("Inducement - stop hunt reversal")
        
        # Minor liquidity that gets swept with small displacement is inducement
        for pool in self.liquidity_pools:
            if pool.is_swept and pool.strength == LiquidityStrength.MINOR:
                if pool.displacement_pct < self.displacement_threshold_pct:
                    pool.is_inducement = True
                    pool.notes.append("Inducement - minor pool, weak displacement")
    
    # ==================== CLASSIFICATION ====================
    
    def _classify_strength(self):
        """Classify liquidity pool strength"""
        for pool in self.liquidity_pools:
            if pool.touched_count >= 4:
                pool.strength = LiquidityStrength.PRIMARY
            elif pool.touched_count >= 3:
                pool.strength = LiquidityStrength.MAJOR
            elif pool.touched_count >= 2:
                pool.strength = LiquidityStrength.MEDIUM
            else:
                pool.strength = LiquidityStrength.MINOR
            
            # HTF levels are always major+
            if pool.liquidity_type in [LiquidityType.DAILY_HIGH, LiquidityType.DAILY_LOW,
                                       LiquidityType.WEEKLY_HIGH, LiquidityType.WEEKLY_LOW]:
                if pool.strength.value < LiquidityStrength.MAJOR.value:
                    pool.strength = LiquidityStrength.MAJOR
    
    def _identify_primary_liquidity(self):
        """
        Identify primary liquidity (main draw)
        
        ICT: "What's the draw on liquidity?"
        """
        # Highest intact buy-side is primary
        intact_buy = [p for p in self.liquidity_pools 
                      if p.side == LiquiditySide.BUY_SIDE and not p.is_swept]
        if intact_buy:
            primary = max(intact_buy, key=lambda p: p.price)
            primary.is_primary = True
            primary.is_draw_on_liquidity = True
            primary.notes.append("Primary buy-side - main draw on liquidity")
        
        # Lowest intact sell-side is primary
        intact_sell = [p for p in self.liquidity_pools
                       if p.side == LiquiditySide.SELL_SIDE and not p.is_swept]
        if intact_sell:
            primary = min(intact_sell, key=lambda p: p.price)
            primary.is_primary = True
            primary.is_draw_on_liquidity = True
            primary.notes.append("Primary sell-side - main draw on liquidity")
        
        # Equal highs/lows with 3+ touches are always primary candidates
        for pool in self.liquidity_pools:
            if pool.liquidity_type in [LiquidityType.EQUAL_HIGHS, LiquidityType.EQUAL_LOWS]:
                if pool.touched_count >= 3 and not pool.is_swept:
                    pool.is_primary = True
    
    # ==================== LIQUIDITY RUN ANALYSIS ====================
    
    def _analyze_liquidity_run(self, df: pd.DataFrame) -> Optional[LiquidityRunAnalysis]:
        """
        Analyze if we're in low or high resistance liquidity run
        
        ICT:
        - Low Resistance: "Down close candles support price, very little give back"
        - High Resistance: "Back and forth, every FVG gets filled"
        """
        if len(df) < 20:
            return None
        
        recent = df.tail(20)
        
        # Determine direction
        if recent.iloc[-1]['close'] > recent.iloc[0]['close']:
            direction = "bullish"
        else:
            direction = "bearish"
        
        # Count retracements
        highs = recent['high'].values
        lows = recent['low'].values
        
        # Calculate retracement depth
        if direction == "bullish":
            # In bullish, look at how deep pullbacks go
            total_range = recent['high'].max() - recent['low'].min()
            pullback_depths = []
            for i in range(5, len(recent) - 1):
                if recent.iloc[i]['low'] < recent.iloc[i - 1]['low']:
                    depth = recent.iloc[i - 1]['high'] - recent.iloc[i]['low']
                    pullback_depths.append(depth / total_range if total_range > 0 else 0)
            
            avg_retracement = np.mean(pullback_depths) if pullback_depths else 0
        else:
            total_range = recent['high'].max() - recent['low'].min()
            pullback_depths = []
            for i in range(5, len(recent) - 1):
                if recent.iloc[i]['high'] > recent.iloc[i - 1]['high']:
                    depth = recent.iloc[i]['high'] - recent.iloc[i - 1]['low']
                    pullback_depths.append(depth / total_range if total_range > 0 else 0)
            
            avg_retracement = np.mean(pullback_depths) if pullback_depths else 0
        
        # Classify run type
        if avg_retracement < 0.3:
            run_type = LiquidityRunType.LOW_RESISTANCE
            stop_management = "Use trailing stops. Don't rush to take partials."
            position_sizing = "Can hold larger positions - less give back expected."
        elif avg_retracement > 0.5:
            run_type = LiquidityRunType.HIGH_RESISTANCE
            stop_management = "Take partials early. Expect deep retracements."
            position_sizing = "Smaller positions - expect chop."
        else:
            run_type = LiquidityRunType.NEUTRAL
            stop_management = "Standard stop management."
            position_sizing = "Normal position sizing."
        
        return LiquidityRunAnalysis(
            run_type=run_type,
            direction=direction,
            retracement_depth=avg_retracement,
            stop_management=stop_management,
            position_sizing=position_sizing
        )
    
    # ==================== DRAW ON LIQUIDITY ====================
    
    def _determine_draw(self, df: pd.DataFrame) -> Optional[LiquidityPool]:
        """
        Determine current draw on liquidity
        
        ICT: "What is price gravitating toward?"
        """
        if len(df) == 0:
            return None
        
        current_price = df.iloc[-1]['close']
        
        # Get all intact pools
        intact = [p for p in self.liquidity_pools if not p.is_swept]
        if not intact:
            return None
        
        # Find nearest above and below
        above = [p for p in intact if p.price > current_price]
        below = [p for p in intact if p.price < current_price]
        
        nearest_above = min(above, key=lambda p: p.price - current_price) if above else None
        nearest_below = max(below, key=lambda p: p.price) if below else None
        
        # Determine which is the likely draw
        # Priority: Equal highs/lows > Primary > Major > others
        candidates = []
        if nearest_above:
            candidates.append(nearest_above)
        if nearest_below:
            candidates.append(nearest_below)
        
        if not candidates:
            return None
        
        # Score each candidate
        def score_pool(p):
            s = 0
            if p.liquidity_type in [LiquidityType.EQUAL_HIGHS, LiquidityType.EQUAL_LOWS]:
                s += 30
            if p.is_primary:
                s += 20
            if p.strength == LiquidityStrength.PRIMARY:
                s += 15
            elif p.strength == LiquidityStrength.MAJOR:
                s += 10
            s += p.touched_count * 3
            return s
        
        best = max(candidates, key=score_pool)
        best.is_draw_on_liquidity = True
        return best
    
    # ==================== UTILITIES ====================
    
    def _detect_pip_value(self, df: pd.DataFrame) -> float:
        """Detect pip value based on price level"""
        avg_price = df['close'].mean()
        if avg_price < 10:
            return 0.0001  # Forex
        elif avg_price < 1000:
            return 0.01    # Small index
        else:
            return 0.25    # Futures
    
    def _get_timestamp(self, df: pd.DataFrame, index: int) -> Optional[datetime]:
        """Get timestamp from dataframe"""
        if isinstance(df.index[index], (datetime, pd.Timestamp)):
            return df.index[index]
        return None
    
    def _pool_exists_near(self, price: float, side: LiquiditySide) -> bool:
        """Check if pool exists near price"""
        threshold = self.equal_threshold_pips * self._pip_value * 2
        for pool in self.liquidity_pools:
            if pool.side == side and abs(pool.price - price) <= threshold:
                return True
        return False
    
    def _build_analysis(self, run_analysis: Optional[LiquidityRunAnalysis],
                        current_draw: Optional[LiquidityPool]) -> LiquidityAnalysis:
        """Build complete analysis result"""
        buy_side = [p for p in self.liquidity_pools if p.side == LiquiditySide.BUY_SIDE]
        sell_side = [p for p in self.liquidity_pools if p.side == LiquiditySide.SELL_SIDE]
        intact = [p for p in self.liquidity_pools if not p.is_swept]
        swept = [p for p in self.liquidity_pools if p.is_swept]
        
        primary_buy = next((p for p in buy_side if p.is_primary and not p.is_swept), None)
        primary_sell = next((p for p in sell_side if p.is_primary and not p.is_swept), None)
        
        stop_hunts = [e for e in self.sweep_events if e.sweep_type == SweepType.STOP_HUNT]
        inducement = [p for p in self.liquidity_pools if p.is_inducement]
        
        return LiquidityAnalysis(
            all_pools=self.liquidity_pools,
            buy_side_pools=buy_side,
            sell_side_pools=sell_side,
            intact_pools=intact,
            swept_pools=swept,
            primary_buy_side=primary_buy,
            primary_sell_side=primary_sell,
            sweep_events=self.sweep_events,
            stop_hunts=stop_hunts,
            inducement_pools=inducement,
            liquidity_voids=self.liquidity_voids,
            run_analysis=run_analysis,
            current_draw=current_draw
        )
    
    # ==================== QUERY METHODS ====================
    
    def get_draw_on_liquidity(self, side: Optional[LiquiditySide] = None) -> Optional[LiquidityPool]:
        """Get current draw on liquidity"""
        pools = [p for p in self.liquidity_pools if p.is_draw_on_liquidity and not p.is_swept]
        if side:
            pools = [p for p in pools if p.side == side]
        return pools[0] if pools else None
    
    def get_nearest_liquidity(self, price: float, side: Optional[LiquiditySide] = None) -> Optional[LiquidityPool]:
        """Get nearest intact liquidity pool"""
        intact = [p for p in self.liquidity_pools if not p.is_swept]
        if side:
            intact = [p for p in intact if p.side == side]
        if not intact:
            return None
        return min(intact, key=lambda p: abs(p.price - price))
    
    def get_summary(self) -> str:
        """Generate text summary"""
        lines = [
            "", "=" * 70, "ICT LIQUIDITY ANALYSIS", "=" * 70,
            f"\nTotal Pools: {len(self.liquidity_pools)}",
            f"  Buy-Side: {len([p for p in self.liquidity_pools if p.side == LiquiditySide.BUY_SIDE])}",
            f"  Sell-Side: {len([p for p in self.liquidity_pools if p.side == LiquiditySide.SELL_SIDE])}",
            f"  Intact: {len([p for p in self.liquidity_pools if not p.is_swept])}",
            f"  Swept: {len([p for p in self.liquidity_pools if p.is_swept])}",
            f"\nSweep Events: {len(self.sweep_events)}",
            f"  Stop Hunts: {len([e for e in self.sweep_events if e.sweep_type == SweepType.STOP_HUNT])}",
            f"\nInducement Pools: {len([p for p in self.liquidity_pools if p.is_inducement])}",
        ]
        
        # Primary liquidity
        primary = [p for p in self.liquidity_pools if p.is_primary and not p.is_swept]
        if primary:
            lines.append("\nPrimary Liquidity (Draw):")
            for p in primary:
                lines.append(f"  {p}")
        
        # Recent sweeps
        if self.sweep_events:
            lines.append("\nRecent Sweeps:")
            for e in self.sweep_events[-5:]:
                lines.append(f"  {e}")
        
        lines.append("=" * 70 + "\n")
        return "\n".join(lines)


if __name__ == "__main__":
    print("ICT Liquidity Handler")
    print("=" * 50)
    print("\nKey Concepts:")
    print("  • Buy-Side Liquidity (above equal highs)")
    print("  • Sell-Side Liquidity (below equal lows)")
    print("  • Stop Hunt (Turtle Soup)")
    print("  • Inducement")
    print("  • Low/High Resistance Liquidity Runs")
    print("  • Draw on Liquidity")
