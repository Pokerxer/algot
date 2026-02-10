"""
ICT Trading Models Handler - Comprehensive Implementation
=========================================================

Implements all ICT trading models, setups, and execution frameworks based on
direct ICT transcript teachings.

ICT TRADING MODELS IMPLEMENTED:
==============================

1. ICT 2022 MODEL (Primary):
   "Buy side taken into a premium, shift in market structure, fair value gap,
   trades up into it, there's your show, go short."
   - Stage 1: Liquidity taken (buy-side for shorts, sell-side for longs)
   - Stage 2: Retrace into premium/discount zone (50%+ retracement)
   - Stage 3: Entry at PD array (FVG, OB, Breaker)
   - Must have structure shift + displacement

2. SILVER BULLET:
   "First FVG that forms after 10:00 inside 10:00-11:00 hour."
   - Asian: 20:00-21:00 EST
   - London: 03:00-04:00 EST  
   - AM: 10:00-11:00 EST (most popular)
   - PM: 14:00-15:00 EST
   - Entry: First Presented FVG (FPFVG)

3. VENOM:
   "Rally up one candle. Remember I told you only takes one, but it can be
   a number of candles up here. And then you have a single pass candle down.
   So, up and down. And then we trade up into that candlestick right there.
   That's venom. Beautiful."
   - STRICT: Single candle penetration, single candle return
   - CIBI (bearish) or BISI (bullish) must be present

4. TURTLE SOUP:
   "If it can trade through the quadrant levels and it doesn't stop... then this
   is probably going to have a turtle soup scenario."
   - Old high/low violated with wick only
   - Quick reversal
   - Range-bound context required

5. POWER OF THREE (AMD):
   "Power three, which is the open, high, low, and close formation, which is
   open, accumulate, manipulate, distribute AMD, power three."
   - Accumulation phase
   - Manipulation (stop hunt)
   - Distribution phase

6. UNICORN MODEL:
   "All the high significant second stage reaccumulation or second stage
   redistribution the unicorn they will always lay on one of these quadrants."
   - Second stage re-accumulation (bullish)
   - Second stage redistribution (bearish)

7. MARKET MAKER BUY/SELL MODEL:
   "Structure my market maker buy and sell model... they match up"
   - Full curve structure
   - Accumulation -> Distribution (Buy model)
   - Distribution -> Accumulation (Sell model)

8. SWING PROJECTION:
   "Take the high here and add your FIB to it here and draw it down to that low
   one standard deviation would be 42.78.75"
   - Fulcrum point projection
   - Standard deviation targets (-1, -2)

9. INSTITUTIONAL ORDER FLOW ENTRY DRILL (IOFED):
   "Institutional order flow entry drill which is a partial return into not a
   complete closure of that Gap"
   - Partial gap fill entry
   - Low resistance liquidity run conditions

10. LOW RESISTANCE LIQUIDITY RUN (LRLR):
    "Low resistance liquidity run conditions... expect them to just really make
    a Mad Dash to get to the liquidity"
    - Large range day conditions
    - Gap not expected to fill
    - Explosive directional move

11. GAP HIERARCHY MODELS:
    "Gaps have an hierarchy... breakaway gap, common gaps, measuring gaps"
    - Breakaway Gap: Between quadrants, doesn't fill
    - Measuring Gap: Midpoint projection
    - Common Gap: Can be reclaimed
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from datetime import datetime, time
import pandas as pd
import numpy as np


# =============================================================================
# ENUMERATIONS
# =============================================================================

class TradingModel(Enum):
    """ICT Trading Model Types"""
    MODEL_2022 = "2022_model"
    SILVER_BULLET = "silver_bullet"
    TURTLE_SOUP = "turtle_soup"
    VENOM = "venom"
    MARKET_MAKER_BUY = "market_maker_buy"
    MARKET_MAKER_SELL = "market_maker_sell"
    POWER_OF_THREE = "power_of_three"
    OPTIMAL_TRADE_ENTRY = "optimal_trade_entry"
    BREAKAWAY_GAP = "breakaway_gap"
    MEASURING_GAP = "measuring_gap"
    UNICORN = "unicorn_model"
    IOFED = "institutional_order_flow_entry_drill"
    LRLR = "low_resistance_liquidity_run"
    SWING_PROJECTION = "swing_projection"
    STAGING_SHORTS = "staging_shorts_ath"


class ModelStage(Enum):
    """Trading model stage"""
    STAGE_1 = 1  # Initial stage (liquidity/accumulation)
    STAGE_2 = 2  # Development stage (premium/manipulation)
    STAGE_3 = 3  # Entry stage (PD array/distribution)
    COMPLETE = 4  # All stages complete


class EntryType(Enum):
    """Entry PD array type"""
    FVG = "fair_value_gap"
    ORDER_BLOCK = "order_block"
    BREAKER = "breaker_block"
    MITIGATION = "mitigation_block"
    INVERSION_FVG = "inversion_fvg"
    VOLUME_IMBALANCE = "volume_imbalance"
    SUSPENSION_BLOCK = "suspension_block"


class SetupQuality(Enum):
    """Setup quality classification"""
    A_PLUS = "A+"    # All criteria met, perfect alignment
    A = "A"          # Most criteria met, strong setup
    B = "B"          # Some criteria met, tradeable
    C = "C"          # Minimal criteria, avoid or reduce size
    INVALID = "Invalid"


class SilverBulletSession(Enum):
    """Silver Bullet session windows"""
    ASIAN = "asian"      # 20:00-21:00 EST
    LONDON = "london"    # 03:00-04:00 EST
    NY_AM = "am"         # 10:00-11:00 EST (most popular)
    NY_PM = "pm"         # 14:00-15:00 EST


class GapHierarchy(Enum):
    """ICT Gap classification hierarchy"""
    BREAKAWAY = "breakaway"      # Between quadrants, doesn't fill, strong trend
    MEASURING = "measuring"      # Midpoint projection, half stays open
    COMMON = "common"            # Can be reclaimed/traded multiple times
    RUNAWAY = "runaway"          # Continuation in trend


# =============================================================================
# DATA CLASSES - MODEL SETUPS
# =============================================================================

@dataclass
class Model2022Setup:
    """
    ICT 2022 Model - Primary trading model
    
    ICT Quote: "Buy side taken into a premium shift in Market structure
    fair value Gap trades up into it there's your show go short what do 
    you aim for relative equal lows and liquidity"
    
    Sequence:
    1. Liquidity taken (buy-side for shorts, sell-side for longs)
    2. Retrace into premium (shorts) or discount (longs)  
    3. Structure shift (CHoCH or BOS)
    4. Entry at PD array (FVG, Order Block, Breaker)
    """
    # Setup identification
    timestamp: datetime
    direction: str  # 'long' or 'short'
    
    # Liquidity stage
    liquidity_side: str  # 'buy_side' or 'sell_side'
    liquidity_level: float
    liquidity_taken: bool
    
    # Premium/Discount stage
    zone_type: str  # 'premium' or 'discount'
    zone_high: float
    zone_low: float
    retracement_percent: float  # How deep into premium/discount
    
    # OTE Zone (62-79%)
    ote_high: float
    ote_low: float
    ote_mid: float  # 70.5% sweet spot
    in_ote: bool
    
    # Entry details
    entry_type: EntryType
    entry_price: float
    stop_loss: float
    target_1: float  # First target (opposite liquidity)
    target_2: Optional[float] = None  # Extended target
    
    # Model stages completion
    stage_1_liquidity: bool = False
    stage_2_retrace: bool = False
    stage_3_structure: bool = False
    stage_4_entry: bool = False
    
    # Quality metrics
    structure_shift: bool = False
    structure_shift_type: str = ""  # 'CHoCH' or 'BOS'
    displacement_present: bool = False
    displacement_size: float = 0.0
    time_aligned: bool = False  # In kill zone or macro time
    
    # Confluence
    has_fvg: bool = False
    has_order_block: bool = False
    has_breaker: bool = False
    has_volume_imbalance: bool = False
    confluence_count: int = 0
    
    # Result
    quality: SetupQuality = SetupQuality.C
    favorability: float = 0.0  # 0-100
    notes: str = ""
    
    def calculate_quality(self):
        """Calculate setup quality based on ICT criteria"""
        score = 0
        
        # Stage completion (40 points)
        if self.stage_1_liquidity:
            score += 10
        if self.stage_2_retrace:
            score += 10
        if self.stage_3_structure:
            score += 10
        if self.stage_4_entry:
            score += 10
            
        # OTE alignment (20 points)
        if self.in_ote:
            score += 20
        elif self.retracement_percent >= 50:
            score += 10
            
        # Structure shift (15 points)
        if self.structure_shift:
            score += 15
            
        # Displacement (10 points)
        if self.displacement_present:
            score += 10
            
        # Time alignment (5 points)
        if self.time_aligned:
            score += 5
            
        # Confluence (10 points)
        score += min(self.confluence_count * 3, 10)
        
        self.favorability = min(score, 100)
        
        if score >= 90:
            self.quality = SetupQuality.A_PLUS
        elif score >= 75:
            self.quality = SetupQuality.A
        elif score >= 60:
            self.quality = SetupQuality.B
        elif score >= 40:
            self.quality = SetupQuality.C
        else:
            self.quality = SetupQuality.INVALID


@dataclass
class SilverBulletSetup:
    """
    Silver Bullet Model - Time-specific setups
    
    ICT Quote: "If you're trading the ICT Silver Bullet which is the model 
    that I've taught you for 2023 for the YouTube... we have a shift in 
    Market structure and it's much more pronounced"
    
    Time Windows:
    - Asian: 20:00-21:00 EST
    - London: 03:00-04:00 EST
    - AM: 10:00-11:00 EST (most popular)
    - PM: 14:00-15:00 EST
    
    Key: First Presented FVG (FPFVG) after window opens
    """
    # Setup identification
    timestamp: datetime
    session: SilverBulletSession
    time_window_start: time
    time_window_end: time
    
    # Direction
    direction: str  # 'long' or 'short'
    
    # First Presented FVG (critical for Silver Bullet)
    first_fvg_price: float
    first_fvg_high: float
    first_fvg_low: float
    first_fvg_type: str  # 'bullish' or 'bearish'
    
    # Entry
    entry_price: float
    stop_loss: float
    target: float
    
    is_first_fvg: bool = True  # Must be FIRST FVG after window opens
    
    # Standard deviation projection
    swing_high: float = 0.0
    swing_low: float = 0.0
    std_dev_target: float = 0.0  # -1 standard deviation target
    
    # Criteria validation
    in_proper_time: bool = True
    has_structure_shift: bool = False
    has_displacement: bool = False
    liquidity_swept: bool = False
    
    # Session flags
    is_asian_sb: bool = False
    is_london_sb: bool = False
    is_am_sb: bool = False
    is_pm_sb: bool = False
    
    # Quality
    quality: SetupQuality = SetupQuality.B
    favorability: float = 0.0
    notes: str = ""
    
    def __post_init__(self):
        """Set session flags based on session"""
        if self.session == SilverBulletSession.ASIAN:
            self.is_asian_sb = True
            self.time_window_start = time(20, 0)
            self.time_window_end = time(21, 0)
        elif self.session == SilverBulletSession.LONDON:
            self.is_london_sb = True
            self.time_window_start = time(3, 0)
            self.time_window_end = time(4, 0)
        elif self.session == SilverBulletSession.NY_AM:
            self.is_am_sb = True
            self.time_window_start = time(10, 0)
            self.time_window_end = time(11, 0)
        elif self.session == SilverBulletSession.NY_PM:
            self.is_pm_sb = True
            self.time_window_start = time(14, 0)
            self.time_window_end = time(15, 0)


@dataclass
class VenomSetup:
    """
    Venom Model - Single pass liquidity injection
    
    ICT Quote: "Here's venom rally up one candle. Remember I told you only 
    takes one, but it can be a number of candles up here. And then you have 
    a single pass candle down. So, up and down. And then we trade up into 
    that candlestick right there. That's venom. Beautiful."
    
    STRICT CRITERIA:
    1. ONE candle penetrates liquidity (single pass down/up)
    2. ONE candle returns (single pass return)
    3. CIBI (bearish) or BISI (bullish) must be present
    4. Entry at injection candle close or better
    """
    # Identification
    timestamp: datetime
    direction: str  # 'long' or 'short'
    
    # Liquidity injection
    liquidity_level: float
    injection_candle_idx: int
    injection_candle_high: float
    injection_candle_low: float
    injection_candle_close: float
    
    # Entry
    entry_price: float  # At injection candle close or better
    stop_loss: float    # Beyond injection candle extreme
    target: float
    
    # STRICT single pass validation
    single_pass_to_liquidity: bool = False  # Only ONE candle penetrates
    single_pass_return: bool = False         # Only ONE candle returns
    
    # Imbalance type
    has_cibi: bool = False  # Sell Imbalance Buy Efficiency (bearish venom)
    has_bisi: bool = False  # Buy Imbalance Sell Efficiency (bullish venom)
    imbalance_high: float = 0.0
    imbalance_low: float = 0.0
    
    # Time distortion (consolidation after injection)
    consolidation_bars: int = 0
    consolidation_broken: bool = False
    
    # Validation
    is_valid_venom: bool = False
    quality: SetupQuality = SetupQuality.C
    favorability: float = 0.0
    notes: str = ""
    
    def validate(self):
        """Validate strict Venom criteria"""
        self.is_valid_venom = (
            self.single_pass_to_liquidity and
            self.single_pass_return and
            (self.has_cibi or self.has_bisi)
        )
        
        if self.is_valid_venom:
            self.quality = SetupQuality.A
            self.favorability = 90.0
        else:
            self.quality = SetupQuality.INVALID
            self.favorability = 0.0


@dataclass
class TurtleSoupSetup:
    """
    Turtle Soup - Stop hunt reversal
    
    ICT Quote: "If it can trade through the quadrant levels and it doesn't 
    stop, like see how this right here went right to it and stopped and went 
    lower. If you see that at anything like the consequent encroachment or 
    the lower quadrant and there's no bodies in the upper half or above the 
    consequent encroachment level, then this is probably going to have a 
    turtle soup scenario."
    """
    # Identification
    timestamp: datetime
    hunt_type: str  # 'long' (hunt below old low) or 'short' (hunt above old high)
    
    # Stop hunt level
    old_level: float  # The old high/low being violated
    new_extreme: float  # The wick that violated it
    violation_size: float  # How far beyond old level
    
    # Entry
    entry_price: float
    stop_loss: float  # Beyond new extreme
    target: float     # Opposite liquidity pool
    
    # Context requirements (ICT specific)
    in_range: bool = False  # Must be range-bound market
    
    # Validation criteria
    old_level_violated: bool = True
    is_wick_only: bool = True  # Body must NOT close beyond
    quick_reversal: bool = False
    opposite_liquidity_available: bool = False  # Target on other side
    
    at_quadrant: bool = False  # At CE or quadrant level
    bodies_below_ce: bool = False  # For bearish turtle soup
    bodies_above_ce: bool = False  # For bullish turtle soup
    
    # FVG support
    has_fvg_support: bool = False
    fvg_price: float = 0.0
    
    quality: SetupQuality = SetupQuality.C
    favorability: float = 0.0
    notes: str = ""


@dataclass
class PowerOfThreeSetup:
    """
    Power of Three (AMD) - Accumulation, Manipulation, Distribution
    
    ICT Quote: "Power three, which is the open, high, low, and close formation, 
    which is open, accumulate, manipulate, distribute AMD, power three. It's the 
    way the candlesticks formed its daily range."
    
    Daily Candle Structure:
    - Bullish Day: Open near low, close near high (O-L-H-C)
    - Bearish Day: Open near high, close near low (O-H-L-C)
    """
    # Identification
    timestamp: datetime
    timeframe: str  # 'daily', 'session', 'hour'
    direction: str  # 'bullish' or 'bearish'
    
    # Daily OHLC
    open_price: float
    high_price: float
    low_price: float
    close_price: float
    
    # Three phases
    accumulation_zone_high: float
    accumulation_zone_low: float
    
    manipulation_level: float  # The stop hunt / fake move
    manipulation_direction: str = ""  # 'upside' or 'downside'
    manipulation_complete: bool = False
    
    accumulation_complete: bool = False
    distribution_zone_high: float = 0.0
    distribution_zone_low: float = 0.0
    in_distribution: bool = False
    
    # Current phase
    current_phase: str = ""  # 'accumulation', 'manipulation', 'distribution'
    
    # Pattern validation
    is_valid_amd: bool = False
    
    # For bullish: O-L-H-C (open near low, close near high)
    # For bearish: O-H-L-C (open near high, close near low)
    open_near_low: bool = False  # Bullish indicator
    open_near_high: bool = False  # Bearish indicator
    close_near_high: bool = False  # Bullish indicator
    close_near_low: bool = False  # Bearish indicator
    
    quality: SetupQuality = SetupQuality.C
    favorability: float = 0.0
    notes: str = ""
    
    def validate_amd_pattern(self, tolerance: float = 0.3):
        """
        Validate AMD pattern based on OHLC structure
        
        tolerance: How close open/close must be to high/low (as fraction of range)
        """
        range_size = self.high_price - self.low_price
        if range_size == 0:
            return
            
        # Check open position relative to range
        open_from_low = (self.open_price - self.low_price) / range_size
        open_from_high = (self.high_price - self.open_price) / range_size
        
        # Check close position relative to range
        close_from_low = (self.close_price - self.low_price) / range_size
        close_from_high = (self.high_price - self.close_price) / range_size
        
        self.open_near_low = open_from_low <= tolerance
        self.open_near_high = open_from_high <= tolerance
        self.close_near_high = close_from_high <= tolerance
        self.close_near_low = close_from_low <= tolerance
        
        # Bullish AMD: Open near low, close near high
        if self.open_near_low and self.close_near_high:
            self.direction = 'bullish'
            self.is_valid_amd = True
            self.manipulation_direction = 'downside'  # Fake move down first
            
        # Bearish AMD: Open near high, close near low
        elif self.open_near_high and self.close_near_low:
            self.direction = 'bearish'
            self.is_valid_amd = True
            self.manipulation_direction = 'upside'  # Fake move up first


@dataclass
class UnicornSetup:
    """
    Unicorn Model - Second stage re-accumulation/redistribution
    
    ICT Quote: "All the high significant second stage reaccumulation or 
    second stage redistribution the unicorn they will always lay on one 
    of these quadrants when you're doing it."
    """
    # Identification
    timestamp: datetime
    direction: str  # 'bullish' (re-accumulation) or 'bearish' (redistribution)
    
    # Structure
    first_stage_high: float
    first_stage_low: float
    
    # Second stage formation
    second_stage_high: float
    second_stage_low: float
    second_stage_type: str  # 're-accumulation' or 'redistribution'
    
    # Quadrant alignment (critical for Unicorn)
    at_quadrant: bool = False
    quadrant_type: str = ""  # '25%', '50% (CE)', '75%'
    quadrant_price: float = 0.0
    
    # Validation
    has_consolidation: bool = False
    consolidation_bars: int = 0
    
    # Entry
    entry_price: float = 0.0
    stop_loss: float = 0.0
    target: float = 0.0
    
    quality: SetupQuality = SetupQuality.C
    favorability: float = 0.0
    notes: str = ""


@dataclass
class SwingProjection:
    """
    ICT Swing Projection Theory
    
    ICT Quote: "Take the high here and add your FIB to it here and draw it 
    down to that low one standard deviation would be 42.78.75 that's a pretty 
    handsome objective if we treat that high to that low this low being a 
    fulcrum point that means if this move from high to that low s like a door 
    okay and it swung this was the hinge of the door this high if it was 
    allowed to swing completely all the way around it would come right down 
    right below that low"
    """
    # Swing identification
    timestamp: datetime
    direction: str  # 'bullish' or 'bearish'
    
    # Swing points
    swing_high: float
    swing_low: float
    swing_range: float = field(init=False)
    
    # Fulcrum point
    fulcrum_price: float  # The "hinge of the door"
    
    # Standard deviation projections
    std_dev_minus_1: float = field(init=False)  # -1 standard deviation
    std_dev_minus_2: float = field(init=False)  # -2 standard deviation
    std_dev_plus_1: float = field(init=False)   # +1 standard deviation
    std_dev_plus_2: float = field(init=False)   # +2 standard deviation
    
    # Gap context (if applicable)
    is_measuring_gap: bool = False
    gap_high: float = 0.0
    gap_low: float = 0.0
    
    notes: str = ""
    
    def __post_init__(self):
        """Calculate swing range and projections"""
        self.swing_range = self.swing_high - self.swing_low
        
        if self.direction == 'bearish':
            # For bearish projection, subtract from fulcrum (usually swing low)
            self.std_dev_minus_1 = self.fulcrum_price - self.swing_range
            self.std_dev_minus_2 = self.fulcrum_price - (self.swing_range * 2)
            self.std_dev_plus_1 = self.fulcrum_price + self.swing_range
            self.std_dev_plus_2 = self.fulcrum_price + (self.swing_range * 2)
        else:
            # For bullish projection, add to fulcrum (usually swing high)
            self.std_dev_plus_1 = self.fulcrum_price + self.swing_range
            self.std_dev_plus_2 = self.fulcrum_price + (self.swing_range * 2)
            self.std_dev_minus_1 = self.fulcrum_price - self.swing_range
            self.std_dev_minus_2 = self.fulcrum_price - (self.swing_range * 2)


@dataclass
class IOFEDSetup:
    """
    Institutional Order Flow Entry Drill
    
    ICT Quote: "Institutional order flow entry drill which is a partial return 
    into not a complete closure of that Gap that candles high that candle is 
    low just a little bit above the low and then consolidates and tears often 
    goes lower"
    
    Key: PARTIAL gap fill, not complete closure
    """
    # Identification
    timestamp: datetime
    direction: str
    
    # Gap being partially filled
    gap_high: float
    gap_low: float
    gap_ce: float  # Consequent encroachment (50%)
    
    # Partial fill level
    fill_level: float  # The level it filled to
    fill_percentage: float  # How much of gap was filled
    
    # Entry
    entry_at_partial: float  # Entry at partial fill level
    stop_loss: float
    target: float
    
    # LRLR context
    in_lrlr_conditions: bool = False  # Low resistance liquidity run
    is_large_range_day: bool = False
    
    # Order block at partial fill
    has_order_block: bool = False
    order_block_price: float = 0.0
    
    quality: SetupQuality = SetupQuality.B
    favorability: float = 0.0
    notes: str = ""


@dataclass
class LRLRConditions:
    """
    Low Resistance Liquidity Run Conditions
    
    ICT Quote: "Low resistance liquidity run conditions... expect them to just 
    really make a Mad Dash to get to the liquidity"
    
    Characteristics:
    - Early session of potential large range day
    - Gap not expected to fill
    - Explosive directional move
    - Breakaway gap conditions
    """
    # Identification
    timestamp: datetime
    direction: str
    
    # Conditions
    is_early_session: bool = False  # First 30-60 minutes
    is_potential_large_day: bool = False
    has_breakaway_gap: bool = False
    
    # Gap context
    opening_gap_size: float = 0.0
    gap_fill_expected: bool = False  # Usually FALSE in LRLR
    
    # Liquidity targets
    buy_side_liquidity: List[float] = field(default_factory=list)
    sell_side_liquidity: List[float] = field(default_factory=list)
    primary_target: float = 0.0
    
    # Entry
    entry_price: float = 0.0
    stop_loss: float = 0.0
    
    # Validation
    is_valid_lrlr: bool = False
    notes: str = ""


@dataclass
class BreakawayGapSetup:
    """
    Breakaway Gap - ICT specific classification
    
    ICT Quote: "The inefficiency that shows up on your chart that does not get 
    filled back in. If it's between a quadrant level like this and another 
    quadrant, expect that when we are, you looking for lower prices, when it 
    breaks lower, if it leaves a quadrant like that and breaks a little bit 
    lower, this is many times going to act as a breakaway gap because it's in 
    no man's land between a quadrant and another quadrant inside the narrative 
    and bias that the market's moving one direction."
    """
    # Identification
    timestamp: datetime
    direction: str  # 'bullish' or 'bearish'
    
    # Gap levels
    gap_high: float
    gap_low: float
    gap_size: float = field(init=False)
    
    # Quadrant context (CRITICAL for breakaway classification)
    is_between_quadrants: bool = True  # MUST be between quadrants
    upper_quadrant: float = 0.0
    lower_quadrant: float = 0.0
    
    # Validation
    breaking_from_structure: bool = False  # Leaving consolidation
    portion_unfilled: bool = True  # Doesn't fully fill
    
    # Projection
    expected_continuation: float = 0.0  # Target based on gap size
    
    notes: str = ""
    
    def __post_init__(self):
        self.gap_size = abs(self.gap_high - self.gap_low)


@dataclass
class MeasuringGapSetup:
    """
    Measuring Gap - Midpoint projection
    
    ICT Quote: "A halfway point when there's a gap that is during a move if 
    it's between a quadrant of either an inefficiency, okay, or an actual gap 
    by a wick... common gaps are a fair value Gap that can be reclaimed treated 
    as support and resistance... measuring gaps tend to leave a portion open 
    just like a breakaway"
    """
    # Identification
    timestamp: datetime
    direction: str
    
    # Gap levels
    gap_high: float
    gap_low: float
    gap_mid: float = field(init=False)
    gap_size: float = field(init=False)
    
    # Position in move
    move_start: float  # Start of the move
    move_current: float  # Current price
    move_projected_end: float = field(init=False)
    
    # Validation
    is_midway_in_move: bool = False
    between_quadrants: bool = False
    portion_stays_open: bool = True  # Half stays open
    
    # Projection
    measured_move_distance: float = field(init=False)
    std_dev_target: float = field(init=False)
    
    notes: str = ""
    
    def __post_init__(self):
        self.gap_size = abs(self.gap_high - self.gap_low)
        self.gap_mid = (self.gap_high + self.gap_low) / 2
        self.measured_move_distance = abs(self.move_current - self.move_start)
        
        if self.direction == 'bearish':
            self.move_projected_end = self.gap_low - self.measured_move_distance
            self.std_dev_target = self.gap_low - (self.gap_size * 2)
        else:
            self.move_projected_end = self.gap_high + self.measured_move_distance
            self.std_dev_target = self.gap_high + (self.gap_size * 2)


@dataclass 
class StagingShortSetup:
    """
    Smart Money Staging Shorts at All-Time Highs
    
    When body closes above previous ATH, smart money starts staging 
    shorts in 3 stages at each higher closing price.
    """
    # Identification
    timestamp: datetime
    
    # All-time high reference
    previous_ath: float
    
    # Staging zones (3 stages)
    stage_1_close: float  # First body close above ATH
    stage_2_close: float = 0.0  # Second higher close
    stage_3_close: float = 0.0  # Third higher close
    
    staging_zones: List[Tuple[float, float]] = field(default_factory=list)
    
    # Current stage
    current_stage: int = 0
    
    # Smart money activity indicators
    wicks_increasing: bool = False  # Upper wicks getting larger
    bodies_closing_lower: bool = False  # Signs of distribution
    
    # Entry
    entry_price: float = 0.0
    stop_loss: float = 0.0
    target: float = 0.0
    
    quality: SetupQuality = SetupQuality.C
    favorability: float = 0.0
    notes: str = ""


@dataclass
class MarketMakerBuyModel:
    """
    Market Maker Buy Model - Full curve structure
    
    ICT Quote: "Structure my market maker buy and sell model... they match up"
    
    Stages:
    1. Drop - Initial sell-off
    2. Accumulation - SM hedging on way down
    3. Reversal - Turn higher  
    4. Distribution - Sell to retail on way up
    """
    # Identification
    timestamp: datetime
    
    # Curve structure
    accumulation_low: float
    accumulation_zone: Tuple[float, float]
    
    distribution_high: float
    distribution_zone: Tuple[float, float]
    
    # Stages
    stage_1_drop_complete: bool = False
    stage_2_accumulation_complete: bool = False
    stage_3_reversal_complete: bool = False
    stage_4_distribution_active: bool = False
    
    current_stage: int = 1
    
    # Reclaimed blocks
    reclaimed_bullish_blocks: List[float] = field(default_factory=list)
    
    # Smart money zones
    sm_long_zones: List[Tuple[float, float]] = field(default_factory=list)
    
    quality: SetupQuality = SetupQuality.C
    favorability: float = 0.0
    notes: str = ""


@dataclass
class MarketMakerSellModel:
    """Market Maker Sell Model - Inverse of buy model"""
    timestamp: datetime
    
    distribution_high: float
    distribution_zone: Tuple[float, float]
    
    accumulation_low: float
    accumulation_zone: Tuple[float, float]
    
    stage_1_rally_complete: bool = False
    stage_2_distribution_complete: bool = False
    stage_3_reversal_complete: bool = False
    stage_4_accumulation_active: bool = False
    
    current_stage: int = 1
    
    reclaimed_bearish_blocks: List[float] = field(default_factory=list)
    sm_short_zones: List[Tuple[float, float]] = field(default_factory=list)
    
    quality: SetupQuality = SetupQuality.C
    favorability: float = 0.0
    notes: str = ""


@dataclass
class OptimalTradeEntry:
    """
    Optimal Trade Entry (OTE) - 62-79% retracement zone
    
    ICT Quote: "You can see the 79% 62% retracement respectively. That's 
    optimal trade entry. Uh very minimum it's got to go above the low to high. 
    It's got to go to 50% or higher for it to be in a premium market."
    """
    # Swing
    impulse_high: float
    impulse_low: float
    swing_range: float = field(init=False)
    
    # OTE Zone (62-79%)
    ote_high: float = field(init=False)  # 79% level
    ote_low: float = field(init=False)   # 62% level
    ote_mid: float = field(init=False)   # 70.5% sweet spot
    
    # Current position
    current_price: float = 0.0
    in_ote_zone: bool = False
    retracement_percent: float = 0.0
    
    # Direction
    direction: str = ""  # 'long' or 'short'
    
    # PD arrays in zone
    fvg_in_zone: bool = False
    order_block_in_zone: bool = False
    
    # Entry
    entry_price: float = 0.0
    stop_loss: float = 0.0
    target: float = 0.0
    
    quality: SetupQuality = SetupQuality.C
    favorability: float = 0.0
    notes: str = ""
    
    def __post_init__(self):
        self.swing_range = self.impulse_high - self.impulse_low
        # For bullish retracement (looking for longs after pullback)
        self.ote_low = self.impulse_low + (self.swing_range * 0.62)
        self.ote_high = self.impulse_low + (self.swing_range * 0.79)
        self.ote_mid = self.impulse_low + (self.swing_range * 0.705)


# =============================================================================
# MAIN HANDLER CLASS
# =============================================================================

class TradingModelsHandler:
    """
    Comprehensive handler for all ICT trading models and setups.
    Implements specific trading frameworks with entry/exit rules.
    
    ICT Key Principles:
    - "It's there every day" - Models repeat consistently
    - Time alignment is critical (kill zones, macro times)
    - Structure shift must be present
    - Displacement confirms commitment
    - Bodies tell the story, wicks do the damage
    """
    
    def __init__(self):
        self.active_setups: Dict[str, List] = {
            'model_2022': [],
            'silver_bullet': [],
            'venom': [],
            'turtle_soup': [],
            'power_of_three': [],
            'unicorn': [],
            'swing_projection': [],
            'iofed': [],
            'lrlr': [],
            'breakaway_gap': [],
            'measuring_gap': [],
            'staging_shorts': [],
            'market_maker_buy': [],
            'market_maker_sell': [],
            'ote': []
        }
        
        # Kill zone times (EST)
        self.kill_zones = {
            'asian': (time(20, 0), time(0, 0)),
            'london_open': (time(1, 0), time(5, 0)),
            'ny_open': (time(7, 0), time(10, 0)),
            'ny_am': (time(9, 30), time(12, 0)),
            'ny_lunch': (time(12, 0), time(13, 30)),
            'ny_pm': (time(13, 30), time(16, 0)),
            'last_hour': (time(15, 0), time(16, 0))
        }
        
        # Silver bullet windows
        self.silver_bullet_windows = {
            SilverBulletSession.ASIAN: (time(20, 0), time(21, 0)),
            SilverBulletSession.LONDON: (time(3, 0), time(4, 0)),
            SilverBulletSession.NY_AM: (time(10, 0), time(11, 0)),
            SilverBulletSession.NY_PM: (time(14, 0), time(15, 0))
        }
        
        # Macro times
        self.macro_times = [
            (time(2, 33), time(3, 0)),
            (time(4, 3), time(4, 30)),
            (time(8, 50), time(9, 10)),
            (time(9, 50), time(10, 10)),
            (time(10, 50), time(11, 10)),
            (time(11, 50), time(12, 10)),
            (time(13, 10), time(13, 40)),
            (time(14, 50), time(15, 10)),
            (time(15, 15), time(15, 45))
        ]
    
    # =========================================================================
    # MODEL 2022 ANALYSIS
    # =========================================================================
    
    def analyze_model_2022(self, df: pd.DataFrame, bias: str,
                          current_time: Optional[datetime] = None) -> Optional[Model2022Setup]:
        """
        Analyze for ICT 2022 Model setup.
        
        ICT Sequence:
        1. Liquidity taken (buy-side for shorts, sell-side for longs)
        2. Retrace into premium (shorts) or discount (longs)
        3. Structure shift (CHoCH or BOS)
        4. Entry at PD array (FVG, Order Block, Breaker)
        
        Args:
            df: OHLC dataframe with columns ['open', 'high', 'low', 'close']
            bias: 'bullish' or 'bearish'
            current_time: Current time for time alignment check
            
        Returns:
            Model2022Setup or None
        """
        if len(df) < 50:
            return None
            
        current_price = df['close'].iloc[-1]
        timestamp = datetime.now() if current_time is None else current_time
        
        try:
            if bias == 'bullish':
                # BULLISH 2022 MODEL
                # 1. Sell-side liquidity taken
                recent_low = df['low'].tail(20).min()
                swing_high = df['high'].tail(20).max()
                swing_range = swing_high - recent_low
                
                liquidity_taken = df['low'].iloc[-5:].min() <= recent_low
                
                if liquidity_taken:
                    # 2. Check if in discount zone
                    discount_boundary = recent_low + (swing_range * 0.5)
                    in_discount = current_price < discount_boundary
                    
                    # Calculate OTE levels
                    ote_low = recent_low + (swing_range * 0.62)
                    ote_high = recent_low + (swing_range * 0.79)
                    ote_mid = recent_low + (swing_range * 0.705)
                    in_ote = ote_low <= current_price <= ote_high
                    
                    # 3. Check for structure shift
                    structure_shift, shift_type = self._check_structure_shift(df, 'bullish')
                    
                    # 4. Check for entry PD arrays
                    fvg_present = self._find_fvg_near_price(df, current_price)
                    ob_present = self._find_order_block_near_price(df, current_price, 'bullish')
                    
                    # Calculate retracement
                    retrace_pct = ((current_price - recent_low) / swing_range) * 100 if swing_range > 0 else 0
                    
                    # Check displacement
                    displacement = self._check_displacement(df)
                    
                    # Time alignment
                    time_aligned = self._check_time_alignment(timestamp) if current_time else False
                    
                    setup = Model2022Setup(
                        timestamp=timestamp,
                        direction='long',
                        liquidity_side='sell_side',
                        liquidity_level=recent_low,
                        liquidity_taken=True,
                        zone_type='discount',
                        zone_high=discount_boundary,
                        zone_low=recent_low,
                        retracement_percent=retrace_pct,
                        ote_high=ote_high,
                        ote_low=ote_low,
                        ote_mid=ote_mid,
                        in_ote=in_ote,
                        entry_type=EntryType.FVG if fvg_present else EntryType.ORDER_BLOCK,
                        entry_price=current_price,
                        stop_loss=recent_low - (swing_range * 0.05),
                        target_1=swing_high,
                        target_2=swing_high + (swing_range * 0.5),
                        stage_1_liquidity=True,
                        stage_2_retrace=in_discount,
                        stage_3_structure=structure_shift,
                        stage_4_entry=(fvg_present or ob_present),
                        structure_shift=structure_shift,
                        structure_shift_type=shift_type,
                        displacement_present=displacement,
                        time_aligned=time_aligned,
                        has_fvg=fvg_present,
                        has_order_block=ob_present,
                        confluence_count=sum([fvg_present, ob_present, in_ote, structure_shift]),
                        notes="2022 Model LONG - Sell-side taken, in discount"
                    )
                    
                    setup.calculate_quality()
                    return setup
                    
            else:  # bearish
                # BEARISH 2022 MODEL
                # 1. Buy-side liquidity taken
                recent_high = df['high'].tail(20).max()
                swing_low = df['low'].tail(20).min()
                swing_range = recent_high - swing_low
                
                liquidity_taken = df['high'].iloc[-5:].max() >= recent_high
                
                if liquidity_taken:
                    # 2. Check if in premium zone
                    premium_boundary = recent_high - (swing_range * 0.5)
                    in_premium = current_price > premium_boundary
                    
                    # Calculate OTE levels (for shorts, invert)
                    ote_high = recent_high - (swing_range * 0.62)
                    ote_low = recent_high - (swing_range * 0.79)
                    ote_mid = recent_high - (swing_range * 0.705)
                    in_ote = ote_low <= current_price <= ote_high
                    
                    # 3. Check for structure shift
                    structure_shift, shift_type = self._check_structure_shift(df, 'bearish')
                    
                    # 4. Check for entry PD arrays
                    fvg_present = self._find_fvg_near_price(df, current_price)
                    ob_present = self._find_order_block_near_price(df, current_price, 'bearish')
                    
                    retrace_pct = ((recent_high - current_price) / swing_range) * 100 if swing_range > 0 else 0
                    displacement = self._check_displacement(df)
                    time_aligned = self._check_time_alignment(timestamp) if current_time else False
                    
                    setup = Model2022Setup(
                        timestamp=timestamp,
                        direction='short',
                        liquidity_side='buy_side',
                        liquidity_level=recent_high,
                        liquidity_taken=True,
                        zone_type='premium',
                        zone_high=recent_high,
                        zone_low=premium_boundary,
                        retracement_percent=retrace_pct,
                        ote_high=ote_high,
                        ote_low=ote_low,
                        ote_mid=ote_mid,
                        in_ote=in_ote,
                        entry_type=EntryType.FVG if fvg_present else EntryType.ORDER_BLOCK,
                        entry_price=current_price,
                        stop_loss=recent_high + (swing_range * 0.05),
                        target_1=swing_low,
                        target_2=swing_low - (swing_range * 0.5),
                        stage_1_liquidity=True,
                        stage_2_retrace=in_premium,
                        stage_3_structure=structure_shift,
                        stage_4_entry=(fvg_present or ob_present),
                        structure_shift=structure_shift,
                        structure_shift_type=shift_type,
                        displacement_present=displacement,
                        time_aligned=time_aligned,
                        has_fvg=fvg_present,
                        has_order_block=ob_present,
                        confluence_count=sum([fvg_present, ob_present, in_ote, structure_shift]),
                        notes="2022 Model SHORT - Buy-side taken, in premium"
                    )
                    
                    setup.calculate_quality()
                    return setup
                    
        except Exception as e:
            print(f"Error in Model 2022 analysis: {e}")
            
        return None
    
    # =========================================================================
    # SILVER BULLET ANALYSIS
    # =========================================================================
    
    def analyze_silver_bullet(self, df: pd.DataFrame, 
                             current_time: datetime,
                             session: SilverBulletSession = SilverBulletSession.NY_AM) -> Optional[SilverBulletSetup]:
        """
        Analyze for Silver Bullet setup.
        
        ICT Key: First Presented FVG (FPFVG) after window opens
        
        Args:
            df: OHLC dataframe
            current_time: Current datetime
            session: Which Silver Bullet window
            
        Returns:
            SilverBulletSetup or None
        """
        if len(df) < 30:
            return None
            
        window_start, window_end = self.silver_bullet_windows[session]
        current_t = current_time.time()
        
        # Check if in proper time window
        in_window = window_start <= current_t <= window_end
        
        if not in_window:
            return None
            
        # Find First Presented FVG after window start
        first_fvg = self._find_first_fvg_after_time(df, window_start)
        
        if first_fvg is None:
            return None
            
        fvg_price, fvg_high, fvg_low, fvg_type = first_fvg
        direction = 'long' if fvg_type == 'bullish' else 'short'
        
        # Find swing points for projection
        swing_high = df['high'].tail(20).max()
        swing_low = df['low'].tail(20).min()
        swing_range = swing_high - swing_low
        
        # Calculate standard deviation target
        if direction == 'short':
            std_dev_target = swing_low - swing_range  # -1 std dev below low
            target = swing_low
            stop_loss = fvg_high + (swing_range * 0.1)
        else:
            std_dev_target = swing_high + swing_range  # +1 std dev above high
            target = swing_high
            stop_loss = fvg_low - (swing_range * 0.1)
            
        # Check for structure shift and displacement
        structure_shift, _ = self._check_structure_shift(df, 'bullish' if direction == 'long' else 'bearish')
        displacement = self._check_displacement(df)
        
        # Check if liquidity was swept
        liquidity_swept = self._check_liquidity_sweep(df, direction)
        
        # Calculate favorability
        favorability = 70.0
        if structure_shift:
            favorability += 10
        if displacement:
            favorability += 10
        if liquidity_swept:
            favorability += 10
            
        quality = SetupQuality.A if favorability >= 90 else (SetupQuality.B if favorability >= 75 else SetupQuality.C)
        
        return SilverBulletSetup(
            timestamp=current_time,
            session=session,
            time_window_start=window_start,
            time_window_end=window_end,
            direction=direction,
            first_fvg_price=fvg_price,
            first_fvg_high=fvg_high,
            first_fvg_low=fvg_low,
            first_fvg_type=fvg_type,
            is_first_fvg=True,
            entry_price=fvg_price,
            stop_loss=stop_loss,
            target=target,
            swing_high=swing_high,
            swing_low=swing_low,
            std_dev_target=std_dev_target,
            in_proper_time=True,
            has_structure_shift=structure_shift,
            has_displacement=displacement,
            liquidity_swept=liquidity_swept,
            quality=quality,
            favorability=favorability,
            notes=f"Silver Bullet {session.value.upper()} - FPFVG {fvg_type}"
        )
    
    # =========================================================================
    # VENOM ANALYSIS
    # =========================================================================
    
    def analyze_venom(self, df: pd.DataFrame, bias: str = 'bullish') -> Optional[VenomSetup]:
        """
        Analyze for Venom setup - Single pass liquidity injection.
        
        STRICT CRITERIA (ICT):
        1. ONE candle penetrates liquidity
        2. ONE candle returns
        3. CIBI (bearish) or BISI (bullish) must be present
        
        Args:
            df: OHLC dataframe
            bias: 'bullish' or 'bearish'
            
        Returns:
            VenomSetup or None
        """
        if len(df) < 20:
            return None
            
        if bias == 'bullish':
            # Venom LONG: Single pass below liquidity, return above
            liquidity_level = df['low'].iloc[:-5].tail(15).min()
            
            for i in range(len(df) - 6, len(df) - 2):
                if i < 1 or i + 2 >= len(df):
                    continue
                    
                # Check if this candle penetrated liquidity
                if df['low'].iloc[i] <= liquidity_level:
                    # Validate SINGLE PASS criteria
                    prev_above = df['low'].iloc[i-1] > liquidity_level
                    next_above = df['low'].iloc[i+1] > liquidity_level
                    
                    if prev_above and next_above:
                        # Single pass confirmed
                        injection_idx = i
                        
                        # Check for BISI (Buy Imbalance Sell Efficiency)
                        has_bisi = False
                        bisi_high = 0.0
                        bisi_low = 0.0
                        
                        if i + 2 < len(df):
                            # BISI: Low of candle+2 > High of candle
                            if df['low'].iloc[i+2] > df['high'].iloc[i]:
                                has_bisi = True
                                bisi_low = df['high'].iloc[i]
                                bisi_high = df['low'].iloc[i+2]
                        
                        if has_bisi:
                            entry_price = df['close'].iloc[injection_idx]
                            stop_loss = df['low'].iloc[injection_idx] - (df['high'].iloc[injection_idx] - df['low'].iloc[injection_idx]) * 0.5
                            target = df['high'].tail(10).max()
                            
                            # Count consolidation bars
                            consol_bars = self._count_consolidation_bars(df, injection_idx)
                            
                            setup = VenomSetup(
                                timestamp=datetime.now(),
                                direction='long',
                                liquidity_level=liquidity_level,
                                injection_candle_idx=injection_idx,
                                injection_candle_high=df['high'].iloc[injection_idx],
                                injection_candle_low=df['low'].iloc[injection_idx],
                                injection_candle_close=df['close'].iloc[injection_idx],
                                single_pass_to_liquidity=True,
                                single_pass_return=True,
                                has_cibi=False,
                                has_bisi=True,
                                imbalance_high=bisi_high,
                                imbalance_low=bisi_low,
                                entry_price=entry_price,
                                stop_loss=stop_loss,
                                target=target,
                                consolidation_bars=consol_bars,
                                notes="Venom LONG - Single pass injection with BISI"
                            )
                            setup.validate()
                            return setup
                            
        else:  # bearish
            # Venom SHORT: Single pass above liquidity, return below
            liquidity_level = df['high'].iloc[:-5].tail(15).max()
            
            for i in range(len(df) - 6, len(df) - 2):
                if i < 1 or i + 2 >= len(df):
                    continue
                    
                if df['high'].iloc[i] >= liquidity_level:
                    prev_below = df['high'].iloc[i-1] < liquidity_level
                    next_below = df['high'].iloc[i+1] < liquidity_level
                    
                    if prev_below and next_below:
                        injection_idx = i
                        
                        # Check for CIBI (Sell Imbalance Buy Efficiency)
                        has_cibi = False
                        cibi_high = 0.0
                        cibi_low = 0.0
                        
                        if i + 2 < len(df):
                            # CIBI: High of candle+2 < Low of candle
                            if df['high'].iloc[i+2] < df['low'].iloc[i]:
                                has_cibi = True
                                cibi_high = df['low'].iloc[i]
                                cibi_low = df['high'].iloc[i+2]
                        
                        if has_cibi:
                            entry_price = df['close'].iloc[injection_idx]
                            stop_loss = df['high'].iloc[injection_idx] + (df['high'].iloc[injection_idx] - df['low'].iloc[injection_idx]) * 0.5
                            target = df['low'].tail(10).min()
                            
                            consol_bars = self._count_consolidation_bars(df, injection_idx)
                            
                            setup = VenomSetup(
                                timestamp=datetime.now(),
                                direction='short',
                                liquidity_level=liquidity_level,
                                injection_candle_idx=injection_idx,
                                injection_candle_high=df['high'].iloc[injection_idx],
                                injection_candle_low=df['low'].iloc[injection_idx],
                                injection_candle_close=df['close'].iloc[injection_idx],
                                single_pass_to_liquidity=True,
                                single_pass_return=True,
                                has_cibi=True,
                                has_bisi=False,
                                imbalance_high=cibi_high,
                                imbalance_low=cibi_low,
                                entry_price=entry_price,
                                stop_loss=stop_loss,
                                target=target,
                                consolidation_bars=consol_bars,
                                notes="Venom SHORT - Single pass injection with CIBI"
                            )
                            setup.validate()
                            return setup
                            
        return None
    
    # =========================================================================
    # TURTLE SOUP ANALYSIS
    # =========================================================================
    
    def analyze_turtle_soup(self, df: pd.DataFrame) -> Optional[TurtleSoupSetup]:
        """
        Analyze for Turtle Soup setup - Stop hunt reversal.
        
        ICT Criteria:
        - Old high/low violated with WICK only
        - Body does NOT close beyond
        - Quick reversal
        - Range-bound context
        
        Args:
            df: OHLC dataframe
            
        Returns:
            TurtleSoupSetup or None
        """
        if len(df) < 30:
            return None
            
        # Check if market is range-bound
        recent_range = df['high'].tail(20).max() - df['low'].tail(20).min()
        avg_candle_range = (df['high'] - df['low']).tail(20).mean()
        
        # Range-bound: overall range < 15x average candle range
        is_range_bound = recent_range < (avg_candle_range * 15)
        
        if not is_range_bound:
            return None
            
        # Look for Turtle Soup LONG (hunt below old low)
        old_low = df['low'].iloc[-20:-5].min()
        recent_candles = df.iloc[-5:]
        
        for i, row in recent_candles.iterrows():
            if row['low'] < old_low and row['close'] > old_low:
                # Wick below, body above - potential Turtle Soup Long
                # Check for quick reversal
                idx = df.index.get_loc(i)
                if idx < len(df) - 1:
                    next_close = df['close'].iloc[idx + 1] if idx + 1 < len(df) else row['close']
                    quick_reversal = next_close > row['close']
                    
                    if quick_reversal:
                        # Find target (opposite liquidity)
                        target = df['high'].tail(20).max()
                        
                        # Check for FVG support
                        fvg_support = self._find_fvg_near_price(df, row['close'])
                        
                        return TurtleSoupSetup(
                            timestamp=datetime.now(),
                            hunt_type='long',
                            old_level=old_low,
                            new_extreme=row['low'],
                            violation_size=old_low - row['low'],
                            old_level_violated=True,
                            is_wick_only=True,
                            quick_reversal=True,
                            opposite_liquidity_available=True,
                            in_range=True,
                            entry_price=row['close'],
                            stop_loss=row['low'] - avg_candle_range,
                            target=target,
                            has_fvg_support=fvg_support,
                            quality=SetupQuality.A if fvg_support else SetupQuality.B,
                            favorability=85.0 if fvg_support else 75.0,
                            notes="Turtle Soup LONG - False break below old low"
                        )
        
        # Look for Turtle Soup SHORT (hunt above old high)
        old_high = df['high'].iloc[-20:-5].max()
        
        for i, row in recent_candles.iterrows():
            if row['high'] > old_high and row['close'] < old_high:
                idx = df.index.get_loc(i)
                if idx < len(df) - 1:
                    next_close = df['close'].iloc[idx + 1] if idx + 1 < len(df) else row['close']
                    quick_reversal = next_close < row['close']
                    
                    if quick_reversal:
                        target = df['low'].tail(20).min()
                        fvg_support = self._find_fvg_near_price(df, row['close'])
                        
                        return TurtleSoupSetup(
                            timestamp=datetime.now(),
                            hunt_type='short',
                            old_level=old_high,
                            new_extreme=row['high'],
                            violation_size=row['high'] - old_high,
                            old_level_violated=True,
                            is_wick_only=True,
                            quick_reversal=True,
                            opposite_liquidity_available=True,
                            in_range=True,
                            entry_price=row['close'],
                            stop_loss=row['high'] + avg_candle_range,
                            target=target,
                            has_fvg_support=fvg_support,
                            quality=SetupQuality.A if fvg_support else SetupQuality.B,
                            favorability=85.0 if fvg_support else 75.0,
                            notes="Turtle Soup SHORT - False break above old high"
                        )
                        
        return None
    
    # =========================================================================
    # POWER OF THREE (AMD) ANALYSIS
    # =========================================================================
    
    def analyze_power_of_three(self, df: pd.DataFrame, 
                               timeframe: str = 'daily') -> Optional[PowerOfThreeSetup]:
        """
        Analyze for Power of Three (AMD) pattern.
        
        ICT Pattern:
        - Bullish Day: Open near low, close near high (O-L-H-C)
        - Bearish Day: Open near high, close near low (O-H-L-C)
        
        Args:
            df: OHLC dataframe
            timeframe: 'daily', 'session', 'hour'
            
        Returns:
            PowerOfThreeSetup or None
        """
        if len(df) < 1:
            return None
            
        # Use last complete candle
        candle = df.iloc[-1]
        
        o = candle['open']
        h = candle['high']
        l = candle['low']
        c = candle['close']
        
        range_size = h - l
        if range_size == 0:
            return None
            
        setup = PowerOfThreeSetup(
            timestamp=datetime.now(),
            timeframe=timeframe,
            direction='',
            open_price=o,
            high_price=h,
            low_price=l,
            close_price=c,
            accumulation_zone_high=0.0,
            accumulation_zone_low=0.0,
            manipulation_level=0.0
        )
        
        setup.validate_amd_pattern(tolerance=0.3)
        
        if setup.is_valid_amd:
            if setup.direction == 'bullish':
                # Bullish AMD: Accumulation at low, manipulation down, distribution up
                setup.accumulation_zone_low = l
                setup.accumulation_zone_high = l + (range_size * 0.25)
                setup.manipulation_level = l  # Fake move down
                setup.distribution_zone_low = h - (range_size * 0.25)
                setup.distribution_zone_high = h
                setup.quality = SetupQuality.A
                setup.favorability = 85.0
                setup.notes = "Bullish AMD - Open near low, close near high"
            else:
                # Bearish AMD: Accumulation at high, manipulation up, distribution down
                setup.accumulation_zone_high = h
                setup.accumulation_zone_low = h - (range_size * 0.25)
                setup.manipulation_level = h  # Fake move up
                setup.distribution_zone_high = l + (range_size * 0.25)
                setup.distribution_zone_low = l
                setup.quality = SetupQuality.A
                setup.favorability = 85.0
                setup.notes = "Bearish AMD - Open near high, close near low"
                
            return setup
            
        return None
    
    # =========================================================================
    # SWING PROJECTION
    # =========================================================================
    
    def calculate_swing_projection(self, swing_high: float, swing_low: float,
                                  direction: str,
                                  fulcrum: Optional[float] = None) -> SwingProjection:
        """
        Calculate ICT Swing Projection with standard deviations.
        
        ICT Quote: "Take the high here and add your FIB to it here and draw 
        it down to that low one standard deviation would be 42.78.75"
        
        Args:
            swing_high: High of the swing
            swing_low: Low of the swing
            direction: 'bullish' or 'bearish'
            fulcrum: The fulcrum point (hinge of the door)
            
        Returns:
            SwingProjection
        """
        if fulcrum is None:
            fulcrum = swing_low if direction == 'bearish' else swing_high
            
        return SwingProjection(
            timestamp=datetime.now(),
            direction=direction,
            swing_high=swing_high,
            swing_low=swing_low,
            fulcrum_price=fulcrum,
            notes=f"Swing projection {direction} - Fulcrum at {fulcrum:.2f}"
        )
    
    # =========================================================================
    # OPTIMAL TRADE ENTRY
    # =========================================================================
    
    def analyze_optimal_trade_entry(self, df: pd.DataFrame,
                                   bias: str) -> Optional[OptimalTradeEntry]:
        """
        Analyze for Optimal Trade Entry (OTE) - 62-79% retracement zone.
        
        ICT Quote: "Very minimum it's got to go above the low to high. 
        It's got to go to 50% or higher for it to be in a premium market."
        
        Args:
            df: OHLC dataframe
            bias: 'bullish' or 'bearish'
            
        Returns:
            OptimalTradeEntry or None
        """
        if len(df) < 20:
            return None
            
        swing_high = df['high'].tail(20).max()
        swing_low = df['low'].tail(20).min()
        current_price = df['close'].iloc[-1]
        
        ote = OptimalTradeEntry(
            impulse_high=swing_high,
            impulse_low=swing_low,
            current_price=current_price
        )
        
        swing_range = swing_high - swing_low
        
        if bias == 'bullish':
            # For bullish OTE, looking for retracement from high to OTE zone
            retrace = (swing_high - current_price) / swing_range if swing_range > 0 else 0
            ote.retracement_percent = retrace * 100
            ote.in_ote_zone = 0.62 <= retrace <= 0.79
            ote.direction = 'long'
            ote.entry_price = current_price
            ote.stop_loss = swing_low - (swing_range * 0.05)
            ote.target = swing_high
            
        else:  # bearish
            retrace = (current_price - swing_low) / swing_range if swing_range > 0 else 0
            ote.retracement_percent = retrace * 100
            ote.in_ote_zone = 0.62 <= retrace <= 0.79
            ote.direction = 'short'
            ote.entry_price = current_price
            ote.stop_loss = swing_high + (swing_range * 0.05)
            ote.target = swing_low
        
        if ote.in_ote_zone:
            ote.fvg_in_zone = self._find_fvg_near_price(df, current_price)
            ote.order_block_in_zone = self._find_order_block_near_price(df, current_price, bias)
            
            favorability = 75.0
            if ote.fvg_in_zone:
                favorability += 10
            if ote.order_block_in_zone:
                favorability += 10
            if 0.68 <= ote.retracement_percent/100 <= 0.72:  # Near sweet spot
                favorability += 5
                
            ote.favorability = favorability
            ote.quality = SetupQuality.A if favorability >= 90 else SetupQuality.B
            ote.notes = f"OTE {ote.direction} - {ote.retracement_percent:.1f}% retracement"
            
            return ote
            
        return None
    
    # =========================================================================
    # HELPER METHODS
    # =========================================================================
    
    def _check_structure_shift(self, df: pd.DataFrame, bias: str) -> Tuple[bool, str]:
        """Check for market structure shift (CHoCH or BOS)"""
        if len(df) < 15:
            return False, ""
            
        if bias == 'bullish':
            # Look for break of previous lower high
            recent_highs = []
            for i in range(len(df) - 10, len(df) - 3):
                if i > 0 and i < len(df) - 1:
                    if df['high'].iloc[i] > df['high'].iloc[i-1] and df['high'].iloc[i] > df['high'].iloc[i+1]:
                        recent_highs.append(df['high'].iloc[i])
            
            if len(recent_highs) >= 2 and df['close'].iloc[-1] > recent_highs[-1]:
                return True, "CHoCH"  # Change of Character
                
        else:  # bearish
            recent_lows = []
            for i in range(len(df) - 10, len(df) - 3):
                if i > 0 and i < len(df) - 1:
                    if df['low'].iloc[i] < df['low'].iloc[i-1] and df['low'].iloc[i] < df['low'].iloc[i+1]:
                        recent_lows.append(df['low'].iloc[i])
            
            if len(recent_lows) >= 2 and df['close'].iloc[-1] < recent_lows[-1]:
                return True, "CHoCH"
                
        return False, ""
    
    def _check_displacement(self, df: pd.DataFrame, threshold: float = 0.005) -> bool:
        """Check for strong displacement move"""
        if len(df) < 5:
            return False
            
        recent = df.tail(5)
        move = abs(recent['close'].iloc[-1] - recent['close'].iloc[0])
        pct_move = move / recent['close'].iloc[0] if recent['close'].iloc[0] > 0 else 0
        
        return pct_move > threshold
    
    def _check_time_alignment(self, current_time: datetime) -> bool:
        """Check if current time is in a kill zone or macro time"""
        t = current_time.time()
        
        # Check kill zones
        for zone_name, (start, end) in self.kill_zones.items():
            if start <= t <= end:
                return True
                
        # Check macro times
        for start, end in self.macro_times:
            if start <= t <= end:
                return True
                
        return False
    
    def _find_fvg_near_price(self, df: pd.DataFrame, price: float, tolerance: float = 0.002) -> bool:
        """Check if FVG exists near given price"""
        for i in range(len(df) - 3, max(0, len(df) - 20), -1):
            # Bullish FVG
            if df['low'].iloc[i+2] > df['high'].iloc[i]:
                fvg_low = df['high'].iloc[i]
                fvg_high = df['low'].iloc[i+2]
                if fvg_low <= price <= fvg_high:
                    return True
                if abs(price - fvg_low) / price < tolerance:
                    return True
                    
            # Bearish FVG
            if df['high'].iloc[i+2] < df['low'].iloc[i]:
                fvg_high = df['low'].iloc[i]
                fvg_low = df['high'].iloc[i+2]
                if fvg_low <= price <= fvg_high:
                    return True
                if abs(price - fvg_high) / price < tolerance:
                    return True
                    
        return False
    
    def _find_first_fvg_after_time(self, df: pd.DataFrame, 
                                   after_time: time) -> Optional[Tuple[float, float, float, str]]:
        """Find first FVG that formed after specified time"""
        for i in range(max(0, len(df) - 15), len(df) - 2):
            # Bullish FVG
            if df['low'].iloc[i+2] > df['high'].iloc[i]:
                fvg_low = df['high'].iloc[i]
                fvg_high = df['low'].iloc[i+2]
                fvg_mid = (fvg_high + fvg_low) / 2
                return (fvg_mid, fvg_high, fvg_low, 'bullish')
                
            # Bearish FVG
            if df['high'].iloc[i+2] < df['low'].iloc[i]:
                fvg_high = df['low'].iloc[i]
                fvg_low = df['high'].iloc[i+2]
                fvg_mid = (fvg_high + fvg_low) / 2
                return (fvg_mid, fvg_high, fvg_low, 'bearish')
                
        return None
    
    def _find_order_block_near_price(self, df: pd.DataFrame, price: float, bias: str) -> bool:
        """Check if order block exists near price"""
        tolerance = abs(df['close'].iloc[-1] - df['open'].iloc[-1]) * 3
        
        for i in range(len(df) - 2, max(0, len(df) - 15), -1):
            if bias == 'bullish':
                # Bullish OB: Last down-close candle before up move
                if df['close'].iloc[i] < df['open'].iloc[i]:  # Down candle
                    ob_entry = df['open'].iloc[i]
                    if abs(price - ob_entry) < tolerance:
                        return True
            else:
                # Bearish OB: Last up-close candle before down move
                if df['close'].iloc[i] > df['open'].iloc[i]:  # Up candle
                    ob_entry = df['open'].iloc[i]
                    if abs(price - ob_entry) < tolerance:
                        return True
                        
        return False
    
    def _check_liquidity_sweep(self, df: pd.DataFrame, direction: str) -> bool:
        """Check if liquidity was swept recently"""
        if len(df) < 10:
            return False
            
        if direction == 'long':
            old_low = df['low'].iloc[-15:-5].min()
            swept = df['low'].iloc[-5:].min() < old_low
            returned = df['close'].iloc[-1] > old_low
            return swept and returned
        else:
            old_high = df['high'].iloc[-15:-5].max()
            swept = df['high'].iloc[-5:].max() > old_high
            returned = df['close'].iloc[-1] < old_high
            return swept and returned
    
    def _count_consolidation_bars(self, df: pd.DataFrame, start_idx: int) -> int:
        """Count bars in consolidation after a move"""
        if start_idx >= len(df) - 1:
            return 0
            
        ref_range = df['high'].iloc[start_idx] - df['low'].iloc[start_idx]
        count = 0
        
        for i in range(start_idx + 1, len(df)):
            move = abs(df['close'].iloc[i] - df['close'].iloc[start_idx])
            if move < ref_range * 0.5:
                count += 1
            else:
                break
                
        return count
    
    # =========================================================================
    # COMPREHENSIVE ANALYSIS
    # =========================================================================
    
    def analyze_all_models(self, df: pd.DataFrame, bias: str,
                          current_time: Optional[datetime] = None) -> Dict[str, any]:
        """
        Run all model analyses and return results.
        
        Args:
            df: OHLC dataframe
            bias: 'bullish' or 'bearish'
            current_time: Current datetime
            
        Returns:
            Dictionary of all model results
        """
        results = {
            'model_2022': None,
            'silver_bullet': None,
            'venom': None,
            'turtle_soup': None,
            'power_of_three': None,
            'optimal_trade_entry': None,
            'timestamp': datetime.now(),
            'bias': bias
        }
        
        # Model 2022
        results['model_2022'] = self.analyze_model_2022(df, bias, current_time)
        
        # Silver Bullet (if in time window)
        if current_time:
            for session in SilverBulletSession:
                sb = self.analyze_silver_bullet(df, current_time, session)
                if sb:
                    results['silver_bullet'] = sb
                    break
        
        # Venom
        results['venom'] = self.analyze_venom(df, bias)
        
        # Turtle Soup
        results['turtle_soup'] = self.analyze_turtle_soup(df)
        
        # Power of Three
        results['power_of_three'] = self.analyze_power_of_three(df)
        
        # OTE
        results['optimal_trade_entry'] = self.analyze_optimal_trade_entry(df, bias)
        
        return results
    
    def print_analysis_summary(self, results: Dict[str, any]):
        """Print formatted summary of all analyses"""
        print(f"\n{'='*70}")
        print(f"ICT TRADING MODELS ANALYSIS SUMMARY")
        print(f"Time: {results['timestamp']}")
        print(f"Bias: {results['bias'].upper()}")
        print(f"{'='*70}")
        
        for model_name, setup in results.items():
            if model_name in ['timestamp', 'bias']:
                continue
                
            print(f"\n{model_name.upper().replace('_', ' ')}:")
            print("-" * 40)
            
            if setup is None:
                print("  No setup detected")
            else:
                if hasattr(setup, 'quality'):
                    print(f"  Quality: {setup.quality.value}")
                if hasattr(setup, 'favorability'):
                    print(f"  Favorability: {setup.favorability:.1f}%")
                if hasattr(setup, 'direction'):
                    print(f"  Direction: {setup.direction}")
                if hasattr(setup, 'entry_price'):
                    print(f"  Entry: {setup.entry_price:.5f}")
                if hasattr(setup, 'stop_loss'):
                    print(f"  Stop Loss: {setup.stop_loss:.5f}")
                if hasattr(setup, 'target'):
                    print(f"  Target: {setup.target:.5f}")
                if hasattr(setup, 'notes'):
                    print(f"  Notes: {setup.notes}")
                    
        print(f"\n{'='*70}\n")
    
    @staticmethod
    def get_model_trading_rules() -> Dict[str, List[str]]:
        """Return comprehensive trading rules for all models"""
        return {
            "Model_2022_Rules": [
                "1. Identify bias on higher timeframe (daily)",
                "2. Wait for liquidity sweep (buy-side for shorts, sell-side for longs)",
                "3. Price must retrace into premium/discount (50%+ retracement)",
                "4. Look for structure shift (CHoCH or BOS)",
                "5. Entry at PD array (FVG, OB, or Breaker)",
                "6. Stop beyond liquidity sweep",
                "7. Target: Opposite liquidity pool",
                "8. Must occur during kill zone for highest probability"
            ],
            
            "Silver_Bullet_Rules": [
                "1. Trade ONLY during designated windows",
                "   - Asian: 20:00-21:00 EST",
                "   - London: 03:00-04:00 EST",
                "   - AM: 10:00-11:00 EST (most popular)",
                "   - PM: 14:00-15:00 EST",
                "2. Look for FIRST PRESENTED FVG after window opens",
                "3. Must have structure shift and displacement",
                "4. Entry at the FVG",
                "5. Use standard deviation for targets"
            ],
            
            "Venom_Rules": [
                "1. STRICT: Single candle penetrates liquidity",
                "2. STRICT: Single candle returns above/below",
                "3. Must have BISI (bullish) or CIBI (bearish)",
                "4. Entry at injection candle close or better",
                "5. Stop beyond injection candle extreme",
                "6. High probability setup when criteria met"
            ],
            
            "Turtle_Soup_Rules": [
                "1. Market must be range-bound",
                "2. Old high/low violated by WICK only",
                "3. Body must NOT close beyond",
                "4. Quick reversal required",
                "5. Target: Opposite liquidity pool",
                "6. Best during consolidation periods"
            ],
            
            "Power_of_Three_Rules": [
                "1. Understand daily candle formation",
                "2. Bullish Day: Open near low, close near high",
                "3. Bearish Day: Open near high, close near low",
                "4. Accumulation → Manipulation → Distribution",
                "5. First move is often the manipulation (fake)",
                "6. Trade the distribution phase"
            ],
            
            "OTE_Rules": [
                "1. 62-79% retracement zone is optimal",
                "2. 70.5% is the sweet spot",
                "3. Must be in premium (shorts) or discount (longs)",
                "4. Look for PD arrays within OTE zone",
                "5. FVG or OB in zone increases probability"
            ]
        }


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    print("ICT Trading Models Handler - Comprehensive Implementation")
    print("=" * 60)
    
    handler = TradingModelsHandler()
    
    # Print trading rules
    print("\nICT TRADING MODEL RULES:")
    rules = handler.get_model_trading_rules()
    for model, model_rules in rules.items():
        print(f"\n{model.replace('_', ' ')}:")
        for rule in model_rules:
            print(f"  {rule}")
    
    print("\n" + "=" * 60)
    print("Handler initialized and ready for use!")
