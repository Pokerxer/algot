"""
Comprehensive ICT Fair Value Gap (FVG) Handler
Based on Inner Circle Trader methodology and transcript teachings

Key ICT Concepts Implemented:
- SIBI (Sell-side Imbalance Buy-side Inefficiency) - Bearish FVG
- BISI (Buy-side Imbalance Sell-side Inefficiency) - Bullish FVG
- Consequent Encroachment (CE) - The 50% midpoint of any inefficiency
- Quadrant Levels - 25% and 75% levels within FVG
- Inversion Fair Value Gap - FVG that changes characteristic
- Reclaimed Fair Value Gap - FVG that regains original characteristic
- High Probability FVG - FVG overlaying a quadrant level
- Volume Imbalance (VI) - Gap between candle bodies
- Suspension Block - Candle with VI at both high and low
- First Presented FVG - First FVG after setup (highest probability)
- IOFED - Institutional Order Flow Entry Drill (partial fill)
- Breakaway Gap - Between quadrant levels, doesn't fill
- Measuring Gap - Used for projection (halfway point of move)
- Common Gap - Can be reclaimed/traded multiple times
- Premium/Discount Sensitivity - FVG position in dealing range

Author: ICT Signal Engine
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Union
from enum import Enum
from datetime import datetime, time
import pandas as pd
import numpy as np


# ==================== ENUMS ====================

class FVGType(Enum):
    """FVG types using ICT terminology"""
    BISI = "bisi"       # Buy-side Imbalance Sell-side Inefficiency (Bullish)
    SIBI = "sibi"       # Sell-side Imbalance Buy-side Inefficiency (Bearish)
    BULLISH = "bullish" # Alias for BISI
    BEARISH = "bearish" # Alias for SIBI


class FVGStatus(Enum):
    """FVG status states"""
    ACTIVE = "active"                   # FVG is still valid and unfilled
    FILLED = "filled"                   # Price traded through the FVG  
    PARTIALLY_FILLED = "partial"        # Price tested but didn't fully fill (IOFED potential)
    INVERTED = "inverted"               # FVG changed polarity
    RECLAIMED = "reclaimed"             # FVG regained original characteristic
    EXPIRED = "expired"                 # Too old or invalidated


class FVGClassification(Enum):
    """ICT Gap Classification"""
    BREAKAWAY = "breakaway"     # Between quadrants, doesn't fill
    MEASURING = "measuring"     # Halfway point, used for projection
    COMMON = "common"           # Can be reclaimed multiple times
    EXHAUSTION = "exhaustion"   # End of move, followed by reversal


class FVGProbability(Enum):
    """FVG probability rating based on ICT criteria"""
    HIGH = "high"           # Overlays quadrant level
    MEDIUM = "medium"       # Standard FVG with confluence
    LOW = "low"             # No special confluence
    AVOID = "avoid"         # Should not be traded


class PremiumDiscountZone(Enum):
    """Position in dealing range"""
    PREMIUM = "premium"         # Above 50% of dealing range
    DISCOUNT = "discount"       # Below 50% of dealing range
    EQUILIBRIUM = "equilibrium" # At 50%


# ==================== DATA CLASSES ====================

@dataclass
class VolumeImbalance:
    """
    Volume Imbalance Structure
    Gap between candle BODIES (not wicks)
    ICT: "The difference between two bodies that are not touching"
    """
    index: int
    imbalance_type: str  # 'bullish' or 'bearish'
    high: float          # Top of body gap
    low: float           # Bottom of body gap
    mid_point: float     # Consequent encroachment
    size: float
    is_filled: bool = False
    timestamp: Optional[datetime] = None
    
    def __str__(self):
        return f"VI {self.imbalance_type.upper()}: {self.low:.5f} - {self.high:.5f}"


@dataclass
class SuspensionBlock:
    """
    ICT Suspension Block
    A single candle with volume imbalance at BOTH the high AND the low
    "Suspended between two volume imbalances"
    ICT: "This is extremely strong. It's one of the most powerful."
    """
    index: int
    candle_open: float
    candle_high: float
    candle_low: float
    candle_close: float
    upper_vi: VolumeImbalance      # Volume imbalance at the high
    lower_vi: VolumeImbalance      # Volume imbalance at the low
    mid_point: float               # Consequent encroachment of entire block
    upper_quadrant: float
    lower_quadrant: float
    is_bullish: bool
    is_filled: bool = False
    timestamp: Optional[datetime] = None
    
    def __str__(self):
        direction = "BULLISH" if self.is_bullish else "BEARISH"
        return f"SUSPENSION BLOCK {direction}: {self.candle_low:.5f} - {self.candle_high:.5f}"


@dataclass
class FairValueGap:
    """
    Complete ICT Fair Value Gap structure with all properties
    
    ICT Terminology:
    - BISI: Buy-side Imbalance Sell-side Inefficiency (Bullish FVG)
    - SIBI: Sell-side Imbalance Buy-side Inefficiency (Bearish FVG)
    - Consequent Encroachment (CE): The 50% midpoint
    - Upper Quadrant: 75% level
    - Lower Quadrant: 25% level
    """
    # Core properties
    start_index: int
    end_index: int
    gap_type: str               # 'bullish'/'bisi' or 'bearish'/'sibi'
    high: float
    low: float
    
    # ICT Key Levels
    consequent_encroachment: float  # The 50% midpoint (CE)
    upper_quadrant: float           # 75% level
    lower_quadrant: float           # 25% level
    size: float = 0.0
    
    # Aliases for compatibility
    @property
    def mid_point(self) -> float:
        """Alias for consequent_encroachment"""
        return self.consequent_encroachment
    
    # Status tracking
    status: FVGStatus = FVGStatus.ACTIVE
    is_filled: bool = False
    is_inverted: bool = False
    is_reclaimed: bool = False      # NEW: Regained original characteristic
    is_partially_filled: bool = False
    fill_percentage: float = 0.0
    
    # ICT-specific classifications
    classification: FVGClassification = FVGClassification.COMMON
    probability: FVGProbability = FVGProbability.MEDIUM
    zone: PremiumDiscountZone = PremiumDiscountZone.EQUILIBRIUM
    
    is_breakaway: bool = False
    is_measuring: bool = False
    is_exhaustion: bool = False
    is_first_presented: bool = False    # First FVG after setup
    is_high_probability: bool = False   # Overlays quadrant level
    
    # Body respect tracking (ICT: "Bodies tell the story")
    body_respects: int = 0              # Times bodies respected the level
    body_violations: int = 0            # Times bodies violated the level
    wick_only_touches: int = 0          # Times only wicks touched (respect)
    
    # IOFED tracking (Institutional Order Flow Entry Drill)
    has_iofed_entry: bool = False       # Partial fill entry opportunity
    iofed_level: Optional[float] = None
    
    # Confluence factors
    touches: int = 0
    rejection_count: int = 0
    has_order_block: bool = False
    has_liquidity_above: bool = False
    has_liquidity_below: bool = False
    overlays_quadrant: bool = False     # NEW: FVG lays on quadrant level
    quadrant_level_overlaid: Optional[float] = None
    
    # Inversion tracking
    inversion_count: int = 0            # Times it has inverted
    original_type: Optional[str] = None # Original gap type before inversion
    
    # Dealing range context
    dealing_range_high: Optional[float] = None
    dealing_range_low: Optional[float] = None
    position_in_range: Optional[float] = None  # 0-100 (0=bottom, 100=top)
    
    # Metadata
    timestamp: Optional[datetime] = None
    timeframe: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate derived values"""
        self.size = self.high - self.low
        self.original_type = self.gap_type
        
    def __str__(self):
        ict_name = "BISI" if self.gap_type in ['bullish', 'bisi'] else "SIBI"
        status = self.status.value.upper()
        prob = f"[{self.probability.value.upper()}]" if self.is_high_probability else ""
        return (f"{ict_name} FVG [{status}] {prob}: {self.low:.5f} - {self.high:.5f} "
                f"(CE: {self.consequent_encroachment:.5f})")
    
    def get_entry_level(self, entry_type: str = "ce") -> float:
        """
        Get optimal entry level for trading this FVG
        
        ICT Entry Types:
        - "ce": Consequent Encroachment (50%) - Standard entry
        - "aggressive": Upper/Lower quadrant (75%/25%) - Earlier entry
        - "conservative": Beyond CE toward invalidation
        - "iofed": Institutional Order Flow Entry Drill (partial fill)
        
        Args:
            entry_type: Type of entry ("ce", "aggressive", "conservative", "iofed")
            
        Returns:
            Entry price level
        """
        if entry_type == "ce":
            return self.consequent_encroachment
        elif entry_type == "aggressive":
            if self.gap_type in ['bullish', 'bisi']:
                return self.lower_quadrant  # Enter at 25% for longs
            else:
                return self.upper_quadrant  # Enter at 75% for shorts
        elif entry_type == "conservative":
            if self.gap_type in ['bullish', 'bisi']:
                return self.low + (self.size * 0.1)  # Near bottom
            else:
                return self.high - (self.size * 0.1)  # Near top
        elif entry_type == "iofed" and self.has_iofed_entry:
            return self.iofed_level
        else:
            return self.consequent_encroachment
    
    def get_invalidation_level(self) -> float:
        """
        Get stop loss / invalidation level
        ICT: Stop goes beyond the FVG range
        """
        if self.gap_type in ['bullish', 'bisi']:
            return self.low
        else:
            return self.high
    
    def is_price_at_ce(self, price: float, tolerance: float = 0.0001) -> bool:
        """Check if price is at consequent encroachment"""
        return abs(price - self.consequent_encroachment) <= tolerance
    
    def is_price_in_premium(self, price: float) -> bool:
        """Check if price is in premium portion of FVG (above CE)"""
        return price > self.consequent_encroachment
    
    def is_price_in_discount(self, price: float) -> bool:
        """Check if price is in discount portion of FVG (below CE)"""
        return price < self.consequent_encroachment


@dataclass
class FVGAnalysis:
    """Complete FVG analysis results with ICT classifications"""
    all_fvgs: List[FairValueGap]
    bisi_fvgs: List[FairValueGap]       # Bullish FVGs
    sibi_fvgs: List[FairValueGap]       # Bearish FVGs
    active_fvgs: List[FairValueGap]
    filled_fvgs: List[FairValueGap]
    inverted_fvgs: List[FairValueGap]
    reclaimed_fvgs: List[FairValueGap]
    
    # High probability FVGs
    high_prob_fvgs: List[FairValueGap]
    
    # Volume imbalances
    volume_imbalances: List[VolumeImbalance]
    
    # Suspension blocks
    suspension_blocks: List[SuspensionBlock]
    
    # Best trading opportunities
    best_bisi_fvg: Optional[FairValueGap] = None
    best_sibi_fvg: Optional[FairValueGap] = None
    first_presented_fvg: Optional[FairValueGap] = None
    
    # Statistics
    total_count: int = 0
    bisi_count: int = 0
    sibi_count: int = 0
    active_count: int = 0
    fill_rate: float = 0.0
    inversion_rate: float = 0.0
    
    def __str__(self):
        return (f"FVG Analysis: {self.total_count} total "
                f"({self.bisi_count} BISI, {self.sibi_count} SIBI) | "
                f"{self.active_count} active | "
                f"Fill rate: {self.fill_rate:.1f}% | "
                f"High Prob: {len(self.high_prob_fvgs)}")


# ==================== MAIN FVG HANDLER ====================

class FVGHandler:
    """
    Comprehensive ICT FVG Handler
    
    Implements all ICT Fair Value Gap concepts:
    - Standard FVG detection (BISI/SIBI)
    - Volume Imbalance detection
    - Suspension Block detection
    - Consequent Encroachment levels
    - Quadrant levels (25%, 50%, 75%)
    - Inversion and Reclaimed FVGs
    - High Probability FVG filtering
    - First Presented FVG identification
    - IOFED entry detection
    - Premium/Discount classification
    - Time-based filtering (macros, 2PM FVG)
    - Body respect analysis
    """
    
    def __init__(self, 
                 sensitivity: float = 0.0001,
                 min_gap_size: float = 0.0,
                 track_body_respect: bool = True,
                 detect_volume_imbalances: bool = True,
                 detect_suspension_blocks: bool = True,
                 use_ict_terminology: bool = True):
        """
        Initialize FVG Handler
        
        Args:
            sensitivity: Minimum gap size to detect (default 0.0001)
            min_gap_size: Minimum gap size in absolute price (0 = no minimum)
            track_body_respect: Track how candle bodies interact with FVG
            detect_volume_imbalances: Also detect volume imbalances (body gaps)
            detect_suspension_blocks: Detect ICT suspension blocks
            use_ict_terminology: Use BISI/SIBI vs bullish/bearish
        """
        self.sensitivity = sensitivity
        self.min_gap_size = min_gap_size
        self.track_body_respect = track_body_respect
        self.detect_volume_imbalances = detect_volume_imbalances
        self.detect_suspension_blocks = detect_suspension_blocks
        self.use_ict_terminology = use_ict_terminology
        
        # Storage
        self.detected_fvgs: List[FairValueGap] = []
        self.volume_imbalances: List[VolumeImbalance] = []
        self.suspension_blocks: List[SuspensionBlock] = []
        
        # Dealing range context
        self.dealing_range_high: Optional[float] = None
        self.dealing_range_low: Optional[float] = None
        
    # ==================== CORE FVG DETECTION ====================
    
    def detect_all_fvgs(self, df: pd.DataFrame) -> List[FairValueGap]:
        """
        Detect all Fair Value Gaps in the provided data
        
        ICT FVG Definition:
        - 3 candle formation
        - Gap between candle 1's high/low and candle 3's low/high
        - Candle 2 is the impulse candle
        
        Args:
            df: DataFrame with OHLC data (must have 'open', 'high', 'low', 'close')
            
        Returns:
            List of all detected FVGs
        """
        fvgs = []
        
        # Calculate dealing range if not set
        if self.dealing_range_high is None:
            self.dealing_range_high = df['high'].max()
            self.dealing_range_low = df['low'].min()
        
        for i in range(2, len(df)):
            # Get three consecutive candles
            candle_1 = df.iloc[i-2]  # First candle
            candle_2 = df.iloc[i-1]  # Impulse candle (middle)
            candle_3 = df.iloc[i]    # Third candle
            
            # Get timestamp if available
            timestamp = None
            if 'timestamp' in df.columns:
                timestamp = df.iloc[i]['timestamp']
            elif isinstance(df.index[i], (datetime, pd.Timestamp)):
                timestamp = df.index[i]
            
            # Check for BULLISH FVG (BISI)
            # ICT: Gap between candle 1's HIGH and candle 3's LOW
            if candle_3['low'] > candle_1['high'] + self.sensitivity:
                gap_low = candle_1['high']
                gap_high = candle_3['low']
                gap_size = gap_high - gap_low
                
                if gap_size >= self.min_gap_size:
                    fvg = self._create_fvg(
                        start_idx=i-2,
                        end_idx=i,
                        gap_type='bisi' if self.use_ict_terminology else 'bullish',
                        high=gap_high,
                        low=gap_low,
                        timestamp=timestamp,
                        timeframe=df.attrs.get('timeframe', None)
                    )
                    fvgs.append(fvg)
            
            # Check for BEARISH FVG (SIBI)
            # ICT: Gap between candle 1's LOW and candle 3's HIGH
            elif candle_3['high'] < candle_1['low'] - self.sensitivity:
                gap_high = candle_1['low']
                gap_low = candle_3['high']
                gap_size = gap_high - gap_low
                
                if gap_size >= self.min_gap_size:
                    fvg = self._create_fvg(
                        start_idx=i-2,
                        end_idx=i,
                        gap_type='sibi' if self.use_ict_terminology else 'bearish',
                        high=gap_high,
                        low=gap_low,
                        timestamp=timestamp,
                        timeframe=df.attrs.get('timeframe', None)
                    )
                    fvgs.append(fvg)
        
        # Mark first presented FVG
        if fvgs:
            fvgs[0].is_first_presented = True
            fvgs[0].notes.append("First Presented FVG - Highest probability entry")
        
        # Update status for all FVGs
        self._update_fvg_status(df, fvgs)
        
        # Classify gap types
        self._classify_gap_types(df, fvgs)
        
        # Determine high probability FVGs
        self._identify_high_probability_fvgs(df, fvgs)
        
        # Set premium/discount zones
        self._set_premium_discount_zones(fvgs)
        
        # Detect volume imbalances if enabled
        if self.detect_volume_imbalances:
            self.volume_imbalances = self._detect_volume_imbalances(df)
        
        # Detect suspension blocks if enabled
        if self.detect_suspension_blocks:
            self.suspension_blocks = self._detect_suspension_blocks(df)
        
        self.detected_fvgs = fvgs
        return fvgs
    
    def _create_fvg(self, start_idx: int, end_idx: int, gap_type: str,
                    high: float, low: float, timestamp=None, timeframe=None) -> FairValueGap:
        """
        Create FVG with all ICT levels calculated
        
        ICT Levels:
        - Consequent Encroachment (CE): 50% midpoint
        - Upper Quadrant: 75% level
        - Lower Quadrant: 25% level
        """
        size = high - low
        ce = low + (size * 0.5)  # Consequent Encroachment
        
        return FairValueGap(
            start_index=start_idx,
            end_index=end_idx,
            gap_type=gap_type,
            high=high,
            low=low,
            consequent_encroachment=ce,
            upper_quadrant=low + (size * 0.75),  # 75% level
            lower_quadrant=low + (size * 0.25),  # 25% level
            size=size,
            timestamp=timestamp,
            timeframe=timeframe,
            dealing_range_high=self.dealing_range_high,
            dealing_range_low=self.dealing_range_low
        )
    
    # ==================== VOLUME IMBALANCE DETECTION ====================
    
    def _detect_volume_imbalances(self, df: pd.DataFrame) -> List[VolumeImbalance]:
        """
        Detect Volume Imbalances (VI)
        
        ICT Definition: Gap between candle BODIES (not wicks)
        "The difference between two bodies that are not touching"
        
        Different from FVG which uses wicks
        """
        vis = []
        
        for i in range(1, len(df)):
            candle_prev = df.iloc[i-1]
            candle_curr = df.iloc[i]
            
            # Get body boundaries
            prev_body_high = max(candle_prev['open'], candle_prev['close'])
            prev_body_low = min(candle_prev['open'], candle_prev['close'])
            curr_body_high = max(candle_curr['open'], candle_curr['close'])
            curr_body_low = min(candle_curr['open'], candle_curr['close'])
            
            # Bullish Volume Imbalance: Gap up between bodies
            if curr_body_low > prev_body_high + self.sensitivity:
                vi = VolumeImbalance(
                    index=i,
                    imbalance_type='bullish',
                    high=curr_body_low,
                    low=prev_body_high,
                    mid_point=(curr_body_low + prev_body_high) / 2,
                    size=curr_body_low - prev_body_high,
                    timestamp=df.iloc[i].get('timestamp', None) if hasattr(df.iloc[i], 'get') else None
                )
                vis.append(vi)
            
            # Bearish Volume Imbalance: Gap down between bodies
            elif curr_body_high < prev_body_low - self.sensitivity:
                vi = VolumeImbalance(
                    index=i,
                    imbalance_type='bearish',
                    high=prev_body_low,
                    low=curr_body_high,
                    mid_point=(prev_body_low + curr_body_high) / 2,
                    size=prev_body_low - curr_body_high,
                    timestamp=df.iloc[i].get('timestamp', None) if hasattr(df.iloc[i], 'get') else None
                )
                vis.append(vi)
        
        return vis
    
    # ==================== SUSPENSION BLOCK DETECTION ====================
    
    def _detect_suspension_blocks(self, df: pd.DataFrame) -> List[SuspensionBlock]:
        """
        Detect ICT Suspension Blocks
        
        ICT Definition: A single candle with volume imbalance at BOTH ends
        "Suspended between two volume imbalances"
        "This is extremely strong. It's one of the most powerful."
        
        Even if wicks have crossed over it to the left, it still acts as FVG
        """
        blocks = []
        
        for i in range(1, len(df) - 1):
            candle_prev = df.iloc[i-1]
            candle_curr = df.iloc[i]
            candle_next = df.iloc[i+1]
            
            # Get body boundaries
            prev_body_high = max(candle_prev['open'], candle_prev['close'])
            prev_body_low = min(candle_prev['open'], candle_prev['close'])
            curr_body_high = max(candle_curr['open'], candle_curr['close'])
            curr_body_low = min(candle_curr['open'], candle_curr['close'])
            next_body_high = max(candle_next['open'], candle_next['close'])
            next_body_low = min(candle_next['open'], candle_next['close'])
            
            # Check for VI at the LOW (gap below current candle)
            has_lower_vi = curr_body_low > prev_body_high + self.sensitivity
            
            # Check for VI at the HIGH (gap above current candle)
            has_upper_vi = next_body_low > curr_body_high + self.sensitivity
            
            if has_lower_vi and has_upper_vi:
                # Create suspension block
                lower_vi = VolumeImbalance(
                    index=i,
                    imbalance_type='bullish',
                    high=curr_body_low,
                    low=prev_body_high,
                    mid_point=(curr_body_low + prev_body_high) / 2,
                    size=curr_body_low - prev_body_high
                )
                
                upper_vi = VolumeImbalance(
                    index=i,
                    imbalance_type='bullish',
                    high=next_body_low,
                    low=curr_body_high,
                    mid_point=(next_body_low + curr_body_high) / 2,
                    size=next_body_low - curr_body_high
                )
                
                block_range = candle_curr['high'] - candle_curr['low']
                
                block = SuspensionBlock(
                    index=i,
                    candle_open=candle_curr['open'],
                    candle_high=candle_curr['high'],
                    candle_low=candle_curr['low'],
                    candle_close=candle_curr['close'],
                    upper_vi=upper_vi,
                    lower_vi=lower_vi,
                    mid_point=(candle_curr['high'] + candle_curr['low']) / 2,
                    upper_quadrant=candle_curr['low'] + (block_range * 0.75),
                    lower_quadrant=candle_curr['low'] + (block_range * 0.25),
                    is_bullish=candle_curr['close'] > candle_curr['open'],
                    timestamp=df.iloc[i].get('timestamp', None) if hasattr(df.iloc[i], 'get') else None
                )
                blocks.append(block)
        
        return blocks
    
    # ==================== FVG STATUS TRACKING ====================
    
    def _update_fvg_status(self, df: pd.DataFrame, fvgs: List[FairValueGap]):
        """
        Update status for all FVGs with ICT-specific tracking
        
        ICT Status Types:
        - Filled: Price traded through CE
        - Partially Filled: IOFED opportunity (price entered but respected)
        - Inverted: Changed characteristic (bullish became bearish or vice versa)
        - Reclaimed: Regained original characteristic after inversion
        """
        for fvg in fvgs:
            if fvg.end_index >= len(df) - 1:
                continue
                
            future_candles = df.iloc[fvg.end_index + 1:]
            
            for idx in range(len(future_candles)):
                candle = future_candles.iloc[idx]
                
                if fvg.gap_type in ['bullish', 'bisi']:
                    self._check_bisi_status(fvg, candle)
                else:
                    self._check_sibi_status(fvg, candle)
                
                # Track body respect (ICT: "Bodies tell the story")
                if self.track_body_respect:
                    self._track_body_respect(fvg, candle)
                
                # Count touches
                if self._is_touching_fvg(fvg, candle):
                    fvg.touches += 1
                    
                # Count rejections (wick only touch)
                if self._is_rejection_from_fvg(fvg, candle):
                    fvg.rejection_count += 1
                    fvg.wick_only_touches += 1
    
    def _check_bisi_status(self, fvg: FairValueGap, candle: pd.Series):
        """
        Check status of BISI (Bullish) FVG
        
        ICT Rules:
        - Filled if price trades to or below CE (consequent encroachment)
        - IOFED if price enters but bodies respect CE
        - Inverted if bodies close below the FVG low
        - Can be reclaimed if price returns and respects
        """
        body_low = min(candle['open'], candle['close'])
        body_high = max(candle['open'], candle['close'])
        
        # Check for fill at Consequent Encroachment
        if candle['low'] <= fvg.consequent_encroachment:
            if body_low <= fvg.consequent_encroachment:
                # Body penetrated CE - fully filled
                fvg.is_filled = True
                fvg.status = FVGStatus.FILLED
                fvg.fill_percentage = 100.0
            else:
                # Only wick touched CE - IOFED opportunity
                fvg.is_partially_filled = True
                fvg.has_iofed_entry = True
                fvg.iofed_level = fvg.consequent_encroachment
                fvg.status = FVGStatus.PARTIALLY_FILLED
                penetration = fvg.high - candle['low']
                fvg.fill_percentage = (penetration / fvg.size) * 100
                fvg.notes.append("IOFED Entry - Wick respected CE")
        
        # Partial fill (price entered upper portion)
        elif candle['low'] < fvg.high and not fvg.is_filled:
            fvg.is_partially_filled = True
            penetration = fvg.high - candle['low']
            fvg.fill_percentage = max(fvg.fill_percentage, (penetration / fvg.size) * 100)
        
        # Check for inversion (bodies close below FVG)
        if body_low < fvg.low and not fvg.is_inverted:
            fvg.is_inverted = True
            fvg.inversion_count += 1
            fvg.status = FVGStatus.INVERTED
            fvg.gap_type = 'sibi' if self.use_ict_terminology else 'bearish'
            fvg.notes.append(f"INVERSION - Now acts as {fvg.gap_type.upper()}")
        
        # Check for reclaim (was inverted, now respecting again)
        if fvg.is_inverted and body_low > fvg.consequent_encroachment:
            fvg.is_reclaimed = True
            fvg.status = FVGStatus.RECLAIMED
            fvg.gap_type = fvg.original_type
            fvg.notes.append("RECLAIMED - Regained original characteristic")
    
    def _check_sibi_status(self, fvg: FairValueGap, candle: pd.Series):
        """
        Check status of SIBI (Bearish) FVG
        
        ICT Rules:
        - Filled if price trades to or above CE
        - IOFED if price enters but bodies respect CE
        - Inverted if bodies close above the FVG high
        """
        body_low = min(candle['open'], candle['close'])
        body_high = max(candle['open'], candle['close'])
        
        # Check for fill at Consequent Encroachment
        if candle['high'] >= fvg.consequent_encroachment:
            if body_high >= fvg.consequent_encroachment:
                # Body penetrated CE - fully filled
                fvg.is_filled = True
                fvg.status = FVGStatus.FILLED
                fvg.fill_percentage = 100.0
            else:
                # Only wick touched CE - IOFED opportunity
                fvg.is_partially_filled = True
                fvg.has_iofed_entry = True
                fvg.iofed_level = fvg.consequent_encroachment
                fvg.status = FVGStatus.PARTIALLY_FILLED
                penetration = candle['high'] - fvg.low
                fvg.fill_percentage = (penetration / fvg.size) * 100
                fvg.notes.append("IOFED Entry - Wick respected CE")
        
        # Partial fill
        elif candle['high'] > fvg.low and not fvg.is_filled:
            fvg.is_partially_filled = True
            penetration = candle['high'] - fvg.low
            fvg.fill_percentage = max(fvg.fill_percentage, (penetration / fvg.size) * 100)
        
        # Check for inversion
        if body_high > fvg.high and not fvg.is_inverted:
            fvg.is_inverted = True
            fvg.inversion_count += 1
            fvg.status = FVGStatus.INVERTED
            fvg.gap_type = 'bisi' if self.use_ict_terminology else 'bullish'
            fvg.notes.append(f"INVERSION - Now acts as {fvg.gap_type.upper()}")
        
        # Check for reclaim
        if fvg.is_inverted and body_high < fvg.consequent_encroachment:
            fvg.is_reclaimed = True
            fvg.status = FVGStatus.RECLAIMED
            fvg.gap_type = fvg.original_type
            fvg.notes.append("RECLAIMED - Regained original characteristic")
    
    def _track_body_respect(self, fvg: FairValueGap, candle: pd.Series):
        """
        Track how candle bodies interact with FVG
        ICT: "Bodies tell the story, wicks do the damage"
        """
        body_low = min(candle['open'], candle['close'])
        body_high = max(candle['open'], candle['close'])
        
        if fvg.gap_type in ['bullish', 'bisi']:
            # For BISI: Bodies should stay above CE
            if body_low >= fvg.consequent_encroachment:
                fvg.body_respects += 1
            elif body_low < fvg.consequent_encroachment:
                fvg.body_violations += 1
        else:
            # For SIBI: Bodies should stay below CE
            if body_high <= fvg.consequent_encroachment:
                fvg.body_respects += 1
            elif body_high > fvg.consequent_encroachment:
                fvg.body_violations += 1
    
    def _is_touching_fvg(self, fvg: FairValueGap, candle: pd.Series) -> bool:
        """Check if candle is touching FVG"""
        if fvg.gap_type in ['bullish', 'bisi']:
            return candle['low'] <= fvg.high and candle['low'] >= fvg.low
        else:
            return candle['high'] >= fvg.low and candle['high'] <= fvg.high
    
    def _is_rejection_from_fvg(self, fvg: FairValueGap, candle: pd.Series) -> bool:
        """
        Check if candle rejected from FVG (wick touch only)
        ICT: "Wick only touches indicate respect"
        """
        body_low = min(candle['open'], candle['close'])
        body_high = max(candle['open'], candle['close'])
        
        if fvg.gap_type in ['bullish', 'bisi']:
            # Wick touched FVG but body stayed above
            wick_touched = candle['low'] <= fvg.high and candle['low'] >= fvg.low
            body_above = body_low > fvg.consequent_encroachment
            return wick_touched and body_above
        else:
            # Wick touched FVG but body stayed below
            wick_touched = candle['high'] >= fvg.low and candle['high'] <= fvg.high
            body_below = body_high < fvg.consequent_encroachment
            return wick_touched and body_below
    
    # ==================== ICT GAP CLASSIFICATION ====================
    
    def _classify_gap_types(self, df: pd.DataFrame, fvgs: List[FairValueGap]):
        """
        Classify FVGs using ICT methodology
        
        ICT Gap Types:
        - Breakaway: Formed between quadrant levels, doesn't fill
        - Measuring: Approximately halfway point of move, used for projection
        - Common: Can be reclaimed/traded multiple times
        - Exhaustion: End of move, followed by reversal
        """
        if len(df) < 20:
            return
            
        for fvg in fvgs:
            # Get context candles
            before_idx = max(0, fvg.start_index - 20)
            after_idx = min(len(df), fvg.end_index + 20)
            
            candles_before = df.iloc[before_idx:fvg.start_index]
            candles_after = df.iloc[fvg.end_index:after_idx] if fvg.end_index < len(df) else pd.DataFrame()
            
            if len(candles_before) < 5:
                continue
            
            # Calculate move context
            if fvg.gap_type in ['bullish', 'bisi']:
                self._classify_bisi_gap(fvg, candles_before, candles_after, df)
            else:
                self._classify_sibi_gap(fvg, candles_before, candles_after, df)
    
    def _classify_bisi_gap(self, fvg: FairValueGap, 
                           before: pd.DataFrame, 
                           after: pd.DataFrame,
                           full_df: pd.DataFrame):
        """
        Classify BISI (Bullish) gap type
        
        ICT Breakaway Classification:
        "If it's between a quadrant and another quadrant, expect it to act as breakaway"
        """
        if len(before) < 5:
            return
            
        # Calculate the move
        move_low = before['low'].min()
        move_high = full_df.iloc[:fvg.end_index + 10]['high'].max() if fvg.end_index + 10 < len(full_df) else before['high'].max()
        move_range = move_high - move_low
        
        if move_range == 0:
            return
        
        # FVG position in the move
        fvg_position = (fvg.consequent_encroachment - move_low) / move_range
        
        # Breakaway: Near the start of move (0-30%)
        if fvg_position < 0.30:
            fvg.is_breakaway = True
            fvg.classification = FVGClassification.BREAKAWAY
            fvg.notes.append("BREAKAWAY GAP - Start of move, unlikely to fill")
        
        # Measuring: Middle of move (40-60%)
        elif 0.40 <= fvg_position <= 0.60:
            fvg.is_measuring = True
            fvg.classification = FVGClassification.MEASURING
            # Calculate projection target
            projection = fvg.consequent_encroachment + (fvg.consequent_encroachment - move_low)
            fvg.notes.append(f"MEASURING GAP - Projection target: {projection:.5f}")
        
        # Exhaustion: End of move (70%+) with reversal
        elif fvg_position > 0.70:
            if len(after) > 3:
                # Check for reversal
                if after['close'].iloc[-1] < fvg.consequent_encroachment:
                    fvg.is_exhaustion = True
                    fvg.classification = FVGClassification.EXHAUSTION
                    fvg.notes.append("EXHAUSTION GAP - End of move, reversal likely")
        
        # Common gap (default)
        if fvg.classification == FVGClassification.COMMON:
            fvg.notes.append("COMMON GAP - Can be reclaimed multiple times")
    
    def _classify_sibi_gap(self, fvg: FairValueGap,
                           before: pd.DataFrame,
                           after: pd.DataFrame,
                           full_df: pd.DataFrame):
        """Classify SIBI (Bearish) gap type"""
        if len(before) < 5:
            return
            
        move_high = before['high'].max()
        move_low = full_df.iloc[:fvg.end_index + 10]['low'].min() if fvg.end_index + 10 < len(full_df) else before['low'].min()
        move_range = move_high - move_low
        
        if move_range == 0:
            return
        
        fvg_position = (move_high - fvg.consequent_encroachment) / move_range
        
        if fvg_position < 0.30:
            fvg.is_breakaway = True
            fvg.classification = FVGClassification.BREAKAWAY
            fvg.notes.append("BREAKAWAY GAP - Start of move, unlikely to fill")
        elif 0.40 <= fvg_position <= 0.60:
            fvg.is_measuring = True
            fvg.classification = FVGClassification.MEASURING
            projection = fvg.consequent_encroachment - (move_high - fvg.consequent_encroachment)
            fvg.notes.append(f"MEASURING GAP - Projection target: {projection:.5f}")
        elif fvg_position > 0.70:
            if len(after) > 3 and after['close'].iloc[-1] > fvg.consequent_encroachment:
                fvg.is_exhaustion = True
                fvg.classification = FVGClassification.EXHAUSTION
                fvg.notes.append("EXHAUSTION GAP - End of move, reversal likely")
    
    # ==================== HIGH PROBABILITY FVG ====================
    
    def _identify_high_probability_fvgs(self, df: pd.DataFrame, fvgs: List[FairValueGap]):
        """
        Identify HIGH PROBABILITY FVGs using ICT criteria
        
        ICT High Probability FVG:
        "If it's a high probability fair value gap, it trades back down into it.
         It needs to be laying on top of a quadrant level."
        
        "Any portion of the candle that makes the FVG must be at or below 
         the quadrant level [for bullish]"
        """
        if len(df) < 20:
            return
        
        # Calculate key quadrant levels from dealing range
        dr_range = self.dealing_range_high - self.dealing_range_low
        dr_ce = self.dealing_range_low + (dr_range * 0.5)
        dr_upper_quad = self.dealing_range_low + (dr_range * 0.75)
        dr_lower_quad = self.dealing_range_low + (dr_range * 0.25)
        
        # Get recent swing highs/lows for additional quadrant levels
        recent_high = df['high'].tail(50).max()
        recent_low = df['low'].tail(50).min()
        recent_range = recent_high - recent_low
        recent_ce = recent_low + (recent_range * 0.5)
        recent_upper_quad = recent_low + (recent_range * 0.75)
        recent_lower_quad = recent_low + (recent_range * 0.25)
        
        quadrant_levels = [
            dr_ce, dr_upper_quad, dr_lower_quad,
            recent_ce, recent_upper_quad, recent_lower_quad
        ]
        
        for fvg in fvgs:
            # Check if FVG overlays any quadrant level
            for quad_level in quadrant_levels:
                if fvg.low <= quad_level <= fvg.high:
                    fvg.overlays_quadrant = True
                    fvg.quadrant_level_overlaid = quad_level
                    fvg.is_high_probability = True
                    fvg.probability = FVGProbability.HIGH
                    fvg.notes.append(f"HIGH PROBABILITY - Overlays quadrant {quad_level:.5f}")
                    break
            
            # Additional high probability factors
            if fvg.is_first_presented:
                if fvg.probability != FVGProbability.HIGH:
                    fvg.probability = FVGProbability.MEDIUM
                fvg.notes.append("First Presented - Increased probability")
            
            if fvg.is_breakaway:
                fvg.probability = FVGProbability.HIGH
            
            if fvg.is_exhaustion:
                fvg.probability = FVGProbability.AVOID
                fvg.notes.append("AVOID - Exhaustion gap")
    
    # ==================== PREMIUM/DISCOUNT ZONES ====================
    
    def _set_premium_discount_zones(self, fvgs: List[FairValueGap]):
        """
        Set premium/discount zone for each FVG
        
        ICT: "50% or higher is premium, below 50% is discount"
        """
        if self.dealing_range_high is None or self.dealing_range_low is None:
            return
        
        dr_range = self.dealing_range_high - self.dealing_range_low
        dr_ce = self.dealing_range_low + (dr_range * 0.5)
        
        for fvg in fvgs:
            # Calculate FVG position in dealing range
            fvg.position_in_range = ((fvg.consequent_encroachment - self.dealing_range_low) / dr_range) * 100
            
            if fvg.consequent_encroachment > dr_ce:
                fvg.zone = PremiumDiscountZone.PREMIUM
            elif fvg.consequent_encroachment < dr_ce:
                fvg.zone = PremiumDiscountZone.DISCOUNT
            else:
                fvg.zone = PremiumDiscountZone.EQUILIBRIUM
    
    # ==================== TIME-BASED FILTERING ====================
    
    def get_first_fvg_after_time(self, target_time: time, gap_type: Optional[str] = None) -> Optional[FairValueGap]:
        """
        Get first FVG formed after a specific time
        
        ICT: "First presented fair value gap post 2 PM"
        Common times: 2:00 PM (14:00), 10:00 AM, etc.
        
        Args:
            target_time: Time to filter after (e.g., time(14, 0) for 2 PM)
            gap_type: Optional filter for gap type
            
        Returns:
            First FVG after the specified time
        """
        matching_fvgs = []
        
        for fvg in self.detected_fvgs:
            if fvg.timestamp is None:
                continue
            
            fvg_time = fvg.timestamp.time() if isinstance(fvg.timestamp, datetime) else None
            if fvg_time is None:
                continue
            
            if fvg_time >= target_time:
                if gap_type is None or fvg.gap_type == gap_type:
                    matching_fvgs.append(fvg)
        
        return matching_fvgs[0] if matching_fvgs else None
    
    def get_fvgs_in_macro_window(self, macro_start: time, macro_end: time) -> List[FairValueGap]:
        """
        Get FVGs formed during ICT macro windows
        
        ICT Macros:
        - 9:50-10:10 (first 10 min after 10:00)
        - 10:50-11:10 
        - 2:50-3:10 PM
        - etc.
        
        Args:
            macro_start: Start time of macro window
            macro_end: End time of macro window
            
        Returns:
            List of FVGs in the macro window
        """
        matching_fvgs = []
        
        for fvg in self.detected_fvgs:
            if fvg.timestamp is None:
                continue
            
            fvg_time = fvg.timestamp.time() if isinstance(fvg.timestamp, datetime) else None
            if fvg_time is None:
                continue
            
            if macro_start <= fvg_time <= macro_end:
                matching_fvgs.append(fvg)
        
        return matching_fvgs
    
    # ==================== ANALYSIS ====================
    
    def analyze_fvgs(self, df: pd.DataFrame) -> FVGAnalysis:
        """
        Complete FVG analysis with ICT classifications
        
        Returns:
            FVGAnalysis object with all categories and statistics
        """
        # Detect all FVGs
        all_fvgs = self.detect_all_fvgs(df)
        
        # Categorize FVGs
        bisi_fvgs = [f for f in all_fvgs if f.gap_type in ['bullish', 'bisi']]
        sibi_fvgs = [f for f in all_fvgs if f.gap_type in ['bearish', 'sibi']]
        active_fvgs = [f for f in all_fvgs if f.status == FVGStatus.ACTIVE]
        filled_fvgs = [f for f in all_fvgs if f.is_filled]
        inverted_fvgs = [f for f in all_fvgs if f.is_inverted]
        reclaimed_fvgs = [f for f in all_fvgs if f.is_reclaimed]
        high_prob_fvgs = [f for f in all_fvgs if f.is_high_probability]
        
        # Find best trading opportunities
        best_bisi = self._find_best_fvg([f for f in bisi_fvgs if f.status == FVGStatus.ACTIVE], 'bisi')
        best_sibi = self._find_best_fvg([f for f in sibi_fvgs if f.status == FVGStatus.ACTIVE], 'sibi')
        first_presented = next((f for f in all_fvgs if f.is_first_presented), None)
        
        # Calculate statistics
        total = len(all_fvgs)
        fill_rate = (len(filled_fvgs) / total * 100) if total > 0 else 0
        inversion_rate = (len(inverted_fvgs) / total * 100) if total > 0 else 0
        
        return FVGAnalysis(
            all_fvgs=all_fvgs,
            bisi_fvgs=bisi_fvgs,
            sibi_fvgs=sibi_fvgs,
            active_fvgs=active_fvgs,
            filled_fvgs=filled_fvgs,
            inverted_fvgs=inverted_fvgs,
            reclaimed_fvgs=reclaimed_fvgs,
            high_prob_fvgs=high_prob_fvgs,
            volume_imbalances=self.volume_imbalances,
            suspension_blocks=self.suspension_blocks,
            best_bisi_fvg=best_bisi,
            best_sibi_fvg=best_sibi,
            first_presented_fvg=first_presented,
            total_count=total,
            bisi_count=len(bisi_fvgs),
            sibi_count=len(sibi_fvgs),
            active_count=len(active_fvgs),
            fill_rate=fill_rate,
            inversion_rate=inversion_rate
        )
    
    def _find_best_fvg(self, fvgs: List[FairValueGap], gap_type: str) -> Optional[FairValueGap]:
        """
        Find best FVG for trading based on ICT criteria
        
        Scoring priority:
        1. High probability (overlays quadrant)
        2. First presented FVG
        3. Breakaway gap
        4. Body respect ratio
        5. Rejection count
        """
        if not fvgs:
            return None
        
        scored_fvgs = []
        for fvg in fvgs:
            score = 0
            
            # Highest priority: High probability FVG
            if fvg.is_high_probability:
                score += 30
            
            # First presented FVG
            if fvg.is_first_presented:
                score += 25
            
            # Breakaway gaps
            if fvg.is_breakaway:
                score += 20
            
            # Measuring gaps (good for continuation)
            if fvg.is_measuring:
                score += 10
            
            # Avoid exhaustion gaps
            if fvg.is_exhaustion:
                score -= 20
            
            # Body respect ratio
            total_body_events = fvg.body_respects + fvg.body_violations
            if total_body_events > 0:
                respect_ratio = fvg.body_respects / total_body_events
                score += int(respect_ratio * 15)
            
            # Rejection count (wick only touches)
            score += min(fvg.rejection_count * 5, 15)
            
            # IOFED opportunity
            if fvg.has_iofed_entry:
                score += 10
            
            # Confluence bonuses
            if fvg.has_order_block:
                score += 10
            if fvg.overlays_quadrant:
                score += 10
            
            # Zone alignment (buy in discount, sell in premium)
            if gap_type in ['bullish', 'bisi'] and fvg.zone == PremiumDiscountZone.DISCOUNT:
                score += 15
            elif gap_type in ['bearish', 'sibi'] and fvg.zone == PremiumDiscountZone.PREMIUM:
                score += 15
            
            scored_fvgs.append((score, fvg))
        
        scored_fvgs.sort(key=lambda x: x[0], reverse=True)
        return scored_fvgs[0][1] if scored_fvgs else None
    
    # ==================== TRADE SIGNALS ====================
    
    def get_fvg_trade_signal(self, df: pd.DataFrame, current_price: float, 
                              bias: str = 'neutral') -> Dict:
        """
        Generate trade signal based on FVG analysis with ICT methodology
        
        Args:
            df: OHLC DataFrame
            current_price: Current market price
            bias: Market bias ('bullish', 'bearish', 'neutral')
            
        Returns:
            Dict with signal, entry, SL, TP, confidence, and reasoning
        """
        analysis = self.analyze_fvgs(df)
        
        signal = {
            'type': 'NO_SIGNAL',
            'entry': None,
            'stop_loss': None,
            'take_profit': None,
            'confidence': 0,
            'fvg_used': None,
            'entry_type': None,
            'reasoning': []
        }
        
        # Prioritize high probability FVGs
        best_fvg = None
        
        if bias in ['bullish', 'neutral'] and analysis.best_bisi_fvg:
            fvg = analysis.best_bisi_fvg
            distance = current_price - fvg.consequent_encroachment
            # Check if price is approaching FVG from above (retracement)
            if 0 < distance < fvg.size * 3:
                best_fvg = fvg
                signal['type'] = 'BUY'
        
        if bias in ['bearish', 'neutral'] and analysis.best_sibi_fvg:
            fvg = analysis.best_sibi_fvg
            distance = fvg.consequent_encroachment - current_price
            # Check if price is approaching FVG from below (retracement)
            if 0 < distance < fvg.size * 3:
                if best_fvg is None or fvg.probability.value < best_fvg.probability.value:
                    best_fvg = fvg
                    signal['type'] = 'SELL'
        
        if best_fvg:
            # Determine entry type based on probability
            if best_fvg.is_high_probability:
                entry_type = "aggressive"  # Enter at quadrant
            elif best_fvg.has_iofed_entry:
                entry_type = "iofed"
            else:
                entry_type = "ce"  # Standard CE entry
            
            signal['entry'] = best_fvg.get_entry_level(entry_type)
            signal['stop_loss'] = best_fvg.get_invalidation_level()
            signal['entry_type'] = entry_type
            
            # Calculate take profit (2:1 minimum RR)
            risk = abs(signal['entry'] - signal['stop_loss'])
            if signal['type'] == 'BUY':
                signal['take_profit'] = signal['entry'] + (risk * 2)
            else:
                signal['take_profit'] = signal['entry'] - (risk * 2)
            
            signal['confidence'] = self._calculate_fvg_confidence(best_fvg)
            signal['fvg_used'] = best_fvg
            
            # Build reasoning
            ict_name = "BISI" if best_fvg.gap_type in ['bullish', 'bisi'] else "SIBI"
            signal['reasoning'].append(f"{ict_name} FVG at CE {best_fvg.consequent_encroachment:.5f}")
            
            if best_fvg.is_high_probability:
                signal['reasoning'].append(f"HIGH PROBABILITY - Overlays quadrant {best_fvg.quadrant_level_overlaid:.5f}")
            if best_fvg.is_first_presented:
                signal['reasoning'].append("First Presented FVG")
            if best_fvg.is_breakaway:
                signal['reasoning'].append("Breakaway Gap - High priority")
            if best_fvg.has_iofed_entry:
                signal['reasoning'].append("IOFED Entry opportunity")
            if best_fvg.body_respects > 0:
                signal['reasoning'].append(f"Body respects: {best_fvg.body_respects}")
            signal['reasoning'].append(f"Zone: {best_fvg.zone.value.upper()}")
        
        return signal
    
    def _calculate_fvg_confidence(self, fvg: FairValueGap) -> int:
        """Calculate confidence score for FVG trade (0-100)"""
        confidence = 50  # Base confidence
        
        # High probability FVG
        if fvg.is_high_probability:
            confidence += 25
        
        # First presented
        if fvg.is_first_presented:
            confidence += 15
        
        # Breakaway
        if fvg.is_breakaway:
            confidence += 15
        
        # Measuring
        if fvg.is_measuring:
            confidence += 5
        
        # Exhaustion (negative)
        if fvg.is_exhaustion:
            confidence -= 25
        
        # Body respect ratio
        total_body = fvg.body_respects + fvg.body_violations
        if total_body > 0:
            respect_ratio = fvg.body_respects / total_body
            confidence += int(respect_ratio * 10)
        
        # Rejections
        confidence += min(fvg.rejection_count * 3, 10)
        
        # IOFED
        if fvg.has_iofed_entry:
            confidence += 10
        
        # Confluence
        if fvg.overlays_quadrant:
            confidence += 10
        if fvg.has_order_block:
            confidence += 10
        
        # Partially filled (lower confidence)
        if fvg.is_partially_filled and not fvg.has_iofed_entry:
            confidence -= 10
        
        return max(0, min(100, confidence))
    
    # ==================== UTILITY METHODS ====================
    
    def set_dealing_range(self, high: float, low: float):
        """
        Set dealing range for premium/discount analysis
        
        Args:
            high: Dealing range high
            low: Dealing range low
        """
        self.dealing_range_high = high
        self.dealing_range_low = low
    
    def get_active_fvgs(self, gap_type: Optional[str] = None) -> List[FairValueGap]:
        """Get all active (unfilled, non-inverted) FVGs"""
        active = [f for f in self.detected_fvgs if f.status == FVGStatus.ACTIVE]
        
        if gap_type:
            gap_types = [gap_type]
            if gap_type == 'bullish':
                gap_types.append('bisi')
            elif gap_type == 'bearish':
                gap_types.append('sibi')
            active = [f for f in active if f.gap_type in gap_types]
        
        return active
    
    def get_nearest_fvg(self, current_price: float, 
                        gap_type: Optional[str] = None) -> Optional[FairValueGap]:
        """Get nearest FVG to current price"""
        fvgs = self.get_active_fvgs(gap_type)
        
        if not fvgs:
            return None
        
        distances = [(abs(f.consequent_encroachment - current_price), f) for f in fvgs]
        distances.sort(key=lambda x: x[0])
        return distances[0][1]
    
    def get_fvgs_in_range(self, low: float, high: float) -> List[FairValueGap]:
        """Get all FVGs within a price range"""
        return [f for f in self.detected_fvgs 
                if f.low >= low and f.high <= high]
    
    def get_inverted_fvgs(self) -> List[FairValueGap]:
        """Get all inverted FVGs (inversion fair value gaps)"""
        return [f for f in self.detected_fvgs if f.is_inverted]
    
    def get_reclaimed_fvgs(self) -> List[FairValueGap]:
        """Get all reclaimed FVGs"""
        return [f for f in self.detected_fvgs if f.is_reclaimed]
    
    def get_fvg_summary(self) -> str:
        """Get comprehensive text summary of all detected FVGs"""
        if not self.detected_fvgs:
            return "No FVGs detected"
        
        lines = [
            "",
            "=" * 80,
            "ICT FAIR VALUE GAP ANALYSIS",
            "=" * 80,
            f"\nTotal FVGs: {len(self.detected_fvgs)}",
        ]
        
        bisi = [f for f in self.detected_fvgs if f.gap_type in ['bullish', 'bisi']]
        sibi = [f for f in self.detected_fvgs if f.gap_type in ['bearish', 'sibi']]
        high_prob = [f for f in self.detected_fvgs if f.is_high_probability]
        inverted = [f for f in self.detected_fvgs if f.is_inverted]
        reclaimed = [f for f in self.detected_fvgs if f.is_reclaimed]
        
        lines.extend([
            f"BISI (Bullish): {len(bisi)}",
            f"SIBI (Bearish): {len(sibi)}",
            f"High Probability: {len(high_prob)}",
            f"Inverted: {len(inverted)}",
            f"Reclaimed: {len(reclaimed)}",
            f"\nVolume Imbalances: {len(self.volume_imbalances)}",
            f"Suspension Blocks: {len(self.suspension_blocks)}",
            "",
            "-" * 40,
            "RECENT FVGs (Last 10):",
            "-" * 40,
        ])
        
        for i, fvg in enumerate(self.detected_fvgs[-10:], 1):
            status_tags = []
            if fvg.is_high_probability:
                status_tags.append("HIGH PROB")
            if fvg.is_first_presented:
                status_tags.append("1ST PRESENTED")
            if fvg.is_breakaway:
                status_tags.append("BREAKAWAY")
            if fvg.is_measuring:
                status_tags.append("MEASURING")
            if fvg.is_inverted:
                status_tags.append("INVERTED")
            if fvg.is_reclaimed:
                status_tags.append("RECLAIMED")
            if fvg.has_iofed_entry:
                status_tags.append("IOFED")
            
            status = f" [{', '.join(status_tags)}]" if status_tags else ""
            lines.append(f"{i}. {fvg}{status}")
            lines.append(f"   Zone: {fvg.zone.value} | Bodies Respected: {fvg.body_respects} | Rejections: {fvg.rejection_count}")
            
            if fvg.notes:
                for note in fvg.notes[:3]:  # Limit notes
                    lines.append(f"   → {note}")
        
        if self.suspension_blocks:
            lines.extend([
                "",
                "-" * 40,
                "SUSPENSION BLOCKS:",
                "-" * 40,
            ])
            for i, block in enumerate(self.suspension_blocks[-5:], 1):
                lines.append(f"{i}. {block}")
        
        lines.append("=" * 80 + "\n")
        return "\n".join(lines)
    
    def reset(self):
        """Reset all detected patterns"""
        self.detected_fvgs = []
        self.volume_imbalances = []
        self.suspension_blocks = []
        self.dealing_range_high = None
        self.dealing_range_low = None


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("ICT Fair Value Gap Handler")
    print("=" * 50)
    print("\nKey ICT Concepts Implemented:")
    print("  • BISI/SIBI (Bullish/Bearish FVG)")
    print("  • Consequent Encroachment (CE) - 50% midpoint")
    print("  • Quadrant Levels (25%, 75%)")
    print("  • Inversion Fair Value Gap")
    print("  • Reclaimed Fair Value Gap")
    print("  • High Probability FVG (overlays quadrant)")
    print("  • Volume Imbalance (body gaps)")
    print("  • Suspension Block (VI at both ends)")
    print("  • First Presented FVG")
    print("  • IOFED Entry (partial fill)")
    print("  • Premium/Discount Zones")
    print("  • Body Respect Analysis")
    print("  • Time-Based Filtering")
    print("\nUsage:")
    print("  handler = FVGHandler()")
    print("  analysis = handler.analyze_fvgs(df)")
    print("  signal = handler.get_fvg_trade_signal(df, current_price)")
