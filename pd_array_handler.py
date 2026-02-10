"""
Comprehensive ICT PD Arrays Handler
Based on Inner Circle Trader methodology and transcript teachings

Key ICT PD Array Concepts:
- Premium = Above 50% (Sell zones)
- Discount = Below 50% (Buy zones)  
- Equilibrium = 50% (Magnet/CE)
- All wicks must be graded with quadrants
- Daily range is primary reference
- Inversion: PD array changes character when traded through
- Suspension Block: Volume imbalance at both ends

ICT Quotes from transcripts:
- "When bullish, ONLY buy from discount. When bearish, ONLY sell from premium"
- "Equilibrium is ALWAYS a magnet. Price seeks it"
- "Grade EVERYTHING. Every range, every wick, every array gets quadrants"
- "Premium wick acts as resistance. Discount wick acts as support"
- "If price trades through array, it becomes inversion (changes character)"
- "Whenever you see price hovering between quadrants don't touch it"
- "The candlestick body respecting it here. Look at the bodies"
- "Suspension block - volume imbalance at top and bottom"
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from datetime import datetime
import pandas as pd
import numpy as np


# ==================== ENUMS ====================

class PDZone(Enum):
    """Premium/Discount zone classification"""
    EXTREME_PREMIUM = "extreme_premium"    # 75-100%
    PREMIUM = "premium"                     # 50-75%
    EQUILIBRIUM = "equilibrium"             # Around 50%
    DISCOUNT = "discount"                   # 25-50%
    EXTREME_DISCOUNT = "extreme_discount"   # 0-25%


class ArrayType(Enum):
    """Type of PD array"""
    ORDER_BLOCK = "order_block"
    FVG = "fair_value_gap"
    BREAKER = "breaker"
    MITIGATION_BLOCK = "mitigation_block"
    WICK = "wick"
    SUSPENSION_BLOCK = "suspension_block"  # ICT special type
    INVERSION_FVG = "inversion_fvg"
    DAILY_RANGE = "daily_range"
    SWING_RANGE = "swing_range"


class WickType(Enum):
    """Type of wick relative to body"""
    PREMIUM_WICK = "premium_wick"      # Wick above body
    DISCOUNT_WICK = "discount_wick"    # Wick below body


class ArrayCharacter(Enum):
    """ICT Array character - can change via inversion"""
    PREMIUM_ARRAY = "premium"          # Resistance - sell from
    DISCOUNT_ARRAY = "discount"        # Support - buy from
    NEUTRAL = "neutral"


# ==================== DATA CLASSES ====================

@dataclass
class Quadrants:
    """
    ICT Quadrant Grading System
    
    ICT: "Grade EVERYTHING - every range, every wick, every array"
    """
    high: float       # 100%
    low: float        # 0%
    
    # Calculated levels
    upper_quadrant: float = field(init=False)     # 75%
    ce: float = field(init=False)                  # 50% - Consequent Encroachment
    lower_quadrant: float = field(init=False)     # 25%
    range_size: float = field(init=False)
    
    # OTE levels (Optimal Trade Entry)
    ote_high: float = field(init=False)           # 79%
    ote_low: float = field(init=False)            # 62%
    
    def __post_init__(self):
        self.range_size = abs(self.high - self.low)
        self.upper_quadrant = self.low + (self.range_size * 0.75)
        self.ce = self.low + (self.range_size * 0.50)
        self.lower_quadrant = self.low + (self.range_size * 0.25)
        self.ote_high = self.low + (self.range_size * 0.79)
        self.ote_low = self.low + (self.range_size * 0.62)
    
    def get_zone(self, price: float) -> PDZone:
        """Determine which PD zone price is in"""
        if price >= self.upper_quadrant:
            return PDZone.EXTREME_PREMIUM
        elif price >= self.ce:
            return PDZone.PREMIUM
        elif price <= self.lower_quadrant:
            return PDZone.EXTREME_DISCOUNT
        elif price <= self.ce:
            return PDZone.DISCOUNT
        else:
            return PDZone.EQUILIBRIUM
    
    def get_position_pct(self, price: float) -> float:
        """Get price position as % of range (0=low, 100=high)"""
        if self.range_size == 0:
            return 50.0
        return ((price - self.low) / self.range_size) * 100
    
    def is_in_ote(self, price: float) -> bool:
        """Check if price is in Optimal Trade Entry zone (62-79%)"""
        return self.ote_low <= price <= self.ote_high
    
    def is_between_quadrants(self, price: float) -> bool:
        """
        ICT: "Whenever you see price hovering between quadrants don't touch it"
        """
        # Not at any specific quadrant level
        tolerance = self.range_size * 0.02
        at_high = abs(price - self.high) <= tolerance
        at_upper = abs(price - self.upper_quadrant) <= tolerance
        at_ce = abs(price - self.ce) <= tolerance
        at_lower = abs(price - self.lower_quadrant) <= tolerance
        at_low = abs(price - self.low) <= tolerance
        
        return not (at_high or at_upper or at_ce or at_lower or at_low)


@dataclass
class GradedWick:
    """
    ICT Graded Wick with Quadrants
    
    ICT: "Grade wicks like gaps - they act the same way"
    "Premium wick is resistance, discount wick is support"
    """
    wick_type: WickType
    wick_high: float
    wick_low: float
    body_level: float       # Where body starts/ends
    
    # Quadrants
    quadrants: Optional[Quadrants] = None
    
    # Character (can change via inversion)
    character: ArrayCharacter = ArrayCharacter.NEUTRAL
    is_inverted: bool = False
    
    # Candle info
    candle_index: Optional[int] = None
    timestamp: Optional[datetime] = None
    
    # Which wick in series (ICT: "lowest reaching wick")
    is_lowest_reaching: bool = False
    is_highest_reaching: bool = False
    
    def __post_init__(self):
        self.quadrants = Quadrants(self.wick_high, self.wick_low)
        
        # Set initial character based on wick type
        if self.wick_type == WickType.PREMIUM_WICK:
            self.character = ArrayCharacter.PREMIUM_ARRAY
        else:
            self.character = ArrayCharacter.DISCOUNT_ARRAY
    
    def invert(self):
        """
        Invert wick character when price trades through
        ICT: "If price is above premium wick, it acts as discount array"
        """
        self.is_inverted = True
        if self.character == ArrayCharacter.PREMIUM_ARRAY:
            self.character = ArrayCharacter.DISCOUNT_ARRAY
        else:
            self.character = ArrayCharacter.PREMIUM_ARRAY


@dataclass
class PDArray:
    """
    Generic PD Array (Order Block, FVG, Breaker, etc.)
    
    ICT: "Premium arrays are for selling, discount arrays are for buying"
    """
    array_type: ArrayType
    high: float
    low: float
    
    # Quadrants
    quadrants: Optional[Quadrants] = None
    
    # Character
    character: ArrayCharacter = ArrayCharacter.NEUTRAL
    is_inverted: bool = False
    
    # Position in daily range
    zone_in_daily: Optional[PDZone] = None
    
    # Timing
    timestamp: Optional[datetime] = None
    candle_index: Optional[int] = None
    timeframe: str = "unknown"
    
    # Status
    is_mitigated: bool = False
    mitigation_index: Optional[int] = None
    respects_count: int = 0
    
    # Body respect tracking (ICT specific)
    bodies_respecting: bool = False
    
    # Notes
    notes: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        self.quadrants = Quadrants(self.high, self.low)
    
    def set_character_from_daily_zone(self, daily_ce: float):
        """Set character based on position relative to daily CE"""
        midpoint = (self.high + self.low) / 2
        if midpoint > daily_ce:
            self.character = ArrayCharacter.PREMIUM_ARRAY
        else:
            self.character = ArrayCharacter.DISCOUNT_ARRAY
    
    def invert(self):
        """
        Invert array character
        ICT: "When price trades through, array changes character"
        """
        self.is_inverted = True
        if self.character == ArrayCharacter.PREMIUM_ARRAY:
            self.character = ArrayCharacter.DISCOUNT_ARRAY
        else:
            self.character = ArrayCharacter.PREMIUM_ARRAY
    
    def check_body_respect(self, body_open: float, body_close: float) -> bool:
        """
        Check if candle body is respecting the array
        ICT: "Look at the bodies. Bodies tell the story"
        """
        body_high = max(body_open, body_close)
        body_low = min(body_open, body_close)
        
        if self.character == ArrayCharacter.PREMIUM_ARRAY:
            # For premium, bodies should stay below
            if body_high <= self.quadrants.ce:
                self.bodies_respecting = True
                self.respects_count += 1
                return True
        else:
            # For discount, bodies should stay above
            if body_low >= self.quadrants.ce:
                self.bodies_respecting = True
                self.respects_count += 1
                return True
        
        self.bodies_respecting = False
        return False


@dataclass
class SuspensionBlock:
    """
    ICT Suspension Block
    
    ICT: "Volume imbalance at top and bottom - suspended between two imbalances"
    "This is extremely strong. One of the most powerful"
    """
    high: float
    low: float
    
    # Volume imbalances
    upper_imbalance: Tuple[float, float]  # (high, low) of upper VI
    lower_imbalance: Tuple[float, float]  # (high, low) of lower VI
    
    # Quadrants
    quadrants: Optional[Quadrants] = None
    
    # Character
    character: ArrayCharacter = ArrayCharacter.NEUTRAL
    is_inverted: bool = False
    
    # Candle info
    candle_index: Optional[int] = None
    timestamp: Optional[datetime] = None
    
    # Strength
    strength: str = "extreme"  # Always extreme per ICT
    
    def __post_init__(self):
        self.quadrants = Quadrants(self.high, self.low)
    
    def invert(self):
        """Invert when price trades through"""
        self.is_inverted = True
        if self.character == ArrayCharacter.PREMIUM_ARRAY:
            self.character = ArrayCharacter.DISCOUNT_ARRAY
        else:
            self.character = ArrayCharacter.PREMIUM_ARRAY


@dataclass
class DailyRange:
    """
    Daily Range - Primary Reference Frame
    
    ICT: "All PD analysis is relative to the daily range"
    """
    date: datetime
    open_price: float
    high: float
    low: float
    close: float
    
    # Quadrants
    quadrants: Optional[Quadrants] = None
    
    # Zones
    premium_zone: Tuple[float, float] = field(init=False)   # 50-100%
    discount_zone: Tuple[float, float] = field(init=False)  # 0-50%
    
    # Wicks
    upper_wick: Optional[GradedWick] = None
    lower_wick: Optional[GradedWick] = None
    
    # Previous day reference
    prev_day_high: Optional[float] = None
    prev_day_low: Optional[float] = None
    prev_day_close: Optional[float] = None
    
    def __post_init__(self):
        self.quadrants = Quadrants(self.high, self.low)
        self.premium_zone = (self.quadrants.ce, self.high)
        self.discount_zone = (self.low, self.quadrants.ce)
        
        # Grade wicks if present
        body_top = max(self.open_price, self.close)
        body_bottom = min(self.open_price, self.close)
        
        if self.high > body_top:
            self.upper_wick = GradedWick(
                wick_type=WickType.PREMIUM_WICK,
                wick_high=self.high,
                wick_low=body_top,
                body_level=body_top
            )
        
        if self.low < body_bottom:
            self.lower_wick = GradedWick(
                wick_type=WickType.DISCOUNT_WICK,
                wick_high=body_bottom,
                wick_low=self.low,
                body_level=body_bottom
            )
    
    def get_current_zone(self, price: float) -> PDZone:
        """Get which PD zone price is currently in"""
        return self.quadrants.get_zone(price)
    
    def is_in_premium(self, price: float) -> bool:
        """Check if price is in premium zone (above 50%)"""
        return price > self.quadrants.ce
    
    def is_in_discount(self, price: float) -> bool:
        """Check if price is in discount zone (below 50%)"""
        return price < self.quadrants.ce


@dataclass
class PDAnalysis:
    """Complete PD Array analysis result"""
    timestamp: datetime
    current_price: float
    
    # Daily range context
    daily_range: Optional[DailyRange] = None
    current_zone: PDZone = PDZone.EQUILIBRIUM
    position_pct: float = 50.0
    
    # Arrays found
    premium_arrays: List[PDArray] = field(default_factory=list)
    discount_arrays: List[PDArray] = field(default_factory=list)
    
    # Graded wicks
    premium_wicks: List[GradedWick] = field(default_factory=list)
    discount_wicks: List[GradedWick] = field(default_factory=list)
    
    # Special blocks
    suspension_blocks: List[SuspensionBlock] = field(default_factory=list)
    
    # Nearest levels
    nearest_premium: Optional[float] = None
    nearest_discount: Optional[float] = None
    
    # Trading bias
    overall_bias: str = "neutral"
    bias_strength: float = 50.0
    
    # Recommendations
    valid_for_longs: bool = False
    valid_for_shorts: bool = False
    avoid_trading: bool = False


# ==================== MAIN HANDLER ====================

class PDArrayHandler:
    """
    Comprehensive ICT PD Array Handler
    
    ICT Rules:
    - When bullish: ONLY buy from discount (below 50%)
    - When bearish: ONLY sell from premium (above 50%)
    - Equilibrium (50%) is ALWAYS a magnet
    - Grade EVERYTHING with quadrants
    - Bodies tell the story - track body respect
    - Inversion changes array character
    """
    
    def __init__(self):
        self.daily_range: Optional[DailyRange] = None
        self.pd_arrays: List[PDArray] = []
        self.graded_wicks: List[GradedWick] = []
        self.suspension_blocks: List[SuspensionBlock] = []
    
    # ==================== DAILY RANGE ====================
    
    def set_daily_range(self, date: datetime, open_price: float,
                        high: float, low: float, close: float,
                        prev_high: Optional[float] = None,
                        prev_low: Optional[float] = None,
                        prev_close: Optional[float] = None) -> DailyRange:
        """
        Set daily range - primary reference frame
        ICT: "All PD analysis is relative to the daily range"
        """
        self.daily_range = DailyRange(
            date=date, open_price=open_price, high=high, low=low, close=close,
            prev_day_high=prev_high, prev_day_low=prev_low, prev_day_close=prev_close
        )
        return self.daily_range
    
    # ==================== WICK GRADING ====================
    
    def grade_wick(self, wick_type: WickType, wick_high: float, 
                   wick_low: float, body_level: float,
                   candle_index: Optional[int] = None,
                   timestamp: Optional[datetime] = None) -> GradedWick:
        """
        Grade a wick with quadrants
        ICT: "Grade wicks like gaps"
        """
        wick = GradedWick(
            wick_type=wick_type,
            wick_high=wick_high,
            wick_low=wick_low,
            body_level=body_level,
            candle_index=candle_index,
            timestamp=timestamp
        )
        self.graded_wicks.append(wick)
        return wick
    
    def grade_candle_wicks(self, df: pd.DataFrame) -> Tuple[List[GradedWick], List[GradedWick]]:
        """
        Grade all wicks in dataframe
        ICT: "Premium wicks are resistance, discount wicks are support"
        """
        premium_wicks = []
        discount_wicks = []
        
        for idx in range(len(df)):
            row = df.iloc[idx]
            body_top = max(row['open'], row['close'])
            body_bottom = min(row['open'], row['close'])
            
            # Premium wick (above body)
            if row['high'] > body_top:
                wick = self.grade_wick(
                    WickType.PREMIUM_WICK,
                    row['high'], body_top, body_top,
                    idx, row.get('timestamp')
                )
                premium_wicks.append(wick)
            
            # Discount wick (below body)
            if row['low'] < body_bottom:
                wick = self.grade_wick(
                    WickType.DISCOUNT_WICK,
                    body_bottom, row['low'], body_bottom,
                    idx, row.get('timestamp')
                )
                discount_wicks.append(wick)
        
        # Mark lowest/highest reaching
        if discount_wicks:
            lowest = min(discount_wicks, key=lambda w: w.wick_low)
            lowest.is_lowest_reaching = True
        
        if premium_wicks:
            highest = max(premium_wicks, key=lambda w: w.wick_high)
            highest.is_highest_reaching = True
        
        return premium_wicks, discount_wicks
    
    def find_lowest_reaching_wick(self, wicks: List[GradedWick]) -> Optional[GradedWick]:
        """
        Find lowest reaching wick
        ICT: "Lowest reaching wick is the one to use"
        """
        if not wicks:
            return None
        discount = [w for w in wicks if w.wick_type == WickType.DISCOUNT_WICK]
        if not discount:
            return None
        return min(discount, key=lambda w: w.wick_low)
    
    def find_highest_reaching_wick(self, wicks: List[GradedWick]) -> Optional[GradedWick]:
        """Find highest reaching wick"""
        if not wicks:
            return None
        premium = [w for w in wicks if w.wick_type == WickType.PREMIUM_WICK]
        if not premium:
            return None
        return max(premium, key=lambda w: w.wick_high)
    
    # ==================== PD ARRAYS ====================
    
    def add_pd_array(self, array_type: ArrayType, high: float, low: float,
                     timestamp: Optional[datetime] = None,
                     candle_index: Optional[int] = None,
                     timeframe: str = "unknown") -> PDArray:
        """Add a PD array and classify it"""
        array = PDArray(
            array_type=array_type, high=high, low=low,
            timestamp=timestamp, candle_index=candle_index, timeframe=timeframe
        )
        
        # Set character based on daily range if available
        if self.daily_range:
            array.set_character_from_daily_zone(self.daily_range.quadrants.ce)
            array.zone_in_daily = self.daily_range.get_current_zone((high + low) / 2)
        
        self.pd_arrays.append(array)
        return array
    
    def create_suspension_block(self, high: float, low: float,
                                 upper_vi: Tuple[float, float],
                                 lower_vi: Tuple[float, float],
                                 candle_index: Optional[int] = None) -> SuspensionBlock:
        """
        Create ICT Suspension Block
        ICT: "Volume imbalance at top and bottom - extremely strong"
        """
        block = SuspensionBlock(
            high=high, low=low,
            upper_imbalance=upper_vi, lower_imbalance=lower_vi,
            candle_index=candle_index
        )
        
        if self.daily_range:
            midpoint = (high + low) / 2
            if midpoint > self.daily_range.quadrants.ce:
                block.character = ArrayCharacter.PREMIUM_ARRAY
            else:
                block.character = ArrayCharacter.DISCOUNT_ARRAY
        
        self.suspension_blocks.append(block)
        return block
    
    # ==================== INVERSION ====================
    
    def check_and_apply_inversion(self, array: PDArray, current_price: float) -> bool:
        """
        Check if array should be inverted
        ICT: "If price trades through, it changes character"
        """
        if array.is_inverted:
            return False
        
        # Check if price has traded completely through
        if array.character == ArrayCharacter.DISCOUNT_ARRAY:
            if current_price < array.low:
                array.invert()
                array.notes.append("Inverted: Was discount, now premium")
                return True
        else:
            if current_price > array.high:
                array.invert()
                array.notes.append("Inverted: Was premium, now discount")
                return True
        
        return False
    
    # ==================== ANALYSIS ====================
    
    def analyze(self, current_price: float, bias: str = "neutral") -> PDAnalysis:
        """
        Comprehensive PD array analysis
        
        ICT Rules applied:
        - Classify all arrays as premium or discount
        - Find nearest levels
        - Determine if valid for longs/shorts based on zone
        """
        analysis = PDAnalysis(
            timestamp=datetime.now(),
            current_price=current_price
        )
        
        if self.daily_range:
            analysis.daily_range = self.daily_range
            analysis.current_zone = self.daily_range.get_current_zone(current_price)
            analysis.position_pct = self.daily_range.quadrants.get_position_pct(current_price)
        
        # Classify arrays
        for array in self.pd_arrays:
            if not array.is_mitigated:
                if array.character == ArrayCharacter.PREMIUM_ARRAY:
                    analysis.premium_arrays.append(array)
                else:
                    analysis.discount_arrays.append(array)
        
        # Classify wicks
        for wick in self.graded_wicks:
            if wick.wick_type == WickType.PREMIUM_WICK:
                analysis.premium_wicks.append(wick)
            else:
                analysis.discount_wicks.append(wick)
        
        analysis.suspension_blocks = self.suspension_blocks.copy()
        
        # Find nearest levels
        premium_levels = [a.low for a in analysis.premium_arrays if a.low > current_price]
        discount_levels = [a.high for a in analysis.discount_arrays if a.high < current_price]
        
        if premium_levels:
            analysis.nearest_premium = min(premium_levels)
        if discount_levels:
            analysis.nearest_discount = max(discount_levels)
        
        # Trading validity based on ICT rules
        analysis.overall_bias = bias
        
        if bias == "bullish":
            # ICT: "When bullish, ONLY buy from discount"
            analysis.valid_for_longs = analysis.current_zone in [
                PDZone.DISCOUNT, PDZone.EXTREME_DISCOUNT
            ]
            analysis.valid_for_shorts = False
            
        elif bias == "bearish":
            # ICT: "When bearish, ONLY sell from premium"
            analysis.valid_for_shorts = analysis.current_zone in [
                PDZone.PREMIUM, PDZone.EXTREME_PREMIUM
            ]
            analysis.valid_for_longs = False
        
        # Check for no-trade zone
        if self.daily_range and self.daily_range.quadrants.is_between_quadrants(current_price):
            analysis.avoid_trading = True
        
        # Calculate bias strength
        if self.daily_range:
            pct = analysis.position_pct
            if pct >= 75:
                analysis.bias_strength = 90 if bias == "bearish" else 30
            elif pct >= 50:
                analysis.bias_strength = 70 if bias == "bearish" else 50
            elif pct >= 25:
                analysis.bias_strength = 50 if bias == "bullish" else 70
            else:
                analysis.bias_strength = 30 if bias == "bearish" else 90
        
        return analysis
    
    # ==================== TRADING RULES ====================
    
    def get_pd_trading_rules(self) -> Dict[str, str]:
        """Get ICT PD Array trading rules"""
        return {
            "Rule_1": "When bullish, ONLY buy from discount (below 50%)",
            "Rule_2": "When bearish, ONLY sell from premium (above 50%)",
            "Rule_3": "Equilibrium (50%) is ALWAYS a magnet - price seeks it",
            "Rule_4": "Grade EVERYTHING with quadrants (25%, 50%, 75%, 100%)",
            "Rule_5": "Bodies tell the story - wicks show the damage",
            "Rule_6": "If array is traded through, it INVERTS character",
            "Rule_7": "Don't trade when price is between quadrants (no man's land)",
            "Rule_8": "OTE (Optimal Trade Entry) = 62-79% retracement",
            "Rule_9": "Premium wicks = resistance, Discount wicks = support",
            "Rule_10": "Lowest reaching wick is most significant for discount",
            "Rule_11": "Suspension block = volume imbalance at both ends (extremely strong)",
            "Confluence": "Best setups: Array + FVG + Wick CE all at same level"
        }
    
    def get_summary(self) -> str:
        """Generate text summary"""
        lines = ["", "=" * 70, "ICT PD ARRAY ANALYSIS", "=" * 70]
        
        if self.daily_range:
            dr = self.daily_range
            lines.extend([
                f"\nDaily Range ({dr.date.strftime('%Y-%m-%d')}):",
                f"  High: {dr.high:.5f}",
                f"  Upper Quad (75%): {dr.quadrants.upper_quadrant:.5f}",
                f"  CE (50%): {dr.quadrants.ce:.5f}",
                f"  Lower Quad (25%): {dr.quadrants.lower_quadrant:.5f}",
                f"  Low: {dr.low:.5f}"
            ])
        
        lines.append(f"\nPD Arrays: {len(self.pd_arrays)}")
        premium = [a for a in self.pd_arrays if a.character == ArrayCharacter.PREMIUM_ARRAY]
        discount = [a for a in self.pd_arrays if a.character == ArrayCharacter.DISCOUNT_ARRAY]
        lines.append(f"  Premium (sell from): {len(premium)}")
        lines.append(f"  Discount (buy from): {len(discount)}")
        
        lines.append(f"\nGraded Wicks: {len(self.graded_wicks)}")
        lines.append(f"Suspension Blocks: {len(self.suspension_blocks)}")
        
        lines.append("=" * 70)
        return "\n".join(lines)


if __name__ == "__main__":
    print("ICT PD Array Handler - Comprehensive")
    print("=" * 50)
    print("\nKey ICT Concepts:")
    print("  • Premium = Above 50% (Sell zones)")
    print("  • Discount = Below 50% (Buy zones)")
    print("  • CE = 50% (Magnet)")
    print("  • Grade ALL wicks with quadrants")
    print("  • Inversion changes array character")
    print("  • Bodies tell the story")
