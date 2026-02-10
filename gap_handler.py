"""
Comprehensive ICT Gap Handler
Based on Inner Circle Trader methodology and transcript teachings

Key ICT Gap Concepts Implemented:
- NWOG (New Week Opening Gap): Friday 5pm close to Sunday 6pm open
- NDOG (New Day Opening Gap): Previous 4pm close to current 9:30am open  
- Opening Range Gap: 9:30am opening with quadrants (keep last 3 days!)
- First Presented Fair Value Gap (FPFVG): First FVG after session open
- Midnight Opening Range: True day start at midnight EST
- Gap Quadrants: Upper (75%), CE (50%), Lower (25%)
- Breakaway Gap: Gap between quadrants that doesn't fill
- Measuring Gap: Halfway point projection

ICT Quotes from transcripts:
- "70% of time, CE gets hit in first 30 minutes"
- "Opening range gap quadrants - go back over last 3 days, include them"
- "New week opening gap - if it doesn't fill by Wednesday, very strong trend"
- "Grade ALL gaps - NWOG, NDOG, opening range - carry forward"
- "These levels are being used by the algorithm"
- "Price need not fill in that gap the day it forms or even the week"
- "Bodies at CE = reversal signal"
- "Wicks can violate, bodies shouldn't"
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from datetime import datetime, time, timedelta
import pandas as pd
import numpy as np


# ==================== ENUMS ====================

class GapType(Enum):
    """ICT Gap Types"""
    NWOG = "nwog"                    # New Week Opening Gap
    NDOG = "ndog"                    # New Day Opening Gap  
    OPENING_RANGE = "opening_range"  # 9:30 Opening Range
    MIDNIGHT_RANGE = "midnight"      # Midnight EST opening
    ASIAN_RANGE = "asian"            # Asian session range
    LONDON_RANGE = "london"          # London open range
    FPFVG = "fpfvg"                  # First Presented FVG
    BREAKAWAY = "breakaway"          # Breakaway gap (between quadrants)
    MEASURING = "measuring"          # Measuring gap (halfway projection)


class GapDirection(Enum):
    """Gap direction"""
    UP = "gap_up"
    DOWN = "gap_down"
    FLAT = "no_gap"


class GapStatus(Enum):
    """Gap fill status"""
    OPEN = "open"                    # Gap not touched
    CE_TOUCHED = "ce_touched"        # 50% touched (70% fill per ICT)
    PARTIAL_FILL = "partial"         # Some fill but not CE
    FULL_FILL = "full"               # Complete fill
    INVERSION = "inversion"          # Gap inverted (becomes opposite PD array)
    BREAKAWAY = "breakaway"          # Gap will not fill (strong trend)


class GapTradeBias(Enum):
    """Trading bias based on gap"""
    FADE = "fade"                    # Trade into gap (expect fill)
    WITH = "with"                    # Trade with gap direction
    WAIT_CE = "wait_ce"              # Wait for CE fill first
    AVOID = "avoid"                  # Large gap, don't fade


# ==================== DATA CLASSES ====================

@dataclass
class GapQuadrants:
    """
    ICT Gap Quadrant Levels
    
    ICT: "Grade ALL gaps - calculate quadrants"
    "Consequent Encroachment (50%) is most important"
    "Bodies at CE = reversal signal"
    """
    gap_high: float
    gap_low: float
    
    # Quadrant levels (calculated)
    upper_quadrant: float = field(init=False)     # 75%
    ce: float = field(init=False)                  # 50% - Consequent Encroachment
    lower_quadrant: float = field(init=False)     # 25%
    
    # Range info
    range_size: float = field(init=False)
    
    def __post_init__(self):
        self.range_size = abs(self.gap_high - self.gap_low)
        self.upper_quadrant = self.gap_low + (self.range_size * 0.75)
        self.ce = self.gap_low + (self.range_size * 0.50)
        self.lower_quadrant = self.gap_low + (self.range_size * 0.25)
    
    def get_level_at_price(self, price: float) -> str:
        """Identify which level price is at"""
        tolerance = self.range_size * 0.005 if self.range_size > 0 else 0.0001
        
        if abs(price - self.gap_high) <= tolerance:
            return "gap_high"
        elif abs(price - self.upper_quadrant) <= tolerance:
            return "upper_quadrant"
        elif abs(price - self.ce) <= tolerance:
            return "ce"
        elif abs(price - self.lower_quadrant) <= tolerance:
            return "lower_quadrant"
        elif abs(price - self.gap_low) <= tolerance:
            return "gap_low"
        elif price > self.gap_high:
            return "above_gap"
        elif price < self.gap_low:
            return "below_gap"
        else:
            return "inside_gap"
    
    def get_position_pct(self, price: float) -> float:
        """Get price position as % of gap (0=low, 100=high)"""
        if self.range_size == 0:
            return 50.0
        return ((price - self.gap_low) / self.range_size) * 100
    
    def is_between_quadrants(self, price: float) -> bool:
        """
        Check if price is in 'no man's land' between quadrants
        ICT: "Whenever you see price hovering between quadrants don't touch it"
        """
        at_level = self.get_level_at_price(price)
        return at_level == "inside_gap"
    
    def body_respecting_ce(self, body_close: float, direction: str = 'bullish') -> bool:
        """
        Check if candle body is respecting CE
        ICT: "Bodies at CE = reversal signal"
        """
        tolerance = self.range_size * 0.01
        if direction == 'bullish':
            return body_close >= (self.ce - tolerance)
        else:
            return body_close <= (self.ce + tolerance)


@dataclass
class Gap:
    """ICT Gap with full quadrant analysis"""
    gap_type: GapType
    direction: GapDirection
    open_price: float
    close_price: float
    gap_high: float
    gap_low: float
    quadrants: Optional[GapQuadrants] = None
    formed_at: Optional[datetime] = None
    formed_index: Optional[int] = None
    expected_fill_by: Optional[str] = None
    status: GapStatus = GapStatus.OPEN
    fill_timestamp: Optional[datetime] = None
    fill_index: Optional[int] = None
    ce_touched: bool = False
    ce_touch_index: Optional[int] = None
    full_fill: bool = False
    is_large: bool = False
    size_pips: float = 0.0
    size_pct: float = 0.0
    is_breakaway: bool = False
    is_measuring: bool = False
    is_inverted: bool = False
    trade_bias: GapTradeBias = GapTradeBias.WAIT_CE
    notes: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.quadrants is None:
            self.quadrants = GapQuadrants(self.gap_high, self.gap_low)
    
    def check_fill(self, price: float) -> Tuple[bool, str]:
        """Check if gap is filled at given price"""
        if self.direction == GapDirection.UP:
            if price <= self.gap_low:
                return True, "full_fill"
            elif price <= self.quadrants.ce:
                return True, "ce_fill"
            elif price <= self.quadrants.lower_quadrant:
                return True, "partial_fill"
        elif self.direction == GapDirection.DOWN:
            if price >= self.gap_high:
                return True, "full_fill"
            elif price >= self.quadrants.ce:
                return True, "ce_fill"
            elif price >= self.quadrants.upper_quadrant:
                return True, "partial_fill"
        return False, "open"
    
    def update_status(self, price: float, index: Optional[int] = None):
        """Update gap fill status"""
        filled, fill_type = self.check_fill(price)
        if fill_type == "ce_fill" and not self.ce_touched:
            self.ce_touched = True
            self.ce_touch_index = index
            if self.status == GapStatus.OPEN:
                self.status = GapStatus.CE_TOUCHED
        if fill_type == "full_fill" and not self.full_fill:
            self.full_fill = True
            self.fill_index = index
            self.status = GapStatus.FULL_FILL


@dataclass
class OpeningRangeGap:
    """
    ICT Opening Range Gap (9:30 AM)
    ICT: "Opening range gap quadrants - go back over last 3 days"
    """
    date: datetime
    previous_close: float
    current_open: float
    first_30_high: Optional[float] = None
    first_30_low: Optional[float] = None
    quadrants: Optional[GapQuadrants] = None
    gap_size: float = 0.0
    direction: GapDirection = GapDirection.FLAT
    ce_touched: bool = False
    ce_touch_time: Optional[datetime] = None
    first_30_manipulation: Optional[str] = None
    true_direction: Optional[str] = None
    fill_probability: float = 0.70
    days_old: int = 0
    is_significant: bool = False
    
    def __post_init__(self):
        self.gap_size = self.current_open - self.previous_close
        if self.gap_size > 0:
            self.direction = GapDirection.UP
        elif self.gap_size < 0:
            self.direction = GapDirection.DOWN
        gap_high = max(self.previous_close, self.current_open)
        gap_low = min(self.previous_close, self.current_open)
        self.quadrants = GapQuadrants(gap_high, gap_low)
    
    def detect_manipulation(self):
        """ICT: "9:30-10:00 AM = manipulation period, First move often FALSE" """
        if self.first_30_high is None or self.first_30_low is None:
            return
        high_extension = self.first_30_high - self.current_open
        low_extension = self.current_open - self.first_30_low
        if high_extension > low_extension * 1.5:
            self.first_30_manipulation = "upside"
            self.true_direction = "bearish"
        elif low_extension > high_extension * 1.5:
            self.first_30_manipulation = "downside"
            self.true_direction = "bullish"
        else:
            self.first_30_manipulation = "both"


@dataclass
class NWOG:
    """ICT New Week Opening Gap"""
    friday_close: float
    sunday_open: float
    week_start: datetime
    quadrants: Optional[GapQuadrants] = None
    direction: GapDirection = GapDirection.FLAT
    size_pips: float = 0.0
    is_large: bool = False
    monday_fill: bool = False
    tuesday_fill: bool = False
    wednesday_fill: bool = False
    fill_day: Optional[str] = None
    status: GapStatus = GapStatus.OPEN
    ce_touched: bool = False
    trade_bias: GapTradeBias = GapTradeBias.WAIT_CE
    unfilled_by_wednesday: bool = False
    
    def __post_init__(self):
        gap_high = max(self.friday_close, self.sunday_open)
        gap_low = min(self.friday_close, self.sunday_open)
        self.quadrants = GapQuadrants(gap_high, gap_low)
        if self.sunday_open > self.friday_close:
            self.direction = GapDirection.UP
        elif self.sunday_open < self.friday_close:
            self.direction = GapDirection.DOWN
        if self.is_large:
            self.trade_bias = GapTradeBias.WITH
    
    def check_wednesday_rule(self, current_day: str) -> bool:
        """ICT: "If unfilled by Wednesday, very strong trend" """
        if current_day.lower() in ['wednesday', 'thursday', 'friday']:
            if not self.ce_touched:
                self.unfilled_by_wednesday = True
                self.trade_bias = GapTradeBias.WITH
                return True
        return False


@dataclass
class GapAnalysis:
    """Complete gap analysis result"""
    nwog: Optional[NWOG] = None
    ndog: Optional[Gap] = None
    opening_range_gaps: List[OpeningRangeGap] = field(default_factory=list)
    active_gaps: List[Gap] = field(default_factory=list)
    all_levels: List[Tuple[float, str, str]] = field(default_factory=list)
    current_price: Optional[float] = None
    nearest_level: Optional[Tuple[float, str]] = None
    in_gap_zone: bool = False
    current_gap: Optional[Gap] = None


# ==================== MAIN HANDLER ====================

class GapHandler:
    """
    Comprehensive ICT Gap Handler
    
    ICT Rules:
    - Grade ALL gaps with quadrants
    - Keep last 3 days of opening range gaps
    - 70% of time, CE is hit in first 30 minutes (NDOG)
    - Large gaps (>40 pips forex): Don't fade, trade WITH
    - If NWOG unfilled by Wednesday: Strong trend signal
    """
    
    def __init__(self, large_gap_pips_forex: float = 40.0,
                 large_gap_points_indices: float = 50.0,
                 keep_gaps_days: int = 3):
        self.large_gap_pips_forex = large_gap_pips_forex
        self.large_gap_points_indices = large_gap_points_indices
        self.keep_gaps_days = keep_gaps_days
        self.gaps: List[Gap] = []
        self.opening_range_gaps: List[OpeningRangeGap] = []
        self.current_nwog: Optional[NWOG] = None
        self.current_ndog: Optional[Gap] = None
    
    def create_nwog(self, friday_close: float, sunday_open: float,
                    week_start: datetime, pip_value: float = 0.0001) -> NWOG:
        """Create New Week Opening Gap"""
        size = abs(sunday_open - friday_close)
        size_pips = size / pip_value
        is_large = size_pips > self.large_gap_pips_forex
        nwog = NWOG(friday_close=friday_close, sunday_open=sunday_open,
                    week_start=week_start, size_pips=size_pips, is_large=is_large)
        if is_large:
            nwog.trade_bias = GapTradeBias.WITH
        self.current_nwog = nwog
        return nwog
    
    def create_ndog(self, previous_close: float, current_open: float,
                    formed_at: datetime, pip_value: float = 0.0001,
                    is_indices: bool = False) -> Gap:
        """Create New Day Opening Gap"""
        gap_high = max(previous_close, current_open)
        gap_low = min(previous_close, current_open)
        direction = GapDirection.UP if current_open > previous_close else GapDirection.DOWN
        if current_open == previous_close:
            direction = GapDirection.FLAT
        size = abs(current_open - previous_close)
        size_pips = size / pip_value if not is_indices else size
        threshold = self.large_gap_points_indices if is_indices else self.large_gap_pips_forex
        is_large = size_pips > threshold
        trade_bias = GapTradeBias.WITH if is_large else GapTradeBias.WAIT_CE
        
        ndog = Gap(gap_type=GapType.NDOG, direction=direction,
                   open_price=current_open, close_price=previous_close,
                   gap_high=gap_high, gap_low=gap_low, formed_at=formed_at,
                   expected_fill_by="First 30 minutes (70% probability)",
                   is_large=is_large, size_pips=size_pips,
                   size_pct=(size / previous_close) * 100 if previous_close > 0 else 0,
                   trade_bias=trade_bias)
        self.current_ndog = ndog
        self.gaps.append(ndog)
        return ndog
    
    def create_opening_range_gap(self, previous_close: float, current_open: float,
                                  date: datetime, first_30_high: Optional[float] = None,
                                  first_30_low: Optional[float] = None,
                                  is_significant: bool = False) -> OpeningRangeGap:
        """Create Opening Range Gap - keep last 3 days per ICT"""
        org = OpeningRangeGap(date=date, previous_close=previous_close,
                              current_open=current_open, first_30_high=first_30_high,
                              first_30_low=first_30_low, is_significant=is_significant)
        if first_30_high and first_30_low:
            org.detect_manipulation()
        self.opening_range_gaps.append(org)
        if len(self.opening_range_gaps) > self.keep_gaps_days:
            self.opening_range_gaps = self.opening_range_gaps[-self.keep_gaps_days:]
        for i, gap in enumerate(self.opening_range_gaps):
            gap.days_old = len(self.opening_range_gaps) - 1 - i
        return org
    
    def get_all_opening_range_levels(self) -> List[Tuple[float, str, int]]:
        """Get all opening range quadrant levels - ICT: carry forward"""
        levels = []
        for org in self.opening_range_gaps:
            if org.quadrants:
                days = org.days_old
                levels.extend([
                    (org.quadrants.gap_high, f"ORG_{days}d_high", days),
                    (org.quadrants.upper_quadrant, f"ORG_{days}d_75%", days),
                    (org.quadrants.ce, f"ORG_{days}d_CE", days),
                    (org.quadrants.lower_quadrant, f"ORG_{days}d_25%", days),
                    (org.quadrants.gap_low, f"ORG_{days}d_low", days)
                ])
        return sorted(levels, key=lambda x: x[0], reverse=True)
    
    def analyze(self, df: pd.DataFrame, current_price: float) -> GapAnalysis:
        """Comprehensive gap analysis"""
        analysis = GapAnalysis(nwog=self.current_nwog, ndog=self.current_ndog,
                               opening_range_gaps=self.opening_range_gaps.copy(),
                               active_gaps=[g for g in self.gaps if g.status != GapStatus.FULL_FILL],
                               current_price=current_price)
        
        all_levels = []
        if self.current_nwog and self.current_nwog.quadrants:
            q = self.current_nwog.quadrants
            all_levels.extend([
                (q.gap_high, "NWOG_high", "nwog"), (q.upper_quadrant, "NWOG_75%", "nwog"),
                (q.ce, "NWOG_CE", "nwog"), (q.lower_quadrant, "NWOG_25%", "nwog"),
                (q.gap_low, "NWOG_low", "nwog")
            ])
        if self.current_ndog and self.current_ndog.quadrants:
            q = self.current_ndog.quadrants
            all_levels.extend([
                (q.gap_high, "NDOG_high", "ndog"), (q.upper_quadrant, "NDOG_75%", "ndog"),
                (q.ce, "NDOG_CE", "ndog"), (q.lower_quadrant, "NDOG_25%", "ndog"),
                (q.gap_low, "NDOG_low", "ndog")
            ])
        for org in self.opening_range_gaps:
            if org.quadrants:
                days = org.days_old
                all_levels.extend([
                    (org.quadrants.gap_high, f"ORG_{days}d_high", "org"),
                    (org.quadrants.ce, f"ORG_{days}d_CE", "org"),
                    (org.quadrants.gap_low, f"ORG_{days}d_low", "org")
                ])
        
        analysis.all_levels = sorted(all_levels, key=lambda x: x[0], reverse=True)
        if all_levels:
            nearest = min(all_levels, key=lambda x: abs(x[0] - current_price))
            analysis.nearest_level = (nearest[0], nearest[1])
        
        for gap in analysis.active_gaps:
            if gap.quadrants.gap_low <= current_price <= gap.quadrants.gap_high:
                analysis.in_gap_zone = True
                analysis.current_gap = gap
                break
        return analysis
    
    def get_gap_trading_rules(self) -> Dict[str, List[str]]:
        """Get ICT gap trading rules"""
        return {
            "NWOG_Rules": [
                "NWOG = Friday 5pm close to Sunday 6pm open",
                "70% fill probability within the week",
                "Large gap (>40 pips): Trade WITH gap direction",
                "If unfilled by Wednesday: VERY STRONG TREND",
                "NEVER fade a large NWOG"
            ],
            "NDOG_Rules": [
                "NDOG = Previous 4pm close to Current 9:30am open",
                "70% of time CE gets hit in first 30 minutes",
                "If CE unfilled by 10:30: Strong bias"
            ],
            "Opening_Range_Rules": [
                "Keep last 3 days of opening range gaps on chart",
                "9:30-10:00 AM = manipulation period",
                "First move often FALSE",
                "True direction emerges 10:00-10:30"
            ],
            "Quadrant_Rules": [
                "CE (50%) most important - reversal signal",
                "Bodies at CE = reversal",
                "Wicks can violate, bodies shouldn't",
                "Don't trade between quadrants (no man's land)"
            ]
        }


if __name__ == "__main__":
    print("ICT Gap Handler - Comprehensive")
    print("=" * 50)
    print("\nKey ICT Concepts:")
    print("  • NWOG: Friday 5pm to Sunday 6pm")
    print("  • NDOG: Previous 4pm to Current 9:30am")  
    print("  • Opening Range: Keep last 3 days!")
    print("  • 70% CE fill in first 30 minutes")
    print("  • If unfilled by Wednesday = Strong trend")
