"""
Comprehensive ICT Market Structure Handler
Based on Inner Circle Trader methodology and transcript teachings

Key ICT Concepts Implemented:
- Market Structure Shift (MSS): "Shift in market structure" - confirmed reversal
- Break of Structure (BOS): Continuation break in trend direction
- Change of Character (CHoCH): First break against trend (warning)
- Internal vs External Structure: Pullback swings vs major swings
- Premium/Discount Zones: Above/below equilibrium of dealing range
- Dealing Range: High to low of significant move
- Market Maker Models: AMD (Accumulation, Manipulation, Distribution)
- Smart Money Reversal: Structure break + displacement + FVG
- Body vs Wick: "Bodies tell the story, wicks do the damage"

ICT Quotes from transcripts:
- "Shift in market structure - that's enough to indicate shift bearishly"
- "Run fib from high to low, look for it to trade to premium (above 50%)"
- "Model 2022 - buy side taken into premium, shift in market structure, FVG"

Author: ICT Signal Engine
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from datetime import datetime
import pandas as pd
import numpy as np


# ==================== ENUMS ====================

class StructureType(Enum):
    HIGHER_HIGH = "HH"
    HIGHER_LOW = "HL"
    LOWER_HIGH = "LH"
    LOWER_LOW = "LL"
    EQUAL_HIGH = "EQH"
    EQUAL_LOW = "EQL"


class StructureBreakType(Enum):
    BOS = "BOS"      # Break of Structure - continuation
    CHOCH = "CHoCH"  # Change of Character - first reversal warning
    MSS = "MSS"      # Market Structure Shift - confirmed reversal
    SMS = "SMS"      # Smart Money Shift - MSS with displacement + FVG


class MarketPhase(Enum):
    ACCUMULATION = "accumulation"
    MANIPULATION = "manipulation"
    DISTRIBUTION = "distribution"
    REACCUMULATION = "reaccumulation"
    REDISTRIBUTION = "redistribution"


class TrendState(Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    RANGING = "ranging"
    TRANSITIONING = "transitioning"


class PriceZone(Enum):
    PREMIUM = "premium"      # Above equilibrium (50%)
    DISCOUNT = "discount"    # Below equilibrium (50%)
    EQUILIBRIUM = "equilibrium"


class SwingStrength(Enum):
    WEAK = 1
    MEDIUM = 2
    STRONG = 3
    EXTREME = 4


# ==================== DATA CLASSES ====================

@dataclass
class SwingPoint:
    """ICT Swing Point with body/wick tracking"""
    index: int
    timestamp: Optional[datetime]
    price: float
    is_high: bool
    
    structure_type: Optional[StructureType] = None
    strength: SwingStrength = SwingStrength.MEDIUM
    
    is_internal: bool = False
    is_external: bool = False
    is_equal: bool = False
    
    is_broken: bool = False
    broken_by_body: bool = False
    broken_by_wick: bool = False
    break_index: Optional[int] = None
    
    formation_displacement: float = 0.0
    zone: Optional[PriceZone] = None
    position_in_range: Optional[float] = None
    
    has_liquidity_above: bool = False
    has_liquidity_below: bool = False
    
    def __str__(self):
        swing_type = "High" if self.is_high else "Low"
        structure = self.structure_type.value if self.structure_type else "N/A"
        status = "BROKEN" if self.is_broken else "INTACT"
        return f"{swing_type} {structure} @ {self.price:.5f} [{status}]"


@dataclass
class StructureBreak:
    """ICT Structure Break (BOS/CHoCH/MSS/SMS)"""
    index: int
    timestamp: Optional[datetime]
    break_type: StructureBreakType
    direction: str
    
    broken_swing: Optional[SwingPoint] = None
    broken_level: float = 0.0
    
    displacement: float = 0.0
    displacement_pct: float = 0.0
    broke_with_body: bool = False
    broke_with_wick: bool = False
    
    is_confirmed: bool = False
    has_fvg: bool = False
    has_displacement: bool = False
    
    retest_occurred: bool = False
    retest_held: bool = False
    
    notes: List[str] = field(default_factory=list)
    
    def __str__(self):
        confirmed = "✓" if self.is_confirmed else "?"
        fvg = "+FVG" if self.has_fvg else ""
        return f"{self.break_type.value} {self.direction} {confirmed} {fvg} @ {self.broken_level:.5f}"


@dataclass
class DealingRange:
    """ICT Dealing Range with premium/discount zones"""
    high: float
    low: float
    timestamp_high: Optional[datetime] = None
    timestamp_low: Optional[datetime] = None
    
    equilibrium: float = field(init=False)
    upper_quadrant: float = field(init=False)
    lower_quadrant: float = field(init=False)
    std_dev_1_high: float = field(init=False)
    std_dev_1_low: float = field(init=False)
    
    def __post_init__(self):
        range_size = self.high - self.low
        self.equilibrium = self.low + (range_size * 0.5)
        self.upper_quadrant = self.low + (range_size * 0.75)
        self.lower_quadrant = self.low + (range_size * 0.25)
        self.std_dev_1_high = self.high + range_size
        self.std_dev_1_low = self.low - range_size
    
    def get_zone(self, price: float) -> PriceZone:
        if price > self.equilibrium:
            return PriceZone.PREMIUM
        elif price < self.equilibrium:
            return PriceZone.DISCOUNT
        return PriceZone.EQUILIBRIUM
    
    def get_position(self, price: float) -> float:
        range_size = self.high - self.low
        if range_size == 0:
            return 50.0
        return ((price - self.low) / range_size) * 100
    
    def __str__(self):
        return f"DR: {self.low:.5f} - {self.high:.5f} | EQ: {self.equilibrium:.5f}"


@dataclass 
class MarketStructureState:
    """Current market structure state"""
    trend: TrendState = TrendState.RANGING
    dealing_range: Optional[DealingRange] = None
    current_zone: Optional[PriceZone] = None
    
    current_high: Optional[SwingPoint] = None
    current_low: Optional[SwingPoint] = None
    external_high: Optional[SwingPoint] = None
    external_low: Optional[SwingPoint] = None
    internal_high: Optional[SwingPoint] = None
    internal_low: Optional[SwingPoint] = None
    
    last_break: Optional[StructureBreak] = None
    last_mss: Optional[StructureBreak] = None
    
    consecutive_hh: int = 0
    consecutive_hl: int = 0
    consecutive_lh: int = 0
    consecutive_ll: int = 0


@dataclass
class MarketStructureAnalysis:
    """Complete analysis result"""
    swing_highs: List[SwingPoint]
    swing_lows: List[SwingPoint]
    structure_breaks: List[StructureBreak]
    state: MarketStructureState
    dealing_range: Optional[DealingRange]
    
    bos_breaks: List[StructureBreak] = field(default_factory=list)
    choch_breaks: List[StructureBreak] = field(default_factory=list)
    mss_breaks: List[StructureBreak] = field(default_factory=list)


# ==================== MAIN HANDLER ====================

class MarketStructureHandler:
    """
    ICT Market Structure Handler
    
    Implements BOS, CHoCH, MSS, SMS detection with:
    - Body vs wick break distinction
    - Internal vs External structure
    - Premium/Discount zone tracking
    - Break confirmation (displacement + FVG)
    """
    
    def __init__(self,
                 swing_lookback: int = 5,
                 min_displacement_pct: float = 0.1,
                 equal_level_tolerance: float = 0.001):
        self.swing_lookback = swing_lookback
        self.min_displacement_pct = min_displacement_pct
        self.equal_level_tolerance = equal_level_tolerance
        
        self.swing_highs: List[SwingPoint] = []
        self.swing_lows: List[SwingPoint] = []
        self.structure_breaks: List[StructureBreak] = []
        self.state = MarketStructureState()
        self.dealing_range: Optional[DealingRange] = None
    
    def analyze(self, df: pd.DataFrame) -> MarketStructureAnalysis:
        """Complete market structure analysis"""
        self.swing_highs = []
        self.swing_lows = []
        self.structure_breaks = []
        
        self._detect_swing_points(df)
        self._classify_swing_types()
        self._identify_equal_levels()
        self._establish_dealing_range(df)
        self._classify_internal_external()
        self._set_zone_context()
        self._detect_structure_breaks(df)
        self._confirm_breaks(df)
        self._detect_retests(df)
        self._update_state(df)
        
        return self._build_analysis()
    
    def _detect_swing_points(self, df: pd.DataFrame):
        """Detect swing highs and lows"""
        for i in range(self.swing_lookback, len(df) - self.swing_lookback):
            timestamp = df.index[i] if isinstance(df.index[i], (datetime, pd.Timestamp)) else None
            
            # Swing high check
            is_swing_high = True
            current_high = df.iloc[i]['high']
            for j in range(1, self.swing_lookback + 1):
                if df.iloc[i - j]['high'] >= current_high or df.iloc[i + j]['high'] >= current_high:
                    is_swing_high = False
                    break
            
            if is_swing_high:
                swing = SwingPoint(index=i, timestamp=timestamp, price=current_high, is_high=True)
                self.swing_highs.append(swing)
            
            # Swing low check
            is_swing_low = True
            current_low = df.iloc[i]['low']
            for j in range(1, self.swing_lookback + 1):
                if df.iloc[i - j]['low'] <= current_low or df.iloc[i + j]['low'] <= current_low:
                    is_swing_low = False
                    break
            
            if is_swing_low:
                swing = SwingPoint(index=i, timestamp=timestamp, price=current_low, is_high=False)
                self.swing_lows.append(swing)
    
    def _classify_swing_types(self):
        """Classify as HH, HL, LH, LL"""
        for i in range(1, len(self.swing_highs)):
            current, previous = self.swing_highs[i], self.swing_highs[i - 1]
            if current.price > previous.price:
                current.structure_type = StructureType.HIGHER_HIGH
            elif current.price < previous.price:
                current.structure_type = StructureType.LOWER_HIGH
            else:
                current.structure_type = StructureType.EQUAL_HIGH
                current.is_equal = previous.is_equal = True
        
        for i in range(1, len(self.swing_lows)):
            current, previous = self.swing_lows[i], self.swing_lows[i - 1]
            if current.price > previous.price:
                current.structure_type = StructureType.HIGHER_LOW
            elif current.price < previous.price:
                current.structure_type = StructureType.LOWER_LOW
            else:
                current.structure_type = StructureType.EQUAL_LOW
                current.is_equal = previous.is_equal = True
    
    def _identify_equal_levels(self):
        """Identify equal highs/lows (liquidity)"""
        for i, swing in enumerate(self.swing_highs):
            for other in self.swing_highs[i + 1:]:
                if abs(swing.price - other.price) / swing.price <= self.equal_level_tolerance:
                    swing.is_equal = other.is_equal = True
                    swing.structure_type = other.structure_type = StructureType.EQUAL_HIGH
                    swing.has_liquidity_above = other.has_liquidity_above = True
        
        for i, swing in enumerate(self.swing_lows):
            for other in self.swing_lows[i + 1:]:
                if abs(swing.price - other.price) / swing.price <= self.equal_level_tolerance:
                    swing.is_equal = other.is_equal = True
                    swing.structure_type = other.structure_type = StructureType.EQUAL_LOW
                    swing.has_liquidity_below = other.has_liquidity_below = True
    
    def _establish_dealing_range(self, df: pd.DataFrame):
        """Establish current dealing range"""
        if not self.swing_highs or not self.swing_lows:
            self.dealing_range = DealingRange(high=df['high'].max(), low=df['low'].min())
            return
        
        unbroken_highs = [s for s in self.swing_highs if not s.is_broken]
        unbroken_lows = [s for s in self.swing_lows if not s.is_broken]
        
        if unbroken_highs and unbroken_lows:
            highest = max(unbroken_highs, key=lambda s: s.price)
            lowest = min(unbroken_lows, key=lambda s: s.price)
            self.dealing_range = DealingRange(
                high=highest.price, low=lowest.price,
                timestamp_high=highest.timestamp, timestamp_low=lowest.timestamp
            )
        else:
            self.dealing_range = DealingRange(high=df['high'].max(), low=df['low'].min())
    
    def _classify_internal_external(self):
        """Classify internal vs external structure"""
        if not self.dealing_range:
            return
        
        for swing in self.swing_highs + self.swing_lows:
            if swing.is_broken:
                swing.is_internal = True
            elif swing.is_high and abs(swing.price - self.dealing_range.high) < 0.0001:
                swing.is_external = True
            elif not swing.is_high and abs(swing.price - self.dealing_range.low) < 0.0001:
                swing.is_external = True
            elif self.dealing_range.low < swing.price < self.dealing_range.high:
                swing.is_internal = True
    
    def _set_zone_context(self):
        """Set premium/discount zone for swings"""
        if not self.dealing_range:
            return
        for swing in self.swing_highs + self.swing_lows:
            swing.zone = self.dealing_range.get_zone(swing.price)
            swing.position_in_range = self.dealing_range.get_position(swing.price)
    
    def _detect_structure_breaks(self, df: pd.DataFrame):
        """Detect BOS, CHoCH, MSS"""
        # Bullish breaks (breaking highs)
        for swing in self.swing_highs:
            if swing.is_broken:
                continue
            for i in range(swing.index + 1, len(df)):
                candle = df.iloc[i]
                body_high = max(candle['open'], candle['close'])
                
                if candle['high'] > swing.price or body_high > swing.price:
                    break_type = self._classify_break_type(is_bullish=True)
                    displacement = candle['high'] - swing.price
                    
                    structure_break = StructureBreak(
                        index=i,
                        timestamp=df.index[i] if isinstance(df.index[i], (datetime, pd.Timestamp)) else None,
                        break_type=break_type,
                        direction="bullish",
                        broken_swing=swing,
                        broken_level=swing.price,
                        displacement=displacement,
                        displacement_pct=(displacement / swing.price) * 100,
                        broke_with_body=body_high > swing.price,
                        broke_with_wick=candle['high'] > swing.price
                    )
                    
                    self.structure_breaks.append(structure_break)
                    swing.is_broken = True
                    swing.break_index = i
                    swing.broken_by_body = structure_break.broke_with_body
                    swing.broken_by_wick = structure_break.broke_with_wick
                    break
        
        # Bearish breaks (breaking lows)
        for swing in self.swing_lows:
            if swing.is_broken:
                continue
            for i in range(swing.index + 1, len(df)):
                candle = df.iloc[i]
                body_low = min(candle['open'], candle['close'])
                
                if candle['low'] < swing.price or body_low < swing.price:
                    break_type = self._classify_break_type(is_bullish=False)
                    displacement = swing.price - candle['low']
                    
                    structure_break = StructureBreak(
                        index=i,
                        timestamp=df.index[i] if isinstance(df.index[i], (datetime, pd.Timestamp)) else None,
                        break_type=break_type,
                        direction="bearish",
                        broken_swing=swing,
                        broken_level=swing.price,
                        displacement=displacement,
                        displacement_pct=(displacement / swing.price) * 100,
                        broke_with_body=body_low < swing.price,
                        broke_with_wick=candle['low'] < swing.price
                    )
                    
                    self.structure_breaks.append(structure_break)
                    swing.is_broken = True
                    swing.break_index = i
                    swing.broken_by_body = structure_break.broke_with_body
                    swing.broken_by_wick = structure_break.broke_with_wick
                    break
    
    def _classify_break_type(self, is_bullish: bool) -> StructureBreakType:
        """Classify break as BOS, CHoCH, or MSS"""
        trend = self._get_current_trend()
        
        if is_bullish:
            if trend == TrendState.BULLISH:
                return StructureBreakType.BOS
            elif trend == TrendState.BEARISH:
                return StructureBreakType.MSS
            return StructureBreakType.BOS
        else:
            if trend == TrendState.BEARISH:
                return StructureBreakType.BOS
            elif trend == TrendState.BULLISH:
                return StructureBreakType.MSS
            return StructureBreakType.BOS
    
    def _get_current_trend(self) -> TrendState:
        """Determine current trend"""
        recent_highs = self.swing_highs[-3:] if len(self.swing_highs) >= 3 else self.swing_highs
        recent_lows = self.swing_lows[-3:] if len(self.swing_lows) >= 3 else self.swing_lows
        
        hh = sum(1 for h in recent_highs if h.structure_type == StructureType.HIGHER_HIGH)
        hl = sum(1 for l in recent_lows if l.structure_type == StructureType.HIGHER_LOW)
        lh = sum(1 for h in recent_highs if h.structure_type == StructureType.LOWER_HIGH)
        ll = sum(1 for l in recent_lows if l.structure_type == StructureType.LOWER_LOW)
        
        if hh >= 2 and hl >= 1:
            return TrendState.BULLISH
        elif lh >= 1 and ll >= 2:
            return TrendState.BEARISH
        return TrendState.RANGING
    
    def _confirm_breaks(self, df: pd.DataFrame):
        """Confirm breaks with displacement and FVG"""
        for b in self.structure_breaks:
            if b.broke_with_body:
                b.is_confirmed = True
            
            # Check displacement
            if b.index + 3 < len(df):
                future = df.iloc[b.index:b.index + 5]
                if b.direction == "bullish":
                    post_disp = future['high'].max() - b.broken_level
                else:
                    post_disp = b.broken_level - future['low'].min()
                
                if post_disp / b.broken_level * 100 >= self.min_displacement_pct:
                    b.has_displacement = True
            
            # Check for FVG
            b.has_fvg = self._check_fvg(df, b)
            
            # Upgrade to SMS if MSS + FVG
            if b.break_type == StructureBreakType.MSS and b.has_fvg:
                b.break_type = StructureBreakType.SMS
                b.notes.append("Smart Money Shift - MSS + FVG")
    
    def _check_fvg(self, df: pd.DataFrame, b: StructureBreak) -> bool:
        """Check for FVG at break"""
        idx = b.index
        if idx < 1 or idx + 1 >= len(df):
            return False
        
        c1, c3 = df.iloc[idx - 1], df.iloc[idx + 1]
        if b.direction == "bullish":
            return c1['high'] < c3['low']
        return c1['low'] > c3['high']
    
    def _detect_retests(self, df: pd.DataFrame):
        """Detect retests of broken levels"""
        for b in self.structure_breaks:
            if b.index + 5 >= len(df):
                continue
            
            future = df.iloc[b.index + 1:min(b.index + 20, len(df))]
            for i, (_, candle) in enumerate(future.iterrows()):
                if candle['low'] <= b.broken_level <= candle['high']:
                    b.retest_occurred = True
                    actual_idx = b.index + 1 + i
                    if actual_idx + 1 < len(df):
                        next_c = df.iloc[actual_idx + 1]
                        if b.direction == "bullish" and next_c['close'] > b.broken_level:
                            b.retest_held = True
                        elif b.direction == "bearish" and next_c['close'] < b.broken_level:
                            b.retest_held = True
                    break
    
    def _update_state(self, df: pd.DataFrame):
        """Update market state"""
        self.state.trend = self._get_current_trend()
        self.state.dealing_range = self.dealing_range
        
        if self.dealing_range and len(df) > 0:
            self.state.current_zone = self.dealing_range.get_zone(df.iloc[-1]['close'])
        
        if self.swing_highs:
            self.state.current_high = self.swing_highs[-1]
        if self.swing_lows:
            self.state.current_low = self.swing_lows[-1]
        
        ext_highs = [s for s in self.swing_highs if s.is_external and not s.is_broken]
        ext_lows = [s for s in self.swing_lows if s.is_external and not s.is_broken]
        if ext_highs:
            self.state.external_high = max(ext_highs, key=lambda s: s.price)
        if ext_lows:
            self.state.external_low = min(ext_lows, key=lambda s: s.price)
        
        if self.structure_breaks:
            self.state.last_break = self.structure_breaks[-1]
            mss = [b for b in self.structure_breaks if b.break_type in [StructureBreakType.MSS, StructureBreakType.SMS]]
            if mss:
                self.state.last_mss = mss[-1]
    
    def _build_analysis(self) -> MarketStructureAnalysis:
        """Build analysis result"""
        bos = [b for b in self.structure_breaks if b.break_type == StructureBreakType.BOS]
        choch = [b for b in self.structure_breaks if b.break_type == StructureBreakType.CHOCH]
        mss = [b for b in self.structure_breaks if b.break_type in [StructureBreakType.MSS, StructureBreakType.SMS]]
        
        return MarketStructureAnalysis(
            swing_highs=self.swing_highs,
            swing_lows=self.swing_lows,
            structure_breaks=self.structure_breaks,
            state=self.state,
            dealing_range=self.dealing_range,
            bos_breaks=bos,
            choch_breaks=choch,
            mss_breaks=mss
        )
    
    def get_bias(self) -> str:
        """Get current market bias"""
        if self.state.trend == TrendState.BULLISH:
            return "BULLISH"
        elif self.state.trend == TrendState.BEARISH:
            return "BEARISH"
        return "NEUTRAL"
    
    def get_summary(self) -> str:
        """Generate text summary"""
        lines = [
            "", "=" * 70, "ICT MARKET STRUCTURE ANALYSIS", "=" * 70,
            f"\nTrend: {self.state.trend.value.upper()}",
            f"Zone: {self.state.current_zone.value if self.state.current_zone else 'N/A'}",
        ]
        if self.dealing_range:
            lines.append(f"\n{self.dealing_range}")
        lines.extend([
            f"\nSwing Highs: {len(self.swing_highs)} | Swing Lows: {len(self.swing_lows)}",
            f"Structure Breaks: {len(self.structure_breaks)}",
            f"  BOS: {sum(1 for b in self.structure_breaks if b.break_type == StructureBreakType.BOS)}",
            f"  CHoCH: {sum(1 for b in self.structure_breaks if b.break_type == StructureBreakType.CHOCH)}",
            f"  MSS/SMS: {sum(1 for b in self.structure_breaks if b.break_type in [StructureBreakType.MSS, StructureBreakType.SMS])}",
        ])
        if self.structure_breaks:
            lines.append("\nRecent Breaks:")
            for b in self.structure_breaks[-5:]:
                lines.append(f"  {b}")
        lines.append("=" * 70 + "\n")
        return "\n".join(lines)


if __name__ == "__main__":
    print("ICT Market Structure Handler")
    print("=" * 50)
    print("\nKey Concepts: BOS, CHoCH, MSS, SMS")
    print("Internal/External Structure, Premium/Discount Zones")
