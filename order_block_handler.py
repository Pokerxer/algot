"""
Comprehensive ICT Order Block Handler
Based on Inner Circle Trader methodology and transcript teachings

Key ICT Concepts Implemented:
- Order Block Definition: "Orders the change in state of delivery" (NOT about orders)
- Last opposite-colored candle before displacement
- Mean Threshold: 50% of OB body range (NOT consequent encroachment)
- Quadrant Levels: 25%, 50%, 75% for entry refinement
- Consecutive Candle Blending: Multiple same-direction candles become one OB
- Body Projection: Select candle with lowest (bullish) or highest (bearish) body
- Entry at Opening Price: "Change in state of delivery"
- Stop Placement: Below close (bullish) or above close (bearish) primarily
- Reclaimed Order Blocks: Market Maker Buy/Sell model OBs
- Breaker Blocks: Failed OBs acting as reversal zones
- Propulsion Blocks: OBs that launch price with strong momentum
- Mitigation Blocks: OBs that have been tested and respected
- Validation: 2-3x OB range rally/decline before retracement
- Quadrant Level Filtering: OB must be at or below dealing range quadrant

Author: ICT Signal Engine
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Dict, Union
from enum import Enum
from datetime import datetime
import pandas as pd
import numpy as np


# ==================== ENUMS ====================

class OrderBlockType(Enum):
    """Order block classification"""
    BULLISH = "bullish"     # Down-close candle before bullish displacement
    BEARISH = "bearish"     # Up-close candle before bearish displacement


class OrderBlockStatus(Enum):
    """Order block current status"""
    UNTESTED = "untested"           # Not yet revisited
    TESTED = "tested"               # Price returned and respected  
    MITIGATED = "mitigated"         # Partially used up
    FAILED = "failed"               # Price traded through completely
    BREAKER = "breaker"             # Failed OB now acting as reversal zone
    RECLAIMED = "reclaimed"         # Used again after initial respect
    PROPULSION = "propulsion"       # Launching price with strong momentum
    INVALIDATED = "invalidated"     # Too old or structurally broken


class OrderBlockStrength(Enum):
    """Order block strength classification"""
    WEAK = 1            # Small displacement
    MEDIUM = 2          # Moderate displacement  
    STRONG = 3          # Large displacement
    EXTREME = 4         # ICT Extreme Order Block - massive displacement


class OrderBlockContext(Enum):
    """Market Maker Model context for the order block"""
    BUY_SIDE_CURVE = "buy_side"     # In accumulation phase
    SELL_SIDE_CURVE = "sell_side"  # In distribution phase
    SMART_MONEY_REVERSAL = "smr"   # At reversal point
    UNKNOWN = "unknown"


# ==================== DATA CLASSES ====================

@dataclass
class ConsecutiveCandleRange:
    """
    ICT Consecutive Candle Blending
    When multiple same-direction candles appear before displacement,
    they are blended into one order block range.
    
    ICT: "Two consecutive down-close candles - that's all one range"
    """
    start_index: int
    end_index: int
    num_candles: int
    combined_open: float        # Open of first candle
    combined_high: float        # Highest high
    combined_low: float         # Lowest low
    combined_close: float       # Close of last candle
    body_range_high: float      # Highest body level
    body_range_low: float       # Lowest body level
    mean_threshold: float       # 50% of body range
    upper_quadrant: float       # 75% of body range
    lower_quadrant: float       # 25% of body range
    
    # The candle with the best body projection
    best_body_candle_index: int
    best_body_projection: float  # Lowest close for bullish, highest close for bearish


@dataclass 
class OrderBlock:
    """
    ICT Order Block Structure
    
    ICT Definition: "The order block is the last opposite-colored candle 
    before a displacement. It orders (coordinates) the change in state of delivery."
    
    Key Levels:
    - Opening Price: Primary entry level
    - Mean Threshold: 50% of body (NOT consequent encroachment)
    - Upper/Lower Quadrants: 25% and 75% levels
    """
    # Core identification
    index: int                              # Primary candle index
    timestamp: Optional[datetime] = None
    block_type: OrderBlockType = OrderBlockType.BULLISH
    
    # Single candle OHLC
    open: float = 0.0                       # CRITICAL: Entry level
    high: float = 0.0
    low: float = 0.0
    close: float = 0.0
    
    # ICT Key Levels (based on BODY, not wick)
    mean_threshold: float = 0.0             # 50% of body range
    upper_quadrant: float = 0.0             # 75% level
    lower_quadrant: float = 0.0             # 25% level
    body_high: float = 0.0                  # Top of body
    body_low: float = 0.0                   # Bottom of body
    body_range: float = 0.0                 # Size of body
    
    # Consecutive candle handling
    is_blended: bool = False                # Multiple candles blended
    blended_range: Optional[ConsecutiveCandleRange] = None
    num_candles_in_block: int = 1
    
    # Classification
    strength: OrderBlockStrength = OrderBlockStrength.MEDIUM
    displacement_size: float = 0.0
    displacement_ratio: float = 0.0         # Displacement / OB body
    
    # Status tracking
    status: OrderBlockStatus = OrderBlockStatus.UNTESTED
    times_tested: int = 0
    times_respected: int = 0
    times_failed: int = 0
    
    # ICT-specific flags
    is_extreme_ob: bool = False             # Extreme order block
    is_breaker: bool = False                # Failed OB acting as reversal
    is_reclaimed: bool = False              # Used multiple times
    is_propulsion: bool = False             # Launched price strongly
    is_mitigated: bool = False              # Partially used
    is_refined: bool = False                # Higher timeframe refined
    
    # Body respect tracking (ICT: "Bodies tell the story")
    body_respected: int = 0                 # Times bodies stayed above/below MT
    body_violated: int = 0                  # Times bodies crossed MT
    wick_only_tests: int = 0                # Times only wicks tested (respect)
    
    # Volume imbalance within OB
    has_volume_imbalance: bool = False
    volume_imbalance_range: Optional[Tuple[float, float]] = None
    
    # Market Maker Model context
    mm_context: OrderBlockContext = OrderBlockContext.UNKNOWN
    is_buy_side_curve: bool = False         # Left of low in MM Buy Model
    is_sell_side_curve: bool = False        # Right of low in MM Buy Model
    
    # Validation tracking
    validated_by_displacement: bool = False # 2-3x OB range movement
    validation_target_reached: bool = False
    
    # Dealing range context
    at_or_below_quadrant: bool = False      # ICT: Must be at/below quadrant for entry
    dealing_range_quadrant: Optional[float] = None
    position_in_dealing_range: Optional[float] = None  # 0-100
    
    # Confluence
    has_fvg_overlap: bool = False
    has_liquidity_nearby: bool = False
    at_premium_discount_level: bool = False
    
    # Notes
    notes: List[str] = field(default_factory=list)
    
    def __str__(self):
        ob_type = "BULLISH" if self.block_type == OrderBlockType.BULLISH else "BEARISH"
        status = self.status.value.upper()
        blended = f" (Blended {self.num_candles_in_block})" if self.is_blended else ""
        extreme = " [EXTREME]" if self.is_extreme_ob else ""
        return (f"{ob_type} OB{blended}{extreme} [{status}] @ Open: {self.open:.5f} "
                f"| MT: {self.mean_threshold:.5f} | Tested: {self.times_tested}x")
    
    def get_entry_price(self, entry_type: str = "open") -> float:
        """
        Get entry price based on ICT methodology
        
        ICT Entry Types:
        - "open": Opening price (Change in State of Delivery) - PRIMARY
        - "mt": Mean Threshold (50%) - More conservative
        - "aggressive": Near the extreme of OB
        - "quadrant": At upper/lower quadrant
        
        Args:
            entry_type: Type of entry
            
        Returns:
            Entry price
        """
        if entry_type == "open":
            return self.open
        elif entry_type == "mt":
            return self.mean_threshold
        elif entry_type == "aggressive":
            if self.block_type == OrderBlockType.BULLISH:
                return self.lower_quadrant  # Deeper into OB
            else:
                return self.upper_quadrant
        elif entry_type == "quadrant":
            if self.block_type == OrderBlockType.BULLISH:
                return self.upper_quadrant  # Upper half for buys
            else:
                return self.lower_quadrant  # Lower half for sells
        else:
            return self.open
    
    def get_stop_loss(self, buffer_pips: float = 0.0001) -> float:
        """
        Get stop loss based on ICT methodology
        
        ICT: "Stop below the close (bullish) or above the close (bearish)"
        Can also use the low/high with buffer
        
        Args:
            buffer_pips: Additional buffer beyond the level
            
        Returns:
            Stop loss price
        """
        if self.block_type == OrderBlockType.BULLISH:
            # Below the low of OB (or close if more conservative)
            return self.low - buffer_pips
        else:
            # Above the high of OB
            return self.high + buffer_pips
    
    def is_price_in_optimal_entry_zone(self, price: float) -> bool:
        """
        Check if price is in optimal entry zone
        
        ICT: "Best setups form at or above mean threshold (bullish)"
        For bullish: Upper half of OB body
        For bearish: Lower half of OB body
        """
        if self.block_type == OrderBlockType.BULLISH:
            # For buys: price should be in upper half (mean threshold to high)
            return self.mean_threshold <= price <= self.body_high
        else:
            # For sells: price should be in lower half (low to mean threshold)
            return self.body_low <= price <= self.mean_threshold
    
    def should_invalidate(self, price: float) -> bool:
        """
        Check if OB should be invalidated based on price action
        
        ICT: "If it trades through mean threshold with bodies, it's weak"
        """
        if self.block_type == OrderBlockType.BULLISH:
            # Invalidate if price closes below the OB low
            return price < self.low
        else:
            # Invalidate if price closes above the OB high
            return price > self.high


@dataclass
class BreakerBlock:
    """
    ICT Breaker Block
    
    A failed order block that now acts as a reversal zone.
    Bullish OB that fails becomes Bearish Breaker.
    Bearish OB that fails becomes Bullish Breaker.
    """
    original_ob: OrderBlock
    breaker_type: str                   # 'bullish_breaker' or 'bearish_breaker'
    break_index: int                    # When it broke
    break_price: float                  # Price that broke it
    
    # Key levels (inverted from original)
    entry_zone_high: float
    entry_zone_low: float
    mean_threshold: float
    
    is_tested: bool = False
    times_tested: int = 0
    is_respected: bool = False
    
    def __str__(self):
        return f"{self.breaker_type.upper()} @ {self.mean_threshold:.5f}"


@dataclass
class ReclaimedOrderBlock:
    """
    ICT Reclaimed Order Block
    
    From Market Maker Buy/Sell Models:
    - Bullish Reclaimed: Down candle on buy side of curve, used again for longs
    - Bearish Reclaimed: Up candle on sell side of curve, used again for shorts
    
    ICT: "These old blocks will be reclaimed for new entries"
    """
    original_ob: OrderBlock
    reclaim_index: int
    reclaim_count: int = 1              # Times it's been reclaimed
    last_reclaim_price: float = 0.0
    mm_model_context: str = ""          # 'mm_buy' or 'mm_sell'
    
    def __str__(self):
        return f"RECLAIMED {self.original_ob.block_type.value.upper()} OB (x{self.reclaim_count})"


@dataclass
class OrderBlockAnalysis:
    """Complete order block analysis results"""
    all_order_blocks: List[OrderBlock]
    bullish_obs: List[OrderBlock]
    bearish_obs: List[OrderBlock]
    active_obs: List[OrderBlock]
    extreme_obs: List[OrderBlock]
    breaker_blocks: List[BreakerBlock]
    reclaimed_obs: List[ReclaimedOrderBlock]
    propulsion_obs: List[OrderBlock]
    
    # Best candidates
    best_bullish_ob: Optional[OrderBlock] = None
    best_bearish_ob: Optional[OrderBlock] = None
    
    # Statistics
    total_count: int = 0
    bullish_count: int = 0
    bearish_count: int = 0
    active_count: int = 0
    respect_rate: float = 0.0
    
    def __str__(self):
        return (f"OB Analysis: {self.total_count} total "
                f"({self.bullish_count} bullish, {self.bearish_count} bearish) | "
                f"{self.active_count} active | "
                f"Respect rate: {self.respect_rate:.1f}%")


# ==================== MAIN HANDLER ====================

class OrderBlockHandler:
    """
    Comprehensive ICT Order Block Handler
    
    Implements all ICT Order Block concepts:
    - Single and blended (consecutive) order block detection
    - Mean Threshold and Quadrant levels
    - Entry at Opening Price (Change in State of Delivery)
    - Body projection analysis for best entry candle
    - Breaker block detection
    - Reclaimed order block tracking
    - Propulsion block identification
    - Mitigation tracking
    - Market Maker Model context
    - Validation requirements (2-3x range movement)
    - Dealing range quadrant filtering
    """
    
    def __init__(self,
                 displacement_threshold: float = 0.0005,
                 extreme_displacement_multiplier: float = 3.0,
                 min_body_size: float = 0.0001,
                 max_consecutive_candles: int = 5,
                 validation_multiplier: float = 2.5,
                 track_body_respect: bool = True):
        """
        Initialize Order Block Handler
        
        Args:
            displacement_threshold: Minimum displacement size
            extreme_displacement_multiplier: Multiplier for extreme OB
            min_body_size: Minimum candle body size
            max_consecutive_candles: Max candles to blend
            validation_multiplier: Required move away (2-3x OB range)
            track_body_respect: Track body vs wick interactions
        """
        self.displacement_threshold = displacement_threshold
        self.extreme_displacement_multiplier = extreme_displacement_multiplier
        self.min_body_size = min_body_size
        self.max_consecutive_candles = max_consecutive_candles
        self.validation_multiplier = validation_multiplier
        self.track_body_respect = track_body_respect
        
        # Storage
        self.order_blocks: List[OrderBlock] = []
        self.breaker_blocks: List[BreakerBlock] = []
        self.reclaimed_obs: List[ReclaimedOrderBlock] = []
        
        # Dealing range context
        self.dealing_range_high: Optional[float] = None
        self.dealing_range_low: Optional[float] = None
    
    # ==================== CORE DETECTION ====================
    
    def detect_order_blocks(self, df: pd.DataFrame) -> List[OrderBlock]:
        """
        Detect all order blocks in the price data
        
        ICT Definition:
        - Bullish OB: Last DOWN-CLOSE candle before BULLISH displacement
        - Bearish OB: Last UP-CLOSE candle before BEARISH displacement
        
        Also handles consecutive candle blending.
        
        Args:
            df: DataFrame with OHLC data
            
        Returns:
            List of detected OrderBlock objects
        """
        order_blocks = []
        
        # Set dealing range if not set
        if self.dealing_range_high is None:
            self.dealing_range_high = df['high'].max()
            self.dealing_range_low = df['low'].min()
        
        i = 1
        while i < len(df) - 1:
            current = df.iloc[i]
            next_candle = df.iloc[i + 1]
            
            # Get timestamp
            timestamp = self._get_timestamp(df, i)
            
            # Check candle direction
            is_down_close = current['close'] < current['open']
            is_up_close = current['close'] > current['open']
            is_next_bullish = next_candle['close'] > next_candle['open']
            is_next_bearish = next_candle['close'] < next_candle['open']
            
            # Calculate displacement
            displacement = abs(next_candle['close'] - next_candle['open'])
            
            # BULLISH Order Block Detection
            if is_down_close and is_next_bullish:
                # Check for displacement (next candle closes above current high)
                if next_candle['close'] > current['high'] and displacement >= self.displacement_threshold:
                    
                    # Check for consecutive down-close candles
                    blended_range = self._check_consecutive_candles(
                        df, i, 'down', OrderBlockType.BULLISH
                    )
                    
                    if blended_range:
                        ob = self._create_blended_order_block(
                            df, blended_range, OrderBlockType.BULLISH,
                            displacement, timestamp
                        )
                    else:
                        ob = self._create_single_order_block(
                            df, i, current, OrderBlockType.BULLISH,
                            displacement, timestamp
                        )
                    
                    order_blocks.append(ob)
            
            # BEARISH Order Block Detection
            elif is_up_close and is_next_bearish:
                # Check for displacement (next candle closes below current low)
                if next_candle['close'] < current['low'] and displacement >= self.displacement_threshold:
                    
                    # Check for consecutive up-close candles
                    blended_range = self._check_consecutive_candles(
                        df, i, 'up', OrderBlockType.BEARISH
                    )
                    
                    if blended_range:
                        ob = self._create_blended_order_block(
                            df, blended_range, OrderBlockType.BEARISH,
                            displacement, timestamp
                        )
                    else:
                        ob = self._create_single_order_block(
                            df, i, current, OrderBlockType.BEARISH,
                            displacement, timestamp
                        )
                    
                    order_blocks.append(ob)
            
            i += 1
        
        # Update status for all OBs
        self._update_order_block_status(df, order_blocks)
        
        # Identify special types
        self._identify_extreme_order_blocks(order_blocks)
        self._identify_propulsion_blocks(df, order_blocks)
        
        # Check for breakers from failed OBs
        self._detect_breaker_blocks(df, order_blocks)
        
        # Set dealing range context
        self._set_dealing_range_context(order_blocks)
        
        self.order_blocks = order_blocks
        return order_blocks
    
    def _get_timestamp(self, df: pd.DataFrame, index: int) -> Optional[datetime]:
        """Extract timestamp from dataframe"""
        if 'timestamp' in df.columns:
            return df.iloc[index]['timestamp']
        elif isinstance(df.index[index], (datetime, pd.Timestamp)):
            return df.index[index]
        return None
    
    def _check_consecutive_candles(self, df: pd.DataFrame, 
                                    current_idx: int,
                                    direction: str,
                                    ob_type: OrderBlockType) -> Optional[ConsecutiveCandleRange]:
        """
        Check for consecutive same-direction candles to blend
        
        ICT: "Two consecutive down-close candles - that's all one range.
             You have to blend them together to get one full order block."
        
        Args:
            df: DataFrame
            current_idx: Current candle index
            direction: 'up' or 'down'
            ob_type: Type of order block
            
        Returns:
            ConsecutiveCandleRange if multiple candles found, else None
        """
        consecutive_indices = [current_idx]
        
        # Look back for consecutive same-direction candles
        for lookback in range(1, self.max_consecutive_candles):
            prev_idx = current_idx - lookback
            if prev_idx < 0:
                break
            
            prev_candle = df.iloc[prev_idx]
            is_same_direction = (
                (direction == 'down' and prev_candle['close'] < prev_candle['open']) or
                (direction == 'up' and prev_candle['close'] > prev_candle['open'])
            )
            
            if is_same_direction:
                consecutive_indices.insert(0, prev_idx)
            else:
                break
        
        # Only create blended range if more than 1 candle
        if len(consecutive_indices) < 2:
            return None
        
        # Build the blended range
        candles = [df.iloc[idx] for idx in consecutive_indices]
        
        combined_high = max(c['high'] for c in candles)
        combined_low = min(c['low'] for c in candles)
        combined_open = candles[0]['open']  # First candle's open
        combined_close = candles[-1]['close']  # Last candle's close
        
        # Body range (ICT uses bodies, not wicks)
        if ob_type == OrderBlockType.BULLISH:
            # For bullish OB (down-close candles): body high is open, body low is close
            body_high = max(c['open'] for c in candles)  # Highest open
            body_low = min(c['close'] for c in candles)  # Lowest close
            
            # Find candle with lowest body projection (lowest close)
            best_idx = min(range(len(candles)), key=lambda x: candles[x]['close'])
            best_body_projection = candles[best_idx]['close']
        else:
            # For bearish OB (up-close candles): body high is close, body low is open
            body_high = max(c['close'] for c in candles)  # Highest close
            body_low = min(c['open'] for c in candles)  # Lowest open
            
            # Find candle with highest body projection (highest close)
            best_idx = max(range(len(candles)), key=lambda x: candles[x]['close'])
            best_body_projection = candles[best_idx]['close']
        
        body_range = body_high - body_low
        mean_threshold = body_low + (body_range * 0.5)
        
        return ConsecutiveCandleRange(
            start_index=consecutive_indices[0],
            end_index=consecutive_indices[-1],
            num_candles=len(consecutive_indices),
            combined_open=combined_open,
            combined_high=combined_high,
            combined_low=combined_low,
            combined_close=combined_close,
            body_range_high=body_high,
            body_range_low=body_low,
            mean_threshold=mean_threshold,
            upper_quadrant=body_low + (body_range * 0.75),
            lower_quadrant=body_low + (body_range * 0.25),
            best_body_candle_index=consecutive_indices[best_idx],
            best_body_projection=best_body_projection
        )
    
    def _create_single_order_block(self, df: pd.DataFrame,
                                    index: int,
                                    candle: pd.Series,
                                    ob_type: OrderBlockType,
                                    displacement: float,
                                    timestamp: Optional[datetime]) -> OrderBlock:
        """Create order block from single candle"""
        
        # Calculate body levels
        body_high = max(candle['open'], candle['close'])
        body_low = min(candle['open'], candle['close'])
        body_range = body_high - body_low
        
        # ICT: Mean Threshold is 50% of BODY (not wick)
        mean_threshold = body_low + (body_range * 0.5)
        
        # Displacement ratio
        displacement_ratio = displacement / body_range if body_range > 0 else 0
        
        return OrderBlock(
            index=index,
            timestamp=timestamp,
            block_type=ob_type,
            open=candle['open'],
            high=candle['high'],
            low=candle['low'],
            close=candle['close'],
            mean_threshold=mean_threshold,
            upper_quadrant=body_low + (body_range * 0.75),
            lower_quadrant=body_low + (body_range * 0.25),
            body_high=body_high,
            body_low=body_low,
            body_range=body_range,
            is_blended=False,
            num_candles_in_block=1,
            displacement_size=displacement,
            displacement_ratio=displacement_ratio,
            validated_by_displacement=True
        )
    
    def _create_blended_order_block(self, df: pd.DataFrame,
                                     blended_range: ConsecutiveCandleRange,
                                     ob_type: OrderBlockType,
                                     displacement: float,
                                     timestamp: Optional[datetime]) -> OrderBlock:
        """
        Create order block from blended consecutive candles
        
        ICT: "When you have a range of candlesticks, you grade it.
             Divide it in half, divide it in quarters."
        """
        body_range = blended_range.body_range_high - blended_range.body_range_low
        displacement_ratio = displacement / body_range if body_range > 0 else 0
        
        # Use the best body projection candle's data for entry
        best_candle = df.iloc[blended_range.best_body_candle_index]
        
        ob = OrderBlock(
            index=blended_range.best_body_candle_index,  # Use best candle index
            timestamp=timestamp,
            block_type=ob_type,
            open=best_candle['open'],  # Entry at best candle's open
            high=blended_range.combined_high,
            low=blended_range.combined_low,
            close=best_candle['close'],
            mean_threshold=blended_range.mean_threshold,
            upper_quadrant=blended_range.upper_quadrant,
            lower_quadrant=blended_range.lower_quadrant,
            body_high=blended_range.body_range_high,
            body_low=blended_range.body_range_low,
            body_range=body_range,
            is_blended=True,
            blended_range=blended_range,
            num_candles_in_block=blended_range.num_candles,
            displacement_size=displacement,
            displacement_ratio=displacement_ratio,
            validated_by_displacement=True
        )
        
        ob.notes.append(f"Blended from {blended_range.num_candles} consecutive candles")
        ob.notes.append(f"Best body projection at index {blended_range.best_body_candle_index}")
        
        return ob
    
    # ==================== STATUS TRACKING ====================
    
    def _update_order_block_status(self, df: pd.DataFrame, order_blocks: List[OrderBlock]):
        """
        Update status for all order blocks
        
        ICT Rules:
        - Best OBs don't trade below mean threshold (bullish)
        - Body closes matter more than wicks
        - Track respect vs violation
        """
        for ob in order_blocks:
            if ob.index >= len(df) - 1:
                continue
            
            future_candles = df.iloc[ob.index + 1:]
            
            for idx in range(len(future_candles)):
                candle = future_candles.iloc[idx]
                actual_idx = ob.index + 1 + idx
                
                body_low = min(candle['open'], candle['close'])
                body_high = max(candle['open'], candle['close'])
                
                if ob.block_type == OrderBlockType.BULLISH:
                    self._check_bullish_ob_status(ob, candle, body_low, body_high, actual_idx)
                else:
                    self._check_bearish_ob_status(ob, candle, body_low, body_high, actual_idx)
                
                # Track validation (2-3x range movement away)
                self._check_validation(ob, candle)
    
    def _check_bullish_ob_status(self, ob: OrderBlock, candle: pd.Series,
                                  body_low: float, body_high: float, actual_idx: int):
        """
        Check bullish OB status
        
        ICT Rules:
        - Best OBs: Price stays above mean threshold
        - Can "stab through" with wick but body should respect
        - Body below close = failed
        """
        # Check if price entered the OB zone
        if candle['low'] <= ob.body_high:
            ob.times_tested += 1
            
            # Wick touched but body respected mean threshold
            if candle['low'] <= ob.body_high and body_low >= ob.mean_threshold:
                ob.times_respected += 1
                ob.wick_only_tests += 1
                ob.body_respected += 1
                if ob.status == OrderBlockStatus.UNTESTED:
                    ob.status = OrderBlockStatus.TESTED
                ob.notes.append(f"Respected at index {actual_idx} - wick test, body held")
            
            # Body entered but stayed above mean threshold
            elif body_low < ob.body_high and body_low >= ob.mean_threshold:
                ob.times_respected += 1
                ob.body_respected += 1
                if ob.status == OrderBlockStatus.UNTESTED:
                    ob.status = OrderBlockStatus.TESTED
            
            # Body went below mean threshold (weak, but not failed yet)
            elif body_low < ob.mean_threshold and body_low >= ob.body_low:
                ob.body_violated += 1
                ob.status = OrderBlockStatus.MITIGATED
                ob.is_mitigated = True
                ob.notes.append(f"Mitigated at index {actual_idx} - body below MT")
            
            # Body went below OB low = FAILED
            elif body_low < ob.body_low:
                ob.times_failed += 1
                ob.status = OrderBlockStatus.FAILED
                ob.notes.append(f"Failed at index {actual_idx}")
                return  # Stop tracking
        
        # Check for reclaim (tested, left, came back)
        if ob.times_respected >= 2 and not ob.is_reclaimed:
            ob.is_reclaimed = True
            ob.status = OrderBlockStatus.RECLAIMED
            ob.notes.append("Reclaimed Order Block - multiple respects")
    
    def _check_bearish_ob_status(self, ob: OrderBlock, candle: pd.Series,
                                  body_low: float, body_high: float, actual_idx: int):
        """Check bearish OB status (inverse of bullish logic)"""
        
        if candle['high'] >= ob.body_low:
            ob.times_tested += 1
            
            # Wick touched but body respected
            if candle['high'] >= ob.body_low and body_high <= ob.mean_threshold:
                ob.times_respected += 1
                ob.wick_only_tests += 1
                ob.body_respected += 1
                if ob.status == OrderBlockStatus.UNTESTED:
                    ob.status = OrderBlockStatus.TESTED
                ob.notes.append(f"Respected at index {actual_idx}")
            
            elif body_high > ob.body_low and body_high <= ob.mean_threshold:
                ob.times_respected += 1
                ob.body_respected += 1
                if ob.status == OrderBlockStatus.UNTESTED:
                    ob.status = OrderBlockStatus.TESTED
            
            elif body_high > ob.mean_threshold and body_high <= ob.body_high:
                ob.body_violated += 1
                ob.status = OrderBlockStatus.MITIGATED
                ob.is_mitigated = True
            
            elif body_high > ob.body_high:
                ob.times_failed += 1
                ob.status = OrderBlockStatus.FAILED
                return
        
        if ob.times_respected >= 2 and not ob.is_reclaimed:
            ob.is_reclaimed = True
            ob.status = OrderBlockStatus.RECLAIMED
    
    def _check_validation(self, ob: OrderBlock, candle: pd.Series):
        """
        Check if OB has been validated by sufficient movement
        
        ICT: "I want to see at least 2-3 times the OB range as a rally away"
        """
        if ob.validation_target_reached:
            return
        
        target_distance = ob.body_range * self.validation_multiplier
        
        if ob.block_type == OrderBlockType.BULLISH:
            if candle['high'] >= ob.body_high + target_distance:
                ob.validation_target_reached = True
                ob.notes.append(f"Validated - price moved {self.validation_multiplier}x OB range")
        else:
            if candle['low'] <= ob.body_low - target_distance:
                ob.validation_target_reached = True
                ob.notes.append(f"Validated - price moved {self.validation_multiplier}x OB range")
    
    # ==================== SPECIAL TYPES ====================
    
    def _identify_extreme_order_blocks(self, order_blocks: List[OrderBlock]):
        """
        Identify ICT Extreme Order Blocks
        
        Extreme OB: Massive displacement that creates significant
        imbalance in the market
        """
        if not order_blocks:
            return
        
        # Calculate average displacement
        avg_displacement = np.mean([ob.displacement_size for ob in order_blocks])
        
        for ob in order_blocks:
            if ob.displacement_size >= avg_displacement * self.extreme_displacement_multiplier:
                ob.is_extreme_ob = True
                ob.strength = OrderBlockStrength.EXTREME
                ob.notes.append("EXTREME ORDER BLOCK - Massive displacement")
            elif ob.displacement_ratio >= 2.0:
                ob.strength = OrderBlockStrength.STRONG
            elif ob.displacement_ratio >= 1.0:
                ob.strength = OrderBlockStrength.MEDIUM
            else:
                ob.strength = OrderBlockStrength.WEAK
    
    def _identify_propulsion_blocks(self, df: pd.DataFrame, order_blocks: List[OrderBlock]):
        """
        Identify Propulsion Blocks
        
        Propulsion: OB that launches price with strong momentum after respect
        Price doesn't just bounce - it PROPELS
        """
        for ob in order_blocks:
            if ob.status != OrderBlockStatus.TESTED or ob.times_respected == 0:
                continue
            
            # Look at movement after test
            test_idx = ob.index + ob.times_tested + 5  # Approximate
            if test_idx >= len(df):
                continue
            
            post_test = df.iloc[ob.index + 1:min(test_idx + 10, len(df))]
            if len(post_test) < 3:
                continue
            
            if ob.block_type == OrderBlockType.BULLISH:
                max_move = post_test['high'].max() - ob.body_high
                if max_move >= ob.body_range * 3:  # 3x range propulsion
                    ob.is_propulsion = True
                    ob.status = OrderBlockStatus.PROPULSION
                    ob.notes.append("PROPULSION BLOCK - Strong momentum after respect")
            else:
                max_move = ob.body_low - post_test['low'].min()
                if max_move >= ob.body_range * 3:
                    ob.is_propulsion = True
                    ob.status = OrderBlockStatus.PROPULSION
                    ob.notes.append("PROPULSION BLOCK - Strong momentum after respect")
    
    # ==================== BREAKER BLOCKS ====================
    
    def _detect_breaker_blocks(self, df: pd.DataFrame, order_blocks: List[OrderBlock]):
        """
        Detect Breaker Blocks from failed order blocks
        
        ICT: Failed OB becomes reversal zone in opposite direction
        - Failed Bullish OB = Bearish Breaker
        - Failed Bearish OB = Bullish Breaker
        """
        self.breaker_blocks = []
        
        for ob in order_blocks:
            if ob.status != OrderBlockStatus.FAILED:
                continue
            
            # Find break point
            break_idx = self._find_break_index(df, ob)
            if break_idx is None:
                continue
            
            # Check if price returned and respected as breaker
            if self._check_breaker_respect(df, ob, break_idx):
                breaker_type = (
                    "bullish_breaker" if ob.block_type == OrderBlockType.BEARISH
                    else "bearish_breaker"
                )
                
                breaker = BreakerBlock(
                    original_ob=ob,
                    breaker_type=breaker_type,
                    break_index=break_idx,
                    break_price=df.iloc[break_idx]['close'],
                    entry_zone_high=ob.body_high,
                    entry_zone_low=ob.body_low,
                    mean_threshold=ob.mean_threshold
                )
                
                ob.is_breaker = True
                ob.status = OrderBlockStatus.BREAKER
                self.breaker_blocks.append(breaker)
    
    def _find_break_index(self, df: pd.DataFrame, ob: OrderBlock) -> Optional[int]:
        """Find the candle index where OB was broken"""
        for i in range(ob.index + 1, len(df)):
            candle = df.iloc[i]
            body_low = min(candle['open'], candle['close'])
            body_high = max(candle['open'], candle['close'])
            
            if ob.block_type == OrderBlockType.BULLISH:
                if body_low < ob.low:
                    return i
            else:
                if body_high > ob.high:
                    return i
        return None
    
    def _check_breaker_respect(self, df: pd.DataFrame, ob: OrderBlock, 
                                break_idx: int) -> bool:
        """Check if price returned and respected the broken OB as breaker"""
        if break_idx + 3 >= len(df):
            return False
        
        future_candles = df.iloc[break_idx + 1:min(break_idx + 20, len(df))]
        
        for _, candle in future_candles.iterrows():
            if ob.block_type == OrderBlockType.BULLISH:
                # Failed bullish OB becomes resistance (bearish breaker)
                if candle['high'] >= ob.body_low and candle['close'] < candle['open']:
                    return True
            else:
                # Failed bearish OB becomes support (bullish breaker)
                if candle['low'] <= ob.body_high and candle['close'] > candle['open']:
                    return True
        
        return False
    
    # ==================== DEALING RANGE CONTEXT ====================
    
    def _set_dealing_range_context(self, order_blocks: List[OrderBlock]):
        """
        Set dealing range context for each OB
        
        ICT: "Order block must be at or below the quadrant level"
        """
        if self.dealing_range_high is None or self.dealing_range_low is None:
            return
        
        dr_range = self.dealing_range_high - self.dealing_range_low
        dr_lower_quadrant = self.dealing_range_low + (dr_range * 0.25)
        dr_upper_quadrant = self.dealing_range_low + (dr_range * 0.75)
        
        for ob in order_blocks:
            # Calculate position in dealing range
            ob.position_in_dealing_range = (
                (ob.mean_threshold - self.dealing_range_low) / dr_range * 100
            )
            
            # Check if at or below appropriate quadrant
            if ob.block_type == OrderBlockType.BULLISH:
                # Bullish OB should be in discount (below 50%) ideally
                ob.at_or_below_quadrant = ob.mean_threshold <= dr_lower_quadrant
                ob.dealing_range_quadrant = dr_lower_quadrant
            else:
                # Bearish OB should be in premium (above 50%) ideally
                ob.at_or_below_quadrant = ob.mean_threshold >= dr_upper_quadrant
                ob.dealing_range_quadrant = dr_upper_quadrant
            
            if ob.at_or_below_quadrant:
                ob.notes.append(f"At/below dealing range quadrant - HIGH PROBABILITY")
    
    def set_dealing_range(self, high: float, low: float):
        """
        Manually set dealing range
        
        Args:
            high: Dealing range high
            low: Dealing range low
        """
        self.dealing_range_high = high
        self.dealing_range_low = low
    
    # ==================== ANALYSIS ====================
    
    def analyze_order_blocks(self, df: pd.DataFrame) -> OrderBlockAnalysis:
        """
        Complete order block analysis
        
        Returns:
            OrderBlockAnalysis with all categorizations
        """
        # Detect all OBs
        all_obs = self.detect_order_blocks(df)
        
        # Categorize
        bullish_obs = [ob for ob in all_obs if ob.block_type == OrderBlockType.BULLISH]
        bearish_obs = [ob for ob in all_obs if ob.block_type == OrderBlockType.BEARISH]
        active_obs = [ob for ob in all_obs if ob.status in [
            OrderBlockStatus.UNTESTED, OrderBlockStatus.TESTED,
            OrderBlockStatus.RECLAIMED, OrderBlockStatus.PROPULSION
        ]]
        extreme_obs = [ob for ob in all_obs if ob.is_extreme_ob]
        propulsion_obs = [ob for ob in all_obs if ob.is_propulsion]
        
        # Find best candidates
        best_bullish = self._find_best_ob(bullish_obs, OrderBlockType.BULLISH)
        best_bearish = self._find_best_ob(bearish_obs, OrderBlockType.BEARISH)
        
        # Calculate statistics
        total = len(all_obs)
        tested = [ob for ob in all_obs if ob.times_tested > 0]
        respected = [ob for ob in tested if ob.times_respected > 0]
        respect_rate = (len(respected) / len(tested) * 100) if tested else 0
        
        # Reclaimed OBs
        reclaimed_list = [
            ReclaimedOrderBlock(
                original_ob=ob,
                reclaim_index=ob.index,
                reclaim_count=ob.times_respected
            )
            for ob in all_obs if ob.is_reclaimed
        ]
        
        return OrderBlockAnalysis(
            all_order_blocks=all_obs,
            bullish_obs=bullish_obs,
            bearish_obs=bearish_obs,
            active_obs=active_obs,
            extreme_obs=extreme_obs,
            breaker_blocks=self.breaker_blocks,
            reclaimed_obs=reclaimed_list,
            propulsion_obs=propulsion_obs,
            best_bullish_ob=best_bullish,
            best_bearish_ob=best_bearish,
            total_count=total,
            bullish_count=len(bullish_obs),
            bearish_count=len(bearish_obs),
            active_count=len(active_obs),
            respect_rate=respect_rate
        )
    
    def _find_best_ob(self, obs: List[OrderBlock], 
                      ob_type: OrderBlockType) -> Optional[OrderBlock]:
        """
        Find best order block for trading
        
        ICT Priority:
        1. At or below dealing range quadrant
        2. Reclaimed/Propulsion status
        3. Extreme OB
        4. Body respect ratio
        5. Validation confirmed
        """
        active = [ob for ob in obs if ob.status in [
            OrderBlockStatus.UNTESTED, OrderBlockStatus.TESTED,
            OrderBlockStatus.RECLAIMED, OrderBlockStatus.PROPULSION
        ]]
        
        if not active:
            return None
        
        scored_obs = []
        for ob in active:
            score = 0
            
            # Highest priority: At dealing range quadrant
            if ob.at_or_below_quadrant:
                score += 30
            
            # Reclaimed (multiple respects)
            if ob.is_reclaimed:
                score += 25
            
            # Propulsion
            if ob.is_propulsion:
                score += 20
            
            # Extreme OB
            if ob.is_extreme_ob:
                score += 20
            
            # Strength
            score += ob.strength.value * 5
            
            # Body respect ratio
            total_body_events = ob.body_respected + ob.body_violated
            if total_body_events > 0:
                respect_ratio = ob.body_respected / total_body_events
                score += int(respect_ratio * 15)
            
            # Validation
            if ob.validation_target_reached:
                score += 10
            
            # Wick-only tests (strong respect signal)
            score += min(ob.wick_only_tests * 5, 15)
            
            # Penalty for mitigation
            if ob.is_mitigated:
                score -= 10
            
            # Blended OBs can be more reliable
            if ob.is_blended:
                score += 5
            
            scored_obs.append((score, ob))
        
        scored_obs.sort(key=lambda x: x[0], reverse=True)
        return scored_obs[0][1] if scored_obs else None
    
    # ==================== TRADE SIGNALS ====================
    
    def get_trade_signal(self, df: pd.DataFrame, current_price: float,
                         bias: str = 'neutral') -> Dict:
        """
        Generate trade signal from order block analysis
        
        Args:
            df: OHLC DataFrame
            current_price: Current market price
            bias: Market bias ('bullish', 'bearish', 'neutral')
            
        Returns:
            Trade signal dictionary
        """
        analysis = self.analyze_order_blocks(df)
        
        signal = {
            'type': 'NO_SIGNAL',
            'entry': None,
            'stop_loss': None,
            'take_profit': None,
            'confidence': 0,
            'order_block': None,
            'reasoning': []
        }
        
        best_ob = None
        
        # Find OB near current price
        if bias in ['bullish', 'neutral'] and analysis.best_bullish_ob:
            ob = analysis.best_bullish_ob
            # Check if price is approaching OB from above
            if ob.body_high <= current_price <= ob.body_high + (ob.body_range * 2):
                best_ob = ob
                signal['type'] = 'BUY'
        
        if bias in ['bearish', 'neutral'] and analysis.best_bearish_ob:
            ob = analysis.best_bearish_ob
            if ob.body_low >= current_price >= ob.body_low - (ob.body_range * 2):
                if best_ob is None:
                    best_ob = ob
                    signal['type'] = 'SELL'
        
        if best_ob:
            # Entry at opening price (ICT rule)
            signal['entry'] = best_ob.get_entry_price("open")
            signal['stop_loss'] = best_ob.get_stop_loss()
            
            # Calculate take profit (2:1 R:R minimum)
            risk = abs(signal['entry'] - signal['stop_loss'])
            if signal['type'] == 'BUY':
                signal['take_profit'] = signal['entry'] + (risk * 2)
            else:
                signal['take_profit'] = signal['entry'] - (risk * 2)
            
            signal['confidence'] = self._calculate_confidence(best_ob)
            signal['order_block'] = best_ob
            
            # Build reasoning
            signal['reasoning'].append(f"{best_ob.block_type.value.upper()} OB at {best_ob.open:.5f}")
            signal['reasoning'].append(f"Mean Threshold: {best_ob.mean_threshold:.5f}")
            
            if best_ob.is_extreme_ob:
                signal['reasoning'].append("EXTREME ORDER BLOCK")
            if best_ob.is_reclaimed:
                signal['reasoning'].append("RECLAIMED - Multiple respects")
            if best_ob.is_propulsion:
                signal['reasoning'].append("PROPULSION BLOCK")
            if best_ob.at_or_below_quadrant:
                signal['reasoning'].append("At dealing range quadrant")
            if best_ob.is_blended:
                signal['reasoning'].append(f"Blended from {best_ob.num_candles_in_block} candles")
            signal['reasoning'].append(f"Body respected: {best_ob.body_respected}x")
        
        return signal
    
    def _calculate_confidence(self, ob: OrderBlock) -> int:
        """Calculate confidence score (0-100)"""
        confidence = 50
        
        if ob.at_or_below_quadrant:
            confidence += 20
        if ob.is_extreme_ob:
            confidence += 15
        if ob.is_reclaimed:
            confidence += 15
        if ob.is_propulsion:
            confidence += 10
        
        confidence += min(ob.body_respected * 3, 15)
        confidence += ob.strength.value * 3
        
        if ob.validation_target_reached:
            confidence += 5
        if ob.is_mitigated:
            confidence -= 10
        
        return max(0, min(100, confidence))
    
    # ==================== UTILITIES ====================
    
    def get_active_order_blocks(self, ob_type: Optional[OrderBlockType] = None) -> List[OrderBlock]:
        """Get all active order blocks"""
        active = [ob for ob in self.order_blocks if ob.status in [
            OrderBlockStatus.UNTESTED, OrderBlockStatus.TESTED,
            OrderBlockStatus.RECLAIMED, OrderBlockStatus.PROPULSION
        ]]
        if ob_type:
            active = [ob for ob in active if ob.block_type == ob_type]
        return active
    
    def get_breakers(self) -> List[BreakerBlock]:
        """Get all breaker blocks"""
        return self.breaker_blocks
    
    def get_nearest_ob(self, price: float, 
                       ob_type: Optional[OrderBlockType] = None) -> Optional[OrderBlock]:
        """Get nearest OB to price"""
        obs = self.get_active_order_blocks(ob_type)
        if not obs:
            return None
        
        distances = [(abs(ob.mean_threshold - price), ob) for ob in obs]
        distances.sort(key=lambda x: x[0])
        return distances[0][1]
    
    def get_summary(self) -> str:
        """Get text summary of all detected order blocks"""
        if not self.order_blocks:
            return "No Order Blocks detected"
        
        lines = [
            "",
            "=" * 80,
            "ICT ORDER BLOCK ANALYSIS",
            "=" * 80,
            f"\nTotal Order Blocks: {len(self.order_blocks)}",
        ]
        
        bullish = [ob for ob in self.order_blocks if ob.block_type == OrderBlockType.BULLISH]
        bearish = [ob for ob in self.order_blocks if ob.block_type == OrderBlockType.BEARISH]
        extreme = [ob for ob in self.order_blocks if ob.is_extreme_ob]
        reclaimed = [ob for ob in self.order_blocks if ob.is_reclaimed]
        propulsion = [ob for ob in self.order_blocks if ob.is_propulsion]
        
        lines.extend([
            f"Bullish OBs: {len(bullish)}",
            f"Bearish OBs: {len(bearish)}",
            f"Extreme OBs: {len(extreme)}",
            f"Reclaimed OBs: {len(reclaimed)}",
            f"Propulsion OBs: {len(propulsion)}",
            f"Breaker Blocks: {len(self.breaker_blocks)}",
            "",
            "-" * 40,
            "ACTIVE ORDER BLOCKS:",
            "-" * 40,
        ])
        
        active = self.get_active_order_blocks()
        for i, ob in enumerate(active[-10:], 1):
            tags = []
            if ob.is_extreme_ob:
                tags.append("EXTREME")
            if ob.is_reclaimed:
                tags.append("RECLAIMED")
            if ob.is_propulsion:
                tags.append("PROPULSION")
            if ob.at_or_below_quadrant:
                tags.append("AT QUADRANT")
            if ob.is_blended:
                tags.append(f"BLENDED x{ob.num_candles_in_block}")
            
            tag_str = f" [{', '.join(tags)}]" if tags else ""
            lines.append(f"{i}. {ob}{tag_str}")
            lines.append(f"   Body Range: {ob.body_low:.5f} - {ob.body_high:.5f}")
            lines.append(f"   Respected: {ob.body_respected}x | Violated: {ob.body_violated}x")
        
        if self.breaker_blocks:
            lines.extend([
                "",
                "-" * 40,
                "BREAKER BLOCKS:",
                "-" * 40,
            ])
            for breaker in self.breaker_blocks[-5:]:
                lines.append(f"  {breaker}")
        
        lines.append("=" * 80 + "\n")
        return "\n".join(lines)
    
    def reset(self):
        """Reset all data"""
        self.order_blocks = []
        self.breaker_blocks = []
        self.reclaimed_obs = []
        self.dealing_range_high = None
        self.dealing_range_low = None


# ==================== EXAMPLE USAGE ====================

if __name__ == "__main__":
    print("ICT Order Block Handler")
    print("=" * 50)
    print("\nKey ICT Concepts Implemented:")
    print("  • Order Block = Last opposite candle before displacement")
    print("  • 'Orders the change in state of delivery' (NOT about orders)")
    print("  • Mean Threshold = 50% of body (NOT CE)")
    print("  • Entry at Opening Price")
    print("  • Consecutive Candle Blending")
    print("  • Body Projection Analysis")
    print("  • Reclaimed Order Blocks (Market Maker Models)")
    print("  • Breaker Blocks (Failed OBs as reversal)")
    print("  • Propulsion Blocks (Strong momentum)")
    print("  • Validation (2-3x range movement)")
    print("  • Dealing Range Quadrant Filtering")
    print("\nUsage:")
    print("  handler = OrderBlockHandler()")
    print("  analysis = handler.analyze_order_blocks(df)")
    print("  signal = handler.get_trade_signal(df, current_price)")
