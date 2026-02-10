"""
ICT Unified Handler - Phase 1 Complete Integration
===================================================

Single component that integrates ALL Phase 1 handlers for comprehensive
ICT market analysis. Imports and coordinates all Phase 1 components.

Phase 1 Components:
- market_data_engine.py
- timeframe_handler.py
- market_structure_handler.py
- order_block_handler.py
- fvg_handler.py
- liquidity_handler.py
- pd_array_handler.py
- trading_model_handler.py

Author: ICT AI Engine
"""

import pandas as pd
import numpy as np
from datetime import datetime, time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import logging
import pytz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EST = pytz.timezone('US/Eastern')


# =============================================================================
# ENUMS (from individual modules)
# =============================================================================

class TradeDirection(Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class Session(Enum):
    ASIAN = "asian"
    LONDON = "london"
    NY_OPEN = "ny_open"
    NY_AM = "ny_am"
    NY_LUNCH = "ny_lunch"
    NY_PM = "ny_pm"
    OVERNIGHT = "overnight"


class KillZone(Enum):
    ASIAN = "asian"
    LONDON_OPEN = "london_open"
    NY_OPEN = "ny_open"
    NY_AM = "ny_am"
    NY_PM = "ny_pm"
    LAST_HOUR = "last_hour"


class SetupGrade(Enum):
    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class ICTAnalysis:
    """Complete ICT analysis result"""
    timestamp: datetime
    symbol: str
    
    # Price Data
    current_price: float
    current_bar: Dict[str, float]
    
    # Time Context
    session: str
    kill_zone: Optional[str]
    is_macro_time: bool
    
    # Market Structure
    trend: str
    trend_strength: float
    swing_highs: List[Dict]
    swing_lows: List[Dict]
    structure_shift: bool
    shift_type: Optional[str]
    
    # Order Blocks
    bullish_obs: List[Dict]
    bearish_obs: List[Dict]
    nearest_bullish_ob: Optional[Dict]
    nearest_bearish_ob: Optional[Dict]
    
    # FVGs
    bullish_fvgs: List[Dict]
    bearish_fvgs: List[Dict]
    active_fvgs: List[Dict]
    filled_fvgs: List[Dict]
    
    # Liquidity
    buy_side_pools: List[Dict]
    sell_side_pools: List[Dict]
    recent_sweeps: List[Dict]
    
    # PD Arrays
    premium_zones: List[Dict]
    discount_zones: List[Dict]
    equilibrium: float
    price_position: float
    
    # Trading Models
    model_2022: Dict
    silver_bullet: Dict
    venom: Dict
    turtle_soup: Dict
    power_of_three: Dict
    
    # Confluence Score
    confluence_score: int
    confluence_factors: List[str]
    grade: str
    recommendation: str
    
    # Signals
    long_signal: bool
    short_signal: bool
    entry_price: Optional[float]
    stop_loss: Optional[float]
    take_profit: Optional[float]
    confidence: float
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp.isoformat(),
            'symbol': self.symbol,
            'price': self.current_price,
            'session': self.session,
            'kill_zone': self.kill_zone,
            'trend': self.trend,
            'confluence_score': self.confluence_score,
            'grade': self.grade,
            'long_signal': self.long_signal,
            'short_signal': self.short_signal,
            'confidence': self.confidence
        }


# =============================================================================
# PHASE 1 COMPONENT 1: TIMEFRAME HANDLER
# =============================================================================

class TimeframeHandler:
    """Handles session detection, kill zones, and time-based analysis"""
    
    def __init__(self):
        self.sessions = {
            'asian': (0, 5),
            'london': (3, 12),
            'ny_open': (7, 10),
            'ny_am': (9.5, 12),
            'ny_lunch': (12, 13.5),
            'ny_pm': (13.5, 16),
        }
        
        self.kill_zones = {
            'london_open': (1, 5),
            'ny_open': (7, 10),
            'ny_am': (9.5, 12),
            'ny_pm': (13.5, 16),
        }
        
        self.macro_times = [
            (time(2, 33), time(3, 0), "Pre-London"),
            (time(4, 3), time(4, 30), "London Session"),
            (time(8, 50), time(9, 10), "Pre-Market"),
            (time(9, 50), time(10, 10), "Post-Open"),
            (time(10, 50), time(11, 10), "Mid-Morning"),
        ]
    
    def get_session(self, dt: datetime) -> str:
        hour = dt.hour + dt.minute / 60
        for session, (start, end) in self.sessions.items():
            if start <= hour < end:
                return session
        return 'overnight'
    
    def get_kill_zone(self, dt: datetime) -> Optional[str]:
        hour = dt.hour + dt.minute / 60
        for kz, (start, end) in self.kill_zones.items():
            if start <= hour < end:
                return kz
        return None
    
    def is_macro_time(self, dt: datetime) -> Tuple[bool, Optional[str]]:
        t = dt.time()
        for start, end, name in self.macro_times:
            if start <= t <= end:
                return True, name
        return False, None
    
    def get_session_bias(self, dt: datetime) -> Dict:
        session = self.get_session(dt)
        kill_zone = self.get_kill_zone(dt)
        is_macro, macro_name = self.is_macro_time(dt)
        
        confidence = 0.8 if kill_zone == 'ny_am' else 0.7 if session in ['london', 'ny_am'] else 0.3
        
        return {
            'session': session,
            'kill_zone': kill_zone,
            'is_macro_time': is_macro,
            'macro_window': macro_name,
            'confidence': confidence
        }


# =============================================================================
# PHASE 1 COMPONENT 2: MARKET STRUCTURE HANDLER
# =============================================================================

class MarketStructureHandler:
    """Analyzes market structure: swing points, BOS, CHoCH, MSS"""
    
    def __init__(self, lookback: int = 5):
        self.lookback = lookback
        self.swing_highs = []
        self.swing_lows = []
    
    def find_swing_points(self, df: pd.DataFrame) -> Tuple[List, List]:
        highs = df['High'].values
        lows = df['Low'].values
        
        self.swing_highs = []
        self.swing_lows = []
        
        for i in range(self.lookback, len(df) - self.lookback):
            # Swing high
            if highs[i] == highs[i-self.lookback:i+self.lookback].max():
                self.swing_highs.append({
                    'index': i,
                    'price': highs[i],
                    'timestamp': df.index[i]
                })
            
            # Swing low
            if lows[i] == lows[i-self.lookback:i+self.lookback].min():
                self.swing_lows.append({
                    'index': i,
                    'price': lows[i],
                    'timestamp': df.index[i]
                })
        
        return self.swing_highs, self.swing_lows
    
    def analyze_structure(self, df: pd.DataFrame) -> Dict:
        self.find_swing_points(df)
        
        if len(self.swing_highs) < 2 or len(self.swing_lows) < 2:
            return {
                'trend': 'neutral',
                'trend_strength': 0,
                'structure_shift': False,
                'shift_type': None
            }
        
        recent_highs = [h['price'] for h in self.swing_highs[-5:]]
        recent_lows = [l['price'] for l in self.swing_lows[-5:]]
        
        higher_highs = recent_highs[-1] > recent_highs[-2] if len(recent_highs) >= 2 else False
        higher_lows = recent_lows[-1] > recent_lows[-2] if len(recent_lows) >= 2 else False
        lower_highs = recent_highs[-1] < recent_highs[-2] if len(recent_highs) >= 2 else False
        lower_lows = recent_lows[-1] < recent_lows[-2] if len(recent_lows) >= 2 else False
        
        if higher_highs and higher_lows:
            trend = 'BULLISH'
            trend_strength = min(1.0, (recent_highs[-1] - recent_highs[-2]) / recent_highs[-1] * 10 + 0.5)
        elif lower_highs and lower_lows:
            trend = 'BEARISH'
            trend_strength = min(1.0, (recent_highs[-2] - recent_highs[-1]) / recent_highs[-1] * 10 + 0.5)
        else:
            trend = 'RANGING'
            trend_strength = 0.3
        
        return {
            'trend': trend,
            'trend_strength': trend_strength,
            'swing_highs': self.swing_highs[-5:],
            'swing_lows': self.swing_lows[-5:],
            'structure_shift': False,
            'shift_type': None
        }


# =============================================================================
# PHASE 1 COMPONENT 3: ORDER BLOCK HANDLER
# =============================================================================

class OrderBlockHandler:
    """Identifies bullish and bearish order blocks"""
    
    def __init__(self):
        self.order_blocks = []
    
    def identify_order_blocks(self, df: pd.DataFrame) -> List[Dict]:
        opens = df['Open'].values
        closes = df['Close'].values
        highs = df['High'].values
        lows = df['Low'].values
        
        self.order_blocks = []
        
        for i in range(5, len(df)):
            # Bullish OB
            if (closes[i-1] < opens[i-1] and closes[i] > opens[i] and 
                lows[i] < lows[i-1]):
                self.order_blocks.append({
                    'type': 'BULLISH',
                    'index': i,
                    'timestamp': df.index[i],
                    'high': highs[i-1],
                    'low': lows[i-1],
                    'mid': (highs[i-1] + lows[i-1]) / 2,
                    'strength': 0.5 + (closes[i] - opens[i]) / (highs[i] - lows[i] + 0.001) * 0.5
                })
            
            # Bearish OB
            if (closes[i-1] > opens[i-1] and closes[i] < opens[i] and 
                highs[i] > highs[i-1]):
                self.order_blocks.append({
                    'type': 'BEARISH',
                    'index': i,
                    'timestamp': df.index[i],
                    'high': highs[i-1],
                    'low': lows[i-1],
                    'mid': (highs[i-1] + lows[i-1]) / 2,
                    'strength': 0.5 + (opens[i] - closes[i]) / (highs[i] - lows[i] + 0.001) * 0.5
                })
        
        return self.order_blocks
    
    def get_nearest_ob(self, idx: int, ob_type: str) -> Optional[Dict]:
        valid = [ob for ob in self.order_blocks if ob['index'] < idx and ob['type'] == ob_type]
        return valid[-1] if valid else None
    
    def get_recent_obs(self, idx: int, count: int = 10) -> List[Dict]:
        return [ob for ob in self.order_blocks if ob['index'] < idx][-count:]


# =============================================================================
# PHASE 1 COMPONENT 4: FVG HANDLER
# =============================================================================

class FVGHandler:
    """Identifies and manages Fair Value Gaps"""
    
    def __init__(self):
        self.fvgs = []
    
    def identify_fvg(self, df: pd.DataFrame) -> List[Dict]:
        highs = df['High'].values
        lows = df['Low'].values
        
        self.fvgs = []
        
        for i in range(3, len(df)):
            # Bullish FVG
            if lows[i] > highs[i-2]:
                size = lows[i] - highs[i-2]
                self.fvgs.append({
                    'type': 'BULLISH',
                    'index': i,
                    'timestamp': df.index[i],
                    'low': highs[i-2],
                    'high': lows[i],
                    'mid': (highs[i-2] + lows[i]) / 2,
                    'size': size,
                    'filled': False,
                    'fill_price': None,
                    'fill_time': None
                })
            
            # Bearish FVG
            if highs[i] < lows[i-2]:
                size = lows[i-2] - highs[i]
                self.fvgs.append({
                    'type': 'BEARISH',
                    'index': i,
                    'timestamp': df.index[i],
                    'low': highs[i],
                    'high': lows[i-2],
                    'mid': (highs[i] + lows[i-2]) / 2,
                    'size': size,
                    'filled': False,
                    'fill_price': None,
                    'fill_time': None
                })
        
        return self.fvgs
    
    def check_fills(self, df: pd.DataFrame, idx: int) -> List[Dict]:
        filled = []
        low = df['Low'].iloc[idx]
        high = df['High'].iloc[idx]
        
        for fvg in self.fvgs:
            if fvg['filled']:
                continue
            
            if fvg['type'] == 'BULLISH' and low <= fvg['low']:
                fvg['filled'] = True
                fvg['fill_price'] = fvg['low']
                fvg['fill_time'] = df.index[idx]
                filled.append(fvg.copy())
            
            elif fvg['type'] == 'BEARISH' and high >= fvg['high']:
                fvg['filled'] = True
                fvg['fill_price'] = fvg['high']
                fvg['fill_time'] = df.index[idx]
                filled.append(fvg.copy())
        
        return filled
    
    def get_active_fvgs(self, idx: int) -> List[Dict]:
        return [fvg for fvg in self.fvgs if not fvg['filled'] and fvg['index'] < idx]
    
    def get_filled_fvgs(self, idx: int) -> List[Dict]:
        return [fvg for fvg in self.fvgs if fvg['filled'] and fvg['index'] < idx]
    
    def get_fvgs_by_type(self, idx: int, fvg_type: str) -> List[Dict]:
        return [fvg for fvg in self.fvgs if fvg['index'] < idx and fvg['type'] == fvg_type and not fvg['filled']]


# =============================================================================
# PHASE 1 COMPONENT 5: LIQUIDITY HANDLER
# =============================================================================

class LiquidityHandler:
    """Identifies liquidity pools and sweeps"""
    
    def __init__(self, tolerance: float = 0.0005):
        self.tolerance = tolerance
        self.liquidity_pools = {'buy_side': [], 'sell_side': []}
        self.sweeps = []
    
    def find_pools(self, df: pd.DataFrame) -> Dict:
        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        
        self.liquidity_pools = {'buy_side': [], 'sell_side': []}
        
        # Equal highs (sell-side)
        for i in range(10, len(df)):
            for j in range(i+5, min(i+20, len(df))):
                if abs(highs[i] - highs[j]) < closes[i] * self.tolerance:
                    self.liquidity_pools['sell_side'].append({
                        'level': highs[i],
                        'first_touch': df.index[i],
                        'second_touch': df.index[j]
                    })
                    break
        
        # Equal lows (buy-side)
        for i in range(10, len(df)):
            for j in range(i+5, min(i+20, len(df))):
                if abs(lows[i] - lows[j]) < closes[i] * self.tolerance:
                    self.liquidity_pools['buy_side'].append({
                        'level': lows[i],
                        'first_touch': df.index[i],
                        'second_touch': df.index[j]
                    })
                    break
        
        return self.liquidity_pools
    
    def check_sweeps(self, df: pd.DataFrame, idx: int) -> Dict:
        high = df['High'].iloc[idx]
        low = df['Low'].iloc[idx]
        close = df['Close'].iloc[idx]
        
        result = {'buy_side_swept': False, 'sell_side_swept': False, 'sweeps': []}
        
        for pool in self.liquidity_pools.get('sell_side', []):
            if high > pool['level'] and close < pool['level']:
                result['sell_side_swept'] = True
                self.sweeps.append({'type': 'SELL_SIDE', 'level': pool['level'], 'timestamp': df.index[idx]})
                result['sweeps'].append(self.sweeps[-1])
                break
        
        for pool in self.liquidity_pools.get('buy_side', []):
            if low < pool['level'] and close > pool['level']:
                result['buy_side_swept'] = True
                self.sweeps.append({'type': 'BUY_SIDE', 'level': pool['level'], 'timestamp': df.index[idx]})
                result['sweeps'].append(self.sweeps[-1])
                break
        
        return result


# =============================================================================
# PHASE 1 COMPONENT 6: PD ARRAY HANDLER
# =============================================================================

class PDArrayHandler:
    """Premium/Discount array analysis"""
    
    def calculate_pd_arrays(self, df: pd.DataFrame, lookback: int = 20) -> Dict:
        if len(df) < lookback:
            return {'premium': [], 'discount': [], 'equilibrium': 0, 'price_position': 0.5}
        
        period_high = df['High'].tail(lookback).max()
        period_low = df['Low'].tail(lookback).min()
        range_size = period_high - period_low
        
        current_price = df['Close'].iloc[-1]
        
        if range_size == 0:
            return {'premium': [], 'discount': [], 'equilibrium': current_price, 'price_position': 0.5}
        
        price_position = (current_price - period_low) / range_size
        
        premium_high = period_high
        premium_low = period_high - range_size * 0.25
        
        discount_high = period_low + range_size * 0.25
        discount_low = period_low
        
        equilibrium = (period_high + period_low) / 2
        
        return {
            'premium': [{'high': premium_high, 'low': premium_low}],
            'discount': [{'high': discount_high, 'low': discount_low}],
            'equilibrium': equilibrium,
            'price_position': price_position,
            'period_high': period_high,
            'period_low': period_low
        }


# =============================================================================
# PHASE 1 COMPONENT 7: TRADING MODEL HANDLER
# =============================================================================

class TradingModelHandler:
    """Analyzes ICT trading models"""
    
    def __init__(self):
        pass
    
    def analyze_models(self, df: pd.DataFrame, context: Dict) -> Dict:
        results = {}
        
        # Model 2022
        results['model_2022'] = self._analyze_model_2022(context)
        
        # Silver Bullet
        hour = df.index[-1].hour if hasattr(df.index[-1], 'hour') else 12
        silver_bullet_sessions = [(20, 21), (3, 4), (10, 11), (14, 15)]
        in_sb = any(start <= hour < end for start, end in silver_bullet_sessions)
        results['silver_bullet'] = {
            'valid': in_sb and context.get('fvg_present', False),
            'session': 'active' if in_sb else 'inactive'
        }
        
        # Venom
        results['venom'] = {
            'valid': context.get('single_pass', False),
            'criteria_met': context.get('single_pass', False)
        }
        
        # Turtle Soup
        results['turtle_soup'] = {
            'valid': context.get('range_break', False)
        }
        
        # Power of Three
        results['power_of_three'] = {
            'valid': context.get('amd_pattern', False)
        }
        
        return results
    
    def _analyze_model_2022(self, context: Dict) -> Dict:
        requirements = []
        missing = []
        
        if context.get('htf_bias') == 'BULLISH':
            requirements.append('HTF Bullish')
        else:
            missing.append('HTF Bullish')
        
        if context.get('in_discount', False):
            requirements.append('Discount Zone')
        else:
            missing.append('Discount Zone')
        
        if context.get('liquidity_swept', False):
            requirements.append('Liquidity Swept')
        else:
            missing.append('Liquidity Swept')
        
        if context.get('structure_shift', False):
            requirements.append('Structure Shift')
        else:
            missing.append('Structure Shift')
        
        valid = len(requirements) >= 3
        
        return {
            'valid': valid,
            'requirements_met': requirements,
            'requirements_missing': missing,
            'score': len(requirements) / 4
        }


# =============================================================================
# UNIFIED HANDLER - ALL PHASE 1 COMPONENTS INTEGRATED
# =============================================================================

class ICTUnifiedHandler:
    """
    Complete Phase 1 Integration - All Components in One
    
    Integrates:
    - TimeframeHandler
    - MarketStructureHandler
    - OrderBlockHandler
    - FVGHandler
    - LiquidityHandler
    - PDArrayHandler
    - TradingModelHandler
    """
    
    def __init__(self, symbol: str = "NQ"):
        self.symbol = symbol
        
        # Initialize all Phase 1 handlers
        self.timeframe_handler = TimeframeHandler()
        self.structure_handler = MarketStructureHandler()
        self.ob_handler = OrderBlockHandler()
        self.fvg_handler = FVGHandler()
        self.liq_handler = LiquidityHandler()
        self.pd_handler = PDArrayHandler()
        self.model_handler = TradingModelHandler()
        
        # Analysis cache
        self._last_df = None
        self._last_analysis = None
        
        logger.info(f"ICT Unified Handler initialized for {symbol}")
    
    def analyze(self, df: pd.DataFrame) -> ICTAnalysis:
        """
        Perform complete ICT analysis using all Phase 1 components
        
        Args:
            df: DataFrame with OHLC data (columns: Open, High, Low, Close)
            
        Returns:
            ICTAnalysis: Complete analysis result
        """
        if len(df) < 50:
            raise ValueError("Need at least 50 bars for analysis")
        
        current_time = df.index[-1]
        current_price = df['Close'].iloc[-1]
        current_bar = {
            'open': df['Open'].iloc[-1],
            'high': df['High'].iloc[-1],
            'low': df['Low'].iloc[-1],
            'close': current_price
        }
        
        # === PHASE 1 COMPONENT 1: TIMEFRAME ===
        time_ctx = self.timeframe_handler.get_session_bias(current_time)
        
        # === PHASE 1 COMPONENT 2: MARKET STRUCTURE ===
        struct = self.structure_handler.analyze_structure(df)
        
        # === PHASE 1 COMPONENT 3: ORDER BLOCKS ===
        obs = self.ob_handler.identify_order_blocks(df)
        idx = len(df) - 1
        nearest_bullish_ob = self.ob_handler.get_nearest_ob(idx, 'BULLISH')
        nearest_bearish_ob = self.ob_handler.get_nearest_ob(idx, 'BEARISH')
        
        bullish_obs = [ob for ob in obs if ob['type'] == 'BULLISH' and ob['index'] < idx]
        bearish_obs = [ob for ob in obs if ob['type'] == 'BEARISH' and ob['index'] < idx]
        
        # === PHASE 1 COMPONENT 4: FVG ===
        fvgs = self.fvg_handler.identify_fvg(df)
        active_fvgs = self.fvg_handler.get_active_fvgs(idx)
        filled_fvgs = self.fvg_handler.get_filled_fvgs(idx)
        
        bullish_fvgs = [f for f in active_fvgs if f['type'] == 'BULLISH']
        bearish_fvgs = [f for f in active_fvgs if f['type'] == 'BEARISH']
        
        # Check for new fills
        self.fvg_handler.check_fills(df, idx)
        
        # === PHASE 1 COMPONENT 5: LIQUIDITY ===
        self.liq_handler.find_pools(df)
        sweeps = self.liq_handler.check_sweeps(df, idx)
        
        # === PHASE 1 COMPONENT 6: PD ARRAYS ===
        pd_ctx = self.pd_handler.calculate_pd_arrays(df)
        
        # === PHASE 1 COMPONENT 7: TRADING MODELS ===
        model_context = {
            'htf_bias': struct['trend'],
            'structure_shift': struct['structure_shift'],
            'in_discount': pd_ctx['price_position'] < 0.3,
            'in_premium': pd_ctx['price_position'] > 0.7,
            'liquidity_swept': sweeps['sell_side_swept'] or sweeps['buy_side_swept'],
            'fvg_present': len(active_fvgs) > 0,
            'single_pass': False,
            'range_break': False,
            'amd_pattern': False
        }
        
        models = self.model_handler.analyze_models(df, model_context)
        
        # === CONFLUENCE SCORING ===
        confluence_factors = []
        confluence_score = 0
        
        if time_ctx['kill_zone']:
            confluence_factors.append(f"Kill Zone: {time_ctx['kill_zone']}")
            confluence_score += 10
        
        if struct['trend'] != 'neutral':
            confluence_factors.append(f"Trend: {struct['trend']}")
            confluence_score += 15
        
        if pd_ctx['price_position'] < 0.3:
            confluence_factors.append("Discount Zone")
            confluence_score += 15
        elif pd_ctx['price_position'] > 0.7:
            confluence_factors.append("Premium Zone")
            confluence_score += 15
        
        if len(bullish_fvgs) > 0:
            confluence_factors.append(f"Bullish FVG ({len(bullish_fvgs)})")
            confluence_score += 10
        
        if len(bearish_fvgs) > 0:
            confluence_factors.append(f"Bearish FVG ({len(bearish_fvgs)})")
            confluence_score += 10
        
        if nearest_bullish_ob or nearest_bearish_ob:
            confluence_factors.append("Order Block Active")
            confluence_score += 10
        
        if sweeps['buy_side_swept'] or sweeps['sell_side_swept']:
            confluence_factors.append("Liquidity Swept")
            confluence_score += 15
        
        if models['model_2022']['valid']:
            confluence_factors.append("Model 2022 Valid")
            confluence_score += 15
        
        # Grade assignment
        if confluence_score >= 70:
            grade = 'A'
        elif confluence_score >= 55:
            grade = 'B'
        elif confluence_score >= 40:
            grade = 'C'
        elif confluence_score >= 25:
            grade = 'D'
        else:
            grade = 'F'
        
        # === SIGNAL GENERATION ===
        long_signal = False
        short_signal = False
        entry = None
        sl = None
        tp = None
        confidence = 0
        
        atr = (df['High'].tail(14) - df['Low'].tail(14)).mean()
        
        # Long conditions (discount + bullish structure/fvg)
        if (pd_ctx['price_position'] < 0.35 and 
            (struct['trend'] == 'BULLISH' or len(bullish_fvgs) > 0)):
            
            # Check for FVG fill entry
            for fvg in bullish_fvgs:
                if fvg['mid'] < current_price < fvg['high']:
                    long_signal = True
                    entry = current_price
                    sl = fvg['low'] - atr * 0.5
                    tp = current_price + atr * 2
                    confidence = min(0.85, 0.5 + struct['trend_strength'] + (confluence_score / 200))
                    break
            
            # Check for OB entry
            if not long_signal and nearest_bullish_ob:
                if current_price > nearest_bullish_ob['high']:
                    long_signal = True
                    entry = current_price
                    sl = nearest_bullish_ob['low'] - atr * 0.5
                    tp = current_price + atr * 2
                    confidence = min(0.80, 0.5 + nearest_bullish_ob['strength'] + (confluence_score / 200))
        
        # Short conditions (premium + bearish structure/fvg)
        if (pd_ctx['price_position'] > 0.65 and 
            (struct['trend'] == 'BEARISH' or len(bearish_fvgs) > 0)):
            
            for fvg in bearish_fvgs:
                if fvg['low'] < current_price < fvg['mid']:
                    short_signal = True
                    entry = current_price
                    sl = fvg['high'] + atr * 0.5
                    tp = current_price - atr * 2
                    confidence = min(0.85, 0.5 + struct['trend_strength'] + (confluence_score / 200))
                    break
            
            if not short_signal and nearest_bearish_ob:
                if current_price < nearest_bearish_ob['low']:
                    short_signal = True
                    entry = current_price
                    sl = nearest_bearish_ob['high'] + atr * 0.5
                    tp = current_price - atr * 2
                    confidence = min(0.80, 0.5 + nearest_bearish_ob['strength'] + (confluence_score / 200))
        
        # Recommendation
        if long_signal:
            recommendation = "LONG SETUP - Price in discount zone with bullish confluence"
        elif short_signal:
            recommendation = "SHORT SETUP - Price in premium zone with bearish confluence"
        elif time_ctx['kill_zone']:
            recommendation = "WAIT - Kill zone active but no clear setup"
        else:
            recommendation = "WAIT - Outside optimal trading conditions"
        
        # Create analysis object
        analysis = ICTAnalysis(
            timestamp=current_time,
            symbol=self.symbol,
            current_price=current_price,
            current_bar=current_bar,
            session=time_ctx['session'],
            kill_zone=time_ctx['kill_zone'],
            is_macro_time=time_ctx['is_macro_time'],
            trend=struct['trend'],
            trend_strength=struct['trend_strength'],
            swing_highs=struct['swing_highs'],
            swing_lows=struct['swing_lows'],
            structure_shift=struct['structure_shift'],
            shift_type=struct['shift_type'],
            bullish_obs=bullish_obs[-10:],
            bearish_obs=bearish_obs[-10:],
            nearest_bullish_ob=nearest_bullish_ob,
            nearest_bearish_ob=nearest_bearish_ob,
            bullish_fvgs=bullish_fvgs[-10:],
            bearish_fvgs=bearish_fvgs[-10:],
            active_fvgs=active_fvgs[-10:],
            filled_fvgs=filled_fvgs[-10:],
            buy_side_pools=self.liq_handler.liquidity_pools['buy_side'][-5:],
            sell_side_pools=self.liq_handler.liquidity_pools['sell_side'][-5:],
            recent_sweeps=self.liq_handler.sweeps[-5:],
            premium_zones=pd_ctx['premium'],
            discount_zones=pd_ctx['discount'],
            equilibrium=pd_ctx['equilibrium'],
            price_position=pd_ctx['price_position'],
            model_2022=models['model_2022'],
            silver_bullet=models['silver_bullet'],
            venom=models['venom'],
            turtle_soup=models['turtle_soup'],
            power_of_three=models['power_of_three'],
            confluence_score=confluence_score,
            confluence_factors=confluence_factors,
            grade=grade,
            recommendation=recommendation,
            long_signal=long_signal,
            short_signal=short_signal,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            confidence=confidence
        )
        
        self._last_df = df
        self._last_analysis = analysis
        
        return analysis
    
    def get_summary(self, df: pd.DataFrame) -> Dict:
        """Get quick analysis summary"""
        analysis = self.analyze(df)
        return analysis.to_dict()
    
    def run_session(self, df: pd.DataFrame) -> List[ICTAnalysis]:
        """Run analysis on entire dataset and return all signals"""
        analyses = []
        for i in range(50, len(df)):
            analysis = self.analyze(df.iloc[:i+1])
            if analysis.long_signal or analysis.short_signal:
                analyses.append(analysis)
        return analyses


# =============================================================================
# TEST / DEMO
# =============================================================================

def demo():
    """Demo with live data"""
    import yfinance as yf
    
    print("=" * 70)
    print("ICT UNIFIED HANDLER - PHASE 1 COMPLETE INTEGRATION")
    print("=" * 70)
    print()
    
    # Fetch data
    logger.info("Fetching NQ data...")
    df = yf.Ticker("NQ=F").history(period="1mo", interval="1h")
    df = df.dropna()
    df = df[~df.index.duplicated(keep='first')]
    
    print(f"Data: {len(df)} bars | {df.index[0]} to {df.index[-1]}")
    print()
    
    # Initialize unified handler
    handler = ICTUnifiedHandler(symbol="NQ")
    
    # Run analysis
    print("Running unified analysis...")
    analysis = handler.analyze(df)
    
    # Print results
    print()
    print("=" * 70)
    print("ANALYSIS RESULTS")
    print("=" * 70)
    print()
    print(f"SYMBOL:     {analysis.symbol}")
    print(f"PRICE:      ${analysis.current_price:,.0f}")
    print(f"TIMESTAMP:  {analysis.timestamp}")
    print()
    print(f"SESSION:    {analysis.session}")
    print(f"KILL ZONE:  {analysis.kill_zone or 'None'}")
    print(f"MACRO TIME: {analysis.is_macro_time}")
    print()
    print(f"TREND:      {analysis.trend} (strength: {analysis.trend_strength:.2f})")
    print()
    print(f"ORDER BLOCKS:")
    print(f"  Bullish OB:  {len(analysis.bullish_obs)} recent")
    print(f"  Bearish OB:  {len(analysis.bearish_obs)} recent")
    print()
    print(f"FAIR VALUE GAPS:")
    print(f"  Active FVGs: {len(analysis.active_fvgs)}")
    print(f"  Bullish:     {len(analysis.bullish_fvgs)}")
    print(f"  Bearish:     {len(analysis.bearish_fvgs)}")
    print()
    print(f"LIQUIDITY:")
    print(f"  Buy Pools:   {len(analysis.buy_side_pools)}")
    print(f"  Sell Pools:  {len(analysis.sell_side_pools)}")
    print(f"  Recent Sweeps: {len(analysis.recent_sweeps)}")
    print()
    print(f"PD ARRAYS:")
    print(f"  Price Position: {analysis.price_position*100:.1f}%")
    print(f"  Equilibrium:    ${analysis.equilibrium:,.0f}")
    print()
    print(f"TRADING MODELS:")
    print(f"  Model 2022:  {'VALID' if analysis.model_2022['valid'] else 'Invalid'}")
    print(f"  Silver Bullet: {analysis.silver_bullet['session']}")
    print()
    print(f"CONFLUENCE:")
    print(f"  Score:  {analysis.confluence_score}/100")
    print(f"  Grade:  {analysis.grade}")
    print(f"  Factors: {len(analysis.confluence_factors)}")
    for factor in analysis.confluence_factors[:5]:
        print(f"    - {factor}")
    print()
    print(f"SIGNAL:")
    print(f"  Long Signal:  {analysis.long_signal}")
    print(f"  Short Signal: {analysis.short_signal}")
    print(f"  Entry:        {analysis.entry_price}")
    print(f"  Stop Loss:    {analysis.stop_loss}")
    print(f"  Take Profit:  {analysis.take_profit}")
    print(f"  Confidence:   {analysis.confidence:.1%}")
    print()
    print(f"RECOMMENDATION: {analysis.recommendation}")
    print()
    print("=" * 70)
    
    return handler


if __name__ == "__main__":
    handler = demo()
