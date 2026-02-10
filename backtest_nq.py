"""
ICT Backtester - Phase 1 Components Test
=========================================

Backtest all Phase 1 ICT components on NQ with 6 months of live data.
Starting capital: $10,000

Phase 1 Components:
- Market Data Engine
- Timeframe Handler
- Market Structure Handler
- Order Block Handler
- FVG Handler
- Liquidity Handler
- PD Array Handler
- Trading Model Handler
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import json
import logging
import pytz

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

EST = pytz.timezone('US/Eastern')


# =============================================================================
# ENUMS
# =============================================================================

class TradeDirection(Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class SignalType(Enum):
    ENTRY = "entry"
    EXIT = "exit"
    STOP = "stop"


class SetupGrade(Enum):
    A_PLUS = "A+"
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    F = "F"


# =============================================================================
# ORDER BLOCK HANDLER
# =============================================================================

class OrderBlockHandler:
    """Identifies and analyzes Order Blocks"""
    
    def __init__(self):
        self.order_blocks = []
    
    def identify_order_blocks(self, df: pd.DataFrame) -> List[Dict]:
        """Identify order blocks in price data"""
        objs = []
        
        for i in range(5, len(df)):
            # Look for bullish order block (last candle before up move is bearish)
            if (df['Close'].iloc[i-1] < df['Open'].iloc[i-1] and  # Bearish candle
                df['Close'].iloc[i] > df['Open'].iloc[i] and  # Bullish candle
                df['Low'].iloc[i] < df['Low'].iloc[i-1]):  # Low taken
                
                objs.append({
                    'type': 'BULLISH',
                    'index': i,
                    'timestamp': df.index[i],
                    'high': df['High'].iloc[i-1],
                    'low': df['Low'].iloc[i-1],
                    'close': df['Close'].iloc[i-1],
                    'open': df['Open'].iloc[i-1],
                    'fair_value_gap': df['Close'].iloc[i] - df['Low'].iloc[i-1],
                    'strength': self._calculate_strength(df, i, 'BULLISH')
                })
            
            # Look for bearish order block
            if (df['Close'].iloc[i-1] > df['Open'].iloc[i-1] and  # Bullish candle
                df['Close'].iloc[i] < df['Open'].iloc[i] and  # Bearish candle
                df['High'].iloc[i] > df['High'].iloc[i-1]):  # High taken
                
                objs.append({
                    'type': 'BEARISH',
                    'index': i,
                    'timestamp': df.index[i],
                    'high': df['High'].iloc[i-1],
                    'low': df['Low'].iloc[i-1],
                    'close': df['Close'].iloc[i-1],
                    'open': df['Open'].iloc[i-1],
                    'fair_value_gap': df['High'].iloc[i-1] - df['Close'].iloc[i],
                    'strength': self._calculate_strength(df, i, 'BEARISH')
                })
        
        self.order_blocks = objs[-50:]  # Keep last 50
        return self.order_blocks
    
    def _calculate_strength(self, df: pd.DataFrame, idx: int, ob_type: str) -> float:
        """Calculate order block strength based on follow-through"""
        if idx + 5 >= len(df):
            return 0.5
        
        if ob_type == 'BULLISH':
            return min(1.0, (df['Close'].iloc[idx+5] - df['Close'].iloc[idx]) / (df['Close'].iloc[idx] * 0.01) + 0.5)
        else:
            return min(1.0, (df['Close'].iloc[idx] - df['Close'].iloc[idx+5]) / (df['Close'].iloc[idx] * 0.01) + 0.5)
    
    def get_nearest_ob(self, current_idx: int, direction: str) -> Optional[Dict]:
        """Get nearest order block in direction of trade"""
        valid_obs = [ob for ob in self.order_blocks 
                    if ob['index'] < current_idx and ob['type'] == direction]
        return valid_obs[-1] if valid_obs else None


# =============================================================================
# FVG HANDLER
# =============================================================================

class FVGHandler:
    """Identifies and manages Fair Value Gaps"""
    
    def __init__(self):
        self.fvgs = []
    
    def identify_fvg(self, df: pd.DataFrame) -> List[Dict]:
        """Identify all FVGs in price data"""
        fvgs = []
        
        for i in range(3, len(df)):
            high_1 = df['High'].iloc[i-2]
            low_1 = df['Low'].iloc[i-2]
            high_2 = df['High'].iloc[i-1]
            low_2 = df['Low'].iloc[i-1]
            high_3 = df['High'].iloc[i]
            low_3 = df['Low'].iloc[i]
            
            # Bullish FVG
            if low_3 > high_1:
                size = low_3 - high_1
                fvgs.append({
                    'type': 'BULLISH',
                    'index': i,
                    'timestamp': df.index[i],
                    'low': high_1,
                    'high': low_3,
                    'mid': (high_1 + low_3) / 2,
                    'size': size,
                    'filled': False,
                    'fill_price': None,
                    'fill_time': None
                })
            
            # Bearish FVG
            if high_3 < low_1:
                size = low_1 - high_3
                fvgs.append({
                    'type': 'BEARISH',
                    'index': i,
                    'timestamp': df.index[i],
                    'low': high_3,
                    'high': low_1,
                    'mid': (high_3 + low_1) / 2,
                    'size': size,
                    'filled': False,
                    'fill_price': None,
                    'fill_time': None
                })
        
        self.fvgs = fvgs
        return fvgs
    
    def check_fvg_fill(self, df: pd.DataFrame, idx: int) -> List[Dict]:
        """Check if any FVGs were filled at this bar"""
        filled = []
        for fvg in self.fvgs:
            if fvg['filled']:
                continue
            
            if fvg['type'] == 'BULLISH':
                if df['Low'].iloc[idx] <= fvg['low']:
                    fvg['filled'] = True
                    fvg['fill_price'] = fvg['low']
                    fvg['fill_time'] = df.index[idx]
                    filled.append(fvg.copy())
            else:
                if df['High'].iloc[idx] >= fvg['high']:
                    fvg['filled'] = True
                    fvg['fill_price'] = fvg['high']
                    fvg['fill_time'] = df.index[idx]
                    filled.append(fvg.copy())
        
        return filled
    
    def get_active_fvgs(self, idx: int) -> List[Dict]:
        """Get unfilled FVGs above/below current price"""
        current_price = 0  # Will be set by caller
        return [fvg for fvg in self.fvgs if not fvg['filled'] and fvg['index'] < idx]


# =============================================================================
# MARKET STRUCTURE HANDLER
# =============================================================================

class MarketStructureHandler:
    """Analyzes market structure (BOS, CHoCH, MSS)"""
    
    def __init__(self):
        self.swing_highs = []
        self.swing_lows = []
        self.breaks = []
    
    def find_swing_points(self, df: pd.DataFrame, lookback: int = 5) -> Tuple[List, List]:
        """Identify swing highs and lows"""
        highs = []
        lows = []
        
        for i in range(lookback, len(df) - lookback):
            # Swing high
            if df['High'].iloc[i] == df['High'].iloc[i-lookback:i+lookback].max():
                highs.append({
                    'index': i,
                    'price': df['High'].iloc[i],
                    'timestamp': df.index[i]
                })
            
            # Swing low
            if df['Low'].iloc[i] == df['Low'].iloc[i-lookback:i+lookback].min():
                lows.append({
                    'index': i,
                    'price': df['Low'].iloc[i],
                    'timestamp': df.index[i]
                })
        
        self.swing_highs = highs
        self.swing_lows = lows
        return highs, lows
    
    def analyze_structure(self, df: pd.DataFrame) -> Dict:
        """Analyze current market structure"""
        highs, lows = self.find_swing_points(df)
        
        if len(highs) < 2 or len(lows) < 2:
            return {'trend': 'neutral', 'break_type': None, 'mss': None}
        
        # Determine trend
        recent_highs = [h['price'] for h in highs[-5:]]
        recent_lows = [l['price'] for l in lows[-5:]]
        
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            higher_highs = recent_highs[-1] > recent_highs[-2]
            higher_lows = recent_lows[-1] > recent_lows[-2]
            lower_highs = recent_highs[-1] < recent_highs[-2]
            lower_lows = recent_lows[-1] < recent_lows[-2]
            
            if higher_highs and higher_lows:
                trend = 'BULLISH'
            elif lower_highs and lower_lows:
                trend = 'BEARISH'
            else:
                trend = 'RANGING'
        else:
            trend = 'neutral'
        
        return {
            'trend': trend,
            'swing_highs': highs[-5:],
            'swing_lows': lows[-5:],
            'break_type': None,
            'mss': None
        }


# =============================================================================
# LIQUIDITY HANDLER
# =============================================================================

class LiquidityHandler:
    """Identifies liquidity pools and sweeps"""
    
    def __init__(self):
        self.liquidity_pools = []
        self.sweeps = []
    
    def find_liquidity_pools(self, df: pd.DataFrame) -> Dict:
        """Find liquidity pools (equal highs/lows)"""
        pools = {'buy_side': [], 'sell_side': []}
        
        # Find equal highs (sell-side liquidity)
        for i in range(10, len(df)):
            for j in range(i+5, min(i+20, len(df))):
                if abs(df['High'].iloc[i] - df['High'].iloc[j]) < df['Close'].iloc[i] * 0.0005:
                    pools['sell_side'].append({
                        'level': df['High'].iloc[i],
                        'first_touch': df.index[i],
                        'second_touch': df.index[j]
                    })
                    break
        
        # Find equal lows (buy-side liquidity)
        for i in range(10, len(df)):
            for j in range(i+5, min(i+20, len(df))):
                if abs(df['Low'].iloc[i] - df['Low'].iloc[j]) < df['Close'].iloc[i] * 0.0005:
                    pools['buy_side'].append({
                        'level': df['Low'].iloc[i],
                        'first_touch': df.index[i],
                        'second_touch': df.index[j]
                    })
                    break
        
        self.liquidity_pools = pools
        return pools
    
    def check_sweeps(self, df: pd.DataFrame, idx: int) -> Dict:
        """Check if liquidity was swept at this bar"""
        current_price = df['Close'].iloc[idx]
        current_high = df['High'].iloc[idx]
        current_low = df['Low'].iloc[idx]
        
        sweeps = {'buy_side_swept': False, 'sell_side_swept': False}
        
        # Check for sell-side sweeps
        for pool in self.liquidity_pools.get('sell_side', []):
            if current_high > pool['level'] and current_price < pool['level']:
                sweeps['sell_side_swept'] = True
                self.sweeps.append({
                    'type': 'SELL_SIDE',
                    'level': pool['level'],
                    'timestamp': df.index[idx]
                })
                break
        
        # Check for buy-side sweeps
        for pool in self.liquidity_pools.get('buy_side', []):
            if current_low < pool['level'] and current_price > pool['level']:
                sweeps['buy_side_swept'] = True
                self.sweeps.append({
                    'type': 'BUY_SIDE',
                    'level': pool['level'],
                    'timestamp': df.index[idx]
                })
                break
        
        return sweeps


# =============================================================================
# TIMEFRAME HANDLER
# =============================================================================

class TimeframeHandler:
    """Manages multi-timeframe analysis and kill zones"""
    
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
            'asian': (0, 5),
            'london_open': (1, 5),
            'ny_open': (7, 10),
            'ny_am': (9.5, 12),
            'ny_pm': (13.5, 16),
        }
    
    def get_session(self, dt: datetime) -> str:
        """Get current trading session"""
        hour = dt.hour + dt.minute / 60
        
        for session, (start, end) in self.sessions.items():
            if start <= hour < end:
                return session
        return 'overnight'
    
    def get_kill_zone(self, dt: datetime) -> Optional[str]:
        """Get current kill zone"""
        hour = dt.hour + dt.minute / 60
        
        for kz, (start, end) in self.kill_zones.items():
            if start <= hour < end:
                return kz
        return None
    
    def get_time_bias(self, dt: datetime) -> Dict:
        """Get time-based trading bias"""
        session = self.get_session(dt)
        kill_zone = self.get_kill_zone(dt)
        
        # High probability zones
        if session in ['london', 'ny_am']:
            confidence = 0.7
        elif kill_zone in ['ny_am', 'ny_open']:
            confidence = 0.8
        else:
            confidence = 0.3
        
        return {
            'session': session,
            'kill_zone': kill_zone,
            'session_confidence': confidence
        }


# =============================================================================
# PD ARRAY HANDLER
# =============================================================================

class PDArrayHandler:
    """Premium/Discount Array analysis"""
    
    def __init__(self):
        self.pd_zones = []
    
    def calculate_pd_arrays(self, df: pd.DataFrame) -> Dict:
        """Calculate premium and discount arrays"""
        if len(df) < 20:
            return {'premium': [], 'discount': [], 'equilibrium': 0}
        
        # Calculate ADR
        atr = df['High'].tail(20) - df['Low'].tail(20)
        adr = atr.mean()
        
        # Current price position
        current_price = df['Close'].iloc[-1]
        period_high = df['High'].tail(20).max()
        period_low = df['Low'].tail(20).min()
        range_size = period_high - period_low
        
        if range_size == 0:
            return {'premium': [], 'discount': [], 'equilibrium': current_price}
        
        price_position = (current_price - period_low) / range_size
        
        # Premium zone (upper 20% of range - sellers)
        premium_zone_high = period_high
        premium_zone_low = period_high - (range_size * 0.25)
        
        # Discount zone (lower 20% of range - buyers)
        discount_zone_high = period_low + (range_size * 0.25)
        discount_zone_low = period_low
        
        # Equilibrium (middle)
        equilibrium = (period_high + period_low) / 2
        
        return {
            'premium': [{'high': premium_zone_high, 'low': premium_zone_low}],
            'discount': [{'high': discount_zone_high, 'low': discount_zone_low}],
            'equilibrium': equilibrium,
            'price_position': price_position,
            'adr': adr
        }


# =============================================================================
# TRADING MODEL HANDLER
# =============================================================================

class TradingModelHandler:
    """ICT Trading Model analysis (2022, Silver Bullet, etc.)"""
    
    def __init__(self):
        self.models = ['model_2022', 'silver_bullet', 'venom', 'turtle_soup']
    
    def analyze_model_2022(self, df: pd.DataFrame, context: Dict) -> Dict:
        """Analyze Model 2022 setup requirements"""
        analysis = {
            'model': 'model_2022',
            'valid': False,
            'stage': None,
            'requirements_met': [],
            'requirements_missing': []
        }
        
        # Check requirements
        if context.get('htf_bias') == 'BULLISH':
            analysis['requirements_met'].append('HTF Bullish Bias')
        else:
            analysis['requirements_missing'].append('HTF Bullish Bias')
        
        if context.get('structure_shift', False):
            analysis['requirements_met'].append('Structure Shift')
        else:
            analysis['requirements_missing'].append('Structure Shift')
        
        if context.get('in_discount', False):
            analysis['requirements_met'].append('Price in Discount')
        else:
            analysis['requirements_missing'].append('Price in Discount')
        
        if context.get('liquidity_swept', False):
            analysis['requirements_met'].append('Liquidity Swept')
        else:
            analysis['requirements_missing'].append('Liquidity Swept')
        
        # Model valid if at least 3 requirements met
        if len(analysis['requirements_met']) >= 3:
            analysis['valid'] = True
            analysis['stage'] = 'SETUP'
        
        return analysis
    
    def analyze_all_models(self, df: pd.DataFrame, context: Dict) -> Dict:
        """Analyze all trading models"""
        results = {}
        
        # Model 2022
        results['model_2022'] = self.analyze_model_2022(df, context)
        
        # Silver Bullet (time-based)
        hour = df.index[-1].hour if hasattr(df.index[-1], 'hour') else 12
        silver_bullet_sessions = [(20, 21), (3, 4), (10, 11), (14, 15)]
        in_sb_window = any(start <= hour < end for start, end in silver_bullet_sessions)
        
        results['silver_bullet'] = {
            'model': 'silver_bullet',
            'valid': in_sb_window and context.get('fvg_present', False),
            'session': 'active' if in_sb_window else 'inactive'
        }
        
        # Venom (single pass)
        results['venom'] = {
            'model': 'venom',
            'valid': False,
            'criteria_met': context.get('single_pass', False)
        }
        
        # Turtle Soup (range break)
        results['turtle_soup'] = {
            'model': 'turtle_soup',
            'valid': context.get('range_break', False)
        }
        
        return results


# =============================================================================
# BACKTESTER
# =============================================================================

@dataclass
class Trade:
    entry_time: datetime
    entry_price: float
    direction: TradeDirection
    size: float
    stop_loss: float
    take_profit: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: float = 0.0
    status: str = "OPEN"
    setup_type: str = ""
    grade: str = "C"
    confidence: float = 0.5


class ICTBacktester:
    """Main backtesting engine"""
    
    def __init__(self, initial_capital: float = 10000):
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.trades: List[Trade] = []
        self.equity_curve = []
        self.stats = {}
        
        # Initialize handlers
        self.ob_handler = OrderBlockHandler()
        self.fvg_handler = FVGHandler()
        self.struct_handler = MarketStructureHandler()
        self.liq_handler = LiquidityHandler()
        self.tf_handler = TimeframeHandler()
        self.pd_handler = PDArrayHandler()
        self.model_handler = TradingModelHandler()
    
    def run_backtest(self, df: pd.DataFrame, symbol: str = "NQ") -> Dict:
        """Run full backtest on data"""
        logger.info(f"Starting backtest on {symbol}")
        logger.info(f"Period: {df.index[0]} to {df.index[-1]}")
        logger.info(f"Bars: {len(df)}")
        logger.info(f"Initial Capital: ${self.capital:,.2f}")
        
        # Pre-analyze data
        logger.info("Pre-analyzing data...")
        self.fvg_handler.identify_fvg(df)
        self.ob_handler.identify_order_blocks(df)
        self.liq_handler.find_liquidity_pools(df)
        self.struct_handler.find_swing_points(df)
        
        # Run simulation
        logger.info("Running simulation...")
        open_trade = None
        
        for idx in range(50, len(df)):  # Skip first 50 bars for analysis
            current_bar = df.iloc[idx]
            current_time = df.index[idx]
            current_price = current_bar['Close']
            
            # Update equity curve
            self.equity_curve.append({
                'timestamp': current_time,
                'equity': self.capital,
                'drawdown': (self.capital - self.initial_capital) / self.initial_capital
            })
            
            # Get time context
            time_ctx = self.tf_handler.get_time_bias(current_time)
            
            # Get PD array context
            pd_ctx = self.pd_handler.calculate_pd_arrays(df.iloc[:idx+1])
            
            # Get structure context
            struct = self.struct_handler.analyze_structure(df.iloc[:idx+1])
            
            # Build context for models
            context = {
                'htf_bias': struct['trend'],
                'structure_shift': False,
                'in_discount': pd_ctx.get('price_position', 0.5) < 0.25,
                'in_premium': pd_ctx.get('price_position', 0.5) > 0.75,
                'liquidity_swept': False,
                'fvg_present': False,
                'single_pass': False,
                'range_break': False
            }
            
            # Check for fills
            self.fvg_handler.check_fvg_fill(df, idx)
            
            # Check for liquidity sweeps
            sweeps = self.liq_handler.check_sweeps(df, idx)
            context['liquidity_swept'] = sweeps['sell_side_swept'] or sweeps['buy_side_swept']
            
            # Generate signals
            if open_trade is None:
                signal = self._check_for_entry(df, idx, context, time_ctx, struct)
                if signal:
                    open_trade = signal
            else:
                self._check_exit(open_trade, df, idx, current_price)
                
                if open_trade.status == "CLOSED":
                    self.trades.append(open_trade)
                    open_trade = None
            
            # Log progress every 500 bars
            if idx % 500 == 0:
                logger.info(f"Progress: {idx}/{len(df)} bars | Equity: ${self.capital:,.2f}")
        
        # Close any open trade
        if open_trade and open_trade.status == "OPEN":
            final_price = df['Close'].iloc[-1]
            open_trade.exit_time = df.index[-1]
            open_trade.exit_price = final_price
            open_trade.pnl = self._calculate_pnl(open_trade, final_price)
            open_trade.status = "CLOSED (EOD)"
            self.trades.append(open_trade)
        
        # Calculate stats
        self._calculate_stats()
        
        logger.info("Backtest complete!")
        return self.stats
    
    def _check_for_entry(self, df: pd.DataFrame, idx: int, context: Dict, 
                        time_ctx: Dict, struct: Dict) -> Optional[Trade]:
        """Check for entry signal"""
        current_price = df['Close'].iloc[idx]
        current_bar = df.iloc[idx]
        
        # Get recent FVG
        recent_fvgs = [f for f in self.fvg_handler.fvgs 
                      if f['index'] < idx and not f['filled']]
        
        # Get nearest OB
        nearest_bullish_ob = self.ob_handler.get_nearest_ob(idx, 'BULLISH')
        nearest_bearish_ob = self.ob_handler.get_nearest_ob(idx, 'BEARISH')
        
        # Entry conditions
        entry_signal = None
        direction = TradeDirection.NEUTRAL
        setup_type = ""
        confidence = 0.5
        
        # Check bullish setup
        if context['htf_bias'] in ['BULLISH', 'neutral'] and context['in_discount']:
            # Look for bullish reversal at OB or FVG
            for fvg in recent_fvgs:
                if fvg['type'] == 'BULLISH' and fvg['mid'] < current_price < fvg['high']:
                    direction = TradeDirection.LONG
                    setup_type = "FVG Long"
                    confidence = 0.65
                    break
            
            if nearest_bullish_ob and nearest_bullish_ob['index'] > idx - 20:
                if current_price > nearest_bullish_ob['high']:
                    direction = TradeDirection.LONG
                    setup_type = "OB Break"
                    confidence = 0.70
        
        # Check bearish setup
        if context['htf_bias'] in ['BEARISH', 'neutral'] and context['in_premium']:
            for fvg in recent_fvgs:
                if fvg['type'] == 'BEARISH' and fvg['low'] < current_price < fvg['mid']:
                    direction = TradeDirection.SHORT
                    setup_type = "FVG Short"
                    confidence = 0.65
                    break
            
            if nearest_bearish_ob and nearest_bearish_ob['index'] > idx - 20:
                if current_price < nearest_bearish_ob['low']:
                    direction = TradeDirection.SHORT
                    setup_type = "OB Break"
                    confidence = 0.70
        
        # Create trade if signal
        if direction != TradeDirection.NEUTRAL and time_ctx['session_confidence'] > 0.4:
            atr = (df['High'].tail(14) - df['Low'].tail(14)).mean()
            risk_per_trade = self.capital * 0.02  # 2% risk
            
            if direction == TradeDirection.LONG:
                stop_loss = current_price - (atr * 1.5)
                take_profit = current_price + (atr * 3)
            else:
                stop_loss = current_price + (atr * 1.5)
                take_profit = current_price - (atr * 3)
            
            # Size position
            risk_amount = abs(current_price - stop_loss)
            position_size = risk_per_trade / risk_amount if risk_amount > 0 else 1
            
            grade = "B" if confidence > 0.6 else "C"
            
            return Trade(
                entry_time=df.index[idx],
                entry_price=current_price,
                direction=direction,
                size=position_size,
                stop_loss=stop_loss,
                take_profit=take_profit,
                setup_type=setup_type,
                grade=grade,
                confidence=confidence
            )
        
        return None
    
    def _check_exit(self, trade: Trade, df: pd.DataFrame, idx: int, current_price: float):
        """Check exit conditions"""
        # Stop loss hit
        if trade.direction == TradeDirection.LONG:
            if current_price <= trade.stop_loss:
                trade.exit_time = df.index[idx]
                trade.exit_price = trade.stop_loss
                trade.pnl = self._calculate_pnl(trade, trade.stop_loss)
                trade.status = "STOP_HIT"
                self.capital += trade.pnl
                return
            
            if current_price >= trade.take_profit:
                trade.exit_time = df.index[idx]
                trade.exit_price = trade.take_profit
                trade.pnl = self._calculate_pnl(trade, trade.take_profit)
                trade.status = "TP_HIT"
                self.capital += trade.pnl
                return
        
        else:  # SHORT
            if current_price >= trade.stop_loss:
                trade.exit_time = df.index[idx]
                trade.exit_price = trade.stop_loss
                trade.pnl = self._calculate_pnl(trade, trade.stop_loss)
                trade.status = "STOP_HIT"
                self.capital += trade.pnl
                return
            
            if current_price <= trade.take_profit:
                trade.exit_time = df.index[idx]
                trade.exit_price = trade.take_profit
                trade.pnl = self._calculate_pnl(trade, trade.take_profit)
                trade.status = "TP_HIT"
                self.capital += trade.pnl
                return
        
        # Time exit (after 20 bars)
        if trade.entry_time and hasattr(df.index[idx], 'timestamp'):
            bars_held = idx - list(df.index).index(trade.entry_time) if trade.entry_time in df.index.values else 0
            if bars_held > 20:
                trade.exit_time = df.index[idx]
                trade.exit_price = current_price
                trade.pnl = self._calculate_pnl(trade, current_price)
                trade.status = "TIME_EXIT"
                self.capital += trade.pnl
    
    def _calculate_pnl(self, trade: Trade, exit_price: float) -> float:
        """Calculate PnL for trade"""
        if trade.direction == TradeDirection.LONG:
            return (exit_price - trade.entry_price) * trade.size * 20  # NQ = 20x multiplier
        else:
            return (trade.entry_price - exit_price) * trade.size * 20
    
    def _calculate_stats(self):
        """Calculate backtest statistics"""
        closed_trades = [t for t in self.trades if t.status != "OPEN"]
        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl <= 0]
        
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
        
        win_rate = len(winning_trades) / len(closed_trades) * 100 if closed_trades else 0
        avg_win = sum(t.pnl for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t.pnl for t in losing_trades) / len(losing_trades) if losing_trades else 0
        
        max_equity = max(e['equity'] for e in self.equity_curve) if self.equity_curve else self.initial_capital
        min_equity = min(e['equity'] for e in self.equity_curve) if self.equity_curve else self.initial_capital
        max_drawdown = (max_equity - min_equity) / max_equity * 100 if max_equity > 0 else 0
        
        profit_factor = abs(sum(t.pnl for t in winning_trades) / sum(t.pnl for t in losing_trades)) if losing_trades and sum(t.pnl for t in losing_trades) != 0 else float('inf')
        
        self.stats = {
            'symbol': 'NQ',
            'period': {
                'start': str(self.equity_curve[0]['timestamp']) if self.equity_curve else 'N/A',
                'end': str(self.equity_curve[-1]['timestamp']) if self.equity_curve else 'N/A'
            },
            'capital': {
                'initial': self.initial_capital,
                'final': self.capital,
                'return_pct': total_return
            },
            'trades': {
                'total': len(closed_trades),
                'winners': len(winning_trades),
                'losers': len(losing_trades),
                'win_rate': win_rate
            },
            'pnl': {
                'gross_profit': sum(t.pnl for t in winning_trades),
                'gross_loss': sum(t.pnl for t in losing_trades),
                'net_pnl': self.capital - self.initial_capital,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor
            },
            'risk': {
                'max_drawdown_pct': max_drawdown,
                'avg_risk_per_trade': 200  # 2% of $10k
            },
            'setup_breakdown': self._analyze_setups()
        }
    
    def _analyze_setups(self) -> Dict:
        """Analyze performance by setup type"""
        setups = {}
        for trade in self.trades:
            if trade.setup_type:
                if trade.setup_type not in setups:
                    setups[trade.setup_type] = {'count': 0, 'wins': 0, 'pnl': 0}
                setups[trade.setup_type]['count'] += 1
                if trade.pnl > 0:
                    setups[trade.setup_type]['wins'] += 1
                setups[trade.setup_type]['pnl'] += trade.pnl
        return setups
    
    def save_results(self, filepath: str = "backtest_results.json"):
        """Save results to JSON"""
        results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0'
            },
            'statistics': self.stats,
            'equity_curve': self.equity_curve,
            'trades': [
                {
                    'entry_time': str(t.entry_time) if t.entry_time else None,
                    'entry_price': t.entry_price,
                    'direction': t.direction.value,
                    'size': t.size,
                    'stop_loss': t.stop_loss,
                    'take_profit': t.take_profit,
                    'exit_time': str(t.exit_time) if t.exit_time else None,
                    'exit_price': t.exit_price,
                    'pnl': t.pnl,
                    'status': t.status,
                    'setup_type': t.setup_type,
                    'grade': t.grade
                }
                for t in self.trades
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
        return filepath


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("ICT BACKTESTER - NQ 6-MONTH TEST")
    print("=" * 70)
    print(f"Starting Capital: $10,000")
    print()
    
    # Fetch data
    logger.info("Fetching NQ data from Yahoo Finance...")
    ticker = yf.Ticker("NQ=F")
    df = ticker.history(period="6mo", interval="1h")
    
    if df.empty:
        logger.error("Failed to fetch data!")
        return
    
    # Clean data
    df = df.dropna()
    df = df[~df.index.duplicated(keep='first')]
    
    logger.info(f"Data fetched: {len(df)} bars")
    logger.info(f"Period: {df.index[0]} to {df.index[-1]}")
    print()
    
    # Run backtest
    backtester = ICTBacktester(initial_capital=10000)
    stats = backtester.run_backtest(df, symbol="NQ")
    
    # Save results
    output_file = backtester.save_results("backtest_results.json")
    
    # Print summary
    print()
    print("=" * 70)
    print("BACKTEST RESULTS SUMMARY")
    print("=" * 70)
    print(f"Period: {stats['period']['start']} to {stats['period']['end']}")
    print()
    print(f"CAPITAL:")
    print(f"  Initial:    ${stats['capital']['initial']:>12,.2f}")
    print(f"  Final:      ${stats['capital']['final']:>12,.2f}")
    print(f"  Return:     {stats['capital']['return_pct']:>12.2f}%")
    print()
    print(f"TRADES:")
    print(f"  Total:      {stats['trades']['total']:>12}")
    print(f"  Winners:    {stats['trades']['winners']:>12}")
    print(f"  Losers:     {stats['trades']['losers']:>12}")
    print(f"  Win Rate:   {stats['trades']['win_rate']:>12.2f}%")
    print()
    print(f"P&L:")
    print(f"  Gross Profit:  ${stats['pnl']['gross_profit']:>12,.2f}")
    print(f"  Gross Loss:    ${stats['pnl']['gross_loss']:>12,.2f}")
    print(f"  Net PnL:       ${stats['pnl']['net_pnl']:>12,.2f}")
    print(f"  Profit Factor: {stats['pnl']['profit_factor']:>12.2f}")
    print()
    print(f"RISK:")
    print(f"  Max Drawdown:  {stats['risk']['max_drawdown_pct']:>12.2f}%")
    print()
    print(f"SETUP BREAKDOWN:")
    for setup, data in stats['setup_breakdown'].items():
        win_rate = (data['wins'] / data['count'] * 100) if data['count'] > 0 else 0
        print(f"  {setup:15} | Count: {data['count']:3} | Win: {win_rate:5.1f}% | PnL: ${data['pnl']:>10,.2f}")
    
    print()
    print(f"Results saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
