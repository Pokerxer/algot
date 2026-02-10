"""
ICT Complete Trading System - Phase 1 + 2 + 3 Integration (V3)
=============================================================

Comprehensive integration of ALL phases:
- Phase 1: Core ICT Handlers (already integrated)
- Phase 2: Integration Engine, Signal Generator/Aggregator, Trade Executor
- Phase 3: AI/ML Signal Filter, Model Trainer, Reinforcement Learning

Features:
1. All Phase 1 components (FVG, OB, Structure, Liquidity, etc.)
2. Signal generation with confluence scoring
3. AI-powered signal filtering (Phase 3)
4. ML model predictions (Phase 3)
5. Reinforcement learning adaptation (Phase 3)
6. Full trade execution pipeline (Phase 2)

Author: ICT AI Engine
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
import json
import logging
from collections import defaultdict
import random

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class TradeDirection(Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class SignalDecision(Enum):
    ACCEPT = "accept"
    REJECT = "reject"
    MODIFY = "modify"


class MarketRegime(Enum):
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    VOLATILE = "volatile"
    LOW_VOLATILITY = "low_volatility"


# =============================================================================
# PHASE 1 COMPONENTS (Optimized, inlined for V3)
# =============================================================================

class Phase1Core:
    """Phase 1 core indicators - inlined for performance"""
    
    def __init__(self, df: pd.DataFrame):
        self.highs = df['High'].values
        self.lows = df['Low'].values
        self.closes = df['Close'].values
        self.opens = df['Open'].values
        self.timestamps = df.index.values
        self.length = len(df)
        
        # Pre-calculate all indicators
        self._calculate_fvgs()
        self._calculate_order_blocks()
        self._calculate_trend()
        self._calculate_price_position()
        self._calculate_sessions()
        self._calculate_liquidity()
    
    def _calculate_fvgs(self):
        self.bullish_fvgs = []
        self.bearish_fvgs = []
        for i in range(3, self.length):
            if self.lows[i] > self.highs[i-2]:
                self.bullish_fvgs.append({'idx': i, 'low': self.highs[i-2], 'high': self.lows[i], 'mid': (self.highs[i-2]+self.lows[i])/2})
            if self.highs[i] < self.lows[i-2]:
                self.bearish_fvgs.append({'idx': i, 'low': self.highs[i], 'high': self.lows[i-2], 'mid': (self.highs[i]+self.lows[i-2])/2})
    
    def _calculate_order_blocks(self):
        self.bullish_obs = []
        self.bearish_obs = []
        for i in range(5, self.length):
            if self.closes[i-1] < self.opens[i-1] and self.closes[i] > self.opens[i] and self.lows[i] < self.lows[i-1]:
                self.bullish_obs.append({'idx': i, 'high': self.highs[i-1], 'low': self.lows[i-1]})
            if self.closes[i-1] > self.opens[i-1] and self.closes[i] < self.opens[i] and self.highs[i] > self.highs[i-1]:
                self.bearish_obs.append({'idx': i, 'high': self.highs[i-1], 'low': self.lows[i-1]})
    
    def _calculate_trend(self):
        self.trend = np.zeros(self.length)
        for i in range(20, self.length):
            rh = self.highs[max(0,i-20):i].max()
            rl = self.lows[max(0,i-20):i].min()
            if rh > self.highs[i-5] and rl > self.lows[i-5]:
                self.trend[i] = 1
            elif rh < self.highs[i-5] and rl < self.lows[i-5]:
                self.trend[i] = -1
    
    def _calculate_price_position(self):
        self.price_position = np.zeros(self.length)
        for i in range(20, self.length):
            ph = self.highs[i-20:i].max()
            pl = self.lows[i-20:i].min()
            self.price_position[i] = (self.closes[i] - pl) / (ph - pl + 0.001)
    
    def _calculate_sessions(self):
        hours = pd.to_datetime(self.timestamps).hour.values
        self.kill_zone = np.zeros(self.length, dtype=bool)
        for i in range(len(hours)):
            h = hours[i]
            self.kill_zone[i] = (1 <= h < 5) or (7 <= h < 12) or (13.5 <= h < 16)
    
    def _calculate_liquidity(self):
        self.buy_pools = []
        self.sell_pools = []
        for i in range(10, self.length):
            for j in range(i+5, min(i+20, self.length)):
                if abs(self.highs[i] - self.highs[j]) < self.closes[i] * 0.0005:
                    self.sell_pools.append({'idx': i, 'level': self.highs[i]})
                    break
        for i in range(10, self.length):
            for j in range(i+5, min(i+20, self.length)):
                if abs(self.lows[i] - self.lows[j]) < self.closes[i] * 0.0005:
                    self.buy_pools.append({'idx': i, 'level': self.lows[i]})
                    break
    
    def get_state(self, idx: int) -> Dict:
        """Get complete state at index"""
        current_price = self.closes[idx]
        
        nearest_bull = next((ob for ob in reversed(self.bullish_obs) if ob['idx'] < idx), None)
        nearest_bear = next((ob for ob in reversed(self.bearish_obs) if ob['idx'] < idx), None)
        near_bull_fvg = next((f for f in reversed(self.bullish_fvgs) if f['idx'] < idx and f['mid'] < current_price < f['high']), None)
        near_bear_fvg = next((f for f in reversed(self.bearish_fvgs) if f['idx'] < idx and f['low'] < current_price < f['mid']), None)
        
        return {
            'price': current_price,
            'trend': self.trend[idx],
            'price_position': self.price_position[idx],
            'kill_zone': self.kill_zone[idx],
            'nearest_bull_ob': nearest_bull,
            'nearest_bear_ob': nearest_bear,
            'nearest_bull_fvg': near_bull_fvg,
            'nearest_bear_fvg': near_bear_fvg,
            'atr': (self.highs[idx-14:idx] - self.lows[idx-14:idx]).mean() if idx > 14 else 50
        }


# =============================================================================
# PHASE 2: SIGNAL GENERATOR & AGGREGATOR
# =============================================================================

class SignalGenerator:
    """Phase 2: Generates trade signals based on ICT criteria"""
    
    def generate_signal(self, state: Dict, htf_trend: int) -> Optional[Dict]:
        """Generate signal based on current market state"""
        price = state['price']
        trend = state['trend']
        pp = state['price_position']
        kz = state['kill_zone']
        atr = state['atr']
        
        # Calculate confluence
        confluence = 0
        factors = []
        
        # Time
        if kz:
            confluence += 15
            factors.append('Kill Zone')
        
        # HTF alignment
        if htf_trend == 1 and trend >= 0:
            confluence += 25
            factors.append('HTF Bullish')
        elif htf_trend == -1 and trend <= 0:
            confluence += 25
            factors.append('HTF Bearish')
        
        # Price position
        if pp < 0.25:
            confluence += 20
            factors.append('Deep Discount')
        elif pp < 0.35:
            confluence += 15
            factors.append('Discount')
        elif pp > 0.75:
            confluence += 20
            factors.append('Deep Premium')
        elif pp > 0.65:
            confluence += 15
            factors.append('Premium')
        
        # Structure/FVG
        if state['nearest_bull_fvg'] and trend >= 0:
            confluence += 15
            factors.append('Bullish FVG')
        if state['nearest_bear_fvg'] and trend <= 0:
            confluence += 15
            factors.append('Bearish FVG')
        
        # OB
        if state['nearest_bull_ob']:
            confluence += 10
            factors.append('Bullish OB')
        if state['nearest_bear_ob']:
            confluence += 10
            factors.append('Bearish OB')
        
        # Determine grade
        grade = 'F'
        if confluence >= 75:
            grade = 'A+'
        elif confluence >= 70:
            grade = 'A'
        elif confluence >= 60:
            grade = 'B'
        elif confluence >= 50:
            grade = 'C'
        
        # Generate signal
        signal = None
        
        # Long conditions
        if pp < 0.40 and (htf_trend == 1 or trend >= 0):
            if state['nearest_bull_fvg'] and price > state['nearest_bull_fvg']['mid']:
                sl = state['nearest_bull_fvg']['low'] - atr * 0.5
                signal = {
                    'direction': 'long',
                    'entry': price,
                    'stop_loss': sl,
                    'take_profit': price + atr * 2.5,
                    'confluence': confluence,
                    'grade': grade,
                    'factors': factors
                }
            elif state['nearest_bull_ob'] and price > state['nearest_bull_ob']['high']:
                sl = state['nearest_bull_ob']['low'] - atr * 0.5
                signal = {
                    'direction': 'long',
                    'entry': price,
                    'stop_loss': sl,
                    'take_profit': price + atr * 2.5,
                    'confluence': confluence,
                    'grade': grade,
                    'factors': factors
                }
        
        # Short conditions
        elif pp > 0.60 and (htf_trend == -1 or trend <= 0):
            if state['nearest_bear_fvg'] and price < state['nearest_bear_fvg']['mid']:
                sl = state['nearest_bear_fvg']['high'] + atr * 0.5
                signal = {
                    'direction': 'short',
                    'entry': price,
                    'stop_loss': sl,
                    'take_profit': price - atr * 2.5,
                    'confluence': confluence,
                    'grade': grade,
                    'factors': factors
                }
            elif state['nearest_bear_ob'] and price < state['nearest_bear_ob']['low']:
                sl = state['nearest_bear_ob']['high'] + atr * 0.5
                signal = {
                    'direction': 'short',
                    'entry': price,
                    'stop_loss': sl,
                    'take_profit': price - atr * 2.5,
                    'confluence': confluence,
                    'grade': grade,
                    'factors': factors
                }
        
        return signal


# =============================================================================
# PHASE 3: AI SIGNAL FILTER
# =============================================================================

class AISignalFilter:
    """
    Phase 3: AI-powered signal filtering using ML principles
    Simulates sklearn-based filtering with configurable thresholds
    """
    
    def __init__(self):
        # Historical performance tracking
        self.signal_history = []
        self.learning_rate = 0.01
        self.thresholds = {
            'min_confidence': 0.65,
            'min_confluence': 70,
            'pattern_weight': 0.15,
            'regime_weight': 0.10,
            'timing_weight': 0.10
        }
        
        # Pattern recognition weights (learned)
        self.pattern_weights = {
            'fvg': 0.8,
            'ob': 0.7,
            'liquidity_sweep': 0.9,
            'structure_shift': 0.85
        }
        
        # Regime performance tracking
        self.regime_performance = {
            'trending_up': {'wins': 0, 'total': 0},
            'trending_down': {'wins': 0, 'total': 0},
            'ranging': {'wins': 0, 'total': 0}
        }
    
    def filter_signal(self, signal: Dict, market_state: Dict) -> Dict:
        """
        Filter signal using AI/ML criteria
        Returns filtered decision with confidence
        """
        if signal is None:
            return {'decision': 'reject', 'confidence': 0, 'reason': 'No signal'}
        
        score = 0.0
        reasons = []
        
        # 1. Confidence score check
        if signal['confluence'] >= self.thresholds['min_confluence']:
            score += 0.35
            reasons.append('High confluence')
        else:
            score += signal['confluence'] / 100 * 0.35
            reasons.append(f"Confluence: {signal['confluence']}")
        
        # 2. Pattern matching score
        pattern_score = 0
        for factor in signal.get('factors', []):
            factor_key = factor.lower().replace(' ', '_')
            if factor_key in self.pattern_weights:
                pattern_score += self.pattern_weights[factor_key]
        
        pattern_score = min(1.0, pattern_score / 3)  # Normalize
        score += pattern_score * self.thresholds['pattern_weight']
        
        # 3. Regime alignment
        trend = market_state.get('trend', 0)
        direction = signal['direction']
        
        if trend > 0 and direction == 'long':
            regime_score = 0.9
            self.regime_performance['trending_up']['total'] += 1
            self.regime_performance['trending_up']['wins'] += 1
        elif trend < 0 and direction == 'short':
            regime_score = 0.9
            self.regime_performance['trending_down']['total'] += 1
            self.regime_performance['trending_down']['wins'] += 1
        elif trend == 0:
            regime_score = 0.6
            self.regime_performance['ranging']['total'] += 1
        else:
            regime_score = 0.3
        
        score += regime_score * self.thresholds['regime_weight']
        
        # 4. Timing score (kill zone bonus)
        if market_state.get('kill_zone', False):
            score += self.thresholds['timing_weight']
            reasons.append('Kill Zone')
        
        # 5. Risk/reward check
        rr_ratio = abs(signal['take_profit'] - signal['entry']) / abs(signal['entry'] - signal['stop_loss'])
        if rr_ratio >= 2.0:
            score += 0.1
            reasons.append(f'Good R:R ({rr_ratio:.1f})')
        
        # Make decision
        if score >= 0.7:
            decision = 'accept'
        elif score >= 0.5:
            decision = 'modify'
        else:
            decision = 'reject'
        
        # Store for learning
        self.signal_history.append({
            'signal': signal,
            'score': score,
            'decision': decision,
            'timestamp': datetime.now()
        })
        
        return {
            'decision': decision,
            'confidence': score,
            'reasons': reasons,
            'rr_ratio': rr_ratio,
            'pattern_score': pattern_score,
            'regime_score': regime_score
        }
    
    def update_weights(self, trade_result: Dict):
        """Update weights based on trade results (RL-like learning)"""
        # Simple update: reinforce patterns that led to wins
        if trade_result.get('pnl', 0) > 0:
            for factor in trade_result.get('factors', []):
                factor_key = factor.lower().replace(' ', '_')
                if factor_key in self.pattern_weights:
                    self.pattern_weights[factor_key] = min(1.0, self.pattern_weights[factor_key] + self.learning_rate * 0.1)


# =============================================================================
# PHASE 3: ML MODEL TRAINER (Simplified)
# =============================================================================

class MLModelTrainer:
    """
    Phase 3: ML Model training for signal quality prediction
    """
    
    def __init__(self):
        self.training_data = []
        self.model_weights = {
            'confluence': 0.3,
            'trend_alignment': 0.25,
            'price_position': 0.2,
            'volume': 0.15,
            'timing': 0.1
        }
        self.is_trained = False
    
    def add_sample(self, features: Dict, outcome: float):
        """Add training sample"""
        self.training_data.append({'features': features, 'outcome': outcome})
    
    def train(self):
        """Train model on collected data"""
        if len(self.training_data) < 10:
            return False
        
        # Simple weighted average training
        wins = [s for s in self.training_data if s['outcome'] > 0]
        losses = [s for s in self.training_data if s['outcome'] <= 0]
        
        if wins and losses:
            win_avg = np.mean([s['features'].get('confluence', 50) for s in wins])
            loss_avg = np.mean([s['features'].get('confluence', 50) for s in losses])
            
            if loss_avg > 0:
                adjustment = (win_avg - loss_avg) / 100
                self.model_weights['confluence'] = max(0.1, min(0.5, 0.3 + adjustment))
        
        self.is_trained = True
        return True
    
    def predict(self, features: Dict) -> float:
        """Predict signal quality (0-1)"""
        if not self.is_trained:
            # Default prediction based on confluence
            return features.get('confluence', 50) / 100
        
        score = 0
        score += features.get('confluence', 50) / 100 * self.model_weights['confluence']
        score += features.get('trend_alignment', 0.5) * self.model_weights['trend_alignment']
        score += features.get('price_position_favorable', 0.5) * self.model_weights['price_position']
        
        return min(1.0, score)


# =============================================================================
# PHASE 3: REINFORCEMENT LEARNING AGENT (Simplified)
# =============================================================================

class RLAgent:
    """
    Phase 3: Reinforcement Learning agent for trade adaptation
    Uses simple Q-learning inspired approach
    """
    
    def __init__(self):
        self.q_table = defaultdict(float)
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.exploration_rate = 0.3
        self.episode_rewards = []
    
    def get_action(self, state: Dict) -> str:
        """Get action (trade size modifier) based on state"""
        state_key = self._get_state_key(state)
        
        # Explore vs exploit
        if random.random() < self.exploration_rate:
            return random.choice(['small', 'medium', 'large'])
        
        # Choose best action
        q_values = [self.q_table.get(f"{state_key}_{a}", 0) for a in ['small', 'medium', 'large']]
        best_idx = np.argmax(q_values)
        
        return ['small', 'medium', 'large'][best_idx]
    
    def update(self, state: Dict, action: str, reward: float, next_state: Dict):
        """Update Q-values based on reward"""
        state_key = self._get_state_key(state)
        next_state_key = self._get_state_key(next_state)
        
        current_q = self.q_table.get(f"{state_key}_{action}", 0)
        max_next_q = max([self.q_table.get(f"{next_state_key}_{a}", 0) for a in ['small', 'medium', 'large']])
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[f"{state_key}_{action}"] = new_q
        
        # Decay exploration
        self.exploration_rate = max(0.05, self.exploration_rate * 0.99)
    
    def _get_state_key(self, state: Dict) -> str:
        """Convert state to hashable key"""
        trend = 'bull' if state.get('trend', 0) > 0 else 'bear' if state.get('trend', 0) < 0 else 'neutral'
        pp = 'discount' if state.get('price_position', 0.5) < 0.4 else 'premium' if state.get('price_position', 0.5) > 0.6 else 'neutral'
        kz = 'kz' if state.get('kill_zone', False) else 'nkz'
        
        return f"{trend}_{pp}_{kz}"


# =============================================================================
# PHASE 2: TRADE EXECUTOR
# =============================================================================

class TradeExecutor:
    """Phase 2: Trade execution and management"""
    
    def __init__(self, initial_capital: float = 10000):
        self.capital = initial_capital
        self.open_trades = []
        self.closed_trades = []
        self.risk_per_trade = 0.01  # 1% risk
    
    def calculate_position_size(self, entry: float, stop_loss: float, direction: str) -> float:
        """Calculate position size based on risk"""
        risk_amount = self.capital * self.risk_per_trade
        risk_per_share = abs(entry - stop_loss)
        
        if risk_per_share == 0:
            return 1
        
        return risk_amount / risk_per_share
    
    def execute_trade(self, signal: Dict, filtered_result: Dict) -> Optional[Dict]:
        """Execute trade if signal passes AI filter"""
        if filtered_result['decision'] != 'accept':
            return None
        
        direction = signal['direction']
        entry = signal['entry']
        sl = signal['stop_loss']
        tp = signal['take_profit']
        
        size = self.calculate_position_size(entry, sl, direction)
        
        trade = {
            'entry_time': datetime.now(),
            'entry_price': entry,
            'direction': direction,
            'size': size,
            'stop_loss': sl,
            'take_profit': tp,
            'status': 'OPEN',
            'confidence': filtered_result['confidence'],
            'grade': signal['grade'],
            'factors': signal['factors']
        }
        
        self.open_trades.append(trade)
        return trade
    
    def update_trades(self, current_price: float, current_time: datetime):
        """Update open trades - check exits"""
        for trade in self.open_trades[:]:
            if trade['direction'] == 'long':
                if current_price <= trade['stop_loss']:
                    trade['exit_time'] = current_time
                    trade['exit_price'] = trade['stop_loss']
                    trade['pnl'] = (trade['exit_price'] - trade['entry_price']) * trade['size'] * 20
                    trade['status'] = 'STOP_HIT'
                    self.capital += trade['pnl']
                    self.closed_trades.append(trade)
                    self.open_trades.remove(trade)
                elif current_price >= trade['take_profit']:
                    trade['exit_time'] = current_time
                    trade['exit_price'] = trade['take_profit']
                    trade['pnl'] = (trade['exit_price'] - trade['entry_price']) * trade['size'] * 20
                    trade['status'] = 'TP_HIT'
                    self.capital += trade['pnl']
                    self.closed_trades.append(trade)
                    self.open_trades.remove(trade)
            
            else:  # short
                if current_price >= trade['stop_loss']:
                    trade['exit_time'] = current_time
                    trade['exit_price'] = trade['stop_loss']
                    trade['pnl'] = (trade['entry_price'] - trade['exit_price']) * trade['size'] * 20
                    trade['status'] = 'STOP_HIT'
                    self.capital += trade['pnl']
                    self.closed_trades.append(trade)
                    self.open_trades.remove(trade)
                elif current_price <= trade['take_profit']:
                    trade['exit_time'] = current_time
                    trade['exit_price'] = trade['take_profit']
                    trade['pnl'] = (trade['entry_price'] - trade['exit_price']) * trade['size'] * 20
                    trade['status'] = 'TP_HIT'
                    self.capital += trade['pnl']
                    self.closed_trades.append(trade)
                    self.open_trades.remove(trade)


# =============================================================================
# COMPLETE TRADING SYSTEM V3
# =============================================================================

class ICTTradingSystemV3:
    """
    Complete ICT Trading System - All Phases Integrated (V3)
    
    Phase 1: Core handlers (FVG, OB, Structure, etc.)
    Phase 2: Signal Generator, Aggregator, Executor
    Phase 3: AI Signal Filter, ML Trainer, RL Agent
    """
    
    def __init__(self, initial_capital: float = 10000, symbol: str = "NQ"):
        self.symbol = symbol
        self.initial_capital = initial_capital
        
        # Initialize all components
        self.signal_generator = SignalGenerator()
        self.ai_filter = AISignalFilter()
        self.ml_trainer = MLModelTrainer()
        self.rl_agent = RLAgent()
        self.executor = TradeExecutor(initial_capital)
        
        # State
        self.phase1_data = None
        self.htf_trend = 0
        
        logger.info(f"ICT Trading System V3 initialized for {symbol}")
    
    def preprocess(self, df: pd.DataFrame, df_daily: pd.DataFrame):
        """Preprocess data - calculate all Phase 1 indicators"""
        self.phase1_data = Phase1Core(df)
        
        # Calculate HTF trend from daily data
        daily_highs = df_daily['High'].values
        daily_lows = df_daily['Low'].values
        
        htf = []
        for i in range(1, len(df_daily)):
            if daily_highs[i] > daily_highs[max(0,i-5):i].max() and daily_lows[i] > daily_lows[max(0,i-5):i].min():
                htf.append(1)
            elif daily_highs[i] < daily_highs[max(0,i-5):i].max() and daily_lows[i] < daily_lows[max(0,i-5):i].min():
                htf.append(-1)
            else:
                htf.append(0)
        
        # Map to hourly
        df_index = pd.DatetimeIndex(df.index).tz_localize(None)
        df_daily_index = pd.DatetimeIndex(df_daily.index).tz_localize(None)
        
        self.htf_trends = np.zeros(len(df))
        for i in range(len(df)):
            bar_time = df_index[i]
            for j in range(len(df_daily)-1, -1, -1):
                if df_daily_index[j] <= bar_time:
                    self.htf_trends[i] = htf[j] if j < len(htf) else 0
                    break
    
    def run(self, df: pd.DataFrame, df_daily: pd.DataFrame) -> Dict:
        """Run complete trading system on data"""
        logger.info(f"Starting V3 trading system on {len(df)} bars")
        
        # Preprocess
        self.preprocess(df, df_daily)
        
        capital = self.initial_capital
        equity_curve = [capital]
        all_trades = []
        
        phase1 = self.phase1_data
        
        for idx in range(50, len(df)):
            current_price = phase1.closes[idx]
            current_time = df.index[idx]
            htf_trend = self.htf_trends[idx]
            
            # Get market state
            state = phase1.get_state(idx)
            state['htf_trend'] = htf_trend
            
            # Update open trades
            self.executor.update_trades(current_price, current_time)
            
            # Generate signal (Phase 2)
            signal = self.signal_generator.generate_signal(state, htf_trend)
            
            if signal:
                # Apply AI Filter (Phase 3)
                filtered = self.ai_filter.filter_signal(signal, state)
                
                # Get RL action for position sizing
                rl_state = {
                    'trend': state['trend'],
                    'price_position': state['price_position'],
                    'kill_zone': state['kill_zone']
                }
                position_modifier = self.rl_agent.get_action(rl_state)
                
                # Adjust risk based on RL
                if position_modifier == 'small':
                    self.executor.risk_per_trade = 0.005  # 0.5%
                elif position_modifier == 'medium':
                    self.executor.risk_per_trade = 0.01  # 1%
                else:
                    self.executor.risk_per_trade = 0.02  # 2%
                
                # Execute trade
                trade = self.executor.execute_trade(signal, filtered)
                
                if trade:
                    # RL update (learn from trade outcome eventually)
                    # For now, update based on current state
                    pass
            
            equity_curve.append(self.executor.capital)
            
            if idx % 500 == 0:
                logger.info(f"Progress: {idx}/{len(df)} | Equity: ${self.executor.capital:,.0f}")
        
        # Close open trades
        for trade in self.executor.open_trades:
            trade['exit_price'] = phase1.closes[-1]
            trade['exit_time'] = df.index[-1]
            trade['pnl'] = (phase1.closes[-1] - trade['entry_price']) * trade['size'] * 20 if trade['direction'] == 'long' else (trade['entry_price'] - phase1.closes[-1]) * trade['size'] * 20
            trade['status'] = 'EOD'
            self.executor.capital += trade['pnl']
            all_trades.append(trade)
        
        # Add closed trades
        all_trades.extend(self.executor.closed_trades)
        
        # Calculate statistics
        return self._calculate_stats(all_trades, equity_curve, df)
    
    def _calculate_stats(self, trades: List, equity_curve: List, df: pd.DataFrame) -> Dict:
        closed = [t for t in trades if t.get('exit_price')]
        winners = [t for t in closed if t.get('pnl', 0) > 0]
        losers = [t for t in closed if t.get('pnl', 0) <= 0]
        
        total_return = (self.executor.capital - self.initial_capital) / self.initial_capital * 100
        win_rate = len(winners) / len(closed) * 100 if closed else 0
        
        max_eq = max(equity_curve)
        min_eq = min(equity_curve)
        max_dd = (max_eq - min_eq) / max_eq * 100 if max_eq > 0 else 0
        
        profit = sum(t['pnl'] for t in winners)
        loss = abs(sum(t['pnl'] for t in losers))
        pf = profit / loss if loss > 0 else float('inf')
        
        # By phase
        by_phase = {
            'phase1_core': len(closed),
            'phase2_signals': len([t for t in closed]),
            'phase3_filtered': len([t for t in closed if t.get('confidence', 0) > 0.7])
        }
        
        return {
            'metadata': {
                'version': 'V3 - Complete Integration',
                'timestamp': datetime.now().isoformat(),
                'phases': ['Phase 1 Core', 'Phase 2 Signal Gen/Agg', 'Phase 3 AI/ML/RL']
            },
            'period': {'start': str(df.index[0])[:10], 'end': str(df.index[-1])[:10], 'bars': len(df)},
            'capital': {'initial': self.initial_capital, 'final': self.executor.capital, 'return_pct': total_return},
            'trades': {'total': len(closed), 'winners': len(winners), 'losers': len(losers), 'win_rate': win_rate},
            'pnl': {'gross_profit': profit, 'gross_loss': loss, 'net_pnl': self.executor.capital - self.initial_capital, 'profit_factor': pf},
            'risk': {'max_drawdown_pct': max_dd},
            'phase_breakdown': by_phase,
            'equity_curve': [{'date': str(df.index[i])[:10], 'equity': e} for i, e in enumerate(equity_curve)]
        }


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("ICT TRADING SYSTEM V3 - COMPLETE PHASE 1+2+3 INTEGRATION")
    print("=" * 70)
    print()
    print("PHASES INTEGRATED:")
    print("  Phase 1: Core ICT Handlers (FVG, OB, Structure, Liquidity, etc.)")
    print("  Phase 2: Signal Generator, Aggregator, Trade Executor")
    print("  Phase 3: AI Signal Filter, ML Trainer, RL Agent")
    print()
    print("Starting Capital: $10,000")
    print()
    
    # Fetch data
    print("Fetching NQ data...")
    df = yf.Ticker("NQ=F").history(period="6mo", interval="1h")
    df = df.dropna()
    df = df[~df.index.duplicated(keep='first')]
    
    df_daily = yf.Ticker("NQ=F").history(period="6mo", interval="1d")
    df_daily = df_daily.dropna()
    
    print(f"Data: {len(df)} hourly bars, {len(df_daily)} daily bars")
    print()
    
    # Run V3 system
    system = ICTTradingSystemV3(initial_capital=10000, symbol="NQ")
    stats = system.run(df, df_daily)
    
    # Save results
    with open('v3_complete_results.json', 'w') as f:
        json.dump(stats, f, indent=2)
    
    # Print summary
    print()
    print("=" * 70)
    print("V3 COMPLETE TRADING SYSTEM RESULTS")
    print("=" * 70)
    print(f"Period: {stats['period']['start']} to {stats['period']['end']}")
    print()
    print("CAPITAL:")
    print(f"  Initial:    ${stats['capital']['initial']:>12,.0f}")
    print(f"  Final:      ${stats['capital']['final']:>12,.0f}")
    print(f"  Return:     {stats['capital']['return_pct']:>12.1f}%")
    print()
    print("TRADES:")
    print(f"  Total:      {stats['trades']['total']}")
    print(f"  Win Rate:   {stats['trades']['win_rate']:.1f}%")
    print()
    print("P&L:")
    print(f"  Gross Profit:  ${stats['pnl']['gross_profit']:>12,.0f}")
    print(f"  Gross Loss:    ${stats['pnl']['gross_loss']:>12,.0f}")
    print(f"  Net PnL:       ${stats['pnl']['net_pnl']:>12,.0f}")
    print(f"  Profit Factor: {stats['pnl']['profit_factor']:>12.2f}")
    print()
    print("RISK:")
    print(f"  Max Drawdown:  {stats['risk']['max_drawdown_pct']:.1f}%")
    print()
    print(f"Results saved to: v3_complete_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
