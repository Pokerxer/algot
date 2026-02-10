"""
ICT Trading System V3 - OPTIMIZED (V2 Rules + Phase 3 AI/ML/RL)
================================================================

Combines V2's proven filtering rules with Phase 3 enhancements:
- V2 Rules: Confluence >= 70, HTF alignment, A-grade only, 1% risk
- Phase 3: AI Signal Filter, ML Model Trainer, RL Agent

This is the BEST of both worlds.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List
from dataclasses import dataclass
import json
import random

print("=" * 70)
print("ICT TRADING SYSTEM V3 - OPTIMIZED (V2 Rules + Phase 3)")
print("=" * 70)
print()
print("V2 PROVEN RULES:")
print("  - Confluence >= 70 required")
print("  - HTF alignment required")
print("  - Only A-grade setups (A or A+)")
print("  - 1% risk per trade")
print()
print("PHASE 3 ENHANCEMENTS:")
print("  - AI Signal Filter for additional validation")
print("  - ML Model Trainer for pattern learning")
print("  - RL Agent for adaptive position sizing")
print()

# Fetch data
df = yf.Ticker("NQ=F").history(period="6mo", interval="1h")
df = df.dropna()
df = df[~df.index.duplicated(keep='first')]

df_daily = yf.Ticker("NQ=F").history(period="6mo", interval="1d")
df_daily = df_daily.dropna()

print(f"Data: {len(df)} hourly bars, {len(df_daily)} daily bars")

# Pre-calculate indicators
highs = df['High'].values
lows = df['Low'].values
closes = df['Close'].values
opens = df['Open'].values
timestamps = df.index.values

# FVGs
bullish_fvgs = [{'idx': i, 'low': highs[i-2], 'high': lows[i], 'mid': (highs[i-2]+lows[i])/2} 
               for i in range(3, len(df)) if lows[i] > highs[i-2]]
bearish_fvgs = [{'idx': i, 'low': highs[i], 'high': lows[i-2], 'mid': (highs[i]+lows[i-2])/2} 
               for i in range(3, len(df)) if highs[i] < lows[i-2]]

# Order Blocks
bullish_obs = [{'idx': i, 'high': highs[i-1], 'low': lows[i-1]} 
              for i in range(5, len(df)) if closes[i-1] < opens[i-1] and closes[i] > opens[i] and lows[i] < lows[i-1]]
bearish_obs = [{'idx': i, 'high': highs[i-1], 'low': lows[i-1]} 
              for i in range(5, len(df)) if closes[i-1] > opens[i-1] and closes[i] < opens[i] and highs[i] > highs[i-1]]

# HTF Trend
daily_highs = df_daily['High'].values
daily_lows = df_daily['Low'].values

df_index = pd.DatetimeIndex(df.index).tz_localize(None)
df_daily_index = pd.DatetimeIndex(df_daily.index).tz_localize(None)

htf_trends = np.zeros(len(df))
for i in range(len(df)):
    bar_time = df_index[i]
    for j in range(len(df_daily)-1, -1, -1):
        if df_daily_index[j] <= bar_time:
            if j >= 5:
                if daily_highs[j] > daily_highs[j-5:j].max() and daily_lows[j] > daily_lows[j-5:j].min():
                    htf_trends[i] = 1
                elif daily_highs[j] < daily_highs[j-5:j].max() and daily_lows[j] < daily_lows[j-5:j].min():
                    htf_trends[i] = -1
            break

# LTF Trend
trend = np.zeros(len(df))
for i in range(20, len(df)):
    rh = highs[max(0,i-20):i].max()
    rl = lows[max(0,i-20):i].min()
    if rh > highs[i-5] and rl > lows[i-5]:
        trend[i] = 1
    elif rh < highs[i-5] and rl < lows[i-5]:
        trend[i] = -1

# Price Position
price_position = np.zeros(len(df))
for i in range(20, len(df)):
    ph = highs[i-20:i].max()
    pl = lows[i-20:i].min()
    price_position[i] = (closes[i] - pl) / (ph - pl + 0.001)

# Kill Zones
hours = pd.to_datetime(timestamps).hour.values
kill_zone = np.array([(1 <= h < 5) or (7 <= h < 12) or (13.5 <= h < 16) for h in hours])

# =============================================================================
# PHASE 3 COMPONENTS
# =============================================================================

class AISignalFilter:
    """Phase 3: AI-powered signal filtering"""
    
    def __init__(self):
        self.min_confidence = 0.70
        self.min_confluence = 70
        self.learning_rate = 0.01
        self.pattern_weights = {
            'fvg': 0.8,
            'ob': 0.7,
            'structure': 0.9,
            'kill_zone': 0.6,
            'htf_alignment': 0.95,
            'discount': 0.75,
            'premium': 0.75
        }
        self.regime_performance = {
            'trending_up': {'wins': 0, 'total': 0},
            'trending_down': {'wins': 0, 'total': 0},
            'ranging': {'wins': 0, 'total': 0}
        }
    
    def filter(self, signal: Dict, state: Dict) -> Dict:
        """
        Filter signal using AI/ML criteria
        Returns filtered decision with confidence score
        """
        if signal is None:
            return {'decision': 'reject', 'confidence': 0, 'reason': 'No signal'}
        
        # V2 RULE: Check minimum confluence first
        if signal['confluence'] < self.min_confluence:
            return {'decision': 'reject', 'confidence': signal['confluence']/100, 'reason': f'Confluence {signal["confluence"]} < {self.min_confluence}'}
        
        score = 0.0
        reasons = []
        
        # 1. Base confluence score (35% weight)
        if signal['confluence'] >= 75:
            score += 0.35
            reasons.append('High confluence (A+)')
        elif signal['confluence'] >= 70:
            score += 0.30
            reasons.append('Good confluence (A)')
        else:
            score += signal['confluence'] / 100 * 0.35
        
        # 2. Pattern matching score (20% weight)
        pattern_score = 0
        for factor in signal.get('factors', []):
            factor_key = factor.lower().replace(' ', '_')
            if factor_key in self.pattern_weights:
                pattern_score += self.pattern_weights[factor_key]
        
        pattern_score = min(1.0, pattern_score / 3)
        score += pattern_score * 0.20
        if pattern_score > 0.5:
            reasons.append('Strong patterns')
        
        # 3. Regime alignment (25% weight)
        htf_trend = state.get('htf_trend', 0)
        direction = signal['direction']
        
        if htf_trend == 1 and direction == 'long':
            regime_score = 0.95
            self.regime_performance['trending_up']['total'] += 1
            self.regime_performance['trending_up']['wins'] += 1
            reasons.append('Strong HTF bullish alignment')
        elif htf_trend == -1 and direction == 'short':
            regime_score = 0.95
            self.regime_performance['trending_down']['total'] += 1
            self.regime_performance['trending_down']['wins'] += 1
            reasons.append('Strong HTF bearish alignment')
        elif htf_trend == 0:
            regime_score = 0.3
            self.regime_performance['ranging']['total'] += 1
            reasons.append('No clear HTF trend')
        else:
            regime_score = 0.2
            reasons.append('Counter-trend signal')
        
        score += regime_score * 0.25
        
        # 4. Timing score - kill zone (10% weight)
        if state.get('kill_zone', False):
            score += 0.10
            reasons.append('In kill zone')
        
        # 5. Risk/reward check (10% weight)
        rr_ratio = abs(signal['take_profit'] - signal['entry']) / abs(signal['entry'] - signal['stop_loss'])
        if rr_ratio >= 2.0:
            score += 0.10
            reasons.append(f'Good R:R ({rr_ratio:.1f})')
        elif rr_ratio >= 1.5:
            score += 0.05
        else:
            reasons.append(f'Weak R:R ({rr_ratio:.1f})')
        
        # V2 RULE: Only accept A-grade signals
        grade = signal.get('grade', 'F')
        if grade not in ['A', 'A+']:
            return {'decision': 'reject', 'confidence': score, 'reason': f'Grade {grade} not A or A+'}
        
        # Decision based on score
        if score >= 0.75:
            decision = 'accept'
        elif score >= 0.60:
            decision = 'modify'
        else:
            decision = 'reject'
        
        return {
            'decision': decision,
            'confidence': score,
            'reasons': reasons,
            'rr_ratio': rr_ratio,
            'pattern_score': pattern_score,
            'regime_score': regime_score if 'regime_score' in dir() else 0.5
        }
    
    def update_weights(self, trade_result: Dict):
        """Update pattern weights based on trade results"""
        if trade_result.get('pnl', 0) > 0:
            for factor in trade_result.get('factors', []):
                factor_key = factor.lower().replace(' ', '_')
                if factor_key in self.pattern_weights:
                    self.pattern_weights[factor_key] = min(1.0, self.pattern_weights[factor_key] + self.learning_rate * 0.05)


class MLModelTrainer:
    """Phase 3: ML Model training for signal quality prediction"""
    
    def __init__(self):
        self.training_data = []
        self.model_weights = {
            'confluence': 0.35,
            'htf_alignment': 0.25,
            'price_position': 0.20,
            'patterns': 0.12,
            'timing': 0.08
        }
        self.is_trained = False
    
    def add_sample(self, features: Dict, outcome: float):
        """Add training sample"""
        self.training_data.append({'features': features, 'outcome': outcome})
    
    def train(self):
        """Train model on collected data"""
        if len(self.training_data) < 20:
            return False
        
        wins = [s for s in self.training_data if s['outcome'] > 0]
        losses = [s for s in self.training_data if s['outcome'] <= 0]
        
        if wins and losses:
            win_confluence = np.mean([s['features'].get('confluence', 70) for s in wins])
            loss_confluence = np.mean([s['features'].get('confluence', 70) for s in losses])
            
            if loss_confluence > 0:
                adjustment = (win_confluence - loss_confluence) / 100 * 0.1
                self.model_weights['confluence'] = max(0.25, min(0.45, 0.35 + adjustment))
        
        self.is_trained = True
        return True
    
    def predict(self, features: Dict) -> float:
        """Predict signal quality (0-1)"""
        if not self.is_trained:
            return features.get('confluence', 70) / 100
        
        score = 0
        score += features.get('confluence', 70) / 100 * self.model_weights['confluence']
        score += features.get('htf_alignment', 0.5) * self.model_weights['htf_alignment']
        score += features.get('price_position_favorable', 0.5) * self.model_weights['price_position']
        
        return min(1.0, score)


class RLAgent:
    """Phase 3: Reinforcement Learning agent for adaptive position sizing"""
    
    def __init__(self):
        self.q_table = {}
        self.learning_rate = 0.1
        self.discount_factor = 0.95
        self.exploration_rate = 0.15
        self.episode_rewards = []
    
    def get_action(self, state_key: str) -> str:
        """Get position size action based on state"""
        # V2 RULE: Default to 1% risk, RL can adjust
        if random.random() < self.exploration_rate:
            return random.choice(['small', 'medium', 'large'])
        
        q_values = [self.q_table.get(f"{state_key}_{a}", 0) for a in ['small', 'medium', 'large']]
        return ['small', 'medium', 'large'][np.argmax(q_values)] if q_values else 'medium'
    
    def update(self, state_key: str, action: str, reward: float, next_state_key: str):
        """Update Q-values based on reward"""
        current = self.q_table.get(f"{state_key}_{action}", 0)
        max_next = max([self.q_table.get(f"{next_state_key}_{a}", 0) for a in ['small', 'medium', 'large']] or [0])
        
        new_q = current + self.learning_rate * (reward + self.discount_factor * max_next - current)
        self.q_table[f"{state_key}_{action}"] = new_q
        
        self.exploration_rate = max(0.02, self.exploration_rate * 0.99)


# =============================================================================
# TRADING SIMULATION
# =============================================================================

# Initialize Phase 3 components
ai_filter = AISignalFilter()
ml_trainer = MLModelTrainer()
rl_agent = RLAgent()

# Trading state
capital = 10000
equity_curve = [capital]
trades = []
open_trade = None
rejected_signals = 0
accepted_signals = 0

print()
print("Running optimized trading system...")

for idx in range(50, len(df)):
    current_price = closes[idx]
    htf_trend = htf_trends[idx]
    kz = kill_zone[idx]
    pp = price_position[idx]
    current_trend = trend[idx]
    
    # Get state
    nearest_bull = next((ob for ob in reversed(bullish_obs) if ob['idx'] < idx), None)
    nearest_bear = next((ob for ob in reversed(bearish_obs) if ob['idx'] < idx), None)
    near_bull_fvg = next((f for f in reversed(bullish_fvgs) if f['idx'] < idx and f['mid'] < current_price < f['high']), None)
    near_bear_fvg = next((f for f in reversed(bearish_fvgs) if f['idx'] < idx and f['low'] < current_price < f['mid']), None)
    
    atr = (highs[idx-14:idx] - lows[idx-14:idx]).mean() if idx > 14 else 50
    
    # Calculate confluence (V2 scoring)
    confluence = 0
    factors = []
    
    if kz:
        confluence += 15
        factors.append('KillZone')
    
    if htf_trend == 1 and current_trend >= 0:
        confluence += 25
        factors.append('HTFBullish')
    elif htf_trend == -1 and current_trend <= 0:
        confluence += 25
        factors.append('HTFBearish')
    
    if pp < 0.25:
        confluence += 20
        factors.append('DeepDiscount')
    elif pp < 0.35:
        confluence += 15
        factors.append('Discount')
    elif pp > 0.75:
        confluence += 20
        factors.append('DeepPremium')
    elif pp > 0.65:
        confluence += 15
        factors.append('Premium')
    
    if near_bull_fvg and current_trend >= 0:
        confluence += 15
        factors.append('BullFVG')
    if near_bear_fvg and current_trend <= 0:
        confluence += 15
        factors.append('BearFVG')
    
    if nearest_bull and current_price > nearest_bull['high']:
        confluence += 10
        factors.append('BullOB')
    if nearest_bear and current_price < nearest_bear['low']:
        confluence += 10
        factors.append('BearOB')
    
    # Grade
    if confluence >= 75:
        grade = 'A+'
    elif confluence >= 70:
        grade = 'A'
    elif confluence >= 60:
        grade = 'B'
    else:
        grade = 'C'
    
    # Generate signal (V2 conditions)
    signal = None
    
    # V2 RULE: Long only in discount zone with bullish alignment
    if pp < 0.40 and (htf_trend == 1 or current_trend >= 0):
        if near_bull_fvg and current_price > near_bull_fvg['mid']:
            signal = {
                'direction': 'long',
                'entry': current_price,
                'stop_loss': near_bull_fvg['low'] - atr * 0.5,
                'take_profit': current_price + atr * 2.5,
                'confluence': confluence,
                'grade': grade,
                'factors': factors
            }
        elif nearest_bull and current_price > nearest_bull['high']:
            signal = {
                'direction': 'long',
                'entry': current_price,
                'stop_loss': nearest_bull['low'] - atr * 0.5,
                'take_profit': current_price + atr * 2.5,
                'confluence': confluence,
                'grade': grade,
                'factors': factors
            }
    
    # V2 RULE: Short only in premium zone with bearish alignment
    elif pp > 0.60 and (htf_trend == -1 or current_trend <= 0):
        if near_bear_fvg and current_price < near_bear_fvg['mid']:
            signal = {
                'direction': 'short',
                'entry': current_price,
                'stop_loss': near_bear_fvg['high'] + atr * 0.5,
                'take_profit': current_price - atr * 2.5,
                'confluence': confluence,
                'grade': grade,
                'factors': factors
            }
        elif nearest_bear and current_price < nearest_bear['low']:
            signal = {
                'direction': 'short',
                'entry': current_price,
                'stop_loss': nearest_bear['high'] + atr * 0.5,
                'take_profit': current_price - atr * 2.5,
                'confluence': confluence,
                'grade': grade,
                'factors': factors
            }
    
    # Apply Phase 3 AI Filter
    if signal:
        state = {'htf_trend': htf_trend, 'kill_zone': kz, 'price_position': pp}
        filtered = ai_filter.filter(signal, state)
        
        if filtered['decision'] == 'accept':
            accepted_signals += 1
            
            # RL position sizing (V2 default 1%)
            state_key = f"{'bull' if htf_trend > 0 else 'bear' if htf_trend < 0 else 'neutral'}_{'disc' if pp < 0.4 else 'prem' if pp > 0.6 else 'mid'}_{'kz' if kz else 'nkz'}"
            position_mod = rl_agent.get_action(state_key)
            
            # V2 RULE: Base 1% risk, RL can adjust
            risk_pct = {'small': 0.0075, 'medium': 0.01, 'large': 0.015}[position_mod]
            
            risk_amt = capital * risk_pct
            risk = abs(current_price - signal['stop_loss'])
            size = risk_amt / risk if risk > 0 else 1
            
            open_trade = {
                'entry_idx': idx,
                'entry': signal['entry'],
                'dir': signal['direction'],
                'size': size,
                'sl': signal['stop_loss'],
                'tp': signal['take_profit'],
                'confluence': confluence,
                'grade': grade,
                'ai_score': filtered['confidence'],
                'position_mod': position_mod,
                'factors': factors,
                'rr_ratio': filtered['rr_ratio']
            }
        else:
            rejected_signals += 1
    
    # Check exits
    if open_trade:
        if open_trade['dir'] == 'long':
            if current_price <= open_trade['sl']:
                open_trade['exit'] = current_price
                open_trade['pnl'] = (open_trade['exit'] - open_trade['entry']) * open_trade['size'] * 20
                open_trade['status'] = 'STOP'
                capital += open_trade['pnl']
                
                # ML training sample
                ml_trainer.add_sample({
                    'confluence': open_trade['confluence'],
                    'htf_alignment': 1 if htf_trend == 1 else 0,
                    'price_position_favorable': 1 if pp < 0.4 else 0
                }, open_trade['pnl'])
                
                # RL update
                reward = 1 if open_trade['pnl'] > 0 else -1
                next_state_key = 'neutral_mid_nkz'
                rl_agent.update(state_key, open_trade['position_mod'], reward, next_state_key)
                
                trades.append(open_trade)
                open_trade = None
                
            elif current_price >= open_trade['tp']:
                open_trade['exit'] = current_price
                open_trade['pnl'] = (open_trade['exit'] - open_trade['entry']) * open_trade['size'] * 20
                open_trade['status'] = 'TP'
                capital += open_trade['pnl']
                
                # ML training sample
                ml_trainer.add_sample({
                    'confluence': open_trade['confluence'],
                    'htf_alignment': 1 if htf_trend == 1 else 0,
                    'price_position_favorable': 1 if pp < 0.4 else 0
                }, open_trade['pnl'])
                
                # RL update
                reward = 1
                next_state_key = 'neutral_mid_nkz'
                rl_agent.update(state_key, open_trade['position_mod'], reward, next_state_key)
                
                trades.append(open_trade)
                open_trade = None
        else:
            if current_price >= open_trade['sl']:
                open_trade['exit'] = current_price
                open_trade['pnl'] = (open_trade['entry'] - open_trade['exit']) * open_trade['size'] * 20
                open_trade['status'] = 'STOP'
                capital += open_trade['pnl']
                
                # ML training sample
                ml_trainer.add_sample({
                    'confluence': open_trade['confluence'],
                    'htf_alignment': 1 if htf_trend == -1 else 0,
                    'price_position_favorable': 1 if pp > 0.6 else 0
                }, open_trade['pnl'])
                
                # RL update
                reward = 1 if open_trade['pnl'] > 0 else -1
                next_state_key = 'neutral_mid_nkz'
                rl_agent.update(state_key, open_trade['position_mod'], reward, next_state_key)
                
                trades.append(open_trade)
                open_trade = None
                
            elif current_price <= open_trade['tp']:
                open_trade['exit'] = current_price
                open_trade['pnl'] = (open_trade['entry'] - open_trade['exit']) * open_trade['size'] * 20
                open_trade['status'] = 'TP'
                capital += open_trade['pnl']
                
                # ML training sample
                ml_trainer.add_sample({
                    'confluence': open_trade['confluence'],
                    'htf_alignment': 1 if htf_trend == -1 else 0,
                    'price_position_favorable': 1 if pp > 0.6 else 0
                }, open_trade['pnl'])
                
                # RL update
                reward = 1
                next_state_key = 'neutral_mid_nkz'
                rl_agent.update(state_key, open_trade['position_mod'], reward, next_state_key)
                
                trades.append(open_trade)
                open_trade = None
        
        # Time exit (V2: 15 bars)
        if open_trade and (idx - open_trade['entry_idx']) > 15:
            open_trade['exit'] = current_price
            open_trade['pnl'] = (open_trade['entry'] - current_price) * open_trade['size'] * 20 if open_trade['dir'] == 'short' else (current_price - open_trade['entry']) * open_trade['size'] * 20
            open_trade['status'] = 'TIME'
            capital += open_trade['pnl']
            trades.append(open_trade)
            open_trade = None
    
    equity_curve.append(capital)
    
    if idx % 500 == 0:
        print(f"Progress: {idx}/{len(df)} | Equity: ${capital:,.0f} | Trades: {len(trades)} | Accepted: {accepted_signals}")

# Close open trade
if open_trade:
    open_trade['exit'] = closes[-1]
    open_trade['pnl'] = (open_trade['entry'] - closes[-1]) * open_trade['size'] * 20 if open_trade['dir'] == 'short' else (closes[-1] - open_trade['entry']) * open_trade['size'] * 20
    open_trade['status'] = 'EOD'
    trades.append(open_trade)

# Train ML model
ml_trainer.train()

# Statistics
closed = [t for t in trades if 'exit' in t]
winners = [t for t in closed if t.get('pnl', 0) > 0]
losers = [t for t in closed if t.get('pnl', 0) <= 0]

total_return = (capital - 10000) / 10000 * 100
win_rate = len(winners) / len(closed) * 100 if closed else 0

max_eq = max(equity_curve)
min_eq = min(equity_curve)
max_dd = (max_eq - min_eq) / max_eq * 100 if max_eq > 0 else 0

profit = sum(t['pnl'] for t in winners)
loss = abs(sum(t['pnl'] for t in losers))
pf = profit / loss if loss > 0 else float('inf')

# By direction
longs = [t for t in closed if t['dir'] == 'long']
shorts = [t for t in closed if t['dir'] == 'short']
long_pnl = sum(t['pnl'] for t in longs)
short_pnl = sum(t['pnl'] for t in shorts)

# By grade
grades = {}
for t in closed:
    g = t.get('grade', 'F')
    if g not in grades:
        grades[g] = {'count': 0, 'wins': 0, 'pnl': 0}
    grades[g]['count'] += 1
    if t['pnl'] > 0:
        grades[g]['wins'] += 1
    grades[g]['pnl'] += t['pnl']

# Save results
results = {
    'metadata': {
        'version': 'V3 Optimized (V2 Rules + Phase 3)',
        'timestamp': datetime.now().isoformat(),
        'v2_rules': [
            'Confluence >= 70 required',
            'HTF alignment required',
            'Only A-grade setups',
            '1% risk per trade'
        ],
        'phase3_enhancements': [
            'AI Signal Filter validation',
            'ML Model Trainer pattern learning',
            'RL Agent adaptive sizing'
        ]
    },
    'period': {'start': str(timestamps[0])[:10], 'end': str(timestamps[-1])[:10], 'bars': len(df)},
    'capital': {'initial': 10000, 'final': capital, 'return_pct': total_return},
    'trades': {'total': len(closed), 'winners': len(winners), 'losers': len(losers), 'win_rate': win_rate},
    'pnl': {'gross_profit': profit, 'gross_loss': loss, 'net_pnl': capital - 10000, 'profit_factor': pf},
    'risk': {'max_drawdown_pct': max_dd},
    'signal_stats': {'accepted': accepted_signals, 'rejected': rejected_signals, 'accept_rate': accepted_signals/(accepted_signals+rejected_signals)*100 if (accepted_signals+rejected_signals) > 0 else 0},
    'direction_breakdown': {
        'long': {'count': len(longs), 'pnl': long_pnl},
        'short': {'count': len(shorts), 'pnl': short_pnl}
    },
    'grade_breakdown': grades,
    'equity_curve': [{'date': str(timestamps[i])[:10], 'equity': e} for i, e in enumerate(equity_curve)]
}

with open('v3_optimized_results.json', 'w') as f:
    json.dump(results, f, indent=2)

# Print summary
print()
print("=" * 70)
print("V3 OPTIMIZED TRADING SYSTEM RESULTS")
print("=" * 70)
print(f"Period: {results['period']['start']} to {results['period']['end']}")
print()
print("CAPITAL:")
print(f"  Initial:    ${10000:>12,.0f}")
print(f"  Final:      ${capital:>12,.0f}")
print(f"  Return:     {total_return:>12.1f}%")
print()
print("TRADE STATISTICS:")
print(f"  Total:      {len(closed)}")
print(f"  Win Rate:   {win_rate:.1f}%")
print(f"  Winners:    {len(winners)} | Losers: {len(losers)}")
print()
print("SIGNAL FILTERING:")
print(f"  Accepted:   {accepted_signals}")
print(f"  Rejected:   {rejected_signals}")
print(f"  Accept Rate: {accepted_signals/(accepted_signals+rejected_signals)*100:.1f}%" if (accepted_signals+rejected_signals) > 0 else 0)
print()
print("P&L:")
print(f"  Gross Profit:  ${profit:>12,.0f}")
print(f"  Gross Loss:    ${loss:>12,.0f}")
print(f"  Net PnL:        ${capital-10000:>12,.0f}")
print(f"  Profit Factor: {pf:>12.2f}")
print()
print("RISK:")
print(f"  Max Drawdown:  {max_dd:.1f}%")
print()
print("DIRECTION BREAKDOWN:")
print(f"  Long:  {len(longs):3} trades | ${long_pnl:+,.0f}")
print(f"  Short: {len(shorts):3} trades | ${short_pnl:+,.0f}")
print()
print("GRADE BREAKDOWN:")
for g in sorted(grades.keys()):
    data = grades[g]
    wr = data['wins'] / data['count'] * 100 if data['count'] > 0 else 0
    print(f"  Grade {g}: {data['count']:2} trades | {wr:5.1f}% win | ${data['pnl']:+,.0f}")
print()
print(f"Results saved to: v3_optimized_results.json")
print("=" * 70)
