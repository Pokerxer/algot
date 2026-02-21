"""
ICT V8 - With RL Agent for Entry/Exit Timing
=============================================
Combines V7 PD Array analysis + FVG + RL agent for optimal timing
"""

import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import sys
sys.path.insert(0, '.')

import json
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional
import importlib.util

# Import V5 base
spec = importlib.util.spec_from_file_location("ict_v5", "./ict_v5_ibkr.py")
ict_v5 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ict_v5)

fetch_ibkr_data = ict_v5.fetch_ibkr_data
prepare_data_ibkr = ict_v5.prepare_data_ibkr
get_signal = ict_v5.get_signal
calculate_position_size = ict_v5.calculate_position_size
get_contract_info = ict_v5.get_contract_info
get_ibkr_contract = ict_v5.get_ibkr_contract

# Import RL Agent
from reinforcement_learning_agent import (
    ICTReinforcementLearningAgent,
    TrainingConfig,
    RLState,
    MarketState,
    PositionState,
    MarketRegime,
    SessionType,
    EntryAction,
    ExitAction,
    PositionStatus,
    DQNAgent,
    Experience,
    RewardCalculator
)


class V8SignalGenerator:
    """V7 Signal Generator with PD Array analysis + RL Agent"""
    
    def __init__(self, use_rl=True):
        self.min_fvg_size = 0.0
        self.use_rl = use_rl
        
        # Initialize RL agent if enabled
        if use_rl:
            config = TrainingConfig(
                state_size_entry=31,
                state_size_exit=40,
                reward_scale=1.0,
                penalty_scale=1.0,
                epsilon_start=0.1,  # Less exploration in backtest
                epsilon_end=0.01,
                epsilon_decay=0.995
            )
            self.rl_agent = ICTReinforcementLearningAgent(config)
        else:
            self.rl_agent = None
    
    def calculate_daily_quadrants(self, highs, lows, lookback=24):
        """Calculate daily range quadrants (ICT: Grade everything)"""
        if len(highs) < lookback:
            lookback = len(highs)
        
        recent_highs = highs[-lookback:]
        recent_lows = lows[-lookback:]
        
        daily_high = np.max(recent_highs)
        daily_low = np.min(recent_lows)
        range_size = daily_high - daily_low
        
        return {
            'high': daily_high,
            'low': daily_low,
            'ce': daily_low + (range_size * 0.5),
            'upper_quad': daily_low + (range_size * 0.75),
            'lower_quad': daily_low + (range_size * 0.25),
            'ote_high': daily_low + (range_size * 0.79),
            'ote_low': daily_low + (range_size * 0.62),
            'range_size': range_size
        }
    
    def get_pd_zone(self, price, quadrants):
        """Determine Premium/Discount zone for price"""
        if price >= quadrants['upper_quad']:
            return 'extreme_premium'
        elif price >= quadrants['ce']:
            return 'premium'
        elif price <= quadrants['lower_quad']:
            return 'extreme_discount'
        elif price <= quadrants['ce']:
            return 'discount'
        else:
            return 'equilibrium'
    
    def is_in_ote(self, price, quadrants, direction):
        """Check if price is in Optimal Trade Entry zone (62-79%)"""
        if direction == 1:
            return quadrants['ote_low'] <= price <= quadrants['ote_high']
        else:
            ote_short_high = quadrants['high'] - (quadrants['range_size'] * 0.62)
            ote_short_low = quadrants['high'] - (quadrants['range_size'] * 0.79)
            return ote_short_low <= price <= ote_short_high
    
    def detect_fvgs_fast(self, highs, lows, closes, lookback=20):
        """Fast FVG detection"""
        fvgs = []
        n = len(closes)
        start = max(0, n - lookback)
        
        for i in range(max(2, start), n):
            if lows[i] > highs[i-2]:
                gap_size = lows[i] - highs[i-2]
                if gap_size >= self.min_fvg_size:
                    fvgs.append({
                        'type': 'bullish',
                        'high': lows[i],
                        'low': highs[i-2],
                        'ce': (lows[i] + highs[i-2]) / 2,
                        'idx': i
                    })
            
            if highs[i] < lows[i-2]:
                gap_size = lows[i-2] - highs[i]
                if gap_size >= self.min_fvg_size:
                    fvgs.append({
                        'type': 'bearish',
                        'high': lows[i-2],
                        'low': highs[i],
                        'ce': (lows[i-2] + highs[i]) / 2,
                        'idx': i
                    })
        
        return fvgs
    
    def find_liquidity_levels(self, highs, lows, window=5):
        """Find equal highs/lows"""
        n = len(highs)
        if n < window * 2:
            return [], []
        
        buy_side = []
        sell_side = []
        tolerance = 0.001
        
        for i in range(window, n - window):
            recent_highs = highs[i-window:i+window]
            max_high = np.max(recent_highs)
            touches = np.sum(np.abs(recent_highs - max_high) / max_high < tolerance)
            if touches >= 2:
                buy_side.append(max_high)
            
            recent_lows = lows[i-window:i+window]
            min_low = np.min(recent_lows)
            touches = np.sum(np.abs(recent_lows - min_low) / min_low < tolerance)
            if touches >= 2:
                sell_side.append(min_low)
        
        return list(set(buy_side))[:3], list(set(sell_side))[:3]
    
    def build_rl_state(self, data: Dict, idx: int, signal: Dict, 
                       current_price: float, quadrants: Dict, 
                       position: Optional[Dict] = None) -> RLState:
        """Build RL state from current market data"""
        
        # Calculate momentum
        if idx >= 5:
            momentum_5 = (data['closes'][idx] - data['closes'][idx-5]) / data['closes'][idx-5]
        else:
            momentum_5 = 0.0
        
        if idx >= 20:
            momentum_20 = (data['closes'][idx] - data['closes'][idx-20]) / data['closes'][idx-20]
        else:
            momentum_20 = 0.0
        
        # Calculate ATR
        if idx >= 14:
            trs = []
            for j in range(idx - 13, idx + 1):
                high_low = data['highs'][j] - data['lows'][j]
                high_close = abs(data['highs'][j] - data['closes'][j-1]) if j > 0 else high_low
                low_close = abs(data['lows'][j] - data['closes'][j-1]) if j > 0 else high_low
                trs.append(max(high_low, high_close, low_close))
            atr = np.mean(trs)
        else:
            atr = (data['highs'][idx] - data['lows'][idx]) * 0.5
        
        # Determine market regime
        if abs(momentum_20) > 0.02:
            regime = MarketRegime.TRENDING_BULL if momentum_20 > 0 else MarketRegime.TRENDING_BEAR
        else:
            regime = MarketRegime.RANGING
        
        # PD zone
        pd_zone = self.get_pd_zone(current_price, quadrants)
        in_ote = self.is_in_ote(current_price, quadrants, signal['direction'])
        
        # Kill zone (simplified - assume NY session)
        hour = 14  # Assume NY session for simplicity
        in_kill_zone = 8 <= hour <= 12 or 13 <= hour <= 16
        
        # Range position (0-1)
        range_size = quadrants['high'] - quadrants['low']
        if range_size > 0:
            range_position = (current_price - quadrants['low']) / range_size
        else:
            range_position = 0.5
        
        market_state = MarketState(
            current_price=current_price,
            momentum_5=momentum_5,
            momentum_20=momentum_20,
            volatility=atr / current_price if current_price > 0 else 0,
            atr=atr,
            session_high=quadrants['high'],
            session_low=quadrants['low'],
            daily_high=quadrants['high'],
            daily_low=quadrants['low'],
            range_position=range_position,
            rsi=50.0,
            macd=0.0,
            macd_signal=0.0,
            in_premium=pd_zone in ['premium', 'extreme_premium'],
            in_discount=pd_zone in ['discount', 'extreme_discount'],
            premium_discount_depth=abs(range_position - 0.5) * 2,
            in_ote_zone=in_ote,
            trend="bullish" if signal['direction'] == 1 else "bearish" if signal['direction'] == -1 else "ranging",
            session=SessionType.NEW_YORK,
            in_kill_zone=in_kill_zone,
            regime=regime,
            signal_quality=signal['confluence'],
            signal_confluence=int(signal['confluence'])
        )
        
        if position:
            entry_price = position['entry']
            direction = position['direction']
            unrealized_pnl = (current_price - entry_price) * direction
            unrealized_pnl_r = unrealized_pnl / position.get('stop_distance', 1)
            
            position_state = PositionState(
                status=PositionStatus.LONG if direction == 1 else PositionStatus.SHORT,
                direction="long" if direction == 1 else "short",
                entry_price=entry_price,
                current_price=current_price,
                stop_loss=position['stop'],
                take_profit=position['target'],
                position_size=position['qty'],
                unrealized_pnl=unrealized_pnl,
                unrealized_pnl_r=unrealized_pnl_r,
                bars_held=position.get('bars_held', 0),
                remaining_size=1.0
            )
        else:
            position_state = PositionState(
                status=PositionStatus.FLAT,
                direction="",
                entry_price=current_price,
                current_price=current_price,
                stop_loss=0,
                take_profit=0,
                position_size=0,
                unrealized_pnl=0,
                unrealized_pnl_r=0,
                bars_held=0,
                remaining_size=1.0
            )
        
        return RLState(market=market_state, position=position_state)
    
    def generate_signal(self, data: Dict, idx: int) -> Optional[Dict]:
        """Generate V8 signal with PD Array analysis + RL timing"""
        
        v5_signal = get_signal(data, idx)
        if not v5_signal:
            return None
        
        if v5_signal['confluence'] < 50:
            return None
        
        current_price = data['closes'][idx]
        
        # Calculate daily quadrants
        quadrants = self.calculate_daily_quadrants(
            data['highs'][:idx+1],
            data['lows'][:idx+1],
            lookback=24
        )
        
        # Get PD zone
        pd_zone = self.get_pd_zone(current_price, quadrants)
        
        pd_boost = 0
        pd_valid = False
        
        if v5_signal['direction'] == 1:
            if pd_zone in ['discount', 'extreme_discount']:
                pd_boost = 20
                pd_valid = True
                if self.is_in_ote(current_price, quadrants, 1):
                    pd_boost += 10
            elif pd_zone == 'equilibrium':
                pd_boost = 5
                pd_valid = True
            else:
                pd_boost = -10
                pd_valid = False
        else:
            if pd_zone in ['premium', 'extreme_premium']:
                pd_boost = 20
                pd_valid = True
                if self.is_in_ote(current_price, quadrants, -1):
                    pd_boost += 10
            elif pd_zone == 'equilibrium':
                pd_boost = 5
                pd_valid = True
            else:
                pd_boost = -10
                pd_valid = False
        
        # Fast FVG detection
        fvgs = self.detect_fvgs_fast(
            data['highs'][:idx+1],
            data['lows'][:idx+1],
            data['closes'][:idx+1],
            lookback=20
        )
        
        fvg_boost = 0
        fvg_info = None
        for fvg in reversed(fvgs):
            if fvg['type'] == 'bullish' and v5_signal['direction'] == 1:
                if abs(current_price - fvg['ce']) < fvg['high'] - fvg['low']:
                    fvg_boost = 10
                    fvg_info = f"BISI@{fvg['ce']:.2f}"
                    break
            elif fvg['type'] == 'bearish' and v5_signal['direction'] == -1:
                if abs(current_price - fvg['ce']) < fvg['high'] - fvg['low']:
                    fvg_boost = 10
                    fvg_info = f"SIBI@{fvg['ce']:.2f}"
                    break
        
        # Find liquidity levels
        buy_side, sell_side = self.find_liquidity_levels(
            data['highs'][:idx+1],
            data['lows'][:idx+1],
            window=5
        )
        
        liquidity_info = None
        if v5_signal['direction'] == 1 and sell_side:
            for level in sell_side:
                if abs(current_price - level) / level < 0.005:
                    liquidity_info = f"SSL@{level:.2f}"
                    fvg_boost += 5
                    break
        elif v5_signal['direction'] == -1 and buy_side:
            for level in buy_side:
                if abs(current_price - level) / level < 0.005:
                    liquidity_info = f"BSL@{level:.2f}"
                    fvg_boost += 5
                    break
        
        # Combine confluence
        total_confluence = v5_signal['confluence'] + pd_boost + fvg_boost
        total_confluence = max(0, min(total_confluence, 100))
        
        if total_confluence >= 75 and pd_valid:
            confidence = 'HIGH'
        elif total_confluence >= 60:
            confidence = 'MEDIUM'
        elif total_confluence >= 45:
            confidence = 'LOW'
        else:
            confidence = 'LOW'
            if not pd_valid:
                return None
        
        # Get RL entry action if enabled
        rl_entry_action = None
        rl_action_info = None
        if self.use_rl and self.rl_agent:
            signal_dict = {
                'direction': v5_signal['direction'],
                'confluence': total_confluence,
                'model': 'ict_standard'
            }
            
            rl_state = self.build_rl_state(data, idx, {
                'direction': v5_signal['direction'],
                'confluence': total_confluence,
                'fvg_boost': fvg_boost,
                'liquidity_info': liquidity_info
            }, current_price, quadrants)
            
            try:
                rl_entry_action, rl_action_info = self.rl_agent.select_entry_action(
                    rl_state, signal_dict, training=False
                )
            except:
                rl_entry_action = EntryAction.ENTER_NOW
                rl_action_info = {}
        
        return {
            'direction': v5_signal['direction'],
            'confluence': total_confluence,
            'v5_confluence': v5_signal['confluence'],
            'pd_boost': pd_boost,
            'fvg_boost': fvg_boost,
            'confidence': confidence,
            'pd_zone': pd_zone,
            'daily_ce': quadrants['ce'],
            'fvg_info': fvg_info,
            'liquidity_info': liquidity_info,
            'in_ote': self.is_in_ote(current_price, quadrants, v5_signal['direction']),
            'rl_entry_action': rl_entry_action.name if rl_entry_action else 'ENTER_NOW',
            'rl_action_info': rl_action_info
        }
    
    def evaluate_exit(self, data: Dict, idx: int, position: Dict) -> Optional[Dict]:
        """Use RL agent to evaluate if we should exit/manage position"""
        
        if not self.use_rl or not self.rl_agent:
            return None
        
        current_price = data['closes'][idx]
        
        quadrants = self.calculate_daily_quadrants(
            data['highs'][:idx+1],
            data['lows'][:idx+1],
            lookback=24
        )
        
        signal_dict = {'direction': position['direction'], 'confluence': 70}
        
        rl_state = self.build_rl_state(
            data, idx, signal_dict, current_price, quadrants, position
        )
        
        try:
            rl_exit_action, rl_action_info = self.rl_agent.select_exit_action(
                rl_state, training=False
            )
            return {
                'action': rl_exit_action.name,
                'action_info': rl_action_info,
                'unrealized_pnl_r': rl_state.position.unrealized_pnl_r
            }
        except:
            return None


def train_rl_agent(symbols, training_episodes=500, config=None):
    """Train RL agent on historical market data"""
    
    print(f"\n{'='*80}")
    print(f"RL Agent Training")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Episodes: {training_episodes}")
    print(f"{'='*80}\n")
    
    if config is None:
        config = TrainingConfig(
            state_size_entry=31,
            state_size_exit=40,
            reward_scale=1.0,
            penalty_scale=1.0,
            epsilon_start=1.0,
            epsilon_end=0.02,
            epsilon_decay=0.995,
            batch_size=64,
            min_experiences=100,
            buffer_size=50000
        )
    
    agent = ICTReinforcementLearningAgent(config)
    reward_calc = RewardCalculator(reward_scale=1.0, penalty_scale=1.0)
    
    # Load historical data
    all_data = {}
    for symbol in symbols:
        print(f"Loading {symbol} for training...", end=' ')
        data = prepare_data_ibkr(symbol)
        if data and len(data.get('closes', [])) >= 100:
            all_data[symbol] = data
            print(f"✓ {len(data['closes'])} bars")
        else:
            print(f"✗")
    
    if not all_data:
        print("No data loaded for training!")
        return agent
    
    # Generate training episodes from historical data
    print(f"\nGenerating {training_episodes} training episodes...")
    
    episode_rewards = []
    for episode in range(training_episodes):
        # Randomly select a symbol and time period
        symbol = np.random.choice(list(all_data.keys()))
        data = all_data[symbol]
        n = len(data['closes'])
        
        if n < 100:
            continue
        
        # Random start index
        start_idx = np.random.randint(50, n - 50)
        
        # Generate market state from historical data
        idx = start_idx
        current_price = data['closes'][idx]
        
        # Calculate features
        if idx >= 5:
            momentum_5 = (data['closes'][idx] - data['closes'][idx-5]) / data['closes'][idx-5]
        else:
            momentum_5 = 0.0
        
        if idx >= 20:
            momentum_20 = (data['closes'][idx] - data['closes'][idx-20]) / data['closes'][idx-20]
        else:
            momentum_20 = 0.0
        
        # ATR
        if idx >= 14:
            trs = []
            for j in range(idx - 13, idx + 1):
                if j > 0:
                    high_low = data['highs'][j] - data['lows'][j]
                    high_close = abs(data['highs'][j] - data['closes'][j-1])
                    low_close = abs(data['lows'][j] - data['closes'][j-1])
                    trs.append(max(high_low, high_close, low_close))
            atr = np.mean(trs) if trs else 0
        else:
            atr = (data['highs'][idx] - data['lows'][idx]) * 0.5
        
        # Daily quadrants
        highs = data['highs'][:idx+1]
        lows = data['lows'][:idx+1]
        daily_high = np.max(highs[-24:]) if len(highs) >= 24 else np.max(highs)
        daily_low = np.min(lows[-24:]) if len(lows) >= 24 else np.min(lows)
        range_size = daily_high - daily_low
        range_pos = (current_price - daily_low) / range_size if range_size > 0 else 0.5
        
        # Regime
        if abs(momentum_20) > 0.02:
            regime = MarketRegime.TRENDING_BULL if momentum_20 > 0 else MarketRegime.TRENDING_BEAR
        else:
            regime = MarketRegime.RANGING
        
        # PD zone
        if range_pos >= 0.75:
            pd_zone = 'premium'
        elif range_pos >= 0.5:
            pd_zone = 'equilibrium_above'
        elif range_pos >= 0.25:
            pd_zone = 'equilibrium_below'
        else:
            pd_zone = 'discount'
        
        in_ote = 0.25 <= range_pos <= 0.79
        
        # Random direction (simulate signal)
        direction = np.random.choice([1, -1])
        
        # Create market state
        market_state = MarketState(
            current_price=float(current_price),
            momentum_5=float(momentum_5),
            momentum_20=float(momentum_20),
            volatility=float(atr / current_price) if current_price > 0 else 0,
            atr=float(atr),
            session_high=float(daily_high),
            session_low=float(daily_low),
            daily_high=float(daily_high),
            daily_low=float(daily_low),
            range_position=float(range_pos),
            rsi=50.0,
            in_premium=pd_zone in ['premium'],
            in_discount=pd_zone in ['discount'],
            premium_discount_depth=abs(range_pos - 0.5) * 2,
            in_ote_zone=in_ote,
            trend="bullish" if momentum_20 > 0.01 else "bearish" if momentum_20 < -0.01 else "ranging",
            session=SessionType.NEW_YORK,
            in_kill_zone=True,
            regime=regime,
            signal_quality=np.random.uniform(40, 90),
            signal_confluence=int(np.random.uniform(40, 90))
        )
        
        # Create position state (sometimes in position, sometimes flat)
        if np.random.random() < 0.3:  # 30% chance of being in position
            entry_price = current_price * (1 - 0.002 * direction)
            stop_distance = abs(current_price - entry_price)
            unrealized_pnl = (current_price - entry_price) * direction
            unrealized_pnl_r = unrealized_pnl / stop_distance if stop_distance > 0 else 0
            
            position_state = PositionState(
                status=PositionStatus.LONG if direction == 1 else PositionStatus.SHORT,
                direction="long" if direction == 1 else "short",
                entry_price=float(entry_price),
                current_price=float(current_price),
                stop_loss=float(current_price - stop_distance * direction),
                take_profit=float(current_price + stop_distance * 2 * direction),
                position_size=1.0,
                unrealized_pnl=float(unrealized_pnl),
                unrealized_pnl_r=float(unrealized_pnl_r),
                bars_held=np.random.randint(1, 20),
                remaining_size=1.0
            )
        else:
            position_state = PositionState(
                status=PositionStatus.FLAT,
                direction="",
                entry_price=float(current_price),
                current_price=float(current_price),
                stop_loss=0,
                take_profit=0,
                position_size=0,
                unrealized_pnl=0,
                unrealized_pnl_r=0,
                bars_held=0,
                remaining_size=1.0
            )
        
        state = RLState(market=market_state, position=position_state)
        
        # Get entry action
        state_vector = state.to_entry_vector()
        action_idx = agent.entry_agent.select_action(state_vector, training=True)
        entry_action = EntryAction(action_idx)
        
        # Get exit action (if in position)
        if position_state.status != PositionStatus.FLAT:
            exit_vector = state.to_exit_vector()
            exit_idx = agent.exit_agent.select_action(exit_vector, training=True)
            exit_action = ExitAction(exit_idx)
        else:
            exit_action = ExitAction.HOLD
        
        # Simulate reward based on market outcome
        if position_state.status != PositionStatus.FLAT:
            # In position - reward based on P/L
            pnl_r = position_state.unrealized_pnl_r
            if pnl_r > 0:
                reward = 0.2 * pnl_r
            else:
                reward = -0.1 * abs(pnl_r)
        else:
            # Flat - reward based on signal quality and action taken
            if entry_action == EntryAction.PASS:
                reward = -0.1  # Penalty for missing good signal
            elif entry_action == EntryAction.ENTER_NOW:
                reward = 0.1  # Good timing
            else:
                reward = 0.0  # Waited for better entry
        
        # Store experience
        next_state_vector = state_vector + np.random.randn(len(state_vector)) * 0.01
        experience = Experience(
            state=state_vector,
            action=action_idx,
            reward=reward,
            next_state=next_state_vector,
            done=False
        )
        agent.entry_agent.replay_buffer.add(experience)
        
        # Train if enough experiences
        if len(agent.entry_agent.replay_buffer) >= config.min_experiences:
            agent.entry_agent.train_step()
        
        episode_rewards.append(reward)
        
        if episode % 100 == 0:
            recent_avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards)
            print(f"  Episode {episode}/{training_episodes} | Avg Reward: {recent_avg:.3f} | ε: {agent.entry_agent.epsilon:.3f}")
    
    # Final training pass
    print("\nFinal training pass...")
    for _ in range(50):
        if len(agent.entry_agent.replay_buffer) >= config.batch_size:
            agent.entry_agent.train_step()
    
    print(f"\nTraining complete!")
    print(f"Final ε: {agent.entry_agent.epsilon:.3f}")
    print(f"Total experiences: {len(agent.entry_agent.replay_buffer)}")
    
    return agent


def run_v8_backtest(symbols, days=30, initial_capital=50000, risk_per_trade=0.02, use_rl=True, train_first=True, training_episodes=500):
    """Run V8 backtest with PD Array + FVG + RL Agent"""
    
    print(f"\n{'='*80}")
    print(f"V8 Backtest - PD Arrays + FVG + RL Agent")
    print(f"Symbols: {', '.join(symbols)}")
    print(f"Capital: ${initial_capital:,} | Risk: {risk_per_trade*100}%")
    print(f"RL Agent: {'Enabled' if use_rl else 'Disabled'}")
    print(f"Train First: {'Yes' if train_first else 'No'} ({training_episodes} episodes)")
    print(f"{'='*80}\n")
    
    # Train RL agent first if enabled
    trained_agent = None
    if use_rl and train_first:
        trained_agent = train_rl_agent(symbols, training_episodes=training_episodes)
    
    signal_gen = V8SignalGenerator(use_rl=use_rl)
    
    # If we trained an agent, use it
    if trained_agent is not None:
        signal_gen.rl_agent = trained_agent
    
    # Load data
    all_data = {}
    for symbol in symbols:
        print(f"Loading {symbol}...", end=' ')
        data = prepare_data_ibkr(symbol)
        if data and len(data.get('closes', [])) >= 50:
            all_data[symbol] = data
            print(f"✓ {len(data['closes'])} bars")
        else:
            print(f"✗")
    
    if not all_data:
        print("No data loaded!")
        return None
    
    # Get all timestamps
    all_timestamps = sorted(set().union(*[set(data['df'].index) for data in all_data.values()]))
    print(f"\nProcessing {len(all_timestamps)} timestamps...")
    
    # Trading state
    balance = initial_capital
    positions = {}
    trades = []
    rl_decisions = {'entry': [], 'exit': []}
    
    # Process each timestamp
    for i, timestamp in enumerate(all_timestamps):
        if i % 1000 == 0 and i > 0:
            print(f"  [{i}/{len(all_timestamps)}] Balance: ${balance:,.2f} | Trades: {len(trades)}")
        
        for symbol, data in all_data.items():
            if timestamp not in data['df'].index:
                continue
            
            idx = data['df'].index.get_loc(timestamp)
            if idx < 50 or idx >= len(data['closes']) - 1:
                continue
            
            current_price = data['closes'][idx]
            
            # Check exits
            if symbol in positions:
                pos = positions[symbol]
                next_bar = data['df'].iloc[idx + 1]
                
                next_high = next_bar.get('high', next_bar.get('High', 0))
                next_low = next_bar.get('low', next_bar.get('Low', 0))
                
                # Update bars held
                pos['bars_held'] = pos.get('bars_held', 0) + 1
                
                # RL exit evaluation
                exit_decision = signal_gen.evaluate_exit(data, idx, pos)
                
                exit_price = None
                exit_reason = 'stop_target'
                
                if pos['direction'] == 1:
                    if next_low <= pos['stop']:
                        exit_price = pos['stop']
                        exit_reason = 'stop'
                    elif next_high >= pos['target']:
                        exit_price = pos['target']
                        exit_reason = 'target'
                    elif exit_decision:
                        # RL exit logic
                        if exit_decision['action'] == 'EXIT_NOW':
                            exit_price = current_price
                            exit_reason = 'rl_exit'
                        elif exit_decision['action'] == 'MOVE_STOP_BE':
                            pos['stop'] = pos['entry']
                            exit_reason = 'rl_be'
                        elif exit_decision['action'] == 'TRAIL_STOP_TIGHT':
                            pos['stop'] = current_price - (pos['entry'] - pos['stop']) * 0.5
                            exit_reason = 'rl_trail'
                else:
                    if next_high >= pos['stop']:
                        exit_price = pos['stop']
                        exit_reason = 'stop'
                    elif next_low <= pos['target']:
                        exit_price = pos['target']
                        exit_reason = 'target'
                    elif exit_decision:
                        if exit_decision['action'] == 'EXIT_NOW':
                            exit_price = current_price
                            exit_reason = 'rl_exit'
                        elif exit_decision['action'] == 'MOVE_STOP_BE':
                            pos['stop'] = pos['entry']
                            exit_reason = 'rl_be'
                        elif exit_decision['action'] == 'TRAIL_STOP_TIGHT':
                            pos['stop'] = current_price + (pos['stop'] - pos['entry']) * 0.5
                            exit_reason = 'rl_trail'
                
                if exit_price:
                    contract_info = get_contract_info(symbol)
                    
                    if pos['direction'] == 1:
                        price_change = exit_price - pos['entry']
                    else:
                        price_change = pos['entry'] - exit_price
                    
                    if contract_info['type'] == 'futures':
                        pnl = price_change * pos['qty'] * contract_info['multiplier']
                    else:
                        pnl = price_change * pos['qty']
                    
                    balance += pnl
                    trades.append({
                        'symbol': symbol,
                        'direction': 'LONG' if pos['direction'] == 1 else 'SHORT',
                        'entry': pos['entry'],
                        'exit': exit_price,
                        'pnl': pnl,
                        'confidence': pos.get('confidence', 'MEDIUM'),
                        'pd_zone': pos.get('pd_zone'),
                        'exit_reason': exit_reason,
                        'rl_entry': pos.get('rl_entry', 'N/A'),
                        'bars_held': pos.get('bars_held', 0)
                    })
                    
                    if exit_decision:
                        rl_decisions['exit'].append({
                            'action': exit_decision['action'],
                            'pnl_r': exit_decision.get('unrealized_pnl_r', 0),
                            'reason': exit_reason
                        })
                    
                    del positions[symbol]
            
            # Check entries
            elif symbol not in positions:
                signal = signal_gen.generate_signal(data, idx)
                
                if signal and signal['confluence'] >= 60:
                    # Check RL entry decision
                    rl_entry_ok = True
                    if signal.get('rl_entry_action'):
                        entry_action = signal['rl_entry_action']
                        # Only skip if RL specifically says PASS
                        if entry_action == 'PASS':
                            rl_entry_ok = False
                        # For pullback/limit entries, we still enter but track it
                        elif entry_action in ['ENTER_PULLBACK', 'ENTER_LIMIT', 'WAIT_CONFIRMATION']:
                            # Accept but log the preference
                            pass
                    
                    if not rl_entry_ok:
                        continue
                    
                    # Calculate stops/targets
                    if signal['direction'] == 1:
                        stop = data['lows'][idx]
                        target = current_price + (current_price - stop) * 4
                    else:
                        stop = data['highs'][idx]
                        target = current_price - (stop - current_price) * 4
                    
                    stop_distance = abs(current_price - stop)
                    if stop_distance > 0:
                        qty, _ = calculate_position_size(
                            symbol, initial_capital, risk_per_trade, 
                            stop_distance, current_price
                        )
                        
                        if qty > 0:
                            positions[symbol] = {
                                'entry': current_price,
                                'stop': stop,
                                'target': target,
                                'direction': signal['direction'],
                                'qty': qty,
                                'confidence': signal['confidence'],
                                'pd_zone': signal['pd_zone'],
                                'fvg_info': signal.get('fvg_info'),
                                'in_ote': signal.get('in_ote'),
                                'rl_entry': signal.get('rl_entry_action', 'ENTER_NOW'),
                                'stop_distance': stop_distance,
                                'bars_held': 0
                            }
                            
                            if signal.get('rl_action_info'):
                                rl_decisions['entry'].append({
                                    'action': signal['rl_entry_action'],
                                    'q_values': signal['rl_action_info'].get('q_values', {}),
                                    'confidence': signal['rl_action_info'].get('confidence', 0)
                                })
    
    # Results
    total_trades = len(trades)
    wins = len([t for t in trades if t['pnl'] > 0])
    losses = total_trades - wins
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    total_pnl = balance - initial_capital
    return_pct = (total_pnl / initial_capital) * 100
    
    # RL stats
    rl_entry_counts = {}
    rl_exit_counts = {}
    for d in rl_decisions['entry']:
        action = d['action']
        rl_entry_counts[action] = rl_entry_counts.get(action, 0) + 1
    for d in rl_decisions['exit']:
        action = d['action']
        rl_exit_counts[action] = rl_exit_counts.get(action, 0) + 1
    
    # Symbol stats
    symbol_stats = {}
    for symbol in all_data.keys():
        symbol_trades = [t for t in trades if t['symbol'] == symbol]
        symbol_wins = len([t for t in symbol_trades if t['pnl'] > 0])
        symbol_stats[symbol] = {
            'trades': len(symbol_trades),
            'wins': symbol_wins,
            'losses': len(symbol_trades) - symbol_wins,
            'win_rate': (symbol_wins / len(symbol_trades) * 100) if symbol_trades else 0,
            'pnl': sum(t['pnl'] for t in symbol_trades)
        }
    
    print(f"\n{'='*80}")
    print("V8 BACKTEST RESULTS")
    print(f"{'='*80}")
    print(f"Initial: ${initial_capital:,}")
    print(f"Final: ${balance:,.2f}")
    print(f"Return: {return_pct:.2f}%")
    print(f"\nTrades: {total_trades} | Win Rate: {win_rate:.1f}%")
    print(f"Wins: {wins} | Losses: {losses}")
    print(f"Avg Trade: ${total_pnl/total_trades:.2f}" if total_trades > 0 else "N/A")
    
    if use_rl:
        print(f"\nRL Entry Decisions: {rl_entry_counts}")
        print(f"RL Exit Decisions: {rl_exit_counts}")
    
    print(f"\nSymbol Performance:")
    for symbol, stats in sorted(symbol_stats.items(), key=lambda x: x[1]['pnl'], reverse=True):
        print(f"  {symbol}: {stats['trades']}T {stats['win_rate']:.0f}%WR ${stats['pnl']:,.0f}")
    print(f"{'='*80}\n")
    
    return {
        'summary': {
            'initial': initial_capital,
            'final': balance,
            'return_pct': return_pct,
            'trades': total_trades,
            'win_rate': win_rate,
            'wins': wins,
            'losses': losses
        },
        'symbol_stats': symbol_stats,
        'trades': trades,
        'rl_stats': {
            'entry_decisions': rl_entry_counts,
            'exit_decisions': rl_exit_counts
        }
    }


if __name__ == "__main__":
    # Use IBKR for historical data
    from ib_insync import IB, util
    from datetime import datetime, timedelta
    import time as time_module
    
    def fetch_ibkr_extended_data(symbol, days=540, ib=None):
        """Fetch extended historical data from IBKR (up to 540 days = 18 months)"""
        try:
            contract = get_ibkr_contract(symbol)
            
            # IBKR limits: max 1 year of hourly data per request
            # We need to make multiple requests for longer periods
            all_bars = []
            
            # For 18 months, we need to fetch in chunks
            # IBKR allows ~365 days of hourly data per request
            chunks = []
            remaining_days = days
            end_date = ''  # Empty string means "now"
            
            while remaining_days > 0:
                chunk_days = min(remaining_days, 365)
                chunks.append((chunk_days, end_date))
                remaining_days -= chunk_days
                # Calculate new end date (go back chunk_days)
                if end_date == '':
                    from datetime import datetime, timedelta
                    end_dt = datetime.now() - timedelta(days=chunk_days)
                else:
                    end_dt = datetime.strptime(end_date, '%Y%m%d %H:%M:%S') - timedelta(days=chunk_days)
                end_date = end_dt.strftime('%Y%m%d %H:%M:%S')
            
            # Fetch chunks (most recent first)
            for chunk_days, end_dt in chunks:
                print(f"    Fetching {chunk_days} days...", end=' ')
                try:
                    bars = ib.reqHistoricalData(
                        contract,
                        endDateTime=end_dt,
                        durationStr=f"{chunk_days} D",
                        barSizeSetting="1 hour",
                        whatToShow='MIDPOINT',
                        useRTH=False,
                        formatDate=2
                    )
                    if bars:
                        all_bars.extend(bars)
                        print(f"got {len(bars)} bars")
                    else:
                        print("no data")
                    ib.sleep(1)  # Rate limiting
                except Exception as e:
                    print(f"error: {e}")
                    continue
            
            if not all_bars:
                return None
            
            # Convert to DataFrame
            df = util.df(all_bars)
            df.set_index('date', inplace=True)
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df = df[~df.index.duplicated(keep='first')]
            
            if len(df) < 100:
                print(f"Not enough data for {symbol}: {len(df)} rows")
                return None
            
            # Prepare data arrays
            highs = np.array(df['high'])
            lows = np.array(df['low'])
            closes = np.array(df['close'])
            opens = np.array(df['open'])
            
            # Calculate FVG
            bullish_fvgs = []
            bearish_fvgs = []
            for i in range(3, len(df)):
                if lows[i] > highs[i-2]:
                    bullish_fvgs.append({'idx': i, 'mid': (highs[i-2] + lows[i]) / 2, 'high': lows[i]})
                if highs[i] < lows[i-2]:
                    bearish_fvgs.append({'idx': i, 'mid': (highs[i] + lows[i-2]) / 2, 'low': highs[i]})
            
            # Daily data for HTF trend (fetch separately)
            print(f"    Fetching daily data...", end=' ')
            try:
                daily_bars = ib.reqHistoricalData(
                    contract,
                    endDateTime='',
                    durationStr=f"{days} D",
                    barSizeSetting="1 day",
                    whatToShow='MIDPOINT',
                    useRTH=False,
                    formatDate=2
                )
                if daily_bars:
                    df_daily = util.df(daily_bars)
                    df_daily.set_index('date', inplace=True)
                    df_daily.index = pd.to_datetime(df_daily.index)
                    print(f"got {len(df_daily)} days")
                else:
                    df_daily = None
                    print("no data")
            except Exception as e:
                df_daily = None
                print(f"error: {e}")
            
            # Calculate HTF trend - try daily data first, fallback to resampling hourly
            if df_daily is not None and len(df_daily) >= 5:
                daily_highs = np.array(df_daily['high'])
                daily_lows = np.array(df_daily['low'])
                htf = []
                for i in range(1, len(df_daily)):
                    if daily_highs[i] > np.max(daily_highs[max(0,i-5):i]) and daily_lows[i] > np.min(daily_lows[max(0,i-5):i]):
                        htf.append(1)
                    elif daily_highs[i] < np.max(daily_highs[max(0,i-5):i]) and daily_lows[i] < np.min(daily_lows[max(0,i-5):i]):
                        htf.append(-1)
                    else:
                        htf.append(0)
                
                htf_trend = np.zeros(len(df))
                df_daily_index = pd.DatetimeIndex(df_daily.index).tz_localize(None) if df_daily.index.tz else df_daily.index
                df_index = pd.DatetimeIndex(df.index).tz_localize(None) if df.index.tz else df.index
                
                for i in range(len(df)):
                    bar_time = df_index[i]
                    for j in range(len(df_daily) - 1, -1, -1):
                        if df_daily_index[j] <= bar_time:
                            htf_trend[i] = htf[j] if j < len(htf) else 0
                            break
            else:
                # Fallback: Compute daily bars from hourly data
                print("    Computing HTF from hourly data...")
                df_resampled = df.resample('1D').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last'
                }).dropna()
                
                daily_highs = np.array(df_resampled['high'])
                daily_lows = np.array(df_resampled['low'])
                htf = []
                for i in range(1, len(df_resampled)):
                    if i >= 5:
                        if daily_highs[i] > np.max(daily_highs[max(0,i-5):i]) and daily_lows[i] > np.min(daily_lows[max(0,i-5):i]):
                            htf.append(1)
                        elif daily_highs[i] < np.max(daily_highs[max(0,i-5):i]) and daily_lows[i] < np.min(daily_lows[max(0,i-5):i]):
                            htf.append(-1)
                        else:
                            htf.append(0)
                    else:
                        htf.append(0)
                
                htf_trend = np.zeros(len(df))
                df_resampled_index = df_resampled.index.tz_localize(None) if df_resampled.index.tz else df_resampled.index
                df_index = df.index.tz_localize(None) if df.index.tz else df.index
                
                for i in range(len(df)):
                    bar_time = df_index[i]
                    for j in range(len(df_resampled) - 1, -1, -1):
                        if df_resampled_index[j] <= bar_time:
                            htf_trend[i] = htf[j] if j < len(htf) else 0
                            break
            
            ltf_trend = htf_trend.copy()
            
            # Kill zone
            kill_zone = np.zeros(len(df), dtype=int)
            for i in range(len(df)):
                hour = df.index[i].hour
                if (13 <= hour < 17) or (18 <= hour < 21):
                    kill_zone[i] = 1
            
            # Price position
            price_position = np.zeros(len(df))
            for i in range(len(df)):
                day_bars = df.iloc[max(0, i-24):i+1]
                if len(day_bars) > 0:
                    day_range = day_bars['high'].max() - day_bars['low'].min()
                    if day_range > 0:
                        price_position[i] = (closes[i] - day_bars['low'].min()) / day_range
                    else:
                        price_position[i] = 0.5
                else:
                    price_position[i] = 0.5
            
            return {
                'opens': opens,
                'highs': highs,
                'lows': lows,
                'closes': closes,
                'df': df,
                'bullish_fvgs': bullish_fvgs,
                'bearish_fvgs': bearish_fvgs,
                'htf_trend': htf_trend,
                'ltf_trend': ltf_trend,
                'kill_zone': kill_zone,
                'price_position': price_position
            }
            
        except Exception as e:
            print(f"Error fetching {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    # Symbols: Crypto + Futures (as requested)
    # Train: 1 year, Test: 6 months (need 18 months = 540 days)
    symbols = ['SOLUSD', 'LINKUSD', 'LTCUSD', 'UNIUSD', 'BTCUSD', 'ETHUSD', 'SI', 'NQ', 'ES', 'GC']
    
    print("\n" + "="*80)
    print("V8 RL TRAINING & TESTING - IBKR DATA")
    print("="*80)
    print("Training Period: 12 months (1 year)")
    print("Testing Period: 6 months")
    print("Symbols:", ', '.join(symbols))
    print("Settings: Confluence >= 60, RR 1:4")
    print("="*80 + "\n")
    
    # Connect to IBKR
    print("Connecting to IBKR...")
    ib = IB()
    try:
        ib.connect('127.0.0.1', 7497, clientId=99)
        print("Connected to IBKR successfully!\n")
    except Exception as e:
        print(f"ERROR: Could not connect to IBKR: {e}")
        print("Make sure IB Gateway or TWS is running with API enabled on port 7497")
        exit(1)
    
    # Load extended data from IBKR (18 months = 540 days)
    print("Loading 18 months of IBKR historical data...")
    extended_data = {}
    for symbol in symbols:
        print(f"  Loading {symbol}...")
        data = fetch_ibkr_extended_data(symbol, days=540, ib=ib)
        if data and len(data.get('closes', [])) >= 500:
            extended_data[symbol] = data
            print(f"    ✓ {len(data['closes'])} bars from {data['df'].index[0]} to {data['df'].index[-1]}")
        else:
            print(f"    ✗ Failed (got {len(data.get('closes', [])) if data else 0} bars)")
        ib.sleep(2)  # Rate limiting between symbols
    
    if not extended_data:
        print("No data loaded!")
        exit(1)
    
    # Split into train/test based on timestamps (4 months train / 2 months test = 67% / 33%)
    all_timestamps = sorted(set().union(*[set(data['df'].index) for data in extended_data.values()]))
    print(f"\nTotal timestamps: {len(all_timestamps)}")
    print(f"Date range: {all_timestamps[0]} to {all_timestamps[-1]}")
    
    # Split: first 12 months for training, last 6 months for testing (67% / 33%)
    split_idx = int(len(all_timestamps) * 0.67)  # 12 months out of 18
    train_timestamps = all_timestamps[:split_idx]
    test_timestamps = all_timestamps[split_idx:]
    
    print(f"\nTrain period: {train_timestamps[0]} to {train_timestamps[-1]} ({len(train_timestamps)} timestamps)")
    print(f"Test period: {test_timestamps[0]} to {test_timestamps[-1]} ({len(test_timestamps)} timestamps)")
    
    # Train RL agent on training data
    print("\n" + "="*80)
    print("PHASE 1: TRAINING RL AGENT")
    print("="*80)
    
    config = TrainingConfig(
        state_size_entry=31,
        state_size_exit=40,
        reward_scale=1.0,
        penalty_scale=1.0,
        epsilon_start=1.0,
        epsilon_end=0.02,
        epsilon_decay=0.995,
        batch_size=32,
        min_experiences=50,
        buffer_size=20000
    )
    
    agent = ICTReinforcementLearningAgent(config)
    reward_calc = RewardCalculator(reward_scale=1.0, penalty_scale=1.0)
    
    # Training loop using actual historical data with ACTUAL TRADE OUTCOMES
    training_episodes = 3000  # Keep at 3000 for faster iteration
    print(f"\nTraining on {len(train_timestamps)} timestamps with {training_episodes} episodes...")
    print("Using actual trade outcome rewards...")
    
    episode_rewards = []
    winning_trades = 0
    losing_trades = 0
    
    for episode in range(training_episodes):
        # Randomly sample from training period
        ts = np.random.choice(train_timestamps[:-50])  # Leave room for forward look
        
        for symbol, data in extended_data.items():
            if ts not in data['df'].index:
                continue
            
            idx = data['df'].index.get_loc(ts)
            if idx < 50 or idx >= len(data['closes']) - 20:
                continue
            
            current_price = data['closes'][idx]
            
            # Calculate features
            if idx >= 5:
                momentum_5 = (data['closes'][idx] - data['closes'][idx-5]) / data['closes'][idx-5]
            else:
                momentum_5 = 0.0
            
            if idx >= 20:
                momentum_20 = (data['closes'][idx] - data['closes'][idx-20]) / data['closes'][idx-20]
            else:
                momentum_20 = 0.0
            
            # ATR
            if idx >= 14:
                trs = []
                for j in range(max(1, idx - 13), idx + 1):
                    high_low = data['highs'][j] - data['lows'][j]
                    high_close = abs(data['highs'][j] - data['closes'][j-1])
                    low_close = abs(data['lows'][j] - data['closes'][j-1])
                    trs.append(max(high_low, high_close, low_close))
                atr = np.mean(trs) if trs else 0
            else:
                atr = (data['highs'][idx] - data['lows'][idx]) * 0.5
            
            # Daily quadrants
            highs = data['highs'][:idx+1]
            lows = data['lows'][:idx+1]
            daily_high = np.max(highs[-24:]) if len(highs) >= 24 else np.max(highs)
            daily_low = np.min(lows[-24:]) if len(lows) >= 24 else np.min(lows)
            range_size = daily_high - daily_low
            range_pos = (current_price - daily_low) / range_size if range_size > 0 else 0.5
            
            # Regime
            if abs(momentum_20) > 0.02:
                regime = MarketRegime.TRENDING_BULL if momentum_20 > 0 else MarketRegime.TRENDING_BEAR
            else:
                regime = MarketRegime.RANGING
            
            # PD zone
            if range_pos >= 0.75:
                pd_zone = 'premium'
            elif range_pos >= 0.5:
                pd_zone = 'equilibrium_above'
            elif range_pos >= 0.25:
                pd_zone = 'equilibrium_below'
            else:
                pd_zone = 'discount'
            
            in_ote = 0.25 <= range_pos <= 0.79
            
            # Determine direction based on trend alignment (not random)
            if momentum_20 > 0.01 and pd_zone in ['discount', 'equilibrium_below']:
                direction = 1  # Long in discount during uptrend
            elif momentum_20 < -0.01 and pd_zone in ['premium', 'equilibrium_above']:
                direction = -1  # Short in premium during downtrend
            else:
                direction = np.random.choice([1, -1])
            
            # Calculate actual signal confluence
            signal_confluence = 50
            if data['htf_trend'][idx] == direction:
                signal_confluence += 20
            if in_ote:
                signal_confluence += 15
            if (direction == 1 and pd_zone == 'discount') or (direction == -1 and pd_zone == 'premium'):
                signal_confluence += 15
            
            # Create market state
            market_state = MarketState(
                current_price=float(current_price),
                momentum_5=float(momentum_5),
                momentum_20=float(momentum_20),
                volatility=float(atr / current_price) if current_price > 0 else 0,
                atr=float(atr),
                session_high=float(daily_high),
                session_low=float(daily_low),
                daily_high=float(daily_high),
                daily_low=float(daily_low),
                range_position=float(range_pos),
                rsi=50.0,
                in_premium=pd_zone in ['premium'],
                in_discount=pd_zone in ['discount'],
                premium_discount_depth=abs(range_pos - 0.5) * 2,
                in_ote_zone=in_ote,
                trend="bullish" if momentum_20 > 0.01 else "bearish" if momentum_20 < -0.01 else "ranging",
                session=SessionType.NEW_YORK,
                in_kill_zone=True,
                regime=regime,
                signal_quality=float(signal_confluence),
                signal_confluence=int(signal_confluence)
            )
            
            # Position state - flat, we're evaluating entry decisions
            position_state = PositionState(
                status=PositionStatus.FLAT,
                direction="",
                entry_price=float(current_price),
                current_price=float(current_price),
                stop_loss=0,
                take_profit=0,
                position_size=0,
                unrealized_pnl=0,
                unrealized_pnl_r=0,
                bars_held=0,
                remaining_size=1.0
            )
            
            state = RLState(market=market_state, position=position_state)
            
            # Get action
            state_vector = state.to_entry_vector()
            action_idx = agent.entry_agent.select_action(state_vector, training=True)
            entry_action = EntryAction(action_idx)
            
            # SIMULATE ACTUAL TRADE OUTCOME by looking forward
            # Calculate stop and target
            if direction == 1:
                stop = data['lows'][idx]
                target = current_price + (current_price - stop) * 3  # 1:3 RR
            else:
                stop = data['highs'][idx]
                target = current_price - (stop - current_price) * 3
            
            stop_distance = abs(current_price - stop)
            
            # Look forward to see what happens
            trade_result = 0  # 0 = no trade, 1 = win, -1 = loss
            if entry_action != EntryAction.PASS and stop_distance > 0:
                for future_idx in range(idx + 1, min(idx + 20, len(data['closes']))):
                    future_high = data['highs'][future_idx]
                    future_low = data['lows'][future_idx]
                    
                    if direction == 1:
                        if future_low <= stop:
                            trade_result = -1  # Stop hit
                            break
                        elif future_high >= target:
                            trade_result = 1  # Target hit
                            break
                    else:
                        if future_high >= stop:
                            trade_result = -1
                            break
                        elif future_low <= target:
                            trade_result = 1
                            break
            
            # Reward based on actual outcome
            if entry_action == EntryAction.PASS:
                # Skipped trade - small penalty if it would have won, small reward if it would have lost
                if trade_result == 1:
                    reward = -0.3  # Missed a winner
                elif trade_result == -1:
                    reward = 0.2  # Avoided a loser
                else:
                    reward = 0.0
            elif entry_action == EntryAction.ENTER_NOW:
                if trade_result == 1:
                    reward = 1.0  # Won!
                    winning_trades += 1
                elif trade_result == -1:
                    reward = -0.5  # Lost
                    losing_trades += 1
                else:
                    reward = 0.0  # No result yet
            elif entry_action == EntryAction.ENTER_PULLBACK:
                # Pullback entry - slightly better risk/reward
                if trade_result == 1:
                    reward = 1.2  # Bonus for waiting
                    winning_trades += 1
                elif trade_result == -1:
                    reward = -0.4  # Slightly less penalty
                    losing_trades += 1
                else:
                    reward = 0.05  # Small reward for patience
            else:
                # Other actions (WAIT_CONFIRMATION, etc.)
                if trade_result == 1:
                    reward = 0.8
                    winning_trades += 1
                elif trade_result == -1:
                    reward = -0.3
                    losing_trades += 1
                else:
                    reward = 0.0
            
            # Store experience
            next_state_vector = state_vector + np.random.randn(len(state_vector)) * 0.01
            experience = Experience(
                state=state_vector,
                action=action_idx,
                reward=reward,
                next_state=next_state_vector,
                done=False
            )
            agent.entry_agent.replay_buffer.add(experience)
            
            # Train
            if len(agent.entry_agent.replay_buffer) >= config.min_experiences:
                agent.entry_agent.train_step()
            
            episode_rewards.append(reward)
        
        if episode % 200 == 0:
            recent_avg = np.mean(episode_rewards[-100:]) if len(episode_rewards) >= 100 else np.mean(episode_rewards) if episode_rewards else 0
            total_sim_trades = winning_trades + losing_trades
            sim_wr = winning_trades / total_sim_trades * 100 if total_sim_trades > 0 else 0
            print(f"  Episode {episode}/{training_episodes} | Avg Reward: {recent_avg:.3f} | ε: {agent.entry_agent.epsilon:.3f} | SimWR: {sim_wr:.1f}%")
    
    # Final training pass
    print("\nFinal training pass...")
    for _ in range(100):
        if len(agent.entry_agent.replay_buffer) >= config.batch_size:
            agent.entry_agent.train_step()
    
    print(f"Training complete! Final ε: {agent.entry_agent.epsilon:.3f}")
    
    # Now test on test period
    print("\n" + "="*80)
    print("PHASE 2: TESTING ON UNSEEN DATA")
    print("="*80)
    
    # Run backtest using only test timestamps
    signal_gen = V8SignalGenerator(use_rl=True)
    signal_gen.rl_agent = agent
    
    balance = initial_capital = 5000
    positions = {}
    trades = []
    rl_decisions = {'entry': [], 'exit': []}
    
    for i, timestamp in enumerate(test_timestamps):
        if i % 500 == 0:
            print(f"  Testing: [{i}/{len(test_timestamps)}] Balance: ${balance:,.2f}")
        
        for symbol, data in extended_data.items():
            if timestamp not in data['df'].index:
                continue
            
            idx = data['df'].index.get_loc(timestamp)
            if idx < 50 or idx >= len(data['closes']) - 1:
                continue
            
            current_price = data['closes'][idx]
            
            # Check exits
            if symbol in positions:
                pos = positions[symbol]
                next_bar = data['df'].iloc[idx + 1]
                next_high = next_bar.get('high', next_bar.get('High', 0))
                next_low = next_bar.get('low', next_bar.get('Low', 0))
                
                pos['bars_held'] = pos.get('bars_held', 0) + 1
                
                exit_price = None
                if pos['direction'] == 1:
                    if next_low <= pos['stop']:
                        exit_price = pos['stop']
                    elif next_high >= pos['target']:
                        exit_price = pos['target']
                else:
                    if next_high >= pos['stop']:
                        exit_price = pos['stop']
                    elif next_low <= pos['target']:
                        exit_price = pos['target']
                
                if exit_price:
                    contract_info = get_contract_info(symbol)
                    if pos['direction'] == 1:
                        price_change = exit_price - pos['entry']
                    else:
                        price_change = pos['entry'] - exit_price
                    
                    if contract_info['type'] == 'futures':
                        pnl = price_change * pos['qty'] * contract_info['multiplier']
                    else:
                        pnl = price_change * pos['qty']
                    
                    balance += pnl
                    trades.append({
                        'symbol': symbol,
                        'direction': 'LONG' if pos['direction'] == 1 else 'SHORT',
                        'entry': pos['entry'],
                        'exit': exit_price,
                        'pnl': pnl,
                        'confidence': pos.get('confidence', 'MEDIUM'),
                        'exit_reason': 'stop_target'
                    })
                    del positions[symbol]
            
            # Check entries
            elif symbol not in positions:
                signal = signal_gen.generate_signal(data, idx)
                
                if signal and signal['confluence'] >= 65:
                    # Check RL entry decision
                    rl_entry_ok = True
                    if signal.get('rl_entry_action'):
                        entry_action = signal['rl_entry_action']
                        if entry_action == 'PASS':
                            rl_entry_ok = False
                    
                    if not rl_entry_ok:
                        continue
                    
                    if signal['direction'] == 1:
                        stop = data['lows'][idx]
                        target = current_price + (current_price - stop) * 3
                    else:
                        stop = data['highs'][idx]
                        target = current_price - (stop - current_price) * 3
                    
                    stop_distance = abs(current_price - stop)
                    if stop_distance > 0:
                        qty, _ = calculate_position_size(
                            symbol, initial_capital, 0.02, 
                            stop_distance, current_price
                        )
                        
                        if qty > 0:
                            positions[symbol] = {
                                'entry': current_price,
                                'stop': stop,
                                'target': target,
                                'direction': signal['direction'],
                                'qty': qty,
                                'confidence': signal['confidence'],
                                'pd_zone': signal['pd_zone'],
                                'rl_entry': signal.get('rl_entry_action', 'ENTER_NOW'),
                                'stop_distance': stop_distance,
                                'bars_held': 0
                            }
                            
                            if signal.get('rl_action_info'):
                                rl_decisions['entry'].append(signal['rl_entry_action'])
    
    # Results
    total_trades = len(trades)
    wins = len([t for t in trades if t['pnl'] > 0])
    losses = total_trades - wins
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    total_pnl = balance - initial_capital
    return_pct = (total_pnl / initial_capital) * 100
    
    print(f"\n{'='*80}")
    print("V8 TRAIN/TEST RESULTS")
    print(f"{'='*80}")
    print(f"Training: {len(train_timestamps)} timestamps")
    print(f"Testing: {len(test_timestamps)} timestamps")
    print(f"\nInitial: ${initial_capital:,}")
    print(f"Final: ${balance:,.2f}")
    print(f"Return: {return_pct:.2f}%")
    print(f"\nTrades: {total_trades} | Win Rate: {win_rate:.1f}%")
    print(f"Wins: {wins} | Losses: {losses}")
    print(f"Avg Trade: ${total_pnl/total_trades:.2f}" if total_trades > 0 else "N/A")
    
    # RL stats
    from collections import Counter
    rl_entry_counts = Counter(rl_decisions['entry'])
    print(f"\nRL Entry Decisions: {dict(rl_entry_counts)}")
    
    print(f"{'='*80}\n")
    
    # Save results
    results = {
        'summary': {
            'initial': initial_capital,
            'final': balance,
            'return_pct': return_pct,
            'trades': total_trades,
            'win_rate': win_rate,
            'wins': wins,
            'losses': losses,
            'train_timestamps': len(train_timestamps),
            'test_timestamps': len(test_timestamps)
        },
        'rl_entry_decisions': dict(rl_entry_counts),
        'trades': trades
    }
    
    with open('v8_train_test_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print("Results saved to v8_train_test_results.json")
    
    # Save trained RL model for live trading
    import pickle
    model_path = 'v8_rl_model.pkl'
    with open(model_path, 'wb') as f:
        pickle.dump({
            'agent': agent,
            'config': config,
            'train_timestamps': len(train_timestamps),
            'test_results': {
                'return_pct': return_pct,
                'win_rate': win_rate,
                'trades': total_trades
            }
        }, f)
    print(f"RL model saved to {model_path}")
