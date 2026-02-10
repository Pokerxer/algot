"""
ICT Reinforcement Learning Agent - Optimal Entry/Exit Timing
=============================================================

Deep Reinforcement Learning system that learns optimal entry and exit
timing for ICT trading setups through interaction with market data.

REINFORCEMENT LEARNING ARCHITECTURE:
====================================

┌─────────────────────────────────────────────────────────────────────────────┐
│                     REINFORCEMENT LEARNING TRADING AGENT                     │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                         ENVIRONMENT                                   │    │
│  │                                                                       │    │
│  │  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐             │    │
│  │  │   Market    │    │   ICT       │    │  Position   │             │    │
│  │  │   State     │    │   Signals   │    │   Status    │             │    │
│  │  └─────────────┘    └─────────────┘    └─────────────┘             │    │
│  │        │                   │                   │                     │    │
│  │        └───────────────────┴───────────────────┘                     │    │
│  │                            │                                          │    │
│  │                            ▼                                          │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │                    STATE VECTOR                                │  │    │
│  │  │                                                                │  │    │
│  │  │  • Price action features (momentum, volatility, range)        │  │    │
│  │  │  • ICT elements (OB distance, FVG proximity, liquidity)       │  │    │
│  │  │  • Time features (session, kill zone, macro time)             │  │    │
│  │  │  • Position status (entry price, unrealized P/L, time held)   │  │    │
│  │  │  • Signal quality (confluence, HTF alignment)                 │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                  │                                           │
│                                  ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                           AGENT                                       │    │
│  │                                                                       │    │
│  │  ┌───────────────────────────────────────────────────────────────┐  │    │
│  │  │                    DQN NETWORK                                 │  │    │
│  │  │                                                                │  │    │
│  │  │    Input Layer (State) → Hidden Layers → Output Layer (Q-vals)│  │    │
│  │  │         64 neurons    →    [256, 128]  →    Action Space      │  │    │
│  │  └───────────────────────────────────────────────────────────────┘  │    │
│  │                                                                       │    │
│  │  ACTIONS:                                                             │    │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌───────────┐  │    │
│  │  │    WAIT     │  │   ENTER     │  │    HOLD     │  │   EXIT    │  │    │
│  │  │  (no trade) │  │  (execute)  │  │  (continue) │  │  (close)  │  │    │
│  │  └─────────────┘  └─────────────┘  └─────────────┘  └───────────┘  │    │
│  │                                                                       │    │
│  │  ENTRY TIMING:                    EXIT TIMING:                        │    │
│  │  • ENTER_NOW - immediate          • EXIT_NOW - close position         │    │
│  │  • ENTER_PULLBACK - wait for dip  • EXIT_PARTIAL - scale out          │    │
│  │  • ENTER_BREAKOUT - wait for move • MOVE_STOP_BE - breakeven          │    │
│  │  • ENTER_LIMIT - use limit order  • TRAIL_STOP - trailing stop        │    │
│  │  • PASS - skip this signal        • HOLD - maintain position          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                  │                                           │
│                                  ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        REWARD SYSTEM                                  │    │
│  │                                                                       │    │
│  │  Entry Rewards:                   Exit Rewards:                       │    │
│  │  • Better entry price: +reward    • Profit taken: +scaled reward     │    │
│  │  • Missed opportunity: -penalty   • Loss limited: +small reward      │    │
│  │  • Signal quality: +bonus         • Full SL hit: -penalty            │    │
│  │  • Timing alignment: +bonus       • Trailing profit: +bonus          │    │
│  │                                                                       │    │
│  │  Risk-Adjusted Rewards:                                               │    │
│  │  • R-multiple based scoring                                           │    │
│  │  • Drawdown penalty                                                   │    │
│  │  • Opportunity cost consideration                                     │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                  │                                           │
│                                  ▼                                           │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                       EXPERIENCE REPLAY                               │    │
│  │                                                                       │    │
│  │  ┌─────────────────────────────────────────────────────────────┐    │    │
│  │  │  (state, action, reward, next_state, done) → Replay Buffer  │    │    │
│  │  └─────────────────────────────────────────────────────────────┘    │    │
│  │                                                                       │    │
│  │  • Prioritized Experience Replay                                      │    │
│  │  • Batch sampling for training                                        │    │
│  │  • Target network for stability                                       │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘

RL Training Loop:
1. Observe state (market + ICT elements + position)
2. Select action (ε-greedy exploration)
3. Execute action in environment
4. Receive reward
5. Store experience
6. Sample batch and update network
7. Repeat

ICT Principles Integrated:
- "Time and Price" → State includes session/kill zone features
- "Liquidity Draw" → State includes draw on liquidity distances
- "Premium/Discount" → State includes zone position
- "Optimal Trade Entry" → Actions optimize OTE timing
- "Smart Money Concepts" → Rewards based on institutional-style entries
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from enum import Enum
from datetime import datetime, timedelta
from collections import deque, namedtuple
import logging
import json
import math
import random
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class EntryAction(Enum):
    """Entry timing actions"""
    PASS = 0                    # Skip this signal
    ENTER_NOW = 1               # Enter immediately
    ENTER_PULLBACK = 2          # Wait for pullback
    ENTER_BREAKOUT = 3          # Enter on breakout confirmation
    ENTER_LIMIT = 4             # Place limit order at OTE
    WAIT_CONFIRMATION = 5       # Wait for additional confirmation


class ExitAction(Enum):
    """Exit timing actions"""
    HOLD = 0                    # Continue holding
    EXIT_NOW = 1                # Close entire position
    EXIT_PARTIAL_25 = 2         # Close 25% of position
    EXIT_PARTIAL_50 = 3         # Close 50% of position
    EXIT_PARTIAL_75 = 4         # Close 75% of position
    MOVE_STOP_BE = 5            # Move stop to breakeven
    TRAIL_STOP_TIGHT = 6        # Tight trailing stop
    TRAIL_STOP_WIDE = 7         # Wide trailing stop


class PositionStatus(Enum):
    """Current position status"""
    FLAT = "flat"               # No position
    LONG = "long"               # Long position
    SHORT = "short"             # Short position
    PENDING = "pending"         # Pending order


class MarketRegime(Enum):
    """Market regime classification"""
    TRENDING_BULL = "trending_bull"
    TRENDING_BEAR = "trending_bear"
    RANGING = "ranging"
    VOLATILE = "volatile"
    TRANSITIONING = "transitioning"


class SessionType(Enum):
    """Trading session"""
    ASIAN = "asian"
    LONDON = "london"
    NEW_YORK = "new_york"
    OVERLAP = "overlap"
    OFF_HOURS = "off_hours"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MarketState:
    """Current market state observation"""
    # Price data
    current_price: float
    bid: float = 0.0
    ask: float = 0.0
    spread: float = 0.0
    
    # Price action
    last_5_returns: List[float] = field(default_factory=list)
    momentum_5: float = 0.0
    momentum_20: float = 0.0
    volatility: float = 0.0
    atr: float = 0.0
    
    # Range
    session_high: float = 0.0
    session_low: float = 0.0
    daily_high: float = 0.0
    daily_low: float = 0.0
    range_position: float = 0.5  # 0=low, 1=high
    
    # Technical
    rsi: float = 50.0
    macd: float = 0.0
    macd_signal: float = 0.0
    
    # ICT Elements
    nearest_ob_distance: float = 0.0      # Distance to nearest OB
    ob_strength: float = 0.0               # OB strength score
    nearest_fvg_distance: float = 0.0      # Distance to nearest FVG
    fvg_fill_percent: float = 0.0          # FVG fill percentage
    nearest_liquidity_distance: float = 0.0  # Distance to liquidity
    liquidity_size: float = 0.0            # Liquidity pool size
    
    # Zone position
    in_premium: bool = False
    in_discount: bool = False
    premium_discount_depth: float = 0.0    # How deep in zone
    in_ote_zone: bool = False              # In OTE (61.8-78.6)
    
    # Structure
    trend: str = "ranging"                 # bullish/bearish/ranging
    trend_strength: float = 0.0
    bos_distance: float = 0.0              # Distance to structure break
    swing_high_distance: float = 0.0
    swing_low_distance: float = 0.0
    
    # Time
    session: SessionType = SessionType.OFF_HOURS
    in_kill_zone: bool = False
    kill_zone_type: str = ""
    minutes_in_session: int = 0
    minutes_to_close: int = 0
    day_of_week: int = 0
    is_macro_time: bool = False
    
    # Market regime
    regime: MarketRegime = MarketRegime.RANGING
    
    # Signal info (when applicable)
    signal_quality: float = 0.0
    signal_confluence: int = 0
    htf_alignment: float = 0.0
    ltf_alignment: float = 0.0


@dataclass
class PositionState:
    """Current position state"""
    status: PositionStatus = PositionStatus.FLAT
    direction: str = ""  # 'long' or 'short'
    entry_price: float = 0.0
    current_price: float = 0.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    position_size: float = 0.0
    
    # P/L
    unrealized_pnl: float = 0.0
    unrealized_pnl_r: float = 0.0  # In R multiples
    max_favorable: float = 0.0      # MFE
    max_adverse: float = 0.0        # MAE
    
    # Time
    bars_held: int = 0
    time_held: timedelta = field(default_factory=lambda: timedelta(0))
    entry_time: Optional[datetime] = None
    
    # Partial exits
    remaining_size: float = 1.0     # 1.0 = full, 0.5 = half, etc.
    partial_profits: float = 0.0
    
    # Stop management
    stop_moved_be: bool = False
    trailing_active: bool = False


@dataclass
class RLState:
    """Complete state for RL agent"""
    market: MarketState
    position: PositionState
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_entry_vector(self) -> np.ndarray:
        """Convert to entry decision state vector"""
        return np.array([
            # Market features (normalized)
            self.market.momentum_5 / 100 if abs(self.market.momentum_5) < 100 else np.sign(self.market.momentum_5),
            self.market.momentum_20 / 100 if abs(self.market.momentum_20) < 100 else np.sign(self.market.momentum_20),
            min(self.market.volatility / 50, 1.0),
            self.market.range_position,
            (self.market.rsi - 50) / 50,  # Normalize around 50
            
            # ICT features (normalized)
            min(self.market.nearest_ob_distance / 100, 1.0),
            self.market.ob_strength / 100,
            min(self.market.nearest_fvg_distance / 100, 1.0),
            self.market.fvg_fill_percent / 100,
            min(self.market.nearest_liquidity_distance / 100, 1.0),
            
            # Zone features
            float(self.market.in_premium),
            float(self.market.in_discount),
            self.market.premium_discount_depth,
            float(self.market.in_ote_zone),
            
            # Structure features
            1.0 if self.market.trend == 'bullish' else (-1.0 if self.market.trend == 'bearish' else 0.0),
            self.market.trend_strength / 100,
            min(self.market.bos_distance / 100, 1.0),
            
            # Time features (one-hot encoded session)
            float(self.market.session == SessionType.LONDON),
            float(self.market.session == SessionType.NEW_YORK),
            float(self.market.session == SessionType.OVERLAP),
            float(self.market.in_kill_zone),
            float(self.market.is_macro_time),
            min(self.market.minutes_in_session / 240, 1.0),  # Normalize to 4 hours
            
            # Signal quality
            self.market.signal_quality / 100,
            min(self.market.signal_confluence / 10, 1.0),
            self.market.htf_alignment / 100,
            self.market.ltf_alignment / 100,
            
            # Regime (one-hot)
            float(self.market.regime == MarketRegime.TRENDING_BULL),
            float(self.market.regime == MarketRegime.TRENDING_BEAR),
            float(self.market.regime == MarketRegime.RANGING),
            float(self.market.regime == MarketRegime.VOLATILE),
        ], dtype=np.float32)
    
    def to_exit_vector(self) -> np.ndarray:
        """Convert to exit decision state vector"""
        entry_features = self.to_entry_vector()
        
        position_features = np.array([
            # Position P/L features
            min(max(self.position.unrealized_pnl_r / 5, -1), 1),  # Clamp to [-1, 1]
            min(self.position.max_favorable / 100, 1.0),  # Normalize MFE
            min(self.position.max_adverse / 100, 1.0),    # Normalize MAE
            
            # Time features
            min(self.position.bars_held / 100, 1.0),
            
            # Position status
            self.position.remaining_size,
            float(self.position.stop_moved_be),
            float(self.position.trailing_active),
            
            # Distance to levels (normalized)
            0.0,  # Distance to SL (calculated separately)
            0.0,  # Distance to TP (calculated separately)
        ], dtype=np.float32)
        
        # Calculate distances
        if self.position.status != PositionStatus.FLAT:
            if self.position.stop_loss > 0:
                sl_dist = abs(self.position.current_price - self.position.stop_loss)
                entry_dist = abs(self.position.entry_price - self.position.stop_loss)
                position_features[-2] = min(sl_dist / (entry_dist + 0.0001), 2.0) / 2.0
            
            if self.position.take_profit > 0:
                tp_dist = abs(self.position.take_profit - self.position.current_price)
                entry_dist = abs(self.position.take_profit - self.position.entry_price)
                position_features[-1] = min(tp_dist / (entry_dist + 0.0001), 2.0) / 2.0
        
        return np.concatenate([entry_features, position_features])


@dataclass
class Experience:
    """Single experience for replay buffer"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    priority: float = 1.0


@dataclass
class TrainingConfig:
    """RL training configuration"""
    # Network architecture
    state_size_entry: int = 31
    state_size_exit: int = 40
    hidden_layers: List[int] = field(default_factory=lambda: [256, 128, 64])
    
    # Training parameters
    learning_rate: float = 0.001
    gamma: float = 0.99           # Discount factor
    epsilon_start: float = 1.0    # Initial exploration
    epsilon_end: float = 0.01     # Final exploration
    epsilon_decay: float = 0.995  # Decay rate
    
    # Replay buffer
    buffer_size: int = 100000
    batch_size: int = 64
    min_experiences: int = 1000   # Min before training
    
    # Target network
    target_update_freq: int = 100  # Steps between target updates
    
    # Training
    train_freq: int = 4           # Steps between training
    max_episodes: int = 10000
    max_steps_per_episode: int = 1000
    
    # Rewards
    reward_scale: float = 1.0
    penalty_scale: float = 1.0


# =============================================================================
# NEURAL NETWORK (Pure NumPy Implementation)
# =============================================================================

class NeuralNetwork:
    """
    Simple neural network implemented in NumPy for RL
    Supports forward pass and backpropagation
    """
    
    def __init__(self, layer_sizes: List[int], learning_rate: float = 0.001):
        """
        Initialize network
        
        Args:
            layer_sizes: List of layer sizes [input, hidden1, hidden2, ..., output]
            learning_rate: Learning rate for updates
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        
        # Initialize weights and biases using Xavier initialization
        self.weights = []
        self.biases = []
        
        for i in range(len(layer_sizes) - 1):
            # Xavier initialization
            scale = np.sqrt(2.0 / (layer_sizes[i] + layer_sizes[i + 1]))
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
            b = np.zeros((1, layer_sizes[i + 1]))
            self.weights.append(w)
            self.biases.append(b)
        
        # Cache for backpropagation
        self.activations = []
        self.z_values = []
    
    def relu(self, x: np.ndarray) -> np.ndarray:
        """ReLU activation"""
        return np.maximum(0, x)
    
    def relu_derivative(self, x: np.ndarray) -> np.ndarray:
        """ReLU derivative"""
        return (x > 0).astype(float)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass
        
        Args:
            x: Input array of shape (batch_size, input_size)
            
        Returns:
            Output array of shape (batch_size, output_size)
        """
        self.activations = [x]
        self.z_values = []
        
        current = x
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = np.dot(current, w) + b
            self.z_values.append(z)
            
            # ReLU for hidden layers, linear for output
            if i < len(self.weights) - 1:
                current = self.relu(z)
            else:
                current = z  # Linear output for Q-values
            
            self.activations.append(current)
        
        return current
    
    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Backward pass (backpropagation)
        
        Args:
            y_true: Target values
            y_pred: Predicted values
            
        Returns:
            Loss value
        """
        batch_size = y_true.shape[0]
        
        # MSE loss
        loss = np.mean((y_pred - y_true) ** 2)
        
        # Gradient of loss
        delta = (y_pred - y_true) / batch_size
        
        # Backpropagate through layers
        weight_gradients = []
        bias_gradients = []
        
        for i in reversed(range(len(self.weights))):
            # Gradient for weights and biases
            weight_gradients.insert(0, np.dot(self.activations[i].T, delta))
            bias_gradients.insert(0, np.sum(delta, axis=0, keepdims=True))
            
            if i > 0:
                # Backpropagate delta
                delta = np.dot(delta, self.weights[i].T)
                delta *= self.relu_derivative(self.z_values[i - 1])
        
        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] -= self.learning_rate * weight_gradients[i]
            self.biases[i] -= self.learning_rate * bias_gradients[i]
        
        return loss
    
    def copy_from(self, other: 'NeuralNetwork'):
        """Copy weights from another network"""
        for i in range(len(self.weights)):
            self.weights[i] = np.copy(other.weights[i])
            self.biases[i] = np.copy(other.biases[i])
    
    def get_weights(self) -> Dict[str, List]:
        """Get weights as dictionary"""
        return {
            'weights': [w.tolist() for w in self.weights],
            'biases': [b.tolist() for b in self.biases]
        }
    
    def set_weights(self, weights_dict: Dict[str, List]):
        """Set weights from dictionary"""
        self.weights = [np.array(w) for w in weights_dict['weights']]
        self.biases = [np.array(b) for b in weights_dict['biases']]


# =============================================================================
# REPLAY BUFFER
# =============================================================================

class PrioritizedReplayBuffer:
    """
    Prioritized Experience Replay Buffer
    Samples experiences based on their TD error priority
    """
    
    def __init__(self, capacity: int, alpha: float = 0.6, beta: float = 0.4):
        """
        Initialize buffer
        
        Args:
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = full priority)
            beta: Importance sampling exponent
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = 0.001
        
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.max_priority = 1.0
    
    def add(self, experience: Experience):
        """Add experience to buffer"""
        self.buffer.append(experience)
        self.priorities.append(self.max_priority)
    
    def sample(self, batch_size: int) -> Tuple[List[Experience], np.ndarray, List[int]]:
        """
        Sample batch with prioritized sampling
        
        Returns:
            Tuple of (experiences, importance weights, indices)
        """
        n = len(self.buffer)
        if n == 0:
            return [], np.array([]), []
        
        # Calculate sampling probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        # Sample indices
        indices = np.random.choice(n, size=min(batch_size, n), p=probs, replace=False)
        
        # Get experiences
        experiences = [self.buffer[i] for i in indices]
        
        # Calculate importance sampling weights
        weights = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        # Increase beta
        self.beta = min(1.0, self.beta + self.beta_increment)
        
        return experiences, weights, indices.tolist()
    
    def update_priorities(self, indices: List[int], td_errors: np.ndarray):
        """Update priorities based on TD errors"""
        for idx, td_error in zip(indices, td_errors):
            priority = (abs(td_error) + 0.01) ** self.alpha
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
    
    def __len__(self) -> int:
        return len(self.buffer)


# =============================================================================
# DQN AGENT
# =============================================================================

class DQNAgent:
    """
    Deep Q-Network Agent for entry/exit timing
    
    Uses Double DQN with prioritized experience replay
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        config: TrainingConfig,
        name: str = "dqn_agent"
    ):
        """
        Initialize DQN agent
        
        Args:
            state_size: Size of state vector
            action_size: Number of possible actions
            config: Training configuration
            name: Agent name for logging
        """
        self.state_size = state_size
        self.action_size = action_size
        self.config = config
        self.name = name
        
        # Networks
        layer_sizes = [state_size] + config.hidden_layers + [action_size]
        self.q_network = NeuralNetwork(layer_sizes, config.learning_rate)
        self.target_network = NeuralNetwork(layer_sizes, config.learning_rate)
        self.target_network.copy_from(self.q_network)
        
        # Replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(config.buffer_size)
        
        # Exploration
        self.epsilon = config.epsilon_start
        
        # Training state
        self.training_step = 0
        self.episode = 0
        self.total_reward = 0.0
        
        # Metrics
        self.losses = []
        self.rewards = []
        self.q_values = []
        
        logger.info(f"Initialized {name} - State: {state_size}, Actions: {action_size}")
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using ε-greedy policy
        
        Args:
            state: Current state vector
            training: Whether in training mode
            
        Returns:
            Selected action index
        """
        if training and random.random() < self.epsilon:
            # Explore
            return random.randint(0, self.action_size - 1)
        else:
            # Exploit
            state_batch = state.reshape(1, -1)
            q_values = self.q_network.forward(state_batch)
            return int(np.argmax(q_values[0]))
    
    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for a state"""
        state_batch = state.reshape(1, -1)
        return self.q_network.forward(state_batch)[0]
    
    def store_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Store experience in replay buffer"""
        experience = Experience(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            done=done
        )
        self.replay_buffer.add(experience)
        self.total_reward += reward
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step
        
        Returns:
            Loss value or None if not enough experiences
        """
        if len(self.replay_buffer) < self.config.min_experiences:
            return None
        
        # Sample batch
        experiences, weights, indices = self.replay_buffer.sample(
            self.config.batch_size
        )
        
        if not experiences:
            return None
        
        # Prepare batch data
        states = np.array([e.state for e in experiences])
        actions = np.array([e.action for e in experiences])
        rewards = np.array([e.reward for e in experiences])
        next_states = np.array([e.next_state for e in experiences])
        dones = np.array([e.done for e in experiences])
        
        # Double DQN: Use online network to select actions, target network to evaluate
        next_q_online = self.q_network.forward(next_states)
        next_actions = np.argmax(next_q_online, axis=1)
        
        next_q_target = self.target_network.forward(next_states)
        next_q_values = next_q_target[np.arange(len(experiences)), next_actions]
        
        # Calculate targets
        targets = rewards + self.config.gamma * next_q_values * (1 - dones)
        
        # Get current Q-values
        current_q = self.q_network.forward(states)
        
        # Create target Q-values (only update the action taken)
        target_q = current_q.copy()
        target_q[np.arange(len(experiences)), actions] = targets
        
        # Calculate TD errors for priority updates
        td_errors = targets - current_q[np.arange(len(experiences)), actions]
        self.replay_buffer.update_priorities(indices, td_errors)
        
        # Train network
        loss = self.q_network.backward(target_q, current_q)
        self.losses.append(loss)
        
        # Update training step
        self.training_step += 1
        
        # Update target network periodically
        if self.training_step % self.config.target_update_freq == 0:
            self.target_network.copy_from(self.q_network)
        
        # Decay epsilon
        self.epsilon = max(
            self.config.epsilon_end,
            self.epsilon * self.config.epsilon_decay
        )
        
        return loss
    
    def end_episode(self):
        """End current episode"""
        self.rewards.append(self.total_reward)
        self.episode += 1
        self.total_reward = 0.0
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get training metrics"""
        return {
            'episode': self.episode,
            'training_step': self.training_step,
            'epsilon': self.epsilon,
            'avg_loss': np.mean(self.losses[-100:]) if self.losses else 0,
            'avg_reward': np.mean(self.rewards[-100:]) if self.rewards else 0,
            'buffer_size': len(self.replay_buffer),
        }
    
    def save(self, filepath: str):
        """Save agent state"""
        data = {
            'q_network': self.q_network.get_weights(),
            'target_network': self.target_network.get_weights(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'episode': self.episode,
            'config': {
                'state_size': self.state_size,
                'action_size': self.action_size,
            }
        }
        with open(filepath, 'w') as f:
            json.dump(data, f)
        logger.info(f"Agent saved to {filepath}")
    
    def load(self, filepath: str):
        """Load agent state"""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        self.q_network.set_weights(data['q_network'])
        self.target_network.set_weights(data['target_network'])
        self.epsilon = data['epsilon']
        self.training_step = data['training_step']
        self.episode = data['episode']
        logger.info(f"Agent loaded from {filepath}")


# =============================================================================
# REWARD CALCULATOR
# =============================================================================

class RewardCalculator:
    """
    Calculates rewards for RL agent actions
    Based on ICT trading principles and risk-adjusted returns
    """
    
    def __init__(
        self,
        reward_scale: float = 1.0,
        penalty_scale: float = 1.0,
        pip_value: float = 10.0
    ):
        """
        Initialize reward calculator
        
        Args:
            reward_scale: Scale factor for rewards
            penalty_scale: Scale factor for penalties
            pip_value: Value per pip for the instrument
        """
        self.reward_scale = reward_scale
        self.penalty_scale = penalty_scale
        self.pip_value = pip_value
    
    def calculate_entry_reward(
        self,
        action: EntryAction,
        state: RLState,
        outcome: Dict[str, Any]
    ) -> float:
        """
        Calculate reward for entry decision
        
        Args:
            action: Entry action taken
            state: State when action was taken
            outcome: Outcome of the action
            
        Returns:
            Reward value
        """
        reward = 0.0
        
        if action == EntryAction.PASS:
            # Reward for passing on low-quality signals
            if state.market.signal_quality < 60:
                reward += 0.1 * self.reward_scale
            elif state.market.signal_quality > 80:
                # Penalty for missing good signals
                reward -= 0.2 * self.penalty_scale
        
        elif action in [EntryAction.ENTER_NOW, EntryAction.ENTER_PULLBACK, 
                        EntryAction.ENTER_BREAKOUT, EntryAction.ENTER_LIMIT]:
            
            # Entry was executed
            if outcome.get('filled', False):
                entry_price = outcome.get('fill_price', 0)
                target_price = outcome.get('signal_entry', 0)
                
                if entry_price > 0 and target_price > 0:
                    # Better entry price = higher reward
                    slippage = abs(entry_price - target_price) / state.market.atr
                    
                    if state.position.direction == 'long':
                        if entry_price < target_price:
                            # Bought lower = good
                            reward += 0.5 * (1 - slippage) * self.reward_scale
                        else:
                            # Bought higher = less good
                            reward -= 0.2 * slippage * self.penalty_scale
                    else:  # short
                        if entry_price > target_price:
                            # Sold higher = good
                            reward += 0.5 * (1 - slippage) * self.reward_scale
                        else:
                            reward -= 0.2 * slippage * self.penalty_scale
                
                # Bonus for entering in kill zone
                if state.market.in_kill_zone:
                    reward += 0.1 * self.reward_scale
                
                # Bonus for OTE entry
                if state.market.in_ote_zone:
                    reward += 0.15 * self.reward_scale
                
                # Bonus for high confluence
                if state.market.signal_confluence >= 5:
                    reward += 0.1 * self.reward_scale
            else:
                # Order not filled
                if action == EntryAction.ENTER_LIMIT:
                    # Limit order not filled - small penalty
                    reward -= 0.05 * self.penalty_scale
        
        elif action == EntryAction.WAIT_CONFIRMATION:
            # Waiting for confirmation
            if outcome.get('confirmation_received', False):
                reward += 0.2 * self.reward_scale
            elif outcome.get('signal_expired', False):
                reward -= 0.1 * self.penalty_scale
        
        return reward
    
    def calculate_exit_reward(
        self,
        action: ExitAction,
        state: RLState,
        outcome: Dict[str, Any]
    ) -> float:
        """
        Calculate reward for exit decision
        
        Args:
            action: Exit action taken
            state: State when action was taken
            outcome: Outcome of the action
            
        Returns:
            Reward value
        """
        reward = 0.0
        realized_pnl_r = outcome.get('realized_pnl_r', 0)
        
        if action == ExitAction.HOLD:
            # Reward for holding winners, penalty for holding losers too long
            if state.position.unrealized_pnl_r > 0:
                # In profit
                if outcome.get('continued_profit', False):
                    reward += 0.1 * self.reward_scale
                elif outcome.get('profit_decreased', False):
                    # Could have exited better
                    reward -= 0.05 * self.penalty_scale
            else:
                # In loss
                if outcome.get('recovered', False):
                    reward += 0.3 * self.reward_scale  # Good hold
                elif outcome.get('loss_increased', False):
                    reward -= 0.15 * self.penalty_scale
        
        elif action == ExitAction.EXIT_NOW:
            # Full exit
            if realized_pnl_r > 0:
                # Profitable exit
                reward += realized_pnl_r * 0.5 * self.reward_scale
                
                # Bonus if exited near resistance (long) or support (short)
                if outcome.get('near_target', False):
                    reward += 0.2 * self.reward_scale
            else:
                # Loss exit
                if realized_pnl_r > -1:
                    # Small loss is okay
                    reward += 0.1 * self.reward_scale
                else:
                    # Large loss
                    reward += realized_pnl_r * 0.3 * self.penalty_scale
        
        elif action in [ExitAction.EXIT_PARTIAL_25, ExitAction.EXIT_PARTIAL_50, 
                        ExitAction.EXIT_PARTIAL_75]:
            # Partial exit
            if realized_pnl_r > 0:
                # Took partial profit
                reward += realized_pnl_r * 0.4 * self.reward_scale
                
                # Bonus for scaling out at good levels
                if outcome.get('at_resistance', False) or outcome.get('at_ob', False):
                    reward += 0.15 * self.reward_scale
        
        elif action == ExitAction.MOVE_STOP_BE:
            # Moved stop to breakeven
            if outcome.get('stop_hit_be', False):
                # Saved from loss
                reward += 0.3 * self.reward_scale
            elif outcome.get('continued_profit', False):
                # Good decision
                reward += 0.1 * self.reward_scale
        
        elif action in [ExitAction.TRAIL_STOP_TIGHT, ExitAction.TRAIL_STOP_WIDE]:
            # Trailing stop
            if outcome.get('stop_hit', False):
                if realized_pnl_r > 0:
                    reward += realized_pnl_r * 0.4 * self.reward_scale
                else:
                    reward -= 0.1 * self.penalty_scale
        
        return reward


# =============================================================================
# TRADING ENVIRONMENT
# =============================================================================

class TradingEnvironment:
    """
    Simulated trading environment for RL training
    Simulates market dynamics and ICT conditions
    """
    
    def __init__(
        self,
        historical_data: Optional[List[Dict]] = None,
        pip_size: float = 0.0001,
        spread: float = 1.0,
        commission: float = 0.0
    ):
        """
        Initialize environment
        
        Args:
            historical_data: List of OHLCV bars with ICT features
            pip_size: Size of one pip
            spread: Spread in pips
            commission: Commission per trade
        """
        self.historical_data = historical_data or []
        self.pip_size = pip_size
        self.spread = spread * pip_size
        self.commission = commission
        
        # State
        self.current_bar = 0
        self.position = PositionState()
        self.done = False
        
        # Statistics
        self.trades = []
        self.total_pnl = 0.0
        self.wins = 0
        self.losses = 0
    
    def reset(self) -> RLState:
        """Reset environment to initial state"""
        self.current_bar = 0
        self.position = PositionState()
        self.done = False
        self.trades = []
        self.total_pnl = 0.0
        self.wins = 0
        self.losses = 0
        
        return self._get_state()
    
    def _get_state(self) -> RLState:
        """Get current state"""
        if self.current_bar >= len(self.historical_data):
            # Return empty state if past data
            return RLState(
                market=MarketState(current_price=0),
                position=self.position
            )
        
        bar = self.historical_data[self.current_bar]
        
        market_state = MarketState(
            current_price=bar.get('close', 0),
            bid=bar.get('bid', bar.get('close', 0) - self.spread / 2),
            ask=bar.get('ask', bar.get('close', 0) + self.spread / 2),
            spread=self.spread,
            momentum_5=bar.get('momentum_5', 0),
            momentum_20=bar.get('momentum_20', 0),
            volatility=bar.get('volatility', 0),
            atr=bar.get('atr', self.pip_size * 50),
            session_high=bar.get('session_high', bar.get('high', 0)),
            session_low=bar.get('session_low', bar.get('low', 0)),
            range_position=bar.get('range_position', 0.5),
            rsi=bar.get('rsi', 50),
            nearest_ob_distance=bar.get('ob_distance', 0),
            ob_strength=bar.get('ob_strength', 0),
            nearest_fvg_distance=bar.get('fvg_distance', 0),
            fvg_fill_percent=bar.get('fvg_fill_pct', 0),
            nearest_liquidity_distance=bar.get('liquidity_distance', 0),
            in_premium=bar.get('in_premium', False),
            in_discount=bar.get('in_discount', False),
            in_ote_zone=bar.get('in_ote', False),
            trend=bar.get('trend', 'ranging'),
            trend_strength=bar.get('trend_strength', 0),
            in_kill_zone=bar.get('in_kill_zone', False),
            is_macro_time=bar.get('is_macro_time', False),
            signal_quality=bar.get('signal_quality', 0),
            signal_confluence=bar.get('confluence', 0),
            htf_alignment=bar.get('htf_alignment', 0),
            ltf_alignment=bar.get('ltf_alignment', 0),
        )
        
        # Update position with current price
        if self.position.status != PositionStatus.FLAT:
            self.position.current_price = market_state.current_price
            self._update_position_pnl()
        
        return RLState(
            market=market_state,
            position=self.position,
            timestamp=bar.get('timestamp', datetime.now())
        )
    
    def _update_position_pnl(self):
        """Update position P/L"""
        if self.position.status == PositionStatus.FLAT:
            return
        
        if self.position.direction == 'long':
            pnl = self.position.current_price - self.position.entry_price
        else:  # short
            pnl = self.position.entry_price - self.position.current_price
        
        self.position.unrealized_pnl = pnl / self.pip_size
        
        # Calculate R multiple
        risk = abs(self.position.entry_price - self.position.stop_loss)
        if risk > 0:
            self.position.unrealized_pnl_r = pnl / risk
        
        # Update MFE/MAE
        if self.position.unrealized_pnl > self.position.max_favorable:
            self.position.max_favorable = self.position.unrealized_pnl
        if self.position.unrealized_pnl < -self.position.max_adverse:
            self.position.max_adverse = abs(self.position.unrealized_pnl)
        
        self.position.bars_held += 1
    
    def step_entry(self, action: EntryAction, signal: Dict) -> Tuple[RLState, float, bool, Dict]:
        """
        Execute entry action
        
        Args:
            action: Entry action to take
            signal: Signal information with entry/SL/TP
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        info = {'filled': False}
        
        state = self._get_state()
        
        if action == EntryAction.PASS:
            # Skip signal
            info['action'] = 'pass'
            
        elif action == EntryAction.ENTER_NOW:
            # Enter immediately
            self._open_position(signal, state.market)
            info['filled'] = True
            info['fill_price'] = self.position.entry_price
            info['signal_entry'] = signal.get('entry', 0)
            
        elif action == EntryAction.ENTER_LIMIT:
            # Place limit order (simplified: enters at signal entry if available)
            limit_price = signal.get('entry', state.market.current_price)
            if state.market.current_price <= limit_price and signal.get('direction') == 'long':
                self._open_position(signal, state.market, limit_price)
                info['filled'] = True
                info['fill_price'] = limit_price
                info['signal_entry'] = signal.get('entry', 0)
            elif state.market.current_price >= limit_price and signal.get('direction') == 'short':
                self._open_position(signal, state.market, limit_price)
                info['filled'] = True
                info['fill_price'] = limit_price
                info['signal_entry'] = signal.get('entry', 0)
        
        # Move to next bar
        self.current_bar += 1
        self.done = self.current_bar >= len(self.historical_data)
        
        # Get next state and calculate reward
        next_state = self._get_state()
        reward = 0  # Entry reward calculated separately
        
        return next_state, reward, self.done, info
    
    def step_exit(self, action: ExitAction) -> Tuple[RLState, float, bool, Dict]:
        """
        Execute exit action
        
        Args:
            action: Exit action to take
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        info = {}
        realized_pnl_r = 0
        
        state = self._get_state()
        
        if self.position.status == PositionStatus.FLAT:
            # No position to manage
            pass
        
        elif action == ExitAction.HOLD:
            info['action'] = 'hold'
            info['continued_profit'] = self.position.unrealized_pnl_r > 0
            info['profit_decreased'] = False
            info['loss_increased'] = False
            info['recovered'] = False
            
        elif action == ExitAction.EXIT_NOW:
            realized_pnl_r = self._close_position(1.0)
            info['realized_pnl_r'] = realized_pnl_r
            info['near_target'] = abs(state.market.current_price - self.position.take_profit) < state.market.atr
            
        elif action == ExitAction.EXIT_PARTIAL_25:
            realized_pnl_r = self._close_position(0.25)
            info['realized_pnl_r'] = realized_pnl_r
            
        elif action == ExitAction.EXIT_PARTIAL_50:
            realized_pnl_r = self._close_position(0.50)
            info['realized_pnl_r'] = realized_pnl_r
            
        elif action == ExitAction.EXIT_PARTIAL_75:
            realized_pnl_r = self._close_position(0.75)
            info['realized_pnl_r'] = realized_pnl_r
            
        elif action == ExitAction.MOVE_STOP_BE:
            if not self.position.stop_moved_be and self.position.unrealized_pnl_r > 0.5:
                self.position.stop_loss = self.position.entry_price
                self.position.stop_moved_be = True
                info['stop_moved'] = True
            
        elif action == ExitAction.TRAIL_STOP_TIGHT:
            self._apply_trailing_stop(tight=True)
            self.position.trailing_active = True
            
        elif action == ExitAction.TRAIL_STOP_WIDE:
            self._apply_trailing_stop(tight=False)
            self.position.trailing_active = True
        
        # Check if stop or target hit
        self._check_stop_target()
        
        # Move to next bar
        self.current_bar += 1
        self.done = self.current_bar >= len(self.historical_data)
        
        if self.position.status == PositionStatus.FLAT:
            self.done = True
        
        next_state = self._get_state()
        reward = realized_pnl_r  # Basic reward
        
        return next_state, reward, self.done, info
    
    def _open_position(self, signal: Dict, market: MarketState, price: Optional[float] = None):
        """Open a position"""
        direction = signal.get('direction', 'long')
        
        if price is None:
            if direction == 'long':
                price = market.ask
            else:
                price = market.bid
        
        self.position = PositionState(
            status=PositionStatus.LONG if direction == 'long' else PositionStatus.SHORT,
            direction=direction,
            entry_price=price,
            current_price=price,
            stop_loss=signal.get('stop_loss', 0),
            take_profit=signal.get('take_profit', 0),
            position_size=signal.get('size', 1.0),
            entry_time=datetime.now(),
        )
    
    def _close_position(self, fraction: float) -> float:
        """Close position (or fraction)"""
        if self.position.status == PositionStatus.FLAT:
            return 0
        
        pnl_r = self.position.unrealized_pnl_r * fraction
        
        if fraction >= 1.0:
            # Full close
            self.trades.append({
                'direction': self.position.direction,
                'entry': self.position.entry_price,
                'exit': self.position.current_price,
                'pnl_r': pnl_r,
                'bars_held': self.position.bars_held,
            })
            self.total_pnl += pnl_r
            
            if pnl_r > 0:
                self.wins += 1
            else:
                self.losses += 1
            
            self.position = PositionState()
        else:
            # Partial close
            self.position.remaining_size -= fraction
            self.position.partial_profits += pnl_r
        
        return pnl_r
    
    def _check_stop_target(self):
        """Check if stop loss or take profit is hit"""
        if self.position.status == PositionStatus.FLAT:
            return
        
        bar = self.historical_data[self.current_bar] if self.current_bar < len(self.historical_data) else None
        if not bar:
            return
        
        high = bar.get('high', self.position.current_price)
        low = bar.get('low', self.position.current_price)
        
        if self.position.direction == 'long':
            if low <= self.position.stop_loss:
                self.position.current_price = self.position.stop_loss
                self._close_position(1.0)
            elif high >= self.position.take_profit:
                self.position.current_price = self.position.take_profit
                self._close_position(1.0)
        else:  # short
            if high >= self.position.stop_loss:
                self.position.current_price = self.position.stop_loss
                self._close_position(1.0)
            elif low <= self.position.take_profit:
                self.position.current_price = self.position.take_profit
                self._close_position(1.0)
    
    def _apply_trailing_stop(self, tight: bool = False):
        """Apply trailing stop"""
        if self.position.status == PositionStatus.FLAT:
            return
        
        atr = self.historical_data[self.current_bar].get('atr', self.pip_size * 50) if self.current_bar < len(self.historical_data) else self.pip_size * 50
        trail_distance = atr * (1.0 if tight else 2.0)
        
        if self.position.direction == 'long':
            new_stop = self.position.current_price - trail_distance
            if new_stop > self.position.stop_loss:
                self.position.stop_loss = new_stop
        else:  # short
            new_stop = self.position.current_price + trail_distance
            if new_stop < self.position.stop_loss:
                self.position.stop_loss = new_stop
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get environment statistics"""
        win_rate = self.wins / max(self.wins + self.losses, 1) * 100
        return {
            'total_trades': len(self.trades),
            'wins': self.wins,
            'losses': self.losses,
            'win_rate': win_rate,
            'total_pnl_r': self.total_pnl,
            'avg_pnl_r': self.total_pnl / max(len(self.trades), 1),
        }


# =============================================================================
# ICT REINFORCEMENT LEARNING AGENT
# =============================================================================

class ICTReinforcementLearningAgent:
    """
    Complete RL agent for ICT trading entry/exit optimization
    
    Combines entry agent and exit agent with ICT-specific features
    """
    
    def __init__(self, config: Optional[TrainingConfig] = None):
        """
        Initialize ICT RL agent
        
        Args:
            config: Training configuration
        """
        self.config = config or TrainingConfig()
        
        # Entry agent
        self.entry_agent = DQNAgent(
            state_size=self.config.state_size_entry,
            action_size=len(EntryAction),
            config=self.config,
            name="entry_agent"
        )
        
        # Exit agent
        self.exit_agent = DQNAgent(
            state_size=self.config.state_size_exit,
            action_size=len(ExitAction),
            config=self.config,
            name="exit_agent"
        )
        
        # Reward calculator
        self.reward_calculator = RewardCalculator(
            reward_scale=self.config.reward_scale,
            penalty_scale=self.config.penalty_scale
        )
        
        # Statistics
        self.entry_decisions = []
        self.exit_decisions = []
        self.episode_rewards = []
        
        logger.info("ICT RL Agent initialized")
    
    def select_entry_action(
        self,
        state: RLState,
        signal: Dict,
        training: bool = True
    ) -> Tuple[EntryAction, Dict]:
        """
        Select optimal entry action
        
        Args:
            state: Current market/position state
            signal: ICT signal information
            training: Whether in training mode
            
        Returns:
            Tuple of (action, action_info)
        """
        state_vector = state.to_entry_vector()
        
        # Get action from agent
        action_idx = self.entry_agent.select_action(state_vector, training)
        action = EntryAction(action_idx)
        
        # Get Q-values for analysis
        q_values = self.entry_agent.get_q_values(state_vector)
        
        # Build action info
        action_info = {
            'action': action.name,
            'q_values': {EntryAction(i).name: float(q_values[i]) for i in range(len(EntryAction))},
            'confidence': float(np.max(q_values) - np.mean(q_values)),
            'signal_quality': state.market.signal_quality,
            'confluence': state.market.signal_confluence,
            'in_kill_zone': state.market.in_kill_zone,
            'epsilon': self.entry_agent.epsilon if training else 0,
        }
        
        # Apply ICT filters
        action, action_info = self._apply_ict_entry_filters(action, action_info, state, signal)
        
        return action, action_info
    
    def select_exit_action(
        self,
        state: RLState,
        training: bool = True
    ) -> Tuple[ExitAction, Dict]:
        """
        Select optimal exit action
        
        Args:
            state: Current market/position state
            training: Whether in training mode
            
        Returns:
            Tuple of (action, action_info)
        """
        state_vector = state.to_exit_vector()
        
        # Get action from agent
        action_idx = self.exit_agent.select_action(state_vector, training)
        action = ExitAction(action_idx)
        
        # Get Q-values for analysis
        q_values = self.exit_agent.get_q_values(state_vector)
        
        # Build action info
        action_info = {
            'action': action.name,
            'q_values': {ExitAction(i).name: float(q_values[i]) for i in range(len(ExitAction))},
            'confidence': float(np.max(q_values) - np.mean(q_values)),
            'unrealized_pnl_r': state.position.unrealized_pnl_r,
            'bars_held': state.position.bars_held,
            'remaining_size': state.position.remaining_size,
            'epsilon': self.exit_agent.epsilon if training else 0,
        }
        
        # Apply ICT filters
        action, action_info = self._apply_ict_exit_filters(action, action_info, state)
        
        return action, action_info
    
    def _apply_ict_entry_filters(
        self,
        action: EntryAction,
        action_info: Dict,
        state: RLState,
        signal: Dict
    ) -> Tuple[EntryAction, Dict]:
        """Apply ICT-specific entry filters"""
        
        # Rule 1: Don't enter outside kill zones for certain setups
        if signal.get('model') in ['silver_bullet', 'model_2022']:
            if not state.market.in_kill_zone:
                action = EntryAction.WAIT_CONFIRMATION
                action_info['filter_applied'] = 'kill_zone_required'
        
        # Rule 2: Prefer pullback entries when in premium/discount
        if state.market.in_ote_zone and action == EntryAction.ENTER_NOW:
            if state.market.signal_quality > 70:
                action = EntryAction.ENTER_LIMIT
                action_info['filter_applied'] = 'ote_limit_entry'
        
        # Rule 3: Skip low confluence signals
        if state.market.signal_confluence < 3 and action != EntryAction.PASS:
            if random.random() > state.market.signal_confluence / 5:
                action = EntryAction.PASS
                action_info['filter_applied'] = 'low_confluence'
        
        # Rule 4: Don't enter against HTF bias
        if state.market.htf_alignment < 30:
            if action in [EntryAction.ENTER_NOW, EntryAction.ENTER_BREAKOUT]:
                action = EntryAction.WAIT_CONFIRMATION
                action_info['filter_applied'] = 'htf_misalignment'
        
        return action, action_info
    
    def _apply_ict_exit_filters(
        self,
        action: ExitAction,
        action_info: Dict,
        state: RLState
    ) -> Tuple[ExitAction, Dict]:
        """Apply ICT-specific exit filters"""
        
        # Rule 1: Move to BE after 1R profit
        if state.position.unrealized_pnl_r >= 1.0 and not state.position.stop_moved_be:
            action = ExitAction.MOVE_STOP_BE
            action_info['filter_applied'] = '1r_move_be'
        
        # Rule 2: Take partials at opposing PD arrays
        if state.market.nearest_ob_distance < 10 and state.position.unrealized_pnl_r > 0.5:
            if state.position.remaining_size > 0.5:
                action = ExitAction.EXIT_PARTIAL_50
                action_info['filter_applied'] = 'opposing_ob_partial'
        
        # Rule 3: Exit if structure breaks against position
        if state.market.bos_distance < 5:  # Close to structure break
            if state.position.unrealized_pnl_r < 0:
                action = ExitAction.EXIT_NOW
                action_info['filter_applied'] = 'structure_break_exit'
        
        # Rule 4: Trail stop in trending conditions
        if state.market.trend_strength > 70 and state.position.unrealized_pnl_r > 1.5:
            if not state.position.trailing_active:
                action = ExitAction.TRAIL_STOP_WIDE
                action_info['filter_applied'] = 'trend_trail'
        
        # Rule 5: Tighter trail in volatile conditions
        if state.market.regime == MarketRegime.VOLATILE and state.position.trailing_active:
            action = ExitAction.TRAIL_STOP_TIGHT
            action_info['filter_applied'] = 'volatile_tight_trail'
        
        return action, action_info
    
    def train_entry(
        self,
        state: RLState,
        action: EntryAction,
        reward: float,
        next_state: RLState,
        done: bool
    ):
        """Train entry agent on experience"""
        state_vector = state.to_entry_vector()
        next_state_vector = next_state.to_entry_vector()
        
        self.entry_agent.store_experience(
            state_vector, action.value, reward, next_state_vector, done
        )
        
        if self.entry_agent.training_step % self.config.train_freq == 0:
            self.entry_agent.train_step()
    
    def train_exit(
        self,
        state: RLState,
        action: ExitAction,
        reward: float,
        next_state: RLState,
        done: bool
    ):
        """Train exit agent on experience"""
        state_vector = state.to_exit_vector()
        next_state_vector = next_state.to_exit_vector()
        
        self.exit_agent.store_experience(
            state_vector, action.value, reward, next_state_vector, done
        )
        
        if self.exit_agent.training_step % self.config.train_freq == 0:
            self.exit_agent.train_step()
    
    def train_episode(
        self,
        env: TradingEnvironment,
        signal: Dict
    ) -> Dict[str, Any]:
        """
        Train on a single episode (one trade setup)
        
        Args:
            env: Trading environment
            signal: ICT signal to trade
            
        Returns:
            Episode statistics
        """
        state = env.reset()
        total_entry_reward = 0
        total_exit_reward = 0
        steps = 0
        
        # Entry phase
        entry_done = False
        while not entry_done and steps < self.config.max_steps_per_episode:
            # Select entry action
            action, action_info = self.select_entry_action(state, signal, training=True)
            
            # Execute action
            next_state, _, done, info = env.step_entry(action, signal)
            
            # Calculate reward
            reward = self.reward_calculator.calculate_entry_reward(action, state, info)
            total_entry_reward += reward
            
            # Train
            self.train_entry(state, action, reward, next_state, done)
            
            # Record decision
            self.entry_decisions.append({
                'action': action.name,
                'q_values': action_info.get('q_values', {}),
                'reward': reward,
                'filled': info.get('filled', False),
            })
            
            state = next_state
            steps += 1
            
            # Check if entry phase complete
            if action != EntryAction.WAIT_CONFIRMATION or done:
                entry_done = True
        
        # Exit phase (if position opened)
        if env.position.status != PositionStatus.FLAT:
            while not env.done and steps < self.config.max_steps_per_episode:
                # Select exit action
                action, action_info = self.select_exit_action(state, training=True)
                
                # Execute action
                next_state, _, done, info = env.step_exit(action)
                
                # Calculate reward
                reward = self.reward_calculator.calculate_exit_reward(action, state, info)
                total_exit_reward += reward
                
                # Train
                self.train_exit(state, action, reward, next_state, done)
                
                # Record decision
                self.exit_decisions.append({
                    'action': action.name,
                    'q_values': action_info.get('q_values', {}),
                    'reward': reward,
                    'pnl_r': info.get('realized_pnl_r', 0),
                })
                
                state = next_state
                steps += 1
        
        # End episode
        self.entry_agent.end_episode()
        self.exit_agent.end_episode()
        
        episode_reward = total_entry_reward + total_exit_reward
        self.episode_rewards.append(episode_reward)
        
        return {
            'entry_reward': total_entry_reward,
            'exit_reward': total_exit_reward,
            'total_reward': episode_reward,
            'steps': steps,
            'env_stats': env.get_statistics(),
        }
    
    def get_optimal_entry_timing(
        self,
        state: RLState,
        signal: Dict
    ) -> Dict[str, Any]:
        """
        Get optimal entry timing recommendation
        
        Args:
            state: Current state
            signal: ICT signal
            
        Returns:
            Entry timing recommendation
        """
        action, action_info = self.select_entry_action(state, signal, training=False)
        
        # Build recommendation
        recommendation = {
            'action': action.name,
            'confidence': action_info['confidence'],
            'reasoning': [],
        }
        
        # Add reasoning based on state
        if state.market.in_kill_zone:
            recommendation['reasoning'].append("In optimal kill zone for entry")
        
        if state.market.in_ote_zone:
            recommendation['reasoning'].append("Price in OTE zone (61.8-78.6 fib)")
        
        if state.market.signal_confluence >= 5:
            recommendation['reasoning'].append(f"High confluence ({state.market.signal_confluence} factors)")
        
        if state.market.htf_alignment > 70:
            recommendation['reasoning'].append("Strong HTF bias alignment")
        
        if action == EntryAction.PASS:
            recommendation['reasoning'].append("Signal quality below threshold")
        elif action == EntryAction.ENTER_LIMIT:
            recommendation['reasoning'].append("Recommend limit order at OTE level")
        elif action == EntryAction.ENTER_PULLBACK:
            recommendation['reasoning'].append("Wait for pullback before entry")
        
        return recommendation
    
    def get_optimal_exit_timing(
        self,
        state: RLState
    ) -> Dict[str, Any]:
        """
        Get optimal exit timing recommendation
        
        Args:
            state: Current state
            
        Returns:
            Exit timing recommendation
        """
        action, action_info = self.select_exit_action(state, training=False)
        
        recommendation = {
            'action': action.name,
            'confidence': action_info['confidence'],
            'unrealized_pnl_r': state.position.unrealized_pnl_r,
            'reasoning': [],
        }
        
        # Add reasoning
        if state.position.unrealized_pnl_r >= 2.0:
            recommendation['reasoning'].append("Reached 2R profit target")
        
        if state.position.unrealized_pnl_r >= 1.0 and not state.position.stop_moved_be:
            recommendation['reasoning'].append("Consider moving stop to breakeven")
        
        if state.market.regime == MarketRegime.VOLATILE:
            recommendation['reasoning'].append("High volatility - consider tighter management")
        
        if state.position.bars_held > 50:
            recommendation['reasoning'].append("Extended hold time - evaluate continuation")
        
        if action == ExitAction.EXIT_NOW:
            recommendation['reasoning'].append("Recommend full exit")
        elif action == ExitAction.EXIT_PARTIAL_50:
            recommendation['reasoning'].append("Recommend taking partial profits (50%)")
        elif action == ExitAction.TRAIL_STOP_WIDE:
            recommendation['reasoning'].append("Recommend trailing stop to lock in gains")
        
        return recommendation
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get combined agent metrics"""
        return {
            'entry_agent': self.entry_agent.get_metrics(),
            'exit_agent': self.exit_agent.get_metrics(),
            'avg_episode_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'total_entry_decisions': len(self.entry_decisions),
            'total_exit_decisions': len(self.exit_decisions),
        }
    
    def save(self, directory: str):
        """Save agents to directory"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        self.entry_agent.save(f"{directory}/entry_agent.json")
        self.exit_agent.save(f"{directory}/exit_agent.json")
        
        # Save metrics
        with open(f"{directory}/metrics.json", 'w') as f:
            json.dump(self.get_metrics(), f)
        
        logger.info(f"ICT RL Agent saved to {directory}")
    
    def load(self, directory: str):
        """Load agents from directory"""
        self.entry_agent.load(f"{directory}/entry_agent.json")
        self.exit_agent.load(f"{directory}/exit_agent.json")
        logger.info(f"ICT RL Agent loaded from {directory}")


# =============================================================================
# SYNTHETIC DATA GENERATOR FOR TRAINING
# =============================================================================

class SyntheticTrainingDataGenerator:
    """
    Generates synthetic training data with realistic ICT patterns
    """
    
    def __init__(
        self,
        base_price: float = 1.1000,
        volatility: float = 0.001,
        pip_size: float = 0.0001
    ):
        self.base_price = base_price
        self.volatility = volatility
        self.pip_size = pip_size
    
    def generate_episode_data(
        self,
        num_bars: int = 100,
        trend: str = 'bullish',
        include_ict_features: bool = True
    ) -> List[Dict]:
        """
        Generate synthetic OHLCV data with ICT features
        
        Args:
            num_bars: Number of bars to generate
            trend: 'bullish', 'bearish', or 'ranging'
            include_ict_features: Whether to include ICT features
            
        Returns:
            List of bar dictionaries
        """
        bars = []
        price = self.base_price
        
        # Trend drift
        if trend == 'bullish':
            drift = self.volatility * 0.3
        elif trend == 'bearish':
            drift = -self.volatility * 0.3
        else:
            drift = 0
        
        for i in range(num_bars):
            # Generate OHLCV
            change = np.random.randn() * self.volatility + drift
            open_price = price
            close_price = price + change
            high_price = max(open_price, close_price) + abs(np.random.randn() * self.volatility * 0.5)
            low_price = min(open_price, close_price) - abs(np.random.randn() * self.volatility * 0.5)
            
            bar = {
                'timestamp': datetime.now() + timedelta(minutes=i * 5),
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': random.randint(100, 10000),
            }
            
            if include_ict_features:
                # Add ICT features
                bar.update(self._generate_ict_features(i, trend, close_price))
            
            bars.append(bar)
            price = close_price
        
        return bars
    
    def _generate_ict_features(self, bar_index: int, trend: str, price: float) -> Dict:
        """Generate realistic ICT features"""
        
        # Session and time (rotate through sessions)
        sessions = [SessionType.ASIAN, SessionType.LONDON, SessionType.NEW_YORK, SessionType.OVERLAP]
        session = sessions[bar_index % 4]
        
        # Kill zone (30% chance during London/NY)
        in_kill_zone = session in [SessionType.LONDON, SessionType.NEW_YORK] and random.random() < 0.3
        
        # ICT elements
        ob_distance = random.uniform(5, 50)  # Pips to nearest OB
        ob_strength = random.uniform(40, 90)
        fvg_distance = random.uniform(3, 40)
        fvg_fill_pct = random.uniform(0, 100)
        liquidity_distance = random.uniform(10, 100)
        
        # Zones
        in_premium = price > self.base_price and random.random() < 0.4
        in_discount = price < self.base_price and random.random() < 0.4
        in_ote = random.random() < 0.2  # 20% chance in OTE
        
        # Structure
        trend_strength = random.uniform(30, 90) if trend != 'ranging' else random.uniform(10, 40)
        
        # Signal quality (higher in kill zones with good structure)
        base_quality = random.uniform(40, 80)
        if in_kill_zone:
            base_quality += 10
        if in_ote:
            base_quality += 10
        signal_quality = min(100, base_quality)
        
        # Confluence
        confluence = 0
        if in_kill_zone:
            confluence += 1
        if in_ote:
            confluence += 1
        if ob_distance < 20:
            confluence += 1
        if fvg_distance < 15:
            confluence += 1
        if trend_strength > 60:
            confluence += 1
        confluence += random.randint(0, 3)
        
        return {
            'session': session,
            'in_kill_zone': in_kill_zone,
            'is_macro_time': random.random() < 0.15,
            'ob_distance': ob_distance,
            'ob_strength': ob_strength,
            'fvg_distance': fvg_distance,
            'fvg_fill_pct': fvg_fill_pct,
            'liquidity_distance': liquidity_distance,
            'in_premium': in_premium,
            'in_discount': in_discount,
            'in_ote': in_ote,
            'trend': trend,
            'trend_strength': trend_strength,
            'signal_quality': signal_quality,
            'confluence': confluence,
            'htf_alignment': random.uniform(40, 90) if trend != 'ranging' else random.uniform(20, 60),
            'ltf_alignment': random.uniform(50, 95),
            'momentum_5': random.uniform(-50, 50),
            'momentum_20': random.uniform(-30, 30),
            'volatility': random.uniform(10, 50),
            'atr': self.volatility * 10,
            'range_position': random.uniform(0.2, 0.8),
            'rsi': random.uniform(30, 70),
            'bos_distance': random.uniform(5, 50),
        }
    
    def generate_signal(self, trend: str = 'bullish') -> Dict:
        """Generate a realistic ICT signal"""
        direction = 'long' if trend == 'bullish' else ('short' if trend == 'bearish' else random.choice(['long', 'short']))
        
        entry = self.base_price + random.uniform(-0.001, 0.001)
        
        if direction == 'long':
            stop_loss = entry - random.uniform(0.001, 0.003)
            take_profit = entry + random.uniform(0.002, 0.006)
        else:
            stop_loss = entry + random.uniform(0.001, 0.003)
            take_profit = entry - random.uniform(0.002, 0.006)
        
        return {
            'direction': direction,
            'entry': entry,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'model': random.choice(['model_2022', 'silver_bullet', 'power_of_three']),
            'size': 1.0,
        }


# =============================================================================
# TRAINING ORCHESTRATOR
# =============================================================================

class RLTrainingOrchestrator:
    """
    Orchestrates RL training process
    """
    
    def __init__(self, agent: ICTReinforcementLearningAgent):
        """
        Initialize orchestrator
        
        Args:
            agent: ICT RL agent to train
        """
        self.agent = agent
        self.data_generator = SyntheticTrainingDataGenerator()
        self.training_history = []
    
    def run_training(
        self,
        num_episodes: int = 1000,
        log_interval: int = 100,
        save_interval: int = 500,
        save_directory: str = "./rl_checkpoints"
    ) -> Dict[str, Any]:
        """
        Run training for specified number of episodes
        
        Args:
            num_episodes: Number of episodes to train
            log_interval: Episodes between logging
            save_interval: Episodes between saving
            save_directory: Directory to save checkpoints
            
        Returns:
            Training statistics
        """
        logger.info(f"Starting RL training for {num_episodes} episodes")
        
        for episode in range(num_episodes):
            # Generate episode data
            trend = random.choice(['bullish', 'bearish', 'ranging'])
            episode_data = self.data_generator.generate_episode_data(
                num_bars=random.randint(50, 150),
                trend=trend
            )
            
            # Create environment
            env = TradingEnvironment(
                historical_data=episode_data,
                pip_size=0.0001,
                spread=1.0
            )
            
            # Generate signal
            signal = self.data_generator.generate_signal(trend)
            
            # Train episode
            episode_stats = self.agent.train_episode(env, signal)
            
            self.training_history.append({
                'episode': episode,
                **episode_stats
            })
            
            # Logging
            if episode % log_interval == 0:
                metrics = self.agent.get_metrics()
                logger.info(
                    f"Episode {episode}: "
                    f"Reward={episode_stats['total_reward']:.3f}, "
                    f"Entry ε={metrics['entry_agent']['epsilon']:.3f}, "
                    f"Exit ε={metrics['exit_agent']['epsilon']:.3f}, "
                    f"Env Win Rate={episode_stats['env_stats']['win_rate']:.1f}%"
                )
            
            # Save checkpoint
            if episode % save_interval == 0 and episode > 0:
                self.agent.save(f"{save_directory}/episode_{episode}")
        
        # Final save
        self.agent.save(f"{save_directory}/final")
        
        return self._compile_training_stats()
    
    def _compile_training_stats(self) -> Dict[str, Any]:
        """Compile training statistics"""
        if not self.training_history:
            return {}
        
        rewards = [h['total_reward'] for h in self.training_history]
        win_rates = [h['env_stats']['win_rate'] for h in self.training_history]
        
        return {
            'total_episodes': len(self.training_history),
            'avg_reward': np.mean(rewards),
            'max_reward': max(rewards),
            'min_reward': min(rewards),
            'final_avg_reward': np.mean(rewards[-100:]),
            'avg_win_rate': np.mean(win_rates),
            'final_win_rate': np.mean(win_rates[-100:]),
            'agent_metrics': self.agent.get_metrics(),
        }


# =============================================================================
# USAGE EXAMPLES AND TESTS
# =============================================================================

def run_training_example():
    """Example training run"""
    print("=" * 60)
    print("ICT REINFORCEMENT LEARNING AGENT - TRAINING EXAMPLE")
    print("=" * 60)
    
    # Initialize agent
    config = TrainingConfig(
        epsilon_decay=0.999,
        min_experiences=100,
        batch_size=32,
    )
    
    agent = ICTReinforcementLearningAgent(config)
    
    # Create orchestrator
    orchestrator = RLTrainingOrchestrator(agent)
    
    # Run training
    stats = orchestrator.run_training(
        num_episodes=100,
        log_interval=10,
        save_interval=50,
        save_directory="./rl_training"
    )
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total Episodes: {stats['total_episodes']}")
    print(f"Average Reward: {stats['avg_reward']:.3f}")
    print(f"Final Win Rate: {stats['final_win_rate']:.1f}%")
    print(f"Entry Agent Steps: {stats['agent_metrics']['entry_agent']['training_step']}")
    print(f"Exit Agent Steps: {stats['agent_metrics']['exit_agent']['training_step']}")


def run_inference_example():
    """Example inference (decision making)"""
    print("=" * 60)
    print("ICT REINFORCEMENT LEARNING AGENT - INFERENCE EXAMPLE")
    print("=" * 60)
    
    # Initialize agent
    agent = ICTReinforcementLearningAgent()
    
    # Create sample state
    market_state = MarketState(
        current_price=1.1050,
        momentum_5=25.0,
        momentum_20=15.0,
        volatility=30.0,
        atr=0.0015,
        range_position=0.6,
        rsi=55,
        nearest_ob_distance=15.0,
        ob_strength=75.0,
        nearest_fvg_distance=8.0,
        fvg_fill_percent=40.0,
        in_discount=True,
        in_ote_zone=True,
        trend='bullish',
        trend_strength=70.0,
        session=SessionType.LONDON,
        in_kill_zone=True,
        signal_quality=78.0,
        signal_confluence=5,
        htf_alignment=80.0,
        ltf_alignment=85.0,
    )
    
    state = RLState(market=market_state, position=PositionState())
    
    signal = {
        'direction': 'long',
        'entry': 1.1050,
        'stop_loss': 1.1020,
        'take_profit': 1.1140,
        'model': 'silver_bullet',
    }
    
    # Get entry recommendation
    entry_rec = agent.get_optimal_entry_timing(state, signal)
    
    print("\nENTRY RECOMMENDATION:")
    print(f"  Action: {entry_rec['action']}")
    print(f"  Confidence: {entry_rec['confidence']:.3f}")
    print("  Reasoning:")
    for reason in entry_rec['reasoning']:
        print(f"    - {reason}")
    
    # Simulate being in a position
    position_state = PositionState(
        status=PositionStatus.LONG,
        direction='long',
        entry_price=1.1050,
        current_price=1.1085,
        stop_loss=1.1020,
        take_profit=1.1140,
        unrealized_pnl=35.0,
        unrealized_pnl_r=1.17,
        bars_held=25,
    )
    
    state_with_position = RLState(market=market_state, position=position_state)
    
    # Get exit recommendation
    exit_rec = agent.get_optimal_exit_timing(state_with_position)
    
    print("\nEXIT RECOMMENDATION:")
    print(f"  Action: {exit_rec['action']}")
    print(f"  Confidence: {exit_rec['confidence']:.3f}")
    print(f"  Current P/L: {exit_rec['unrealized_pnl_r']:.2f}R")
    print("  Reasoning:")
    for reason in exit_rec['reasoning']:
        print(f"    - {reason}")


if __name__ == "__main__":
    # Run examples
    run_training_example()
    print("\n" * 2)
    run_inference_example()
