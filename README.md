# ICT Algorithmic Trading Bot

## Complete Python Trading System Based on Inner Circle Trader (ICT) Methodology

**Version:** 1.0.0  
**Total Lines of Code:** 32,946  
**Total Files:** 24 Python modules

---

## ⚠️ DISCLAIMER

**Trading involves significant risk of loss. This software is for educational purposes only. Always test in a paper trading account first before using real money. Past performance does not guarantee future results.**

---

## 📁 Project Structure

```
ict_trading_bot/
│
├── # PHASE 1: ICT HANDLERS (Core ICT Concepts)
├── fvg_handler.py              # Fair Value Gaps (BISI/SIBI, CE, inversions)
├── order_block_handler.py      # Order Blocks (breakers, propulsion, mitigation)
├── liquidity_handler.py        # Liquidity pools, sweeps, stop hunts
├── market_structure_handler.py # BOS/CHoCH/MSS, swing detection
├── gap_handler.py              # NWOG/NDOG, opening ranges
├── pd_array_handler.py         # Premium/Discount zones, PD arrays
├── timeframe_handler.py        # Kill zones, sessions, macro times
├── trading_model_handler.py    # ICT Models (Silver Bullet, 2022, P3)
├── market_data_engine.py       # OHLCV management, intermarket analysis
│
├── # PHASE 2: CORE INTEGRATION LAYER
├── ict_core_engine.py          # Master orchestration
├── ict_integration_engine.py   # Confluence analysis, trade setup generation
├── mtf_coordinator.py          # Multi-timeframe alignment
├── signal_generator.py         # Signal generation
├── signal_aggregator.py        # Confluence scoring
├── ai_learning_engine.py       # Pattern recognition
├── trade_executor.py           # Order management
├── main_trading_bot.py         # Main entry point
│
├── # PHASE 3: AI/MACHINE LEARNING MODULE
├── ml_model_trainer.py         # Random Forest, ensemble training
├── lstm_trend_predictor.py     # Deep learning price direction
├── ai_signal_filter.py         # Multi-model validation
├── reinforcement_learning_agent.py  # DQN for entry/exit timing
│
├── # PHASE 4: RISK MANAGEMENT
├── risk_manager.py             # Position sizing, stops, drawdown protection
│
├── # PHASE 5: BROKER INTEGRATION
├── broker_interface.py         # OANDA, MT5, Paper trading
│
├── # PHASE 6: BACKTESTING
├── backtester.py               # Strategy simulation, metrics, walk-forward
│
└── README.md                   # This file
```

---

## 🔧 Installation

### 1. Python Version
```bash
# Requires Python 3.8 or higher
python --version  # Should be 3.8+
```

### 2. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv ict_bot_env

# Activate (Windows)
ict_bot_env\Scripts\activate

# Activate (Mac/Linux)
source ict_bot_env/bin/activate
```

### 3. Install Dependencies
```bash
# Core dependencies
pip install numpy pandas scipy

# Machine Learning (Phase 3)
pip install scikit-learn tensorflow keras

# Broker Integration (Phase 5)
pip install requests  # For OANDA
pip install MetaTrader5  # For MT5 (Windows only)

# Optional: Data visualization
pip install matplotlib plotly

# Optional: Technical analysis
pip install ta-lib  # Requires separate TA-Lib installation
```

### 4. Create requirements.txt
```bash
# Create this file in your project root:
cat > requirements.txt << EOF
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
keras>=2.8.0
requests>=2.26.0
MetaTrader5>=5.0.0  # Windows only, comment out for Mac/Linux
EOF

# Install all at once
pip install -r requirements.txt
```

---

## 📦 How to Import Handlers

### Basic Import Structure

All files should be in the same directory. Import using standard Python imports:

```python
# ============================================
# IMPORTING PHASE 1: ICT HANDLERS
# ============================================

# Fair Value Gap Handler
from fvg_handler import (
    FVGHandler,
    FVGType,
    FVG,
    FVGConfig
)

# Order Block Handler
from order_block_handler import (
    OrderBlockHandler,
    OrderBlock,
    OrderBlockType,
    OrderBlockConfig
)

# Liquidity Handler
from liquidity_handler import (
    LiquidityHandler,
    LiquidityPool,
    LiquidityType,
    LiquiditySweep
)

# Market Structure Handler
from market_structure_handler import (
    MarketStructureHandler,
    StructureBreak,
    StructureType,
    SwingPoint
)

# Gap Handler (NWOG/NDOG)
from gap_handler import (
    GapHandler,
    Gap,
    GapType
)

# PD Array Handler
from pd_array_handler import (
    PDArrayHandler,
    PDArray,
    PremiumDiscount
)

# Timeframe Handler
from timeframe_handler import (
    TimeframeHandler,
    KillZone,
    TradingSession,
    MacroTime
)

# Trading Model Handler
from trading_model_handler import (
    TradingModelHandler,
    ICTModel,
    ModelType,
    ModelSignal
)

# Market Data Engine
from market_data_engine import (
    MarketDataEngine,
    OHLCV,
    MarketData
)
```

```python
# ============================================
# IMPORTING PHASE 2: INTEGRATION LAYER
# ============================================

from ict_core_engine import ICTCoreEngine
from ict_integration_engine import ICTIntegrationEngine
from mtf_coordinator import MTFCoordinator
from signal_generator import SignalGenerator
from signal_aggregator import SignalAggregator
from ai_learning_engine import AILearningEngine
from trade_executor import TradeExecutor
from main_trading_bot import ICTTradingBot
```

```python
# ============================================
# IMPORTING PHASE 3: AI/ML MODULE
# ============================================

from ml_model_trainer import (
    ICTModelTrainer,
    ICTFeatureEngineer,
    ModelType,
    TrainingConfig
)

from lstm_trend_predictor import (
    ICTLSTMTrainer,
    LSTMFeatureBuilder,
    LSTMPrediction
)

from ai_signal_filter import (
    AISignalFilter,
    FilterResult,
    FilterDecision
)

from reinforcement_learning_agent import (
    ICTRLAgent,
    TradingEnvironment,
    RLConfig
)
```

```python
# ============================================
# IMPORTING PHASE 4: RISK MANAGEMENT
# ============================================

from risk_manager import (
    ICTRiskManager,
    RiskParameters,
    PositionSizingEngine,
    DynamicStopLossEngine,
    DrawdownProtectionEngine,
    TrailingStopEngine,
    SessionGuardsEngine,
    StopLossType,
    TrailingStopMethod,
    create_default_risk_manager,
    create_conservative_risk_manager
)
```

```python
# ============================================
# IMPORTING PHASE 5: BROKER INTEGRATION
# ============================================

from broker_interface import (
    BrokerFactory,
    BrokerInterface,
    OANDABroker,
    MT5Broker,
    PaperBroker,
    OrderExecutionEngine,
    PositionTracker,
    OrderRequest,
    OrderType,
    OrderSide,
    BrokerCredentials,
    BrokerType
)
```

```python
# ============================================
# IMPORTING PHASE 6: BACKTESTING
# ============================================

from backtester import (
    BacktestEngine,
    BacktestConfig,
    BacktestResult,
    PerformanceCalculator,
    PerformanceMetrics,
    WalkForwardAnalyzer,
    StrategyComparator,
    BacktestVisualizer,
    SyntheticDataGenerator,
    TradeSignal,
    TradeDirection,
    OHLCV
)
```

---

## 🚀 Quick Start Examples

### Example 1: Basic FVG Detection

```python
"""
Detect Fair Value Gaps in price data
"""
from fvg_handler import FVGHandler, FVGConfig
from market_data_engine import OHLCV
from datetime import datetime

# Create sample data
data = [
    OHLCV(datetime(2024, 1, 1, 10, 0), 1.0850, 1.0855, 1.0845, 1.0852, 100),
    OHLCV(datetime(2024, 1, 1, 11, 0), 1.0852, 1.0870, 1.0850, 1.0868, 150),
    OHLCV(datetime(2024, 1, 1, 12, 0), 1.0875, 1.0880, 1.0872, 1.0878, 120),
    # ... more bars
]

# Initialize handler
config = FVGConfig(min_gap_size=0.0005, lookback_period=50)
fvg_handler = FVGHandler(config)

# Detect FVGs
fvgs = fvg_handler.detect_fvgs(data)

# Print results
for fvg in fvgs:
    print(f"FVG: {fvg.type.name} at {fvg.timestamp}")
    print(f"  Range: {fvg.low:.5f} - {fvg.high:.5f}")
    print(f"  Filled: {fvg.fill_percentage:.1f}%")
```

### Example 2: Complete ICT Analysis

```python
"""
Run complete ICT analysis on price data
"""
from ict_integration_engine import ICTIntegrationEngine
from market_data_engine import MarketDataEngine

# Initialize engines
data_engine = MarketDataEngine()
ict_engine = ICTIntegrationEngine()

# Load your data (example with synthetic data)
# In production, load from broker or CSV
ohlcv_data = data_engine.load_from_csv('EURUSD_H1.csv')

# Run complete ICT analysis
analysis = ict_engine.analyze(
    data=ohlcv_data,
    symbol='EUR/USD',
    timeframe='H1'
)

# Access results
print(f"Trend: {analysis.trend}")
print(f"Key Levels: {analysis.key_levels}")
print(f"Active FVGs: {len(analysis.fvgs)}")
print(f"Order Blocks: {len(analysis.order_blocks)}")
print(f"Liquidity Pools: {len(analysis.liquidity_pools)}")
print(f"Trade Setup: {analysis.trade_setup}")
```

### Example 3: Paper Trading Backtest

```python
"""
Run a backtest with paper trading
"""
from backtester import (
    BacktestEngine, 
    BacktestConfig,
    BacktestVisualizer,
    SyntheticDataGenerator,
    TradeSignal,
    TradeDirection
)
from datetime import datetime

# Generate test data
data = SyntheticDataGenerator.generate(
    start_date=datetime(2024, 1, 1),
    end_date=datetime(2024, 6, 30),
    timeframe_minutes=60,
    initial_price=1.1000,
    volatility=0.0003
)

# Define your strategy
def my_ict_strategy(data, index, engine):
    """Simple example strategy"""
    if index < 20:
        return None
    
    current = data[index]
    
    # Your ICT logic here...
    # Example: Buy on bullish engulfing
    prev = data[index - 1]
    
    if (current.close > current.open and 
        prev.close < prev.open and
        current.close > prev.open and
        current.open < prev.close):
        
        stop = current.low - 0.0010
        target = current.close + 0.0030
        
        return TradeSignal(
            timestamp=current.timestamp,
            direction=TradeDirection.LONG,
            entry_price=current.close,
            stop_loss=stop,
            take_profit=target,
            confidence=0.75,
            model_name="Engulfing",
            confluence_factors=["Bullish_Engulfing"]
        )
    
    return None

# Configure backtest
config = BacktestConfig(
    initial_capital=10000.0,
    position_size_pct=1.0,  # 1% risk per trade
    commission_per_lot=7.0,
    spread_pips=1.0,
    max_positions=1
)

# Run backtest
engine = BacktestEngine(config)
result = engine.run(
    data=data,
    strategy=my_ict_strategy,
    symbol="EUR/USD",
    strategy_name="My ICT Strategy"
)

# Generate report
visualizer = BacktestVisualizer(result)
print(visualizer.generate_report())

# Access metrics
print(f"\nWin Rate: {result.metrics.win_rate:.1%}")
print(f"Profit Factor: {result.metrics.profit_factor:.2f}")
print(f"Sharpe Ratio: {result.metrics.sharpe_ratio:.2f}")
print(f"Max Drawdown: {result.metrics.max_drawdown_pct:.1f}%")
print(f"Total Return: {result.metrics.total_return_pct:.1f}%")
```

### Example 4: Live Trading with OANDA

```python
"""
Connect to OANDA for live/paper trading
"""
from broker_interface import (
    BrokerFactory,
    BrokerCredentials,
    BrokerType,
    OrderExecutionEngine,
    OrderSide
)

# OANDA Credentials (use environment variables in production!)
credentials = BrokerCredentials(
    broker_type=BrokerType.OANDA,
    api_key="YOUR_OANDA_API_KEY",  # Get from OANDA dashboard
    account_id="YOUR_ACCOUNT_ID",
    environment="practice"  # Use "practice" for demo, "live" for real
)

# Create broker
broker = BrokerFactory.create_broker(credentials)

# Connect
if broker.connect():
    print("Connected to OANDA!")
    
    # Get account info
    account = broker.get_account_info()
    print(f"Balance: ${account.balance:,.2f}")
    print(f"Equity: ${account.equity:,.2f}")
    
    # Get current quote
    quote = broker.get_quote("EUR/USD")
    print(f"EUR/USD: Bid={quote.bid:.5f}, Ask={quote.ask:.5f}")
    
    # Create execution engine with retry logic
    executor = OrderExecutionEngine(broker, max_retries=3)
    
    # Place a trade (CAREFUL - this is real money in live mode!)
    # result = executor.execute_market_order(
    #     symbol="EUR/USD",
    #     side=OrderSide.BUY,
    #     quantity=0.01,  # 0.01 lots = 1000 units
    #     stop_loss=quote.bid - 0.0020,
    #     take_profit=quote.bid + 0.0040
    # )
    
    # Disconnect
    broker.disconnect()
else:
    print("Failed to connect")
```

### Example 5: Risk-Managed Trade Assessment

```python
"""
Assess trade risk before execution
"""
from risk_manager import (
    create_default_risk_manager,
    AccountState,
    StopLossType,
    PositionSizingMethod
)
from datetime import datetime

# Create risk manager
risk_manager = create_default_risk_manager()

# Current account state
account = AccountState(
    balance=10000.0,
    equity=10000.0,
    margin_used=0.0,
    margin_available=10000.0,
    unrealized_pnl=0.0,
    daily_pnl=-50.0,  # Down $50 today
    weekly_pnl=-100.0,
    monthly_pnl=200.0,
    peak_balance=10200.0,
    current_drawdown=2.0,
    max_drawdown=5.0,
    open_positions=0,
    pending_orders=0,
    consecutive_losses=1,
    consecutive_wins=0,
    total_trades=50,
    win_rate=0.55,
    avg_win=150.0,
    avg_loss=100.0
)

# Assess potential trade
assessment = risk_manager.assess_trade_risk(
    account=account,
    entry_price=1.0850,
    direction='long',
    take_profit=1.0900,
    signal_confidence=0.75,
    stop_type=StopLossType.ATR_BASED,
    atr=0.0015,
    swing_low=1.0820,
    order_block={'low': 1.0830, 'high': 1.0840}
)

# Check results
print(f"Can Take Trade: {assessment.can_take_trade}")
print(f"Position Size: {assessment.position_size.position_size:.2f} lots")
print(f"Risk Amount: ${assessment.position_size.risk_amount:.2f}")
print(f"Stop Loss: {assessment.stop_loss.stop_price:.5f}")
print(f"Risk/Reward: {assessment.risk_reward_ratio:.2f}")
print(f"Risk Score: {assessment.risk_score:.0f}/100")

if assessment.warnings:
    print("\nWarnings:")
    for w in assessment.warnings:
        print(f"  ⚠️ {w}")

if assessment.recommendations:
    print("\nRecommendations:")
    for r in assessment.recommendations:
        print(f"  → {r}")
```

### Example 6: Train ML Model

```python
"""
Train ML model on historical ICT signals
"""
from ml_model_trainer import (
    ICTModelTrainer,
    ICTFeatureEngineer,
    SyntheticDataGenerator,
    ModelType,
    TrainingMode,
    TargetVariable
)

# Generate synthetic training data (replace with real data)
generator = SyntheticDataGenerator()
training_samples = generator.generate_samples(
    num_samples=1000,
    include_losers=True
)

# Initialize trainer
trainer = ICTModelTrainer()

# Prepare training data
X, y, feature_names = trainer.prepare_training_data(
    signals=training_samples,
    target=TargetVariable.SIGNAL_QUALITY
)

print(f"Training samples: {len(X)}")
print(f"Features: {len(feature_names)}")

# Train model
result = trainer.train(
    X=X,
    y=y,
    model_type=ModelType.RANDOM_FOREST,
    mode=TrainingMode.STANDARD,
    test_size=0.2
)

print(f"\nModel Performance:")
print(f"  Accuracy: {result['accuracy']:.1%}")
print(f"  F1 Score: {result['f1_score']:.3f}")
print(f"  Cross-Val: {result['cv_mean']:.3f} ± {result['cv_std']:.3f}")

# Save model
trainer.save_model('models/ict_signal_quality_model.pkl')

# Later, load and predict
trainer.load_model('models/ict_signal_quality_model.pkl')
prediction = trainer.predict(new_signal_features)
print(f"Prediction: {prediction['label']} (Confidence: {prediction['confidence']:.1%})")
```

### Example 7: Walk-Forward Analysis

```python
"""
Validate strategy with walk-forward analysis
"""
from backtester import (
    WalkForwardAnalyzer,
    BacktestConfig,
    SyntheticDataGenerator,
    SampleICTStrategies
)
from datetime import datetime

# Generate data
data = SyntheticDataGenerator.generate(
    start_date=datetime(2023, 1, 1),
    end_date=datetime(2024, 12, 31),
    timeframe_minutes=60
)

# Strategy factory
def strategy_factory(training_data):
    """Returns trained strategy function"""
    # In real use, you might optimize parameters here
    return SampleICTStrategies.kill_zone_strategy

# Configure
config = BacktestConfig(initial_capital=10000.0)

# Run walk-forward analysis
wfa = WalkForwardAnalyzer(
    in_sample_ratio=0.7,  # 70% train, 30% test
    num_windows=5
)

result = wfa.analyze(data, strategy_factory, config)

# Results
print(f"Robustness Score: {result.robustness_score:.1f}/100")
print(f"Likely Overfit: {'YES ⚠️' if result.is_curve_fit else 'NO ✓'}")
print(f"\nOut-of-Sample Performance:")
print(f"  Win Rate: {result.combined_oos_metrics.win_rate:.1%}")
print(f"  Profit Factor: {result.combined_oos_metrics.profit_factor:.2f}")
print(f"  Total Trades: {result.combined_oos_metrics.total_trades}")

# Window details
for i, (is_m, oos_m) in enumerate(zip(result.in_sample_metrics, result.out_of_sample_metrics)):
    print(f"\nWindow {i+1}:")
    print(f"  In-Sample:  WR={is_m.win_rate:.1%}, PF={is_m.profit_factor:.2f}")
    print(f"  Out-Sample: WR={oos_m.win_rate:.1%}, PF={oos_m.profit_factor:.2f}")
```

---

## 📊 Complete Trading Bot Example

```python
"""
Complete ICT Trading Bot - Full Integration Example
"""
import time
from datetime import datetime

# Phase 1: ICT Handlers
from fvg_handler import FVGHandler
from order_block_handler import OrderBlockHandler
from liquidity_handler import LiquidityHandler
from market_structure_handler import MarketStructureHandler
from timeframe_handler import TimeframeHandler

# Phase 2: Integration
from ict_integration_engine import ICTIntegrationEngine
from signal_generator import SignalGenerator

# Phase 4: Risk Management
from risk_manager import create_default_risk_manager, AccountState

# Phase 5: Broker
from broker_interface import BrokerFactory, OrderExecutionEngine, OrderSide

# Phase 6: Backtest (for validation)
from backtester import BacktestEngine, BacktestConfig


class ICTTradingSystem:
    """Complete ICT Trading System"""
    
    def __init__(self, broker_credentials=None, paper_mode=True):
        # Initialize ICT handlers
        self.fvg_handler = FVGHandler()
        self.ob_handler = OrderBlockHandler()
        self.liquidity_handler = LiquidityHandler()
        self.structure_handler = MarketStructureHandler()
        self.timeframe_handler = TimeframeHandler()
        
        # Integration engine
        self.ict_engine = ICTIntegrationEngine()
        self.signal_generator = SignalGenerator()
        
        # Risk manager
        self.risk_manager = create_default_risk_manager()
        
        # Broker
        if paper_mode:
            self.broker = BrokerFactory.create_paper_broker(initial_balance=10000.0)
        else:
            self.broker = BrokerFactory.create_broker(broker_credentials)
        
        self.executor = OrderExecutionEngine(self.broker)
        
        self.running = False
    
    def analyze_market(self, data, symbol, timeframe):
        """Run complete ICT analysis"""
        analysis = {
            'timestamp': datetime.utcnow(),
            'symbol': symbol,
            'timeframe': timeframe
        }
        
        # Detect ICT elements
        analysis['fvgs'] = self.fvg_handler.detect_fvgs(data)
        analysis['order_blocks'] = self.ob_handler.detect_order_blocks(data)
        analysis['liquidity'] = self.liquidity_handler.detect_liquidity(data)
        analysis['structure'] = self.structure_handler.analyze_structure(data)
        
        # Check session/kill zone
        analysis['session'] = self.timeframe_handler.get_current_session()
        analysis['in_kill_zone'] = self.timeframe_handler.is_kill_zone()
        
        return analysis
    
    def generate_signal(self, analysis, account_state):
        """Generate trade signal from analysis"""
        # Use signal generator
        signal = self.signal_generator.generate(analysis)
        
        if signal is None:
            return None
        
        # Risk assessment
        assessment = self.risk_manager.assess_trade_risk(
            account=account_state,
            entry_price=signal.entry_price,
            direction=signal.direction,
            take_profit=signal.take_profit,
            signal_confidence=signal.confidence
        )
        
        if not assessment.can_take_trade:
            print(f"Trade blocked: {assessment.blocks}")
            return None
        
        # Attach risk info to signal
        signal.position_size = assessment.position_size.position_size
        signal.risk_amount = assessment.position_size.risk_amount
        
        return signal
    
    def execute_signal(self, signal):
        """Execute trade signal"""
        side = OrderSide.BUY if signal.direction == 'long' else OrderSide.SELL
        
        result = self.executor.execute_market_order(
            symbol=signal.symbol,
            side=side,
            quantity=signal.position_size,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit
        )
        
        return result
    
    def run(self, symbol='EUR/USD', timeframe='H1'):
        """Main trading loop"""
        print(f"Starting ICT Trading System...")
        print(f"Symbol: {symbol}, Timeframe: {timeframe}")
        
        if not self.broker.connect():
            print("Failed to connect to broker!")
            return
        
        self.running = True
        
        try:
            while self.running:
                # Get account state
                account_info = self.broker.get_account_info()
                account_state = AccountState(
                    balance=account_info.balance,
                    equity=account_info.equity,
                    # ... fill other fields
                )
                
                # Get market data
                data = self.broker.get_bars(symbol, timeframe, count=100)
                
                # Analyze market
                analysis = self.analyze_market(data, symbol, timeframe)
                
                # Generate signal
                signal = self.generate_signal(analysis, account_state)
                
                if signal:
                    print(f"Signal: {signal.direction} {symbol}")
                    result = self.execute_signal(signal)
                    print(f"Execution: {'Success' if result.success else 'Failed'}")
                
                # Wait for next bar
                time.sleep(60)  # Adjust based on timeframe
                
        except KeyboardInterrupt:
            print("\nStopping...")
        finally:
            self.broker.disconnect()
            print("Disconnected from broker")
    
    def stop(self):
        """Stop the trading system"""
        self.running = False


# Run the system
if __name__ == "__main__":
    # Paper trading mode
    system = ICTTradingSystem(paper_mode=True)
    
    # For live trading (CAREFUL!):
    # from broker_interface import BrokerCredentials, BrokerType
    # creds = BrokerCredentials(
    #     broker_type=BrokerType.OANDA,
    #     api_key="your_key",
    #     account_id="your_account",
    #     environment="practice"  # or "live"
    # )
    # system = ICTTradingSystem(broker_credentials=creds, paper_mode=False)
    
    system.run(symbol='EUR/USD', timeframe='H1')
```

---

## 🔑 Environment Variables (Recommended for Production)

Create a `.env` file (add to `.gitignore`!):

```bash
# .env file
OANDA_API_KEY=your_oanda_api_key_here
OANDA_ACCOUNT_ID=your_account_id
OANDA_ENVIRONMENT=practice

MT5_LOGIN=your_mt5_login
MT5_PASSWORD=your_mt5_password
MT5_SERVER=your_broker_server
```

Load in Python:
```python
import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv('OANDA_API_KEY')
account_id = os.getenv('OANDA_ACCOUNT_ID')
```

---

## 📋 Dependencies Summary

| Package | Version | Purpose |
|---------|---------|---------|
| numpy | >=1.21.0 | Numerical operations |
| pandas | >=1.3.0 | Data manipulation |
| scipy | >=1.7.0 | Statistical functions |
| scikit-learn | >=1.0.0 | ML models |
| tensorflow | >=2.8.0 | Deep learning (LSTM) |
| keras | >=2.8.0 | Neural network API |
| requests | >=2.26.0 | OANDA API |
| MetaTrader5 | >=5.0.0 | MT5 integration (Windows) |
| python-dotenv | >=0.19.0 | Environment variables |

---

## 🧪 Running Tests

```bash
# Test individual handlers
python fvg_handler.py
python order_block_handler.py
python risk_manager.py
python backtester.py

# Test broker connection (paper mode)
python broker_interface.py

# Run backtest demo
python -c "
from backtester import *
from datetime import datetime

data = SyntheticDataGenerator.generate(
    datetime(2024,1,1), datetime(2024,6,30), 60
)
config = BacktestConfig(initial_capital=10000)
engine = BacktestEngine(config)
result = engine.run(data, SampleICTStrategies.kill_zone_strategy)
print(BacktestVisualizer(result).generate_report())
"
```

---

## 📚 ICT Concepts Implemented

| Concept | Handler | Description |
|---------|---------|-------------|
| Fair Value Gaps | `fvg_handler.py` | BISI, SIBI, CE, inversions |
| Order Blocks | `order_block_handler.py` | Breakers, propulsion, mitigation |
| Liquidity | `liquidity_handler.py` | Pools, sweeps, stop hunts |
| Market Structure | `market_structure_handler.py` | BOS, CHoCH, MSS, SMS |
| Kill Zones | `timeframe_handler.py` | London, NY AM/PM, Asian |
| ICT Models | `trading_model_handler.py` | Silver Bullet, 2022, Power of Three |
| Premium/Discount | `pd_array_handler.py` | Equilibrium, zones |
| Gaps | `gap_handler.py` | NWOG, NDOG, opening ranges |

---

## ⚠️ Important Notes

1. **Always test in paper trading first** before using real money
2. **Start with small position sizes** (0.01 lots)
3. **Monitor your bot** - don't leave it unattended initially
4. **Keep API keys secure** - use environment variables
5. **Understand the ICT concepts** before relying on automated signals
6. **Past performance ≠ future results** - markets change

---

## 📞 Support

For issues or questions:
1. Review the code comments (extensive documentation)
2. Check the example usage in each file's `if __name__ == "__main__"` section
3. Ensure all dependencies are installed correctly

---

**Happy Trading! 📈**

*Remember: Risk management is more important than entry signals.*# algot
