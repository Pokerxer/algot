"""
ICT Backtesting Framework
=========================

Comprehensive backtesting system for ICT algorithmic trading including:
- Historical strategy simulation
- Performance metrics (win rate, profit factor, Sharpe, drawdown)
- Walk-forward analysis with rolling window optimization
- Strategy comparator for ICT model evaluation
- Visualization module for charts and analytics

BACKTESTING ARCHITECTURE:
========================

┌─────────────────────────────────────────────────────────────────────────────┐
│                        ICT BACKTESTING FRAMEWORK                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                      BACKTEST ENGINE                                  │    │
│  │                                                                       │    │
│  │  Historical Data ──► Strategy Logic ──► Simulated Trades ──► Results │    │
│  │                                                                       │    │
│  │  Features:                                                            │    │
│  │  • Tick-by-tick or bar-by-bar simulation                             │    │
│  │  • Realistic spread and slippage modeling                            │    │
│  │  • Commission and swap calculation                                   │    │
│  │  • Multi-timeframe data handling                                     │    │
│  │  • Event-driven architecture                                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    PERFORMANCE METRICS                                │    │
│  │                                                                       │    │
│  │  Returns:                    Risk:                                    │    │
│  │  • Total Return              • Max Drawdown                          │    │
│  │  • CAGR                      • Avg Drawdown                          │    │
│  │  • Monthly Returns           • Drawdown Duration                     │    │
│  │                                                                       │    │
│  │  Risk-Adjusted:              Trade Stats:                            │    │
│  │  • Sharpe Ratio              • Win Rate                              │    │
│  │  • Sortino Ratio             • Profit Factor                         │    │
│  │  • Calmar Ratio              • Avg Win/Loss                          │    │
│  │  • Information Ratio         • Expectancy                            │    │
│  │                                                                       │    │
│  │  ICT-Specific:                                                        │    │
│  │  • Kill Zone Performance     • Model Success Rates                   │    │
│  │  • Session Analysis          • Confluence Correlation                │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    WALK-FORWARD ANALYSIS                              │    │
│  │                                                                       │    │
│  │  ┌─────────┬─────────┬─────────┬─────────┬─────────┐                │    │
│  │  │ Train 1 │ Test 1  │         │         │         │                │    │
│  │  └─────────┴─────────┴─────────┴─────────┴─────────┘                │    │
│  │            ┌─────────┬─────────┐                                     │    │
│  │            │ Train 2 │ Test 2  │                                     │    │
│  │            └─────────┴─────────┘                                     │    │
│  │                      ┌─────────┬─────────┐                           │    │
│  │                      │ Train 3 │ Test 3  │                           │    │
│  │                      └─────────┴─────────┘                           │    │
│  │                                                                       │    │
│  │  • Rolling window optimization                                       │    │
│  │  • Out-of-sample validation                                          │    │
│  │  • Parameter stability analysis                                      │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    STRATEGY COMPARATOR                                │    │
│  │                                                                       │    │
│  │  Compare ICT Models:                                                  │    │
│  │  • Silver Bullet vs 2022 Model vs Power of Three                     │    │
│  │  • Session-specific performance                                      │    │
│  │  • Confluence factor impact                                          │    │
│  │  • Risk-adjusted rankings                                            │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                    VISUALIZATION MODULE                               │    │
│  │                                                                       │    │
│  │  Charts:                      Analysis:                               │    │
│  │  • Equity Curve              • Monthly Returns Heatmap               │    │
│  │  • Drawdown Chart            • Win Rate by Hour/Day                  │    │
│  │  • Trade Distribution        • P&L Distribution                      │    │
│  │  • Entry/Exit Markers        • Rolling Sharpe                        │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                               │
└─────────────────────────────────────────────────────────────────────────────┘

Author: Claude (Anthropic)
Version: 1.0.0
"""

import logging
import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
from collections import defaultdict
import json
import csv
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS AND CONSTANTS
# =============================================================================

class TradeDirection(Enum):
    """Trade direction"""
    LONG = auto()
    SHORT = auto()


class TradeStatus(Enum):
    """Trade status"""
    OPEN = auto()
    CLOSED = auto()
    CANCELLED = auto()


class ExitReason(Enum):
    """Reason for trade exit"""
    TAKE_PROFIT = auto()
    STOP_LOSS = auto()
    TRAILING_STOP = auto()
    SIGNAL_EXIT = auto()
    TIME_EXIT = auto()
    END_OF_DATA = auto()
    MANUAL = auto()


class BacktestMode(Enum):
    """Backtest simulation mode"""
    BAR_BY_BAR = auto()      # Process each bar
    TICK_BY_TICK = auto()    # Process each tick (more accurate but slower)
    VECTORIZED = auto()      # Fast vectorized calculation


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class OHLCV:
    """OHLCV candle data"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0
    
    def __post_init__(self):
        if isinstance(self.timestamp, str):
            self.timestamp = datetime.fromisoformat(self.timestamp)


@dataclass
class TradeSignal:
    """Trading signal from strategy"""
    timestamp: datetime
    direction: TradeDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    confidence: float = 1.0
    model_name: str = ""
    confluence_factors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Trade:
    """Completed or open trade"""
    trade_id: int
    symbol: str
    direction: TradeDirection
    entry_time: datetime
    entry_price: float
    exit_time: Optional[datetime] = None
    exit_price: Optional[float] = None
    quantity: float = 1.0
    stop_loss: float = 0.0
    take_profit: float = 0.0
    status: TradeStatus = TradeStatus.OPEN
    exit_reason: Optional[ExitReason] = None
    pnl: float = 0.0
    pnl_pips: float = 0.0
    r_multiple: float = 0.0
    commission: float = 0.0
    swap: float = 0.0
    slippage: float = 0.0
    max_favorable_excursion: float = 0.0  # MFE
    max_adverse_excursion: float = 0.0    # MAE
    bars_held: int = 0
    model_name: str = ""
    confluence_factors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def net_pnl(self) -> float:
        """Net P&L after costs"""
        return self.pnl - self.commission - abs(self.swap) - self.slippage
    
    @property
    def is_winner(self) -> bool:
        """Check if trade was profitable"""
        return self.net_pnl > 0
    
    @property
    def duration(self) -> Optional[timedelta]:
        """Trade duration"""
        if self.exit_time:
            return self.exit_time - self.entry_time
        return None


@dataclass
class BacktestConfig:
    """Backtest configuration"""
    initial_capital: float = 10000.0
    position_size_pct: float = 1.0  # Risk % per trade
    commission_per_lot: float = 7.0
    spread_pips: float = 1.0
    slippage_pips: float = 0.5
    swap_long_per_day: float = -0.5
    swap_short_per_day: float = 0.3
    pip_value: float = 10.0  # Per standard lot
    pip_size: float = 0.0001
    max_positions: int = 1
    use_trailing_stop: bool = False
    trailing_stop_pips: float = 20.0
    mode: BacktestMode = BacktestMode.BAR_BY_BAR
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None


@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    # Basic stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # P&L
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    profit_factor: float = 0.0
    
    # Average trade
    avg_trade_pnl: float = 0.0
    avg_winning_trade: float = 0.0
    avg_losing_trade: float = 0.0
    avg_win_loss_ratio: float = 0.0
    expectancy: float = 0.0
    expectancy_pips: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    max_drawdown_duration: int = 0  # In bars
    avg_drawdown: float = 0.0
    
    # Risk-adjusted returns
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    
    # Returns
    total_return_pct: float = 0.0
    cagr: float = 0.0
    monthly_return_avg: float = 0.0
    monthly_return_std: float = 0.0
    
    # Trade duration
    avg_bars_held: float = 0.0
    avg_bars_winners: float = 0.0
    avg_bars_losers: float = 0.0
    
    # Streaks
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    current_streak: int = 0
    
    # R-multiples
    avg_r_multiple: float = 0.0
    total_r: float = 0.0
    
    # Recovery
    recovery_factor: float = 0.0
    
    # Additional
    total_commission: float = 0.0
    total_swap: float = 0.0
    total_slippage: float = 0.0
    
    # Time analysis
    best_hour: int = 0
    worst_hour: int = 0
    best_day: int = 0
    worst_day: int = 0


@dataclass
class BacktestResult:
    """Complete backtest result"""
    config: BacktestConfig
    metrics: PerformanceMetrics
    trades: List[Trade]
    equity_curve: List[Tuple[datetime, float]]
    drawdown_curve: List[Tuple[datetime, float]]
    monthly_returns: Dict[str, float]
    daily_returns: List[float]
    start_date: datetime
    end_date: datetime
    duration_days: int
    symbol: str
    strategy_name: str
    
    # ICT-specific
    model_performance: Dict[str, Dict] = field(default_factory=dict)
    session_performance: Dict[str, Dict] = field(default_factory=dict)
    confluence_analysis: Dict[str, float] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Walk-forward analysis result"""
    windows: List[Dict]
    in_sample_metrics: List[PerformanceMetrics]
    out_of_sample_metrics: List[PerformanceMetrics]
    combined_oos_metrics: PerformanceMetrics
    parameter_stability: Dict[str, float]
    robustness_score: float
    is_curve_fit: bool


@dataclass
class StrategyComparison:
    """Strategy comparison result"""
    strategies: List[str]
    metrics_comparison: Dict[str, Dict[str, float]]
    rankings: Dict[str, List[str]]
    best_strategy: str
    recommendation: str


# =============================================================================
# PERFORMANCE CALCULATOR
# =============================================================================

class PerformanceCalculator:
    """
    Calculates comprehensive trading performance metrics.
    """
    
    @staticmethod
    def calculate(
        trades: List[Trade],
        equity_curve: List[Tuple[datetime, float]],
        initial_capital: float,
        risk_free_rate: float = 0.02
    ) -> PerformanceMetrics:
        """
        Calculate all performance metrics from trades and equity curve.
        """
        metrics = PerformanceMetrics()
        
        if not trades:
            return metrics
        
        # Filter closed trades
        closed_trades = [t for t in trades if t.status == TradeStatus.CLOSED]
        
        if not closed_trades:
            return metrics
        
        # Basic counts
        metrics.total_trades = len(closed_trades)
        metrics.winning_trades = len([t for t in closed_trades if t.is_winner])
        metrics.losing_trades = metrics.total_trades - metrics.winning_trades
        metrics.win_rate = metrics.winning_trades / metrics.total_trades if metrics.total_trades > 0 else 0
        
        # P&L calculations
        pnls = [t.net_pnl for t in closed_trades]
        metrics.total_pnl = sum(pnls)
        metrics.gross_profit = sum(p for p in pnls if p > 0)
        metrics.gross_loss = abs(sum(p for p in pnls if p < 0))
        metrics.net_profit = metrics.total_pnl
        metrics.profit_factor = metrics.gross_profit / metrics.gross_loss if metrics.gross_loss > 0 else float('inf')
        
        # Average trade
        metrics.avg_trade_pnl = statistics.mean(pnls) if pnls else 0
        winning_pnls = [t.net_pnl for t in closed_trades if t.is_winner]
        losing_pnls = [t.net_pnl for t in closed_trades if not t.is_winner]
        
        metrics.avg_winning_trade = statistics.mean(winning_pnls) if winning_pnls else 0
        metrics.avg_losing_trade = statistics.mean(losing_pnls) if losing_pnls else 0
        metrics.avg_win_loss_ratio = abs(metrics.avg_winning_trade / metrics.avg_losing_trade) if metrics.avg_losing_trade != 0 else float('inf')
        
        # Expectancy
        metrics.expectancy = (metrics.win_rate * metrics.avg_winning_trade) + ((1 - metrics.win_rate) * metrics.avg_losing_trade)
        
        # Pips calculation
        pips_list = [t.pnl_pips for t in closed_trades]
        metrics.expectancy_pips = statistics.mean(pips_list) if pips_list else 0
        
        # R-multiples
        r_multiples = [t.r_multiple for t in closed_trades if t.r_multiple != 0]
        metrics.avg_r_multiple = statistics.mean(r_multiples) if r_multiples else 0
        metrics.total_r = sum(r_multiples)
        
        # Drawdown analysis
        if equity_curve:
            dd_analysis = PerformanceCalculator._calculate_drawdown(equity_curve)
            metrics.max_drawdown = dd_analysis['max_drawdown']
            metrics.max_drawdown_pct = dd_analysis['max_drawdown_pct']
            metrics.max_drawdown_duration = dd_analysis['max_duration']
            metrics.avg_drawdown = dd_analysis['avg_drawdown']
        
        # Returns
        final_equity = equity_curve[-1][1] if equity_curve else initial_capital
        metrics.total_return_pct = ((final_equity - initial_capital) / initial_capital) * 100
        
        # CAGR
        if equity_curve and len(equity_curve) > 1:
            days = (equity_curve[-1][0] - equity_curve[0][0]).days
            years = days / 365.25 if days > 0 else 1
            metrics.cagr = ((final_equity / initial_capital) ** (1 / years) - 1) * 100 if years > 0 else 0
        
        # Daily returns for Sharpe/Sortino
        daily_returns = PerformanceCalculator._calculate_daily_returns(equity_curve)
        
        if daily_returns and len(daily_returns) > 1:
            # Sharpe Ratio (annualized)
            avg_daily_return = statistics.mean(daily_returns)
            std_daily_return = statistics.stdev(daily_returns)
            daily_rf = risk_free_rate / 252
            
            if std_daily_return > 0:
                metrics.sharpe_ratio = (avg_daily_return - daily_rf) / std_daily_return * math.sqrt(252)
            
            # Sortino Ratio (only downside deviation)
            downside_returns = [r for r in daily_returns if r < daily_rf]
            if downside_returns:
                downside_std = statistics.stdev(downside_returns) if len(downside_returns) > 1 else 0
                if downside_std > 0:
                    metrics.sortino_ratio = (avg_daily_return - daily_rf) / downside_std * math.sqrt(252)
        
        # Calmar Ratio
        if metrics.max_drawdown_pct > 0:
            metrics.calmar_ratio = metrics.cagr / metrics.max_drawdown_pct
        
        # Monthly returns
        monthly = PerformanceCalculator._calculate_monthly_returns(equity_curve)
        if monthly:
            metrics.monthly_return_avg = statistics.mean(monthly.values())
            metrics.monthly_return_std = statistics.stdev(monthly.values()) if len(monthly) > 1 else 0
        
        # Trade duration
        bars_held = [t.bars_held for t in closed_trades]
        metrics.avg_bars_held = statistics.mean(bars_held) if bars_held else 0
        
        winner_bars = [t.bars_held for t in closed_trades if t.is_winner]
        loser_bars = [t.bars_held for t in closed_trades if not t.is_winner]
        metrics.avg_bars_winners = statistics.mean(winner_bars) if winner_bars else 0
        metrics.avg_bars_losers = statistics.mean(loser_bars) if loser_bars else 0
        
        # Streaks
        streaks = PerformanceCalculator._calculate_streaks(closed_trades)
        metrics.max_consecutive_wins = streaks['max_wins']
        metrics.max_consecutive_losses = streaks['max_losses']
        metrics.current_streak = streaks['current']
        
        # Costs
        metrics.total_commission = sum(t.commission for t in closed_trades)
        metrics.total_swap = sum(t.swap for t in closed_trades)
        metrics.total_slippage = sum(t.slippage for t in closed_trades)
        
        # Recovery factor
        if metrics.max_drawdown > 0:
            metrics.recovery_factor = metrics.net_profit / metrics.max_drawdown
        
        # Time analysis
        hour_pnl = defaultdict(float)
        day_pnl = defaultdict(float)
        
        for trade in closed_trades:
            hour_pnl[trade.entry_time.hour] += trade.net_pnl
            day_pnl[trade.entry_time.weekday()] += trade.net_pnl
        
        if hour_pnl:
            metrics.best_hour = max(hour_pnl, key=hour_pnl.get)
            metrics.worst_hour = min(hour_pnl, key=hour_pnl.get)
        
        if day_pnl:
            metrics.best_day = max(day_pnl, key=day_pnl.get)
            metrics.worst_day = min(day_pnl, key=day_pnl.get)
        
        return metrics
    
    @staticmethod
    def _calculate_drawdown(equity_curve: List[Tuple[datetime, float]]) -> Dict:
        """Calculate drawdown metrics"""
        if not equity_curve:
            return {'max_drawdown': 0, 'max_drawdown_pct': 0, 'max_duration': 0, 'avg_drawdown': 0}
        
        peak = equity_curve[0][1]
        max_dd = 0
        max_dd_pct = 0
        drawdowns = []
        duration = 0
        max_duration = 0
        
        for timestamp, equity in equity_curve:
            if equity > peak:
                peak = equity
                if duration > max_duration:
                    max_duration = duration
                duration = 0
            else:
                dd = peak - equity
                dd_pct = (dd / peak) * 100 if peak > 0 else 0
                drawdowns.append(dd)
                
                if dd > max_dd:
                    max_dd = dd
                    max_dd_pct = dd_pct
                
                duration += 1
        
        return {
            'max_drawdown': max_dd,
            'max_drawdown_pct': max_dd_pct,
            'max_duration': max_duration,
            'avg_drawdown': statistics.mean(drawdowns) if drawdowns else 0
        }
    
    @staticmethod
    def _calculate_daily_returns(equity_curve: List[Tuple[datetime, float]]) -> List[float]:
        """Calculate daily returns"""
        if len(equity_curve) < 2:
            return []
        
        # Group by day
        daily_equity = {}
        for timestamp, equity in equity_curve:
            date_key = timestamp.date()
            daily_equity[date_key] = equity
        
        # Calculate returns
        dates = sorted(daily_equity.keys())
        returns = []
        
        for i in range(1, len(dates)):
            prev_equity = daily_equity[dates[i-1]]
            curr_equity = daily_equity[dates[i]]
            if prev_equity > 0:
                daily_return = (curr_equity - prev_equity) / prev_equity
                returns.append(daily_return)
        
        return returns
    
    @staticmethod
    def _calculate_monthly_returns(equity_curve: List[Tuple[datetime, float]]) -> Dict[str, float]:
        """Calculate monthly returns"""
        if len(equity_curve) < 2:
            return {}
        
        # Group by month
        monthly_equity = {}
        for timestamp, equity in equity_curve:
            month_key = timestamp.strftime('%Y-%m')
            monthly_equity[month_key] = equity
        
        # Calculate returns
        months = sorted(monthly_equity.keys())
        returns = {}
        
        for i in range(1, len(months)):
            prev_equity = monthly_equity[months[i-1]]
            curr_equity = monthly_equity[months[i]]
            if prev_equity > 0:
                monthly_return = ((curr_equity - prev_equity) / prev_equity) * 100
                returns[months[i]] = monthly_return
        
        return returns
    
    @staticmethod
    def _calculate_streaks(trades: List[Trade]) -> Dict:
        """Calculate win/loss streaks"""
        if not trades:
            return {'max_wins': 0, 'max_losses': 0, 'current': 0}
        
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for trade in trades:
            if trade.is_winner:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        current = current_wins if current_wins > 0 else -current_losses
        
        return {
            'max_wins': max_wins,
            'max_losses': max_losses,
            'current': current
        }


# =============================================================================
# BACKTEST ENGINE
# =============================================================================

class BacktestEngine:
    """
    Core backtesting engine for strategy simulation.
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.reset()
    
    def reset(self):
        """Reset backtester state"""
        self.capital = self.config.initial_capital
        self.equity = self.config.initial_capital
        self.trades: List[Trade] = []
        self.open_trades: List[Trade] = []
        self.equity_curve: List[Tuple[datetime, float]] = []
        self.drawdown_curve: List[Tuple[datetime, float]] = []
        self._trade_id_counter = 0
        self._current_bar: Optional[OHLCV] = None
        self._bar_index = 0
        self._peak_equity = self.config.initial_capital
    
    def run(
        self,
        data: List[OHLCV],
        strategy: Callable[[List[OHLCV], int, 'BacktestEngine'], Optional[TradeSignal]],
        symbol: str = "EUR/USD",
        strategy_name: str = "ICT Strategy"
    ) -> BacktestResult:
        """
        Run backtest with provided data and strategy.
        
        Args:
            data: List of OHLCV bars
            strategy: Strategy function that returns TradeSignal or None
            symbol: Trading symbol
            strategy_name: Name of strategy
            
        Returns:
            BacktestResult with all metrics and trades
        """
        self.reset()
        
        if not data:
            raise ValueError("No data provided for backtest")
        
        # Filter by date range if specified
        if self.config.start_date:
            data = [bar for bar in data if bar.timestamp >= self.config.start_date]
        if self.config.end_date:
            data = [bar for bar in data if bar.timestamp <= self.config.end_date]
        
        if not data:
            raise ValueError("No data in specified date range")
        
        logger.info(f"Starting backtest: {len(data)} bars, {data[0].timestamp} to {data[-1].timestamp}")
        
        # Main simulation loop
        for i, bar in enumerate(data):
            self._bar_index = i
            self._current_bar = bar
            
            # Update open positions
            self._update_open_trades(bar)
            
            # Check for new signals
            if len(self.open_trades) < self.config.max_positions:
                signal = strategy(data[:i+1], i, self)
                
                if signal:
                    self._open_trade(signal, bar)
            
            # Record equity
            self._update_equity(bar)
        
        # Close any remaining open trades
        if data:
            self._close_all_trades(data[-1], ExitReason.END_OF_DATA)
        
        # Calculate metrics
        metrics = PerformanceCalculator.calculate(
            self.trades,
            self.equity_curve,
            self.config.initial_capital
        )
        
        # Calculate monthly returns
        monthly_returns = PerformanceCalculator._calculate_monthly_returns(self.equity_curve)
        daily_returns = PerformanceCalculator._calculate_daily_returns(self.equity_curve)
        
        # ICT-specific analysis
        model_performance = self._analyze_model_performance()
        session_performance = self._analyze_session_performance()
        confluence_analysis = self._analyze_confluence()
        
        result = BacktestResult(
            config=self.config,
            metrics=metrics,
            trades=self.trades,
            equity_curve=self.equity_curve,
            drawdown_curve=self.drawdown_curve,
            monthly_returns=monthly_returns,
            daily_returns=daily_returns,
            start_date=data[0].timestamp,
            end_date=data[-1].timestamp,
            duration_days=(data[-1].timestamp - data[0].timestamp).days,
            symbol=symbol,
            strategy_name=strategy_name,
            model_performance=model_performance,
            session_performance=session_performance,
            confluence_analysis=confluence_analysis
        )
        
        logger.info(f"Backtest complete: {metrics.total_trades} trades, "
                   f"Win rate: {metrics.win_rate:.1%}, "
                   f"Net P&L: ${metrics.net_profit:,.2f}")
        
        return result
    
    def _open_trade(self, signal: TradeSignal, bar: OHLCV):
        """Open a new trade"""
        self._trade_id_counter += 1
        
        # Calculate entry price with spread and slippage
        spread_cost = self.config.spread_pips * self.config.pip_size / 2
        slippage_cost = self.config.slippage_pips * self.config.pip_size
        
        if signal.direction == TradeDirection.LONG:
            entry_price = signal.entry_price + spread_cost + slippage_cost
        else:
            entry_price = signal.entry_price - spread_cost - slippage_cost
        
        # Calculate position size based on risk
        risk_amount = self.equity * (self.config.position_size_pct / 100)
        stop_distance = abs(entry_price - signal.stop_loss)
        
        if stop_distance > 0:
            stop_pips = stop_distance / self.config.pip_size
            quantity = risk_amount / (stop_pips * self.config.pip_value)
            quantity = max(0.01, round(quantity, 2))  # Min 0.01 lots
        else:
            quantity = 0.1  # Default
        
        # Calculate commission
        commission = self.config.commission_per_lot * quantity
        
        trade = Trade(
            trade_id=self._trade_id_counter,
            symbol="",
            direction=signal.direction,
            entry_time=bar.timestamp,
            entry_price=entry_price,
            quantity=quantity,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            status=TradeStatus.OPEN,
            commission=commission,
            slippage=slippage_cost * quantity * 100000,
            model_name=signal.model_name,
            confluence_factors=signal.confluence_factors.copy(),
            metadata=signal.metadata.copy()
        )
        
        self.open_trades.append(trade)
        self.trades.append(trade)
        
        logger.debug(f"Opened trade {trade.trade_id}: {signal.direction.name} @ {entry_price:.5f}")
    
    def _update_open_trades(self, bar: OHLCV):
        """Update open trades with new bar"""
        trades_to_close = []
        
        for trade in self.open_trades:
            trade.bars_held += 1
            
            # Calculate current P&L
            if trade.direction == TradeDirection.LONG:
                current_pnl = (bar.close - trade.entry_price) * trade.quantity * 100000
                mfe = (bar.high - trade.entry_price) * trade.quantity * 100000
                mae = (trade.entry_price - bar.low) * trade.quantity * 100000
            else:
                current_pnl = (trade.entry_price - bar.close) * trade.quantity * 100000
                mfe = (trade.entry_price - bar.low) * trade.quantity * 100000
                mae = (bar.high - trade.entry_price) * trade.quantity * 100000
            
            trade.max_favorable_excursion = max(trade.max_favorable_excursion, mfe)
            trade.max_adverse_excursion = max(trade.max_adverse_excursion, mae)
            
            # Check stop loss
            if trade.direction == TradeDirection.LONG:
                if bar.low <= trade.stop_loss:
                    trades_to_close.append((trade, trade.stop_loss, ExitReason.STOP_LOSS))
                    continue
                if bar.high >= trade.take_profit:
                    trades_to_close.append((trade, trade.take_profit, ExitReason.TAKE_PROFIT))
                    continue
            else:
                if bar.high >= trade.stop_loss:
                    trades_to_close.append((trade, trade.stop_loss, ExitReason.STOP_LOSS))
                    continue
                if bar.low <= trade.take_profit:
                    trades_to_close.append((trade, trade.take_profit, ExitReason.TAKE_PROFIT))
                    continue
            
            # Update trailing stop if enabled
            if self.config.use_trailing_stop:
                self._update_trailing_stop(trade, bar)
            
            # Calculate swap for overnight positions
            if trade.bars_held > 0 and bar.timestamp.hour == 0:
                if trade.direction == TradeDirection.LONG:
                    trade.swap += self.config.swap_long_per_day * trade.quantity
                else:
                    trade.swap += self.config.swap_short_per_day * trade.quantity
        
        # Close trades
        for trade, exit_price, reason in trades_to_close:
            self._close_trade(trade, exit_price, bar.timestamp, reason)
    
    def _update_trailing_stop(self, trade: Trade, bar: OHLCV):
        """Update trailing stop"""
        trail_distance = self.config.trailing_stop_pips * self.config.pip_size
        
        if trade.direction == TradeDirection.LONG:
            new_stop = bar.high - trail_distance
            if new_stop > trade.stop_loss:
                trade.stop_loss = new_stop
        else:
            new_stop = bar.low + trail_distance
            if new_stop < trade.stop_loss:
                trade.stop_loss = new_stop
    
    def _close_trade(
        self, 
        trade: Trade, 
        exit_price: float, 
        exit_time: datetime,
        reason: ExitReason
    ):
        """Close a trade"""
        trade.exit_time = exit_time
        trade.exit_price = exit_price
        trade.exit_reason = reason
        trade.status = TradeStatus.CLOSED
        
        # Calculate P&L
        if trade.direction == TradeDirection.LONG:
            trade.pnl = (exit_price - trade.entry_price) * trade.quantity * 100000
            trade.pnl_pips = (exit_price - trade.entry_price) / self.config.pip_size
        else:
            trade.pnl = (trade.entry_price - exit_price) * trade.quantity * 100000
            trade.pnl_pips = (trade.entry_price - exit_price) / self.config.pip_size
        
        # Calculate R-multiple
        initial_risk = abs(trade.entry_price - trade.stop_loss)
        if initial_risk > 0:
            if trade.direction == TradeDirection.LONG:
                trade.r_multiple = (exit_price - trade.entry_price) / initial_risk
            else:
                trade.r_multiple = (trade.entry_price - exit_price) / initial_risk
        
        # Update capital
        self.capital += trade.net_pnl
        
        # Remove from open trades
        self.open_trades = [t for t in self.open_trades if t.trade_id != trade.trade_id]
        
        logger.debug(f"Closed trade {trade.trade_id}: {reason.name}, P&L: ${trade.net_pnl:.2f}")
    
    def _close_all_trades(self, bar: OHLCV, reason: ExitReason):
        """Close all open trades"""
        for trade in self.open_trades[:]:
            self._close_trade(trade, bar.close, bar.timestamp, reason)
    
    def _update_equity(self, bar: OHLCV):
        """Update equity curve"""
        # Calculate unrealized P&L
        unrealized_pnl = 0
        for trade in self.open_trades:
            if trade.direction == TradeDirection.LONG:
                unrealized_pnl += (bar.close - trade.entry_price) * trade.quantity * 100000
            else:
                unrealized_pnl += (trade.entry_price - bar.close) * trade.quantity * 100000
        
        self.equity = self.capital + unrealized_pnl
        self.equity_curve.append((bar.timestamp, self.equity))
        
        # Update peak and drawdown
        if self.equity > self._peak_equity:
            self._peak_equity = self.equity
        
        drawdown = self._peak_equity - self.equity
        self.drawdown_curve.append((bar.timestamp, drawdown))
    
    def _analyze_model_performance(self) -> Dict[str, Dict]:
        """Analyze performance by ICT model"""
        model_trades = defaultdict(list)
        
        for trade in self.trades:
            if trade.model_name:
                model_trades[trade.model_name].append(trade)
        
        results = {}
        for model, trades in model_trades.items():
            closed = [t for t in trades if t.status == TradeStatus.CLOSED]
            if closed:
                wins = len([t for t in closed if t.is_winner])
                results[model] = {
                    'trades': len(closed),
                    'win_rate': wins / len(closed),
                    'total_pnl': sum(t.net_pnl for t in closed),
                    'avg_r': statistics.mean([t.r_multiple for t in closed]) if closed else 0,
                    'profit_factor': (
                        sum(t.net_pnl for t in closed if t.is_winner) /
                        abs(sum(t.net_pnl for t in closed if not t.is_winner))
                        if any(not t.is_winner for t in closed) else float('inf')
                    )
                }
        
        return results
    
    def _analyze_session_performance(self) -> Dict[str, Dict]:
        """Analyze performance by trading session"""
        session_trades = defaultdict(list)
        
        for trade in self.trades:
            hour = trade.entry_time.hour
            if 2 <= hour < 5:
                session = 'London'
            elif 8 <= hour < 12:
                session = 'NY_AM'
            elif 12 <= hour < 13:
                session = 'NY_Lunch'
            elif 13 <= hour < 17:
                session = 'NY_PM'
            elif 20 <= hour or hour < 2:
                session = 'Asian'
            else:
                session = 'Other'
            
            session_trades[session].append(trade)
        
        results = {}
        for session, trades in session_trades.items():
            closed = [t for t in trades if t.status == TradeStatus.CLOSED]
            if closed:
                wins = len([t for t in closed if t.is_winner])
                results[session] = {
                    'trades': len(closed),
                    'win_rate': wins / len(closed),
                    'total_pnl': sum(t.net_pnl for t in closed),
                    'avg_pnl': statistics.mean([t.net_pnl for t in closed])
                }
        
        return results
    
    def _analyze_confluence(self) -> Dict[str, float]:
        """Analyze confluence factor impact"""
        factor_performance = defaultdict(list)
        
        for trade in self.trades:
            if trade.status == TradeStatus.CLOSED:
                for factor in trade.confluence_factors:
                    factor_performance[factor].append(trade.net_pnl)
        
        results = {}
        for factor, pnls in factor_performance.items():
            results[factor] = {
                'count': len(pnls),
                'avg_pnl': statistics.mean(pnls) if pnls else 0,
                'win_rate': len([p for p in pnls if p > 0]) / len(pnls) if pnls else 0
            }
        
        return results


# =============================================================================
# WALK-FORWARD ANALYZER
# =============================================================================

class WalkForwardAnalyzer:
    """
    Walk-forward analysis for strategy validation.
    Tests strategy on rolling windows to detect overfitting.
    """
    
    def __init__(
        self,
        in_sample_ratio: float = 0.7,
        num_windows: int = 5,
        step_size: Optional[int] = None
    ):
        """
        Initialize walk-forward analyzer.
        
        Args:
            in_sample_ratio: Ratio of data for training (0.7 = 70%)
            num_windows: Number of rolling windows
            step_size: Step size between windows (auto-calculated if None)
        """
        self.in_sample_ratio = in_sample_ratio
        self.num_windows = num_windows
        self.step_size = step_size
    
    def analyze(
        self,
        data: List[OHLCV],
        strategy_factory: Callable[[List[OHLCV]], Callable],
        config: BacktestConfig
    ) -> WalkForwardResult:
        """
        Perform walk-forward analysis.
        
        Args:
            data: Full dataset
            strategy_factory: Function that takes training data and returns strategy function
            config: Backtest configuration
            
        Returns:
            WalkForwardResult with analysis results
        """
        total_bars = len(data)
        window_size = total_bars // self.num_windows
        
        if self.step_size is None:
            step_size = window_size // 2
        else:
            step_size = self.step_size
        
        in_sample_size = int(window_size * self.in_sample_ratio)
        out_of_sample_size = window_size - in_sample_size
        
        windows = []
        in_sample_metrics = []
        out_of_sample_metrics = []
        all_oos_trades = []
        all_oos_equity = []
        
        logger.info(f"Walk-forward analysis: {self.num_windows} windows, "
                   f"IS: {in_sample_size} bars, OOS: {out_of_sample_size} bars")
        
        for i in range(self.num_windows):
            start_idx = i * step_size
            end_idx = start_idx + window_size
            
            if end_idx > total_bars:
                break
            
            # Split into in-sample and out-of-sample
            is_data = data[start_idx:start_idx + in_sample_size]
            oos_data = data[start_idx + in_sample_size:end_idx]
            
            if not is_data or not oos_data:
                continue
            
            window_info = {
                'window': i + 1,
                'is_start': is_data[0].timestamp,
                'is_end': is_data[-1].timestamp,
                'oos_start': oos_data[0].timestamp,
                'oos_end': oos_data[-1].timestamp
            }
            windows.append(window_info)
            
            # Train strategy on in-sample data
            strategy = strategy_factory(is_data)
            
            # Test on in-sample
            is_engine = BacktestEngine(config)
            is_result = is_engine.run(is_data, strategy)
            in_sample_metrics.append(is_result.metrics)
            
            # Test on out-of-sample
            oos_engine = BacktestEngine(config)
            oos_result = oos_engine.run(oos_data, strategy)
            out_of_sample_metrics.append(oos_result.metrics)
            
            # Collect OOS trades
            all_oos_trades.extend(oos_result.trades)
            all_oos_equity.extend(oos_result.equity_curve)
            
            logger.info(f"Window {i+1}: IS Win Rate: {is_result.metrics.win_rate:.1%}, "
                       f"OOS Win Rate: {oos_result.metrics.win_rate:.1%}")
        
        # Calculate combined OOS metrics
        combined_oos_metrics = PerformanceCalculator.calculate(
            all_oos_trades,
            all_oos_equity,
            config.initial_capital
        )
        
        # Parameter stability analysis
        param_stability = self._analyze_stability(in_sample_metrics, out_of_sample_metrics)
        
        # Robustness score
        robustness = self._calculate_robustness(in_sample_metrics, out_of_sample_metrics)
        
        # Detect overfitting
        is_curve_fit = self._detect_overfitting(in_sample_metrics, out_of_sample_metrics)
        
        return WalkForwardResult(
            windows=windows,
            in_sample_metrics=in_sample_metrics,
            out_of_sample_metrics=out_of_sample_metrics,
            combined_oos_metrics=combined_oos_metrics,
            parameter_stability=param_stability,
            robustness_score=robustness,
            is_curve_fit=is_curve_fit
        )
    
    def _analyze_stability(
        self,
        is_metrics: List[PerformanceMetrics],
        oos_metrics: List[PerformanceMetrics]
    ) -> Dict[str, float]:
        """Analyze parameter stability across windows"""
        stability = {}
        
        # Win rate stability
        is_win_rates = [m.win_rate for m in is_metrics]
        oos_win_rates = [m.win_rate for m in oos_metrics]
        
        if is_win_rates:
            stability['is_win_rate_mean'] = statistics.mean(is_win_rates)
            stability['is_win_rate_std'] = statistics.stdev(is_win_rates) if len(is_win_rates) > 1 else 0
        
        if oos_win_rates:
            stability['oos_win_rate_mean'] = statistics.mean(oos_win_rates)
            stability['oos_win_rate_std'] = statistics.stdev(oos_win_rates) if len(oos_win_rates) > 1 else 0
        
        # Profit factor stability
        is_pf = [m.profit_factor for m in is_metrics if m.profit_factor != float('inf')]
        oos_pf = [m.profit_factor for m in oos_metrics if m.profit_factor != float('inf')]
        
        if is_pf:
            stability['is_profit_factor_mean'] = statistics.mean(is_pf)
        if oos_pf:
            stability['oos_profit_factor_mean'] = statistics.mean(oos_pf)
        
        # Degradation ratio (OOS / IS performance)
        if is_win_rates and oos_win_rates:
            stability['win_rate_degradation'] = statistics.mean(oos_win_rates) / statistics.mean(is_win_rates)
        
        return stability
    
    def _calculate_robustness(
        self,
        is_metrics: List[PerformanceMetrics],
        oos_metrics: List[PerformanceMetrics]
    ) -> float:
        """Calculate robustness score (0-100)"""
        if not oos_metrics:
            return 0
        
        score = 50  # Start at 50
        
        # OOS profitability
        profitable_windows = len([m for m in oos_metrics if m.net_profit > 0])
        profitability_ratio = profitable_windows / len(oos_metrics)
        score += (profitability_ratio - 0.5) * 30
        
        # Win rate consistency
        oos_win_rates = [m.win_rate for m in oos_metrics]
        if len(oos_win_rates) > 1:
            consistency = 1 - (statistics.stdev(oos_win_rates) / statistics.mean(oos_win_rates) if statistics.mean(oos_win_rates) > 0 else 1)
            score += consistency * 20
        
        # IS/OOS correlation
        if is_metrics and len(is_metrics) == len(oos_metrics):
            is_profits = [m.net_profit for m in is_metrics]
            oos_profits = [m.net_profit for m in oos_metrics]
            
            # Simple correlation
            if statistics.stdev(is_profits) > 0 and statistics.stdev(oos_profits) > 0:
                correlation = sum((a - statistics.mean(is_profits)) * (b - statistics.mean(oos_profits)) 
                                for a, b in zip(is_profits, oos_profits))
                correlation /= (len(is_profits) * statistics.stdev(is_profits) * statistics.stdev(oos_profits))
                score += correlation * 10
        
        return max(0, min(100, score))
    
    def _detect_overfitting(
        self,
        is_metrics: List[PerformanceMetrics],
        oos_metrics: List[PerformanceMetrics]
    ) -> bool:
        """Detect if strategy is likely overfit"""
        if not is_metrics or not oos_metrics:
            return False
        
        # Compare IS vs OOS performance
        is_avg_win_rate = statistics.mean([m.win_rate for m in is_metrics])
        oos_avg_win_rate = statistics.mean([m.win_rate for m in oos_metrics])
        
        # Significant degradation (>30% drop)
        if oos_avg_win_rate < is_avg_win_rate * 0.7:
            return True
        
        # OOS not profitable while IS is
        is_profitable = statistics.mean([m.net_profit for m in is_metrics]) > 0
        oos_profitable = statistics.mean([m.net_profit for m in oos_metrics]) > 0
        
        if is_profitable and not oos_profitable:
            return True
        
        return False


# =============================================================================
# STRATEGY COMPARATOR
# =============================================================================

class StrategyComparator:
    """
    Compare multiple ICT strategies/models.
    """
    
    def __init__(self, config: BacktestConfig):
        self.config = config
        self.results: Dict[str, BacktestResult] = {}
    
    def add_result(self, name: str, result: BacktestResult):
        """Add backtest result for comparison"""
        self.results[name] = result
    
    def run_comparison(
        self,
        data: List[OHLCV],
        strategies: Dict[str, Callable]
    ) -> StrategyComparison:
        """
        Run multiple strategies and compare.
        
        Args:
            data: Historical data
            strategies: Dict of strategy_name -> strategy_function
            
        Returns:
            StrategyComparison with rankings
        """
        self.results = {}
        
        for name, strategy in strategies.items():
            logger.info(f"Testing strategy: {name}")
            engine = BacktestEngine(self.config)
            result = engine.run(data, strategy, strategy_name=name)
            self.results[name] = result
        
        return self.compare()
    
    def compare(self) -> StrategyComparison:
        """Compare all added results"""
        if not self.results:
            raise ValueError("No results to compare")
        
        strategy_names = list(self.results.keys())
        
        # Collect metrics for comparison
        metrics_comparison = {}
        
        metric_names = [
            'total_return_pct', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
            'win_rate', 'profit_factor', 'max_drawdown_pct', 'expectancy',
            'avg_r_multiple', 'total_trades', 'recovery_factor'
        ]
        
        for metric in metric_names:
            metrics_comparison[metric] = {}
            for name, result in self.results.items():
                value = getattr(result.metrics, metric, 0)
                metrics_comparison[metric][name] = value
        
        # Create rankings for each metric
        rankings = {}
        
        # Higher is better for these
        higher_better = ['total_return_pct', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio',
                        'win_rate', 'profit_factor', 'expectancy', 'avg_r_multiple', 'recovery_factor']
        
        # Lower is better for these
        lower_better = ['max_drawdown_pct']
        
        for metric in metric_names:
            values = metrics_comparison[metric]
            
            if metric in higher_better:
                sorted_names = sorted(values.keys(), key=lambda x: values[x], reverse=True)
            elif metric in lower_better:
                sorted_names = sorted(values.keys(), key=lambda x: values[x])
            else:
                sorted_names = sorted(values.keys(), key=lambda x: values[x], reverse=True)
            
            rankings[metric] = sorted_names
        
        # Calculate composite score
        composite_scores = {name: 0 for name in strategy_names}
        
        weights = {
            'sharpe_ratio': 2.0,
            'sortino_ratio': 1.5,
            'profit_factor': 1.5,
            'win_rate': 1.0,
            'max_drawdown_pct': 1.5,
            'expectancy': 1.0,
            'recovery_factor': 1.0
        }
        
        for metric, ranking in rankings.items():
            weight = weights.get(metric, 1.0)
            for i, name in enumerate(ranking):
                points = (len(ranking) - i) * weight
                composite_scores[name] += points
        
        # Best strategy by composite score
        best_strategy = max(composite_scores.keys(), key=lambda x: composite_scores[x])
        
        # Generate recommendation
        best_result = self.results[best_strategy]
        recommendation = self._generate_recommendation(best_strategy, best_result)
        
        return StrategyComparison(
            strategies=strategy_names,
            metrics_comparison=metrics_comparison,
            rankings=rankings,
            best_strategy=best_strategy,
            recommendation=recommendation
        )
    
    def _generate_recommendation(self, name: str, result: BacktestResult) -> str:
        """Generate strategy recommendation"""
        metrics = result.metrics
        
        lines = [f"Recommended Strategy: {name}"]
        
        # Strengths
        strengths = []
        if metrics.sharpe_ratio > 1.5:
            strengths.append(f"Excellent risk-adjusted returns (Sharpe: {metrics.sharpe_ratio:.2f})")
        if metrics.win_rate > 0.6:
            strengths.append(f"High win rate ({metrics.win_rate:.1%})")
        if metrics.profit_factor > 2.0:
            strengths.append(f"Strong profit factor ({metrics.profit_factor:.2f})")
        if metrics.max_drawdown_pct < 10:
            strengths.append(f"Low drawdown ({metrics.max_drawdown_pct:.1f}%)")
        
        if strengths:
            lines.append("\nStrengths:")
            for s in strengths:
                lines.append(f"  • {s}")
        
        # Concerns
        concerns = []
        if metrics.sharpe_ratio < 1.0:
            concerns.append("Moderate risk-adjusted returns")
        if metrics.win_rate < 0.5:
            concerns.append("Win rate below 50%")
        if metrics.max_drawdown_pct > 20:
            concerns.append(f"High drawdown risk ({metrics.max_drawdown_pct:.1f}%)")
        if metrics.total_trades < 30:
            concerns.append("Low sample size - needs more data")
        
        if concerns:
            lines.append("\nConcerns:")
            for c in concerns:
                lines.append(f"  • {c}")
        
        return "\n".join(lines)
    
    def get_summary_table(self) -> str:
        """Get comparison summary as formatted table"""
        if not self.results:
            return "No results to compare"
        
        # Headers
        headers = ['Strategy', 'Return %', 'Sharpe', 'Win Rate', 'PF', 'Max DD %', 'Trades']
        
        # Data rows
        rows = []
        for name, result in self.results.items():
            m = result.metrics
            row = [
                name[:20],
                f"{m.total_return_pct:.1f}",
                f"{m.sharpe_ratio:.2f}",
                f"{m.win_rate:.1%}",
                f"{m.profit_factor:.2f}" if m.profit_factor < 100 else "∞",
                f"{m.max_drawdown_pct:.1f}",
                str(m.total_trades)
            ]
            rows.append(row)
        
        # Format table
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
        
        lines = []
        
        # Header
        header_line = " | ".join(h.ljust(col_widths[i]) for i, h in enumerate(headers))
        lines.append(header_line)
        lines.append("-" * len(header_line))
        
        # Data
        for row in rows:
            line = " | ".join(str(v).ljust(col_widths[i]) for i, v in enumerate(row))
            lines.append(line)
        
        return "\n".join(lines)


# =============================================================================
# VISUALIZATION MODULE
# =============================================================================

class BacktestVisualizer:
    """
    Visualization tools for backtest results.
    Generates ASCII charts and data exports.
    """
    
    def __init__(self, result: BacktestResult):
        self.result = result
    
    def equity_curve_ascii(self, width: int = 60, height: int = 15) -> str:
        """Generate ASCII equity curve"""
        if not self.result.equity_curve:
            return "No equity data"
        
        equities = [e for _, e in self.result.equity_curve]
        
        if len(equities) < 2:
            return "Insufficient data"
        
        min_eq = min(equities)
        max_eq = max(equities)
        range_eq = max_eq - min_eq if max_eq != min_eq else 1
        
        # Downsample to width
        step = max(1, len(equities) // width)
        sampled = equities[::step][:width]
        
        # Build chart
        chart = []
        chart.append(f"Equity Curve (${min_eq:,.0f} - ${max_eq:,.0f})")
        chart.append("=" * (width + 10))
        
        for row in range(height, 0, -1):
            threshold = min_eq + (range_eq * row / height)
            line = f"${threshold:>8,.0f} |"
            
            for value in sampled:
                if value >= threshold:
                    line += "█"
                else:
                    line += " "
            
            chart.append(line)
        
        chart.append(" " * 10 + "└" + "─" * width)
        
        return "\n".join(chart)
    
    def drawdown_chart_ascii(self, width: int = 60, height: int = 10) -> str:
        """Generate ASCII drawdown chart"""
        if not self.result.drawdown_curve:
            return "No drawdown data"
        
        drawdowns = [d for _, d in self.result.drawdown_curve]
        
        if not drawdowns or max(drawdowns) == 0:
            return "No drawdowns"
        
        max_dd = max(drawdowns)
        
        # Downsample
        step = max(1, len(drawdowns) // width)
        sampled = drawdowns[::step][:width]
        
        chart = []
        chart.append(f"Drawdown Chart (Max: ${max_dd:,.0f})")
        chart.append("=" * (width + 10))
        
        for row in range(height, 0, -1):
            threshold = max_dd * row / height
            line = f"${threshold:>8,.0f} |"
            
            for value in sampled:
                if value >= threshold:
                    line += "▓"
                else:
                    line += " "
            
            chart.append(line)
        
        chart.append(" " * 10 + "└" + "─" * width)
        
        return "\n".join(chart)
    
    def monthly_returns_heatmap(self) -> str:
        """Generate monthly returns heatmap"""
        if not self.result.monthly_returns:
            return "No monthly data"
        
        # Organize by year and month
        years = defaultdict(dict)
        for month_str, ret in self.result.monthly_returns.items():
            year, month = month_str.split('-')
            years[year][int(month)] = ret
        
        # Build heatmap
        lines = []
        lines.append("Monthly Returns Heatmap (%)")
        lines.append("=" * 80)
        
        # Header
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        header = "Year  " + " ".join(f"{m:>6}" for m in months) + "  Total"
        lines.append(header)
        lines.append("-" * 80)
        
        for year in sorted(years.keys()):
            row = f"{year}  "
            year_total = 0
            
            for m in range(1, 13):
                if m in years[year]:
                    ret = years[year][m]
                    year_total += ret
                    
                    # Color coding with ASCII
                    if ret >= 5:
                        cell = f"[{ret:>4.1f}]"
                    elif ret >= 0:
                        cell = f" {ret:>4.1f} "
                    elif ret >= -5:
                        cell = f"({abs(ret):>4.1f})"
                    else:
                        cell = f"<{abs(ret):>4.1f}>"
                else:
                    cell = "   -  "
                
                row += cell
            
            row += f" {year_total:>6.1f}"
            lines.append(row)
        
        return "\n".join(lines)
    
    def trade_distribution(self) -> str:
        """Generate trade distribution analysis"""
        trades = [t for t in self.result.trades if t.status == TradeStatus.CLOSED]
        
        if not trades:
            return "No closed trades"
        
        pnls = [t.net_pnl for t in trades]
        
        lines = []
        lines.append("Trade P&L Distribution")
        lines.append("=" * 50)
        
        # Statistics
        lines.append(f"Total Trades: {len(trades)}")
        lines.append(f"Mean P&L: ${statistics.mean(pnls):.2f}")
        lines.append(f"Median P&L: ${statistics.median(pnls):.2f}")
        lines.append(f"Std Dev: ${statistics.stdev(pnls):.2f}" if len(pnls) > 1 else "")
        lines.append(f"Min P&L: ${min(pnls):.2f}")
        lines.append(f"Max P&L: ${max(pnls):.2f}")
        
        # Histogram
        lines.append("\nHistogram:")
        
        buckets = 10
        min_pnl = min(pnls)
        max_pnl = max(pnls)
        bucket_size = (max_pnl - min_pnl) / buckets if max_pnl != min_pnl else 1
        
        counts = [0] * buckets
        for pnl in pnls:
            idx = min(buckets - 1, int((pnl - min_pnl) / bucket_size))
            counts[idx] += 1
        
        max_count = max(counts) if counts else 1
        
        for i, count in enumerate(counts):
            bucket_start = min_pnl + i * bucket_size
            bucket_end = bucket_start + bucket_size
            bar_len = int(30 * count / max_count)
            bar = "█" * bar_len
            lines.append(f"${bucket_start:>7.0f} - ${bucket_end:>7.0f} | {bar} ({count})")
        
        return "\n".join(lines)
    
    def time_analysis(self) -> str:
        """Generate time-based performance analysis"""
        trades = [t for t in self.result.trades if t.status == TradeStatus.CLOSED]
        
        if not trades:
            return "No closed trades"
        
        lines = []
        lines.append("Time-Based Performance Analysis")
        lines.append("=" * 60)
        
        # By hour
        hour_pnl = defaultdict(list)
        for t in trades:
            hour_pnl[t.entry_time.hour].append(t.net_pnl)
        
        lines.append("\nBy Hour (EST):")
        lines.append("-" * 40)
        
        for hour in sorted(hour_pnl.keys()):
            pnls = hour_pnl[hour]
            total = sum(pnls)
            wins = len([p for p in pnls if p > 0])
            wr = wins / len(pnls) if pnls else 0
            
            bar = "+" * min(20, int(total / 10)) if total > 0 else "-" * min(20, int(abs(total) / 10))
            lines.append(f"{hour:02d}:00 | {bar:20} | ${total:>8.2f} | WR: {wr:.0%}")
        
        # By day of week
        day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        day_pnl = defaultdict(list)
        for t in trades:
            day_pnl[t.entry_time.weekday()].append(t.net_pnl)
        
        lines.append("\nBy Day of Week:")
        lines.append("-" * 40)
        
        for day in sorted(day_pnl.keys()):
            pnls = day_pnl[day]
            total = sum(pnls)
            wins = len([p for p in pnls if p > 0])
            wr = wins / len(pnls) if pnls else 0
            
            lines.append(f"{day_names[day]} | ${total:>10.2f} | Trades: {len(pnls):>3} | WR: {wr:.0%}")
        
        return "\n".join(lines)
    
    def generate_report(self) -> str:
        """Generate comprehensive text report"""
        lines = []
        
        lines.append("=" * 80)
        lines.append(f"BACKTEST REPORT: {self.result.strategy_name}")
        lines.append("=" * 80)
        lines.append(f"\nSymbol: {self.result.symbol}")
        lines.append(f"Period: {self.result.start_date.strftime('%Y-%m-%d')} to {self.result.end_date.strftime('%Y-%m-%d')}")
        lines.append(f"Duration: {self.result.duration_days} days")
        lines.append(f"Initial Capital: ${self.result.config.initial_capital:,.2f}")
        
        # Performance Summary
        m = self.result.metrics
        lines.append("\n" + "-" * 40)
        lines.append("PERFORMANCE SUMMARY")
        lines.append("-" * 40)
        
        lines.append(f"\nReturns:")
        lines.append(f"  Total Return: {m.total_return_pct:.2f}%")
        lines.append(f"  CAGR: {m.cagr:.2f}%")
        lines.append(f"  Net Profit: ${m.net_profit:,.2f}")
        
        lines.append(f"\nRisk Metrics:")
        lines.append(f"  Max Drawdown: {m.max_drawdown_pct:.2f}%")
        lines.append(f"  Sharpe Ratio: {m.sharpe_ratio:.2f}")
        lines.append(f"  Sortino Ratio: {m.sortino_ratio:.2f}")
        lines.append(f"  Calmar Ratio: {m.calmar_ratio:.2f}")
        
        lines.append(f"\nTrade Statistics:")
        lines.append(f"  Total Trades: {m.total_trades}")
        lines.append(f"  Win Rate: {m.win_rate:.1%}")
        lines.append(f"  Profit Factor: {m.profit_factor:.2f}")
        lines.append(f"  Expectancy: ${m.expectancy:.2f}")
        lines.append(f"  Avg R-Multiple: {m.avg_r_multiple:.2f}R")
        
        lines.append(f"\nTrade Duration:")
        lines.append(f"  Avg Bars Held: {m.avg_bars_held:.1f}")
        lines.append(f"  Avg Winners: {m.avg_bars_winners:.1f} bars")
        lines.append(f"  Avg Losers: {m.avg_bars_losers:.1f} bars")
        
        lines.append(f"\nStreaks:")
        lines.append(f"  Max Consecutive Wins: {m.max_consecutive_wins}")
        lines.append(f"  Max Consecutive Losses: {m.max_consecutive_losses}")
        
        lines.append(f"\nCosts:")
        lines.append(f"  Total Commission: ${m.total_commission:.2f}")
        lines.append(f"  Total Swap: ${m.total_swap:.2f}")
        lines.append(f"  Total Slippage: ${m.total_slippage:.2f}")
        
        # ICT Analysis
        if self.result.model_performance:
            lines.append("\n" + "-" * 40)
            lines.append("ICT MODEL PERFORMANCE")
            lines.append("-" * 40)
            
            for model, stats in self.result.model_performance.items():
                lines.append(f"\n{model}:")
                lines.append(f"  Trades: {stats['trades']}, Win Rate: {stats['win_rate']:.1%}")
                lines.append(f"  Total P&L: ${stats['total_pnl']:.2f}, Avg R: {stats['avg_r']:.2f}")
        
        if self.result.session_performance:
            lines.append("\n" + "-" * 40)
            lines.append("SESSION PERFORMANCE")
            lines.append("-" * 40)
            
            for session, stats in self.result.session_performance.items():
                lines.append(f"\n{session}:")
                lines.append(f"  Trades: {stats['trades']}, Win Rate: {stats['win_rate']:.1%}")
                lines.append(f"  Total P&L: ${stats['total_pnl']:.2f}")
        
        # Charts
        lines.append("\n" + "=" * 80)
        lines.append("CHARTS")
        lines.append("=" * 80)
        
        lines.append("\n" + self.equity_curve_ascii())
        lines.append("\n" + self.drawdown_chart_ascii())
        lines.append("\n" + self.monthly_returns_heatmap())
        lines.append("\n" + self.trade_distribution())
        lines.append("\n" + self.time_analysis())
        
        lines.append("\n" + "=" * 80)
        lines.append("END OF REPORT")
        lines.append("=" * 80)
        
        return "\n".join(lines)
    
    def export_trades_csv(self) -> str:
        """Export trades to CSV format"""
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Header
        writer.writerow([
            'trade_id', 'direction', 'entry_time', 'entry_price',
            'exit_time', 'exit_price', 'quantity', 'pnl', 'pnl_pips',
            'r_multiple', 'exit_reason', 'bars_held', 'model_name'
        ])
        
        # Data
        for t in self.result.trades:
            writer.writerow([
                t.trade_id,
                t.direction.name,
                t.entry_time.isoformat(),
                t.entry_price,
                t.exit_time.isoformat() if t.exit_time else '',
                t.exit_price or '',
                t.quantity,
                t.net_pnl,
                t.pnl_pips,
                t.r_multiple,
                t.exit_reason.name if t.exit_reason else '',
                t.bars_held,
                t.model_name
            ])
        
        return output.getvalue()
    
    def export_equity_csv(self) -> str:
        """Export equity curve to CSV"""
        output = io.StringIO()
        writer = csv.writer(output)
        
        writer.writerow(['timestamp', 'equity', 'drawdown'])
        
        for i, (timestamp, equity) in enumerate(self.result.equity_curve):
            dd = self.result.drawdown_curve[i][1] if i < len(self.result.drawdown_curve) else 0
            writer.writerow([timestamp.isoformat(), equity, dd])
        
        return output.getvalue()


# =============================================================================
# SAMPLE ICT STRATEGIES FOR TESTING
# =============================================================================

class SampleICTStrategies:
    """Sample ICT strategies for backtesting"""
    
    @staticmethod
    def simple_fvg_strategy(data: List[OHLCV], index: int, engine: BacktestEngine) -> Optional[TradeSignal]:
        """
        Simple FVG (Fair Value Gap) strategy.
        Buy when price fills a bullish FVG, sell when fills bearish FVG.
        """
        if index < 5:
            return None
        
        # Look for FVG in recent bars
        for i in range(index - 3, index):
            if i < 1:
                continue
            
            bar1 = data[i - 1]
            bar2 = data[i]
            bar3 = data[i + 1] if i + 1 <= index else data[index]
            
            current = data[index]
            
            # Bullish FVG: bar1.high < bar3.low (gap up)
            if bar1.high < bar3.low:
                fvg_top = bar3.low
                fvg_bottom = bar1.high
                
                # Price filling the gap
                if current.low <= fvg_top and current.close > fvg_bottom:
                    stop = fvg_bottom - (fvg_top - fvg_bottom)
                    target = current.close + 2 * (current.close - stop)
                    
                    return TradeSignal(
                        timestamp=current.timestamp,
                        direction=TradeDirection.LONG,
                        entry_price=current.close,
                        stop_loss=stop,
                        take_profit=target,
                        confidence=0.7,
                        model_name="FVG_Fill",
                        confluence_factors=["Bullish_FVG", "Gap_Fill"]
                    )
            
            # Bearish FVG: bar1.low > bar3.high (gap down)
            if bar1.low > bar3.high:
                fvg_top = bar1.low
                fvg_bottom = bar3.high
                
                # Price filling the gap
                if current.high >= fvg_bottom and current.close < fvg_top:
                    stop = fvg_top + (fvg_top - fvg_bottom)
                    target = current.close - 2 * (stop - current.close)
                    
                    return TradeSignal(
                        timestamp=current.timestamp,
                        direction=TradeDirection.SHORT,
                        entry_price=current.close,
                        stop_loss=stop,
                        take_profit=target,
                        confidence=0.7,
                        model_name="FVG_Fill",
                        confluence_factors=["Bearish_FVG", "Gap_Fill"]
                    )
        
        return None
    
    @staticmethod
    def order_block_strategy(data: List[OHLCV], index: int, engine: BacktestEngine) -> Optional[TradeSignal]:
        """
        Simple Order Block strategy.
        Look for strong moves and trade retrace to origin.
        """
        if index < 10:
            return None
        
        current = data[index]
        
        # Look for displacement (strong move)
        for i in range(index - 5, index):
            if i < 1:
                continue
            
            bar = data[i]
            prev_bar = data[i - 1]
            
            # Strong bullish bar (body > 70% of range)
            body = abs(bar.close - bar.open)
            range_size = bar.high - bar.low
            
            if range_size > 0 and body / range_size > 0.7:
                if bar.close > bar.open:  # Bullish
                    ob_high = prev_bar.high
                    ob_low = prev_bar.low
                    
                    # Price retracing to OB
                    if current.low <= ob_high and current.close > ob_low:
                        stop = ob_low - (ob_high - ob_low) * 0.5
                        target = current.close + 3 * (current.close - stop)
                        
                        return TradeSignal(
                            timestamp=current.timestamp,
                            direction=TradeDirection.LONG,
                            entry_price=current.close,
                            stop_loss=stop,
                            take_profit=target,
                            confidence=0.75,
                            model_name="Order_Block",
                            confluence_factors=["Bullish_OB", "Displacement", "Retrace"]
                        )
                
                elif bar.close < bar.open:  # Bearish
                    ob_high = prev_bar.high
                    ob_low = prev_bar.low
                    
                    # Price retracing to OB
                    if current.high >= ob_low and current.close < ob_high:
                        stop = ob_high + (ob_high - ob_low) * 0.5
                        target = current.close - 3 * (stop - current.close)
                        
                        return TradeSignal(
                            timestamp=current.timestamp,
                            direction=TradeDirection.SHORT,
                            entry_price=current.close,
                            stop_loss=stop,
                            take_profit=target,
                            confidence=0.75,
                            model_name="Order_Block",
                            confluence_factors=["Bearish_OB", "Displacement", "Retrace"]
                        )
        
        return None
    
    @staticmethod
    def kill_zone_strategy(data: List[OHLCV], index: int, engine: BacktestEngine) -> Optional[TradeSignal]:
        """
        Trade only during ICT Kill Zones with trend confirmation.
        """
        if index < 20:
            return None
        
        current = data[index]
        hour = current.timestamp.hour
        
        # Check if in kill zone (simplified)
        in_london_kz = 2 <= hour < 5
        in_ny_kz = 8 <= hour < 11 or 13 <= hour < 16
        
        if not (in_london_kz or in_ny_kz):
            return None
        
        # Simple trend: compare to 20-bar SMA
        closes = [data[i].close for i in range(index - 20, index + 1)]
        sma20 = sum(closes) / len(closes)
        
        # Momentum: recent bar direction
        recent_bullish = sum(1 for i in range(index - 3, index + 1) 
                           if data[i].close > data[i].open)
        recent_bearish = 4 - recent_bullish
        
        if current.close > sma20 and recent_bullish >= 3:
            # Bullish setup in kill zone
            recent_low = min(data[i].low for i in range(index - 5, index + 1))
            stop = recent_low - 0.0010
            target = current.close + 2.5 * (current.close - stop)
            
            kz_name = "London_KZ" if in_london_kz else "NY_KZ"
            
            return TradeSignal(
                timestamp=current.timestamp,
                direction=TradeDirection.LONG,
                entry_price=current.close,
                stop_loss=stop,
                take_profit=target,
                confidence=0.8,
                model_name="Kill_Zone",
                confluence_factors=[kz_name, "Trend_Up", "Momentum"]
            )
        
        elif current.close < sma20 and recent_bearish >= 3:
            # Bearish setup in kill zone
            recent_high = max(data[i].high for i in range(index - 5, index + 1))
            stop = recent_high + 0.0010
            target = current.close - 2.5 * (stop - current.close)
            
            kz_name = "London_KZ" if in_london_kz else "NY_KZ"
            
            return TradeSignal(
                timestamp=current.timestamp,
                direction=TradeDirection.SHORT,
                entry_price=current.close,
                stop_loss=stop,
                take_profit=target,
                confidence=0.8,
                model_name="Kill_Zone",
                confluence_factors=[kz_name, "Trend_Down", "Momentum"]
            )
        
        return None


# =============================================================================
# SYNTHETIC DATA GENERATOR
# =============================================================================

class SyntheticDataGenerator:
    """Generate synthetic OHLCV data for testing"""
    
    @staticmethod
    def generate(
        start_date: datetime,
        end_date: datetime,
        timeframe_minutes: int = 60,
        initial_price: float = 1.1000,
        volatility: float = 0.0002,
        trend: float = 0.0,
        add_sessions: bool = True
    ) -> List[OHLCV]:
        """
        Generate synthetic OHLCV data.
        
        Args:
            start_date: Start date
            end_date: End date
            timeframe_minutes: Bar timeframe in minutes
            initial_price: Starting price
            volatility: Price volatility
            trend: Trend component (positive = up)
            add_sessions: Add session-based volatility
            
        Returns:
            List of OHLCV bars
        """
        import random
        
        data = []
        current_time = start_date
        current_price = initial_price
        
        while current_time <= end_date:
            # Skip weekends
            if current_time.weekday() >= 5:
                current_time += timedelta(minutes=timeframe_minutes)
                continue
            
            # Session-based volatility
            hour = current_time.hour
            if add_sessions:
                if 2 <= hour < 5:  # London
                    vol_mult = 1.3
                elif 8 <= hour < 11:  # NY AM
                    vol_mult = 1.5
                elif 13 <= hour < 16:  # NY PM
                    vol_mult = 1.2
                else:
                    vol_mult = 0.7
            else:
                vol_mult = 1.0
            
            # Generate OHLCV
            bar_vol = volatility * vol_mult
            
            # Random walk with trend
            change = random.gauss(trend, bar_vol)
            
            open_price = current_price
            close_price = open_price * (1 + change)
            
            high_price = max(open_price, close_price) * (1 + abs(random.gauss(0, bar_vol * 0.5)))
            low_price = min(open_price, close_price) * (1 - abs(random.gauss(0, bar_vol * 0.5)))
            
            volume = random.randint(100, 1000) * vol_mult
            
            bar = OHLCV(
                timestamp=current_time,
                open=open_price,
                high=high_price,
                low=low_price,
                close=close_price,
                volume=volume
            )
            data.append(bar)
            
            current_price = close_price
            current_time += timedelta(minutes=timeframe_minutes)
        
        return data


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("ICT BACKTESTING FRAMEWORK DEMO")
    print("=" * 80)
    
    # Generate synthetic data
    print("\nGenerating synthetic data...")
    data = SyntheticDataGenerator.generate(
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 6, 30),
        timeframe_minutes=60,
        initial_price=1.1000,
        volatility=0.0003,
        trend=0.00001
    )
    print(f"Generated {len(data)} bars")
    
    # Configure backtest
    config = BacktestConfig(
        initial_capital=10000.0,
        position_size_pct=1.0,
        commission_per_lot=7.0,
        spread_pips=1.0,
        slippage_pips=0.5,
        max_positions=1
    )
    
    # Run backtest with FVG strategy
    print("\n" + "-" * 40)
    print("Running FVG Strategy Backtest...")
    print("-" * 40)
    
    engine = BacktestEngine(config)
    result = engine.run(
        data=data,
        strategy=SampleICTStrategies.simple_fvg_strategy,
        symbol="EUR/USD",
        strategy_name="Simple FVG Strategy"
    )
    
    # Generate report
    visualizer = BacktestVisualizer(result)
    print(visualizer.generate_report())
    
    # Strategy comparison
    print("\n" + "=" * 80)
    print("STRATEGY COMPARISON")
    print("=" * 80)
    
    comparator = StrategyComparator(config)
    
    strategies = {
        "FVG_Strategy": SampleICTStrategies.simple_fvg_strategy,
        "OrderBlock_Strategy": SampleICTStrategies.order_block_strategy,
        "KillZone_Strategy": SampleICTStrategies.kill_zone_strategy
    }
    
    comparison = comparator.run_comparison(data, strategies)
    
    print("\n" + comparator.get_summary_table())
    print(f"\n{comparison.recommendation}")
    
    # Walk-forward analysis
    print("\n" + "=" * 80)
    print("WALK-FORWARD ANALYSIS")
    print("=" * 80)
    
    def strategy_factory(training_data):
        """Factory that returns strategy function"""
        return SampleICTStrategies.kill_zone_strategy
    
    wfa = WalkForwardAnalyzer(
        in_sample_ratio=0.7,
        num_windows=3
    )
    
    wf_result = wfa.analyze(data, strategy_factory, config)
    
    print(f"\nRobustness Score: {wf_result.robustness_score:.1f}/100")
    print(f"Likely Overfit: {'Yes' if wf_result.is_curve_fit else 'No'}")
    print(f"\nOut-of-Sample Combined Metrics:")
    print(f"  Win Rate: {wf_result.combined_oos_metrics.win_rate:.1%}")
    print(f"  Profit Factor: {wf_result.combined_oos_metrics.profit_factor:.2f}")
    print(f"  Total Trades: {wf_result.combined_oos_metrics.total_trades}")
