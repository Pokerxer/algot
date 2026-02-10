"""
ICT Core Engine - Master Integration Layer
===========================================

Central orchestration engine that integrates all ICT handlers into a unified
analysis and signal generation system. This is the brain of the trading bot.

INTEGRATION ARCHITECTURE:
========================

┌─────────────────────────────────────────────────────────────────────┐
│                        ICT CORE ENGINE                               │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐      │
│  │ Market Data     │  │ Market          │  │ Timeframe       │      │
│  │ Engine          │──│ Condition       │──│ Handler         │      │
│  │ (data + time)   │  │ Handler         │  │ (HTF→LTF)      │      │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘      │
│           │                    │                     │                │
│           ▼                    ▼                     ▼                │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    ANALYSIS LAYER                             │    │
│  │  ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐    │    │
│  │  │ Market    │ │ Order     │ │ Liquidity │ │ FVG/Gap   │    │    │
│  │  │ Structure │ │ Block     │ │ Handler   │ │ Handler   │    │    │
│  │  └───────────┘ └───────────┘ └───────────┘ └───────────┘    │    │
│  │                       │                                       │    │
│  │                       ▼                                       │    │
│  │              ┌───────────────┐                               │    │
│  │              │ PD Array      │                               │    │
│  │              │ Handler       │                               │    │
│  │              └───────────────┘                               │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                        │
│                              ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                 TRADING MODEL LAYER                           │    │
│  │  ┌─────────────────────────────────────────────────────────┐│    │
│  │  │ Trading Model Handler (2022, Silver Bullet, Venom, etc.)││    │
│  │  └─────────────────────────────────────────────────────────┘│    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                        │
│                              ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                 SIGNAL AGGREGATION                            │    │
│  │              (Confluence + Quality Scoring)                   │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                        │
│                              ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │              ENTRY/STOP MANAGEMENT                            │    │
│  │           (Entry + Risk + Position Sizing)                    │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                              │                                        │
│                              ▼                                        │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                    FINAL SIGNAL                               │    │
│  │         (Trade Setup with Full Context)                       │    │
│  └─────────────────────────────────────────────────────────────┘    │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘

ICT Core Principles:
- Higher timeframe determines bias (Weekly → Daily → 4H)
- Lower timeframe provides entry (15m → 5m → 1m)
- Time of day is critical (Kill Zones, Macro Times)
- Confluence increases probability
- Wait for setup, don't force trades
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any, Union
from enum import Enum
from datetime import datetime, time, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# ENUMERATIONS
# =============================================================================

class AnalysisMode(Enum):
    """Analysis mode for the engine"""
    FULL = "full"              # Complete analysis
    QUICK = "quick"            # Fast scan
    SIGNAL_ONLY = "signal"     # Just generate signals
    BACKTEST = "backtest"      # Historical analysis


class SetupGrade(Enum):
    """Setup quality grade (ICT-style grading)"""
    A_PLUS = "A+"      # 90-100: Perfect setup, max confidence
    A = "A"            # 80-89: Excellent setup
    B = "B"            # 70-79: Good setup
    C = "C"            # 60-69: Acceptable setup
    D = "D"            # 50-59: Marginal setup
    F = "F"            # <50: Invalid/Poor setup


class TradeDirection(Enum):
    """Trade direction"""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class BiasStrength(Enum):
    """Bias strength level"""
    STRONG = "strong"
    MODERATE = "moderate"
    WEAK = "weak"
    NEUTRAL = "neutral"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TimeframeAnalysis:
    """Analysis for a specific timeframe"""
    timeframe: str
    bias: TradeDirection
    bias_strength: BiasStrength
    market_structure: Optional[Any] = None
    order_blocks: List[Any] = field(default_factory=list)
    fvgs: List[Any] = field(default_factory=list)
    liquidity_pools: List[Any] = field(default_factory=list)
    pd_arrays: List[Any] = field(default_factory=list)
    key_levels: List[float] = field(default_factory=list)


@dataclass
class MultitimeframeContext:
    """Multi-timeframe analysis context"""
    htf_bias: TradeDirection           # Weekly/Daily bias
    itf_bias: TradeDirection           # 4H/1H intermediate bias
    ltf_bias: TradeDirection           # 15m/5m lower timeframe
    bias_alignment: bool               # All timeframes aligned
    alignment_score: float             # 0-100
    htf_analysis: Optional[TimeframeAnalysis] = None
    itf_analysis: Optional[TimeframeAnalysis] = None
    ltf_analysis: Optional[TimeframeAnalysis] = None
    
    def get_overall_bias(self) -> TradeDirection:
        """Get overall bias based on timeframe alignment"""
        if self.bias_alignment:
            return self.htf_bias
        # HTF has priority
        if self.htf_bias != TradeDirection.NEUTRAL:
            return self.htf_bias
        return self.itf_bias


@dataclass
class ConfluenceFactors:
    """Confluence factors for a trade setup"""
    # Structure
    has_structure_shift: bool = False
    structure_shift_type: str = ""      # CHoCH, BOS, SMS
    
    # PD Arrays
    has_order_block: bool = False
    order_block_type: str = ""
    has_fvg: bool = False
    fvg_type: str = ""
    has_breaker: bool = False
    
    # Liquidity
    liquidity_swept: bool = False
    liquidity_target: bool = False
    
    # Premium/Discount
    in_discount_zone: bool = False      # For longs
    in_premium_zone: bool = False       # For shorts
    
    # Time
    in_kill_zone: bool = False
    is_macro_time: bool = False
    kill_zone_name: str = ""
    
    # Model
    model_validated: bool = False
    model_name: str = ""
    
    # Displacement
    has_displacement: bool = False
    
    # OTE Zone
    in_ote_zone: bool = False           # 62-79% retracement
    
    def count_factors(self) -> int:
        """Count number of confluence factors present"""
        count = 0
        if self.has_structure_shift: count += 1
        if self.has_order_block: count += 1
        if self.has_fvg: count += 1
        if self.has_breaker: count += 1
        if self.liquidity_swept: count += 1
        if self.in_discount_zone or self.in_premium_zone: count += 1
        if self.in_kill_zone: count += 1
        if self.is_macro_time: count += 1
        if self.model_validated: count += 1
        if self.has_displacement: count += 1
        if self.in_ote_zone: count += 1
        return count
    
    def get_score(self) -> float:
        """Calculate confluence score (0-100)"""
        weights = {
            'has_structure_shift': 15,
            'has_order_block': 12,
            'has_fvg': 10,
            'has_breaker': 8,
            'liquidity_swept': 12,
            'premium_discount': 10,
            'in_kill_zone': 8,
            'is_macro_time': 5,
            'model_validated': 10,
            'has_displacement': 8,
            'in_ote_zone': 7,
        }
        
        score = 0
        if self.has_structure_shift: score += weights['has_structure_shift']
        if self.has_order_block: score += weights['has_order_block']
        if self.has_fvg: score += weights['has_fvg']
        if self.has_breaker: score += weights['has_breaker']
        if self.liquidity_swept: score += weights['liquidity_swept']
        if self.in_discount_zone or self.in_premium_zone: score += weights['premium_discount']
        if self.in_kill_zone: score += weights['in_kill_zone']
        if self.is_macro_time: score += weights['is_macro_time']
        if self.model_validated: score += weights['model_validated']
        if self.has_displacement: score += weights['has_displacement']
        if self.in_ote_zone: score += weights['in_ote_zone']
        
        return min(score, 100)


@dataclass
class RiskParameters:
    """Risk parameters for a trade"""
    entry_price: float
    stop_loss: float
    take_profit_1: float                # First target
    take_profit_2: Optional[float] = None  # Runner target
    take_profit_3: Optional[float] = None  # Extended target
    
    risk_in_points: float = 0.0
    reward_1_in_points: float = 0.0
    risk_reward_1: float = 0.0
    risk_reward_2: float = 0.0
    
    position_size: float = 0.0          # Calculated position size
    risk_amount: float = 0.0            # Dollar risk
    risk_percent: float = 0.0           # Account % risk
    
    stop_type: str = ""                 # Type of stop placement
    stop_rationale: str = ""            # Why stop is placed there
    
    def calculate_risk_reward(self):
        """Calculate risk/reward ratios"""
        if self.entry_price and self.stop_loss:
            self.risk_in_points = abs(self.entry_price - self.stop_loss)
            if self.take_profit_1:
                self.reward_1_in_points = abs(self.take_profit_1 - self.entry_price)
                if self.risk_in_points > 0:
                    self.risk_reward_1 = self.reward_1_in_points / self.risk_in_points
            if self.take_profit_2 and self.risk_in_points > 0:
                self.risk_reward_2 = abs(self.take_profit_2 - self.entry_price) / self.risk_in_points


@dataclass
class TradeSetup:
    """Complete trade setup with all context"""
    # Identification
    setup_id: str
    timestamp: datetime
    symbol: str
    
    # Direction and Grade
    direction: TradeDirection
    grade: SetupGrade
    confidence_score: float             # 0-100
    
    # Context
    mtf_context: MultitimeframeContext
    confluence: ConfluenceFactors
    
    # Entry/Exit
    risk_params: RiskParameters
    entry_type: str                     # Limit, Market, Stop Entry
    
    # Model Information
    model_type: str                     # 2022 Model, Silver Bullet, etc.
    model_stage: str                    # Current stage in model
    
    # Timing
    session: str
    kill_zone: Optional[str] = None
    macro_window: Optional[str] = None
    
    # PD Array Reference
    pd_array_type: str = ""             # OB, FVG, Breaker, etc.
    pd_array_level: float = 0.0
    
    # Additional Context
    narrative: str = ""                 # Trade narrative/reasoning
    invalidation_level: float = 0.0     # Price that invalidates setup
    notes: List[str] = field(default_factory=list)
    
    def is_valid(self) -> bool:
        """Check if setup is valid for trading"""
        return (
            self.grade in [SetupGrade.A_PLUS, SetupGrade.A, SetupGrade.B] and
            self.confidence_score >= 60 and
            self.risk_params.risk_reward_1 >= 1.5
        )


@dataclass
class EngineState:
    """Current state of the engine"""
    is_running: bool = False
    last_analysis_time: Optional[datetime] = None
    current_session: str = ""
    current_kill_zone: Optional[str] = None
    is_macro_time: bool = False
    active_setups: List[TradeSetup] = field(default_factory=list)
    pending_signals: List[Any] = field(default_factory=list)


# =============================================================================
# MAIN ENGINE CLASS
# =============================================================================

class ICTCoreEngine:
    """
    ICT Core Engine - Master Integration Layer
    
    Orchestrates all ICT handlers to provide unified market analysis
    and high-quality trade signals.
    
    Usage:
        engine = ICTCoreEngine(config)
        engine.initialize()
        
        # Full analysis
        analysis = engine.analyze(symbol, data)
        
        # Get trade setups
        setups = engine.get_trade_setups()
        
        # Quick signal check
        signal = engine.quick_signal_check(symbol)
    """
    
    # =========================================================================
    # INITIALIZATION
    # =========================================================================
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize the ICT Core Engine
        
        Args:
            config: Configuration dictionary with engine settings
        """
        self.config = config or self._default_config()
        self.state = EngineState()
        
        # Handler instances (will be initialized)
        self.market_data_engine = None
        self.market_structure_handler = None
        self.order_block_handler = None
        self.fvg_handler = None
        self.gap_handler = None
        self.liquidity_handler = None
        self.pd_array_handler = None
        self.timeframe_handler = None
        self.trading_model_handler = None
        self.market_condition_handler = None
        self.entry_stop_handler = None
        
        # Analysis cache
        self._analysis_cache: Dict[str, Any] = {}
        self._cache_duration = 60  # seconds
        
        # Trade setups storage
        self._active_setups: List[TradeSetup] = []
        self._setup_history: List[TradeSetup] = []
        
        logger.info("ICT Core Engine initialized")
    
    def _default_config(self) -> Dict:
        """Default engine configuration"""
        return {
            # Timeframes
            'htf_timeframes': ['W1', 'D1'],
            'itf_timeframes': ['H4', 'H1'],
            'ltf_timeframes': ['M15', 'M5', 'M1'],
            'entry_timeframe': 'M5',
            
            # Analysis Settings
            'min_confluence_score': 60,
            'min_grade': 'B',
            'require_kill_zone': True,
            'require_structure_shift': True,
            
            # Risk Settings
            'min_risk_reward': 2.0,
            'max_risk_percent': 1.0,
            'default_position_size': 1,
            
            # Model Preferences
            'preferred_models': ['2022_model', 'silver_bullet', 'venom'],
            'enable_all_models': True,
            
            # Time Settings
            'timezone': 'America/New_York',
            'trading_hours_only': True,
            
            # Logging
            'log_level': 'INFO',
            'log_analysis': True,
        }
    
    def initialize(self, handlers: Optional[Dict] = None):
        """
        Initialize all handlers
        
        Args:
            handlers: Optional dictionary of pre-initialized handlers
        """
        logger.info("Initializing ICT Core Engine handlers...")
        
        if handlers:
            self._load_handlers(handlers)
        else:
            self._initialize_handlers()
        
        self.state.is_running = True
        logger.info("ICT Core Engine ready")
    
    def _initialize_handlers(self):
        """Initialize all handler instances"""
        # In a real implementation, these would import and instantiate
        # the actual handler classes. For now, we'll use placeholder
        # implementations that can be replaced.
        
        logger.info("Initializing handlers (placeholder mode)...")
        
        # These will be replaced with actual imports:
        # from market_data_engine import MarketDataEngine
        # from market_structure_handler import MarketStructureHandler
        # etc.
        
        self._handlers_initialized = True
    
    def _load_handlers(self, handlers: Dict):
        """Load pre-initialized handlers"""
        self.market_data_engine = handlers.get('market_data_engine')
        self.market_structure_handler = handlers.get('market_structure_handler')
        self.order_block_handler = handlers.get('order_block_handler')
        self.fvg_handler = handlers.get('fvg_handler')
        self.gap_handler = handlers.get('gap_handler')
        self.liquidity_handler = handlers.get('liquidity_handler')
        self.pd_array_handler = handlers.get('pd_array_handler')
        self.timeframe_handler = handlers.get('timeframe_handler')
        self.trading_model_handler = handlers.get('trading_model_handler')
        self.market_condition_handler = handlers.get('market_condition_handler')
        self.entry_stop_handler = handlers.get('entry_stop_handler')
    
    # =========================================================================
    # MAIN ANALYSIS METHODS
    # =========================================================================
    
    def analyze(self, symbol: str, data: Dict[str, Any], 
                mode: AnalysisMode = AnalysisMode.FULL) -> Dict[str, Any]:
        """
        Perform comprehensive market analysis
        
        Args:
            symbol: Market symbol (e.g., 'NQ', 'EURUSD')
            data: Dictionary with OHLC data for multiple timeframes
                  {'M5': df, 'M15': df, 'H1': df, 'H4': df, 'D1': df}
            mode: Analysis mode
            
        Returns:
            Complete analysis dictionary
        """
        logger.info(f"Analyzing {symbol} in {mode.value} mode")
        
        analysis_start = datetime.now()
        
        # Step 1: Get time context
        time_context = self._analyze_time_context()
        
        # Step 2: Analyze market conditions
        market_conditions = self._analyze_market_conditions(data)
        
        # Step 3: Multi-timeframe analysis
        mtf_context = self._analyze_multi_timeframe(data)
        
        # Step 4: Identify PD arrays and structures
        pd_arrays = self._identify_pd_arrays(data, mtf_context.get_overall_bias())
        
        # Step 5: Check for liquidity sweeps
        liquidity_analysis = self._analyze_liquidity(data)
        
        # Step 6: Validate trading models
        model_analysis = self._analyze_trading_models(data, mtf_context, pd_arrays)
        
        # Step 7: Generate trade setups
        trade_setups = self._generate_trade_setups(
            symbol, data, mtf_context, pd_arrays, 
            liquidity_analysis, model_analysis, time_context
        )
        
        # Update state
        self.state.last_analysis_time = datetime.now()
        self.state.active_setups = trade_setups
        
        analysis_time = (datetime.now() - analysis_start).total_seconds()
        
        return {
            'symbol': symbol,
            'timestamp': datetime.now(),
            'analysis_time_seconds': analysis_time,
            'time_context': time_context,
            'market_conditions': market_conditions,
            'mtf_context': mtf_context,
            'pd_arrays': pd_arrays,
            'liquidity': liquidity_analysis,
            'models': model_analysis,
            'trade_setups': trade_setups,
            'best_setup': trade_setups[0] if trade_setups else None,
            'setup_count': len(trade_setups),
        }
    
    def _analyze_time_context(self) -> Dict[str, Any]:
        """Analyze current time context"""
        from datetime import datetime
        
        # Get current time in EST
        now = datetime.now()
        current_time = now.time()
        
        # Determine session
        session = self._get_session(current_time)
        
        # Check kill zone
        kill_zone = self._get_kill_zone(current_time)
        
        # Check macro time
        is_macro, macro_window = self._check_macro_time(current_time)
        
        # Get trading recommendation
        recommendation = self._get_time_recommendation(current_time)
        
        # Update state
        self.state.current_session = session
        self.state.current_kill_zone = kill_zone
        self.state.is_macro_time = is_macro
        
        return {
            'timestamp': now,
            'session': session,
            'kill_zone': kill_zone,
            'is_macro_time': is_macro,
            'macro_window': macro_window,
            'recommendation': recommendation,
            'is_optimal_time': kill_zone is not None and session in ['ny_am', 'london'],
        }
    
    def _get_session(self, t: time) -> str:
        """Determine current trading session"""
        sessions = {
            'asian': (time(0, 0), time(5, 0)),
            'london': (time(3, 0), time(12, 0)),
            'ny_open': (time(7, 0), time(10, 0)),
            'ny_am': (time(9, 30), time(12, 0)),
            'ny_lunch': (time(12, 0), time(13, 30)),
            'ny_pm': (time(13, 30), time(16, 0)),
        }
        
        for session, (start, end) in sessions.items():
            if start <= t <= end:
                return session
        return 'overnight'
    
    def _get_kill_zone(self, t: time) -> Optional[str]:
        """Determine current kill zone"""
        kill_zones = {
            'london_open': (time(1, 0), time(5, 0)),
            'ny_open': (time(7, 0), time(10, 0)),
            'ny_am': (time(9, 30), time(12, 0)),
            'ny_pm': (time(13, 30), time(16, 0)),
        }
        
        for kz, (start, end) in kill_zones.items():
            if start <= t <= end:
                return kz
        return None
    
    def _check_macro_time(self, t: time) -> Tuple[bool, Optional[str]]:
        """Check if current time is a macro time window"""
        macro_times = [
            (time(2, 33), time(3, 0), "Pre-London"),
            (time(4, 3), time(4, 30), "London Session"),
            (time(8, 50), time(9, 10), "Pre-Market"),
            (time(9, 50), time(10, 10), "Post-Open"),
            (time(10, 50), time(11, 10), "Mid-Morning"),
            (time(11, 50), time(12, 10), "Pre-Lunch"),
            (time(13, 10), time(13, 40), "Post-Lunch"),
            (time(14, 50), time(15, 10), "Afternoon"),
            (time(15, 15), time(15, 45), "Power Hour"),
        ]
        
        for start, end, name in macro_times:
            if start <= t <= end:
                return True, name
        return False, None
    
    def _get_time_recommendation(self, t: time) -> str:
        """Get ICT-based time recommendation"""
        if time(9, 30) <= t <= time(10, 0):
            return "CAUTION: Opening range manipulation. First move often FALSE."
        elif time(10, 0) <= t <= time(10, 30):
            return "ALERT: True move typically emerges. Watch for reversal."
        elif time(10, 30) <= t <= time(12, 0):
            return "OPTIMAL: Prime trading window. Execute confirmed setups."
        elif time(12, 0) <= t <= time(13, 30):
            return "AVOID: Lunch hour - low probability."
        elif time(13, 30) <= t <= time(15, 0):
            return "SCALPING: PM session - smaller targets."
        elif time(15, 0) <= t <= time(16, 0):
            return "CAUTION: Last hour - take profits."
        elif time(7, 0) <= t <= time(9, 30):
            return "WATCH: Pre-market - identify bias."
        else:
            return "MONITOR: Outside primary hours."
    
    def _analyze_market_conditions(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze overall market conditions"""
        # This would use the market_condition_handler
        return {
            'state': 'trending',
            'volatility': 'normal',
            'favorability': 'good',
            'trend_strength': 0.75,
        }
    
    def _analyze_multi_timeframe(self, data: Dict[str, Any]) -> MultitimeframeContext:
        """Perform multi-timeframe analysis"""
        
        # Analyze each timeframe tier
        htf_bias = self._determine_bias(data.get('D1'), data.get('H4'))
        itf_bias = self._determine_bias(data.get('H4'), data.get('H1'))
        ltf_bias = self._determine_bias(data.get('M15'), data.get('M5'))
        
        # Check alignment
        bias_alignment = (htf_bias == itf_bias == ltf_bias and 
                         htf_bias != TradeDirection.NEUTRAL)
        
        # Calculate alignment score
        alignment_score = self._calculate_alignment_score(htf_bias, itf_bias, ltf_bias)
        
        return MultitimeframeContext(
            htf_bias=htf_bias,
            itf_bias=itf_bias,
            ltf_bias=ltf_bias,
            bias_alignment=bias_alignment,
            alignment_score=alignment_score,
        )
    
    def _determine_bias(self, higher_tf: Any, lower_tf: Any) -> TradeDirection:
        """Determine bias from timeframe data"""
        # Placeholder - would use market_structure_handler
        return TradeDirection.NEUTRAL
    
    def _calculate_alignment_score(self, htf: TradeDirection, 
                                   itf: TradeDirection, 
                                   ltf: TradeDirection) -> float:
        """Calculate multi-timeframe alignment score"""
        score = 0
        
        # Full alignment = 100
        if htf == itf == ltf and htf != TradeDirection.NEUTRAL:
            return 100.0
        
        # HTF + ITF aligned = 80
        if htf == itf and htf != TradeDirection.NEUTRAL:
            score = 80
            if ltf == htf:
                score = 100
            elif ltf == TradeDirection.NEUTRAL:
                score = 85
        
        # HTF defined, others neutral = 60
        elif htf != TradeDirection.NEUTRAL:
            score = 60
            if itf == htf:
                score = 75
        
        return score
    
    def _identify_pd_arrays(self, data: Dict[str, Any], 
                           bias: TradeDirection) -> Dict[str, List[Any]]:
        """Identify all PD arrays across timeframes"""
        return {
            'order_blocks': [],
            'fvgs': [],
            'breakers': [],
            'mitigation_blocks': [],
            'rejection_blocks': [],
            'volume_imbalances': [],
        }
    
    def _analyze_liquidity(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze liquidity pools and sweeps"""
        return {
            'buy_side_liquidity': [],
            'sell_side_liquidity': [],
            'recent_sweeps': [],
            'targets': [],
        }
    
    def _analyze_trading_models(self, data: Dict[str, Any],
                               mtf_context: MultitimeframeContext,
                               pd_arrays: Dict) -> Dict[str, Any]:
        """Analyze and validate trading models"""
        return {
            'model_2022': {'valid': False, 'stage': None},
            'silver_bullet': {'valid': False, 'session': None},
            'venom': {'valid': False, 'criteria_met': False},
            'turtle_soup': {'valid': False},
            'power_of_three': {'valid': False, 'phase': None},
        }
    
    def _generate_trade_setups(self, symbol: str, data: Dict[str, Any],
                              mtf_context: MultitimeframeContext,
                              pd_arrays: Dict, liquidity: Dict,
                              models: Dict, time_context: Dict) -> List[TradeSetup]:
        """Generate trade setups from analysis"""
        setups = []
        
        # This would synthesize all the analysis to create trade setups
        # For each valid setup, create a TradeSetup object
        
        # Sort by confidence score
        setups.sort(key=lambda x: x.confidence_score, reverse=True)
        
        return setups
    
    # =========================================================================
    # QUICK SIGNAL METHODS
    # =========================================================================
    
    def quick_signal_check(self, symbol: str, data: Any = None) -> Optional[TradeSetup]:
        """
        Quick check for immediate trading signals
        
        Args:
            symbol: Market symbol
            data: Optional current market data
            
        Returns:
            Best TradeSetup if one exists, None otherwise
        """
        # Check cache first
        cache_key = f"{symbol}_signal"
        if cache_key in self._analysis_cache:
            cached, cache_time = self._analysis_cache[cache_key]
            if (datetime.now() - cache_time).seconds < self._cache_duration:
                return cached
        
        # Perform quick analysis
        # This would be a streamlined version of full analysis
        
        return None
    
    def get_active_setups(self) -> List[TradeSetup]:
        """Get all active trade setups"""
        return self._active_setups
    
    def get_best_setup(self) -> Optional[TradeSetup]:
        """Get the highest confidence setup"""
        if not self._active_setups:
            return None
        return max(self._active_setups, key=lambda x: x.confidence_score)
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def calculate_position_size(self, entry: float, stop: float,
                               account_balance: float,
                               risk_percent: float = 1.0) -> float:
        """
        Calculate position size based on risk
        
        Args:
            entry: Entry price
            stop: Stop loss price
            account_balance: Account balance
            risk_percent: Risk as percentage of account
            
        Returns:
            Position size (contracts/lots)
        """
        risk_amount = account_balance * (risk_percent / 100)
        risk_per_unit = abs(entry - stop)
        
        if risk_per_unit == 0:
            return 0
        
        position_size = risk_amount / risk_per_unit
        return position_size
    
    def grade_setup(self, confluence_score: float, 
                   rr_ratio: float,
                   alignment_score: float) -> SetupGrade:
        """
        Grade a trade setup
        
        Args:
            confluence_score: Confluence score (0-100)
            rr_ratio: Risk/reward ratio
            alignment_score: MTF alignment score (0-100)
            
        Returns:
            SetupGrade enum
        """
        # Weighted scoring
        final_score = (
            confluence_score * 0.4 +
            min(rr_ratio * 15, 30) +  # Cap RR contribution at 30
            alignment_score * 0.3
        )
        
        if final_score >= 90:
            return SetupGrade.A_PLUS
        elif final_score >= 80:
            return SetupGrade.A
        elif final_score >= 70:
            return SetupGrade.B
        elif final_score >= 60:
            return SetupGrade.C
        elif final_score >= 50:
            return SetupGrade.D
        else:
            return SetupGrade.F
    
    def validate_setup(self, setup: TradeSetup) -> Tuple[bool, List[str]]:
        """
        Validate a trade setup against rules
        
        Args:
            setup: TradeSetup to validate
            
        Returns:
            Tuple of (is_valid, list of validation messages)
        """
        messages = []
        is_valid = True
        
        # Grade check
        valid_grades = [SetupGrade.A_PLUS, SetupGrade.A, SetupGrade.B]
        if setup.grade not in valid_grades:
            is_valid = False
            messages.append(f"Grade {setup.grade.value} below minimum")
        
        # Risk/Reward check
        if setup.risk_params.risk_reward_1 < self.config['min_risk_reward']:
            is_valid = False
            messages.append(f"R:R {setup.risk_params.risk_reward_1:.2f} below minimum {self.config['min_risk_reward']}")
        
        # Kill zone check (if required)
        if self.config['require_kill_zone'] and not setup.kill_zone:
            is_valid = False
            messages.append("Not in kill zone")
        
        # Structure shift check (if required)
        if self.config['require_structure_shift'] and not setup.confluence.has_structure_shift:
            is_valid = False
            messages.append("No structure shift present")
        
        # Confluence score check
        if setup.confluence.get_score() < self.config['min_confluence_score']:
            is_valid = False
            messages.append(f"Confluence score {setup.confluence.get_score():.0f} below minimum")
        
        return is_valid, messages
    
    def get_state(self) -> EngineState:
        """Get current engine state"""
        return self.state
    
    def clear_cache(self):
        """Clear analysis cache"""
        self._analysis_cache.clear()
        logger.info("Analysis cache cleared")
    
    def shutdown(self):
        """Shutdown the engine"""
        self.state.is_running = False
        self.clear_cache()
        logger.info("ICT Core Engine shutdown complete")
    
    # =========================================================================
    # REPORTING
    # =========================================================================
    
    def generate_analysis_report(self, analysis: Dict[str, Any]) -> str:
        """Generate a text report from analysis"""
        report = []
        report.append("=" * 60)
        report.append("ICT MARKET ANALYSIS REPORT")
        report.append("=" * 60)
        report.append(f"Symbol: {analysis.get('symbol', 'N/A')}")
        report.append(f"Time: {analysis.get('timestamp', 'N/A')}")
        report.append("")
        
        # Time context
        time_ctx = analysis.get('time_context', {})
        report.append("TIME CONTEXT:")
        report.append(f"  Session: {time_ctx.get('session', 'N/A')}")
        report.append(f"  Kill Zone: {time_ctx.get('kill_zone', 'None')}")
        report.append(f"  Macro Time: {time_ctx.get('is_macro_time', False)}")
        report.append(f"  Recommendation: {time_ctx.get('recommendation', 'N/A')}")
        report.append("")
        
        # MTF Context
        mtf = analysis.get('mtf_context')
        if mtf:
            report.append("MULTI-TIMEFRAME ANALYSIS:")
            report.append(f"  HTF Bias: {mtf.htf_bias.value}")
            report.append(f"  ITF Bias: {mtf.itf_bias.value}")
            report.append(f"  LTF Bias: {mtf.ltf_bias.value}")
            report.append(f"  Alignment: {'YES' if mtf.bias_alignment else 'NO'} ({mtf.alignment_score:.0f}%)")
            report.append("")
        
        # Trade setups
        setups = analysis.get('trade_setups', [])
        report.append(f"TRADE SETUPS FOUND: {len(setups)}")
        
        for i, setup in enumerate(setups[:3], 1):  # Top 3
            report.append(f"\n  Setup #{i}:")
            report.append(f"    Direction: {setup.direction.value.upper()}")
            report.append(f"    Grade: {setup.grade.value}")
            report.append(f"    Confidence: {setup.confidence_score:.0f}%")
            report.append(f"    Model: {setup.model_type}")
            report.append(f"    Entry: {setup.risk_params.entry_price}")
            report.append(f"    Stop: {setup.risk_params.stop_loss}")
            report.append(f"    Target: {setup.risk_params.take_profit_1}")
            report.append(f"    R:R: {setup.risk_params.risk_reward_1:.2f}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_engine(config: Optional[Dict] = None) -> ICTCoreEngine:
    """
    Factory function to create and initialize ICT Core Engine
    
    Args:
        config: Optional configuration dictionary
        
    Returns:
        Initialized ICTCoreEngine instance
    """
    engine = ICTCoreEngine(config)
    engine.initialize()
    return engine


# =============================================================================
# EXAMPLE USAGE
# =============================================================================

if __name__ == "__main__":
    print("ICT Core Engine - Master Integration Layer")
    print("=" * 60)
    
    # Create engine
    engine = create_engine()
    
    # Get current state
    state = engine.get_state()
    print(f"\nEngine Running: {state.is_running}")
    
    # Analyze time context
    time_ctx = engine._analyze_time_context()
    print(f"\nCurrent Session: {time_ctx['session']}")
    print(f"Kill Zone: {time_ctx['kill_zone']}")
    print(f"Macro Time: {time_ctx['is_macro_time']}")
    print(f"Recommendation: {time_ctx['recommendation']}")
    
    # Test confluence scoring
    confluence = ConfluenceFactors(
        has_structure_shift=True,
        structure_shift_type="CHoCH",
        has_order_block=True,
        order_block_type="bullish",
        has_fvg=True,
        fvg_type="bullish",
        liquidity_swept=True,
        in_discount_zone=True,
        in_kill_zone=True,
        model_validated=True,
        model_name="2022_model",
        has_displacement=True,
    )
    
    print(f"\nConfluence Factors: {confluence.count_factors()}")
    print(f"Confluence Score: {confluence.get_score():.0f}")
    
    # Test grading
    grade = engine.grade_setup(
        confluence_score=confluence.get_score(),
        rr_ratio=3.0,
        alignment_score=85
    )
    print(f"Setup Grade: {grade.value}")
    
    # Test position sizing
    pos_size = engine.calculate_position_size(
        entry=21500,
        stop=21450,
        account_balance=100000,
        risk_percent=1.0
    )
    print(f"\nPosition Size (1% risk): {pos_size:.2f} contracts")
    
    print("\n" + "=" * 60)
    print("ICT Core Engine ready for integration!")
