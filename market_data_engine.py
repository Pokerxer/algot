"""
ICT Market Data Engine - Comprehensive Implementation
=====================================================

Enhanced market data fetching engine with ICT-specific features including:
- Session time tracking (Midnight, Asian, London, NY)
- Key opening prices (True Day, 9:30, 6pm Forex)
- DXY correlation tracking
- Intermarket analysis
- Kill zone data aggregation
- Historical pattern tracking

ICT TIME CONCEPTS IMPLEMENTED:
==============================

1. TRUE DAY START (Midnight - 00:00 EST):
   "Midnight Eastern time, 9:30 in the morning, 6 o'clock reopen time, all assets."
   "True Day start which is midnight Eastern time."

2. KEY OPENING PRICES:
   - Midnight Open (00:00 EST) - True institutional day
   - Regular Open (9:30 EST) - Equities opening
   - Forex Reopen (5:00 PM EST) - New session start
   - Previous Close (4:00 PM EST) - For gap calculation

3. SESSIONS:
   - Asian: 00:00-05:00 EST (20:00-00:00 local in Asia)
   - London: 03:00-12:00 EST (08:00-17:00 UK)
   - New York: 07:00-16:00 EST
   - PM Session: 13:30-16:00 EST

4. INTERMARKET CORRELATIONS:
   "Higher dollar is what we were looking for... Dollar was going higher. 
   It moved above that range... sending euro and cable lower."
   - DXY vs EUR/USD (inverse)
   - DXY vs GBP/USD (inverse)  
   - DXY vs Gold (generally inverse)
   - ES vs NQ (correlation)
"""

import requests
import json
from datetime import datetime, timedelta, time, timezone
from typing import Dict, List, Optional, Union, Tuple
import time as time_module
from dataclasses import dataclass, asdict, field
from enum import Enum
import pytz


# =============================================================================
# ENUMERATIONS
# =============================================================================

class MarketType(Enum):
    """Market type enumeration"""
    FOREX = "forex"
    INDEX = "index"
    COMMODITY = "commodity"
    CRYPTO = "crypto"


class DataSource(Enum):
    """Available data sources"""
    YAHOO_FINANCE = "yahoo"
    ALPHA_VANTAGE = "alphavantage"
    TWELVE_DATA = "twelvedata"
    FINNHUB = "finnhub"


class Session(Enum):
    """ICT Trading Sessions"""
    ASIAN = "asian"           # 00:00-05:00 EST (builds liquidity)
    LONDON = "london"         # 03:00-12:00 EST (most volatile Forex)
    NY_OPEN = "ny_open"       # 07:00-10:00 EST (key algorithms fire)
    NY_AM = "ny_am"           # 09:30-12:00 EST (equities focus)
    NY_LUNCH = "ny_lunch"     # 12:00-13:30 EST (avoid - low probability)
    NY_PM = "ny_pm"           # 13:30-16:00 EST (scalping environment)
    OVERNIGHT = "overnight"   # 16:00-00:00 EST


class KillZone(Enum):
    """ICT Kill Zones - High probability trading windows"""
    ASIAN = "asian"           # 20:00-00:00 EST (limited for Forex)
    LONDON_OPEN = "london_open"   # 01:00-05:00 EST
    NY_OPEN = "ny_open"       # 07:00-10:00 EST (all algorithms fire)
    NY_AM = "ny_am"           # 09:30-12:00 EST
    NY_PM = "ny_pm"           # 13:30-16:00 EST
    LAST_HOUR = "last_hour"   # 15:00-16:00 EST


class TimeframeInterval(Enum):
    """Data timeframe intervals"""
    M1 = "1m"
    M5 = "5m"
    M15 = "15m"
    M30 = "30m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"
    W1 = "1wk"
    MN = "1mo"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class MarketData:
    """Standard market data structure with ICT-specific fields"""
    symbol: str
    market_type: MarketType
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: Optional[int] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    spread: Optional[float] = None
    source: Optional[str] = None
    
    # ICT-specific fields
    session: Optional[Session] = None
    kill_zone: Optional[KillZone] = None
    is_macro_time: bool = False
    
    def to_dict(self) -> dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['timestamp'] = self.timestamp.isoformat()
        data['market_type'] = self.market_type.value
        if self.session:
            data['session'] = self.session.value
        if self.kill_zone:
            data['kill_zone'] = self.kill_zone.value
        return data


@dataclass
class ICTOpeningPrices:
    """
    ICT Key Opening Prices
    
    ICT Quote: "Midnight Eastern time, 9:30 in the morning, 6 o'clock 
    reopen time, all assets. Yeah. But Forex, I didn't say anything about 
    that. I said this is the time you have on your chart. Five o'clock PM 
    and 4:14 PM Eastern time."
    """
    date: datetime
    symbol: str
    
    # Critical opening prices
    midnight_open: Optional[float] = None       # 00:00 EST - True Day start
    asian_open: Optional[float] = None          # 00:00 EST (same as midnight)
    london_open: Optional[float] = None         # 03:00 EST
    ny_open_7am: Optional[float] = None         # 07:00 EST - Algorithms fire
    equities_open: Optional[float] = None       # 09:30 EST - Regular trading
    pm_session_open: Optional[float] = None     # 13:30 EST
    
    # Close prices for gap calculation
    previous_4pm_close: Optional[float] = None  # 16:00 EST previous day
    previous_5pm_close: Optional[float] = None  # 17:00 EST (Forex)
    
    # Forex specific
    forex_reopen: Optional[float] = None        # 17:00 EST Sunday
    
    # High/Low for each session
    asian_high: Optional[float] = None
    asian_low: Optional[float] = None
    london_high: Optional[float] = None
    london_low: Optional[float] = None
    ny_am_high: Optional[float] = None
    ny_am_low: Optional[float] = None
    
    # Gaps
    midnight_gap: Optional[float] = None        # From previous close to midnight
    opening_gap: Optional[float] = None         # From 4pm close to 9:30 open (NDOG)
    
    def calculate_gaps(self):
        """Calculate ICT gaps from opening prices"""
        if self.midnight_open and self.previous_5pm_close:
            self.midnight_gap = self.midnight_open - self.previous_5pm_close
        if self.equities_open and self.previous_4pm_close:
            self.opening_gap = self.equities_open - self.previous_4pm_close


@dataclass
class SessionData:
    """Data aggregated for a specific ICT session"""
    session: Session
    date: datetime
    symbol: str
    
    # OHLC for session
    session_open: float
    session_high: float
    session_low: float
    session_close: float
    
    # Range
    range_size: float = field(init=False)
    range_percent: float = 0.0
    
    # ICT Analysis
    manipulation_direction: Optional[str] = None  # 'upside' or 'downside'
    true_direction: Optional[str] = None          # After manipulation
    liquidity_swept: bool = False
    
    # First 30 minutes (for NY AM session)
    first_30_high: Optional[float] = None
    first_30_low: Optional[float] = None
    
    def __post_init__(self):
        self.range_size = self.session_high - self.session_low


@dataclass
class IntermarketCorrelation:
    """
    Intermarket correlation tracking
    
    ICT Quote: "Higher dollar is what we were looking for. I was looking for. 
    Let me say it that way. So this is a December contract mini NASDAQ futures... 
    Dollar was going higher. It moved above that range I told you I would I 
    wanted to see and it would treat that inversion fair gap as a discount array."
    """
    timestamp: datetime
    
    # Dollar Index correlation
    dxy_price: Optional[float] = None
    dxy_direction: str = ""  # 'bullish', 'bearish', 'neutral'
    
    # Forex correlations (inverse to DXY)
    eurusd_correlation: float = 0.0  # Should be negative when DXY bullish
    gbpusd_correlation: float = 0.0
    usdjpy_correlation: float = 0.0  # Positive with DXY
    
    # Index correlations
    es_nq_correlation: float = 0.0  # Usually positive
    
    # Risk on/off
    is_risk_on: bool = False
    is_risk_off: bool = False
    
    def determine_risk_sentiment(self):
        """
        Determine risk on/off based on DXY direction
        
        ICT: "Higher dollar is... risk off for uh all assets"
        """
        if self.dxy_direction == 'bullish':
            self.is_risk_off = True
            self.is_risk_on = False
        elif self.dxy_direction == 'bearish':
            self.is_risk_on = True
            self.is_risk_off = False


@dataclass
class MacroTimeWindow:
    """ICT Macro Time windows"""
    start_time: time
    end_time: time
    name: str
    is_active: bool = False
    
    def check_active(self, current_time: time) -> bool:
        """Check if macro time is currently active"""
        self.is_active = self.start_time <= current_time <= self.end_time
        return self.is_active


@dataclass
class OpeningRangeData:
    """
    Opening Range Data (9:30-10:00 AM)
    
    ICT Quote: "9:30-10:00 = manipulation period. First move often FALSE. 
    True direction emerges 10:00-10:30."
    """
    date: datetime
    symbol: str
    
    # Opening range (first 30 minutes)
    range_high: float
    range_low: float
    range_open: float  # 9:30 open
    range_close: float  # 10:00 close
    
    range_size: float = field(init=False)
    range_midpoint: float = field(init=False)
    
    # Manipulation analysis
    first_direction: str = ""  # First 30 min direction (often FALSE)
    expected_true_direction: str = ""  # Opposite of first direction
    
    # After 10:00
    post_range_high: Optional[float] = None
    post_range_low: Optional[float] = None
    range_broken: Optional[str] = None  # 'upside', 'downside', or None
    
    def __post_init__(self):
        self.range_size = self.range_high - self.range_low
        self.range_midpoint = (self.range_high + self.range_low) / 2
        
        # Determine first direction
        if self.range_close > self.range_open:
            self.first_direction = 'bullish'
            self.expected_true_direction = 'bearish'  # Often opposite
        else:
            self.first_direction = 'bearish'
            self.expected_true_direction = 'bullish'


# =============================================================================
# MAIN ENGINE CLASS
# =============================================================================

class MarketDataEngine:
    """
    Enhanced Market Data Engine with ICT-specific features.
    
    Key ICT Enhancements:
    - Session time tracking
    - Key opening price tracking
    - Kill zone identification
    - Macro time windows
    - Intermarket correlations
    - Opening range analysis
    - Historical pattern data
    """
    
    # =========================================================================
    # SYMBOL MAPPINGS
    # =========================================================================
    
    FOREX_PAIRS = {
        'EURUSD': {'yahoo': 'EURUSD=X', 'name': 'Euro/US Dollar', 'pip_value': 0.0001},
        'GBPUSD': {'yahoo': 'GBPUSD=X', 'name': 'British Pound/US Dollar', 'pip_value': 0.0001},
        'USDJPY': {'yahoo': 'USDJPY=X', 'name': 'US Dollar/Japanese Yen', 'pip_value': 0.01},
        'USDCHF': {'yahoo': 'USDCHF=X', 'name': 'US Dollar/Swiss Franc', 'pip_value': 0.0001},
        'AUDUSD': {'yahoo': 'AUDUSD=X', 'name': 'Australian Dollar/US Dollar', 'pip_value': 0.0001},
        'USDCAD': {'yahoo': 'USDCAD=X', 'name': 'US Dollar/Canadian Dollar', 'pip_value': 0.0001},
        'NZDUSD': {'yahoo': 'NZDUSD=X', 'name': 'New Zealand Dollar/US Dollar', 'pip_value': 0.0001},
        'EURGBP': {'yahoo': 'EURGBP=X', 'name': 'Euro/British Pound', 'pip_value': 0.0001},
        'EURJPY': {'yahoo': 'EURJPY=X', 'name': 'Euro/Japanese Yen', 'pip_value': 0.01},
        'GBPJPY': {'yahoo': 'GBPJPY=X', 'name': 'British Pound/Japanese Yen', 'pip_value': 0.01},
        'DXY': {'yahoo': 'DX-Y.NYB', 'name': 'US Dollar Index', 'pip_value': 0.01},
    }
    
    INDEXES = {
        'ES': {'yahoo': 'ES=F', 'name': 'E-mini S&P 500 Futures', 'point_value': 0.25},
        'NQ': {'yahoo': 'NQ=F', 'name': 'E-mini NASDAQ 100 Futures', 'point_value': 0.25},
        'YM': {'yahoo': 'YM=F', 'name': 'E-mini Dow Futures', 'point_value': 1.0},
        'RTY': {'yahoo': 'RTY=F', 'name': 'E-mini Russell 2000 Futures', 'point_value': 0.1},
        'SPX': {'yahoo': '^GSPC', 'name': 'S&P 500 Index', 'point_value': 1.0},
        'NDX': {'yahoo': '^NDX', 'name': 'NASDAQ 100 Index', 'point_value': 1.0},
        'DJI': {'yahoo': '^DJI', 'name': 'Dow Jones Industrial Average', 'point_value': 1.0},
        'VIX': {'yahoo': '^VIX', 'name': 'CBOE Volatility Index', 'point_value': 0.01},
    }
    
    COMMODITIES = {
        'GOLD': {'yahoo': 'GC=F', 'name': 'Gold Futures', 'point_value': 0.1},
        'SILVER': {'yahoo': 'SI=F', 'name': 'Silver Futures', 'point_value': 0.005},
        'CRUDE': {'yahoo': 'CL=F', 'name': 'Crude Oil Futures', 'point_value': 0.01},
    }
    
    # =========================================================================
    # ICT TIME CONFIGURATIONS
    # =========================================================================
    
    # Session times (EST)
    SESSIONS = {
        Session.ASIAN: (time(0, 0), time(5, 0)),
        Session.LONDON: (time(3, 0), time(12, 0)),
        Session.NY_OPEN: (time(7, 0), time(10, 0)),
        Session.NY_AM: (time(9, 30), time(12, 0)),
        Session.NY_LUNCH: (time(12, 0), time(13, 30)),
        Session.NY_PM: (time(13, 30), time(16, 0)),
        Session.OVERNIGHT: (time(16, 0), time(23, 59)),
    }
    
    # Kill zones (high probability trading windows)
    KILL_ZONES = {
        KillZone.ASIAN: (time(20, 0), time(0, 0)),  # Limited for most
        KillZone.LONDON_OPEN: (time(1, 0), time(5, 0)),
        KillZone.NY_OPEN: (time(7, 0), time(10, 0)),  # "All algorithms fire up"
        KillZone.NY_AM: (time(9, 30), time(12, 0)),
        KillZone.NY_PM: (time(13, 30), time(16, 0)),
        KillZone.LAST_HOUR: (time(15, 0), time(16, 0)),
    }
    
    # ICT Macro times
    MACRO_TIMES = [
        MacroTimeWindow(time(2, 33), time(3, 0), "Pre-London"),
        MacroTimeWindow(time(4, 3), time(4, 30), "London Session"),
        MacroTimeWindow(time(8, 50), time(9, 10), "Pre-Market"),
        MacroTimeWindow(time(9, 50), time(10, 10), "Post-Open"),  # "9:50-10:10 often completes CE fill"
        MacroTimeWindow(time(10, 50), time(11, 10), "Mid-Morning"),
        MacroTimeWindow(time(11, 50), time(12, 10), "Pre-Lunch"),
        MacroTimeWindow(time(13, 10), time(13, 40), "Post-Lunch"),
        MacroTimeWindow(time(14, 50), time(15, 10), "Afternoon"),  # "2:50-3:10 macro time"
        MacroTimeWindow(time(15, 15), time(15, 45), "Power Hour"),
    ]
    
    # Key opening times
    KEY_TIMES = {
        'midnight': time(0, 0),       # True Day start
        'london_open': time(3, 0),    # London session
        'ny_open_7am': time(7, 0),    # "7:00 AM is when all algorithms fire up"
        'equities_open': time(9, 30), # Regular trading hours
        'pm_session': time(13, 30),   # PM session start
        'equities_close': time(16, 0),# Close
        'forex_reopen': time(17, 0),  # Forex Sunday reopen
    }
    
    def __init__(self, api_keys: Optional[Dict[str, str]] = None):
        """
        Initialize the ICT Market Data Engine
        
        Args:
            api_keys: Dictionary of API keys for different services
        """
        self.api_keys = api_keys or {}
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'ICTMarketDataEngine/2.0'
        })
        
        # Data caches
        self.cache = {}
        self.cache_duration = 60  # seconds
        
        # ICT-specific data storage
        self.opening_prices: Dict[str, ICTOpeningPrices] = {}
        self.session_data: Dict[str, List[SessionData]] = {}
        self.intermarket: Optional[IntermarketCorrelation] = None
        self.opening_ranges: Dict[str, OpeningRangeData] = {}
        
        # Timezone
        self.tz_est = pytz.timezone('America/New_York')
        
    # =========================================================================
    # TIME ANALYSIS
    # =========================================================================
    
    def get_current_session(self, current_time: Optional[datetime] = None) -> Session:
        """
        Determine current ICT session based on time.
        
        Args:
            current_time: Time to check (defaults to now)
            
        Returns:
            Current Session enum
        """
        if current_time is None:
            current_time = datetime.now(self.tz_est)
        elif current_time.tzinfo is None:
            current_time = self.tz_est.localize(current_time)
            
        t = current_time.time()
        
        for session, (start, end) in self.SESSIONS.items():
            if start <= t <= end:
                return session
                
        return Session.OVERNIGHT
    
    def get_current_kill_zone(self, current_time: Optional[datetime] = None) -> Optional[KillZone]:
        """
        Determine if currently in a kill zone.
        
        ICT Quote: "This is the ICT New York open kill zone 7 o'clock to 
        10 o'clock in the morning for all markets."
        
        Args:
            current_time: Time to check
            
        Returns:
            KillZone if in one, None otherwise
        """
        if current_time is None:
            current_time = datetime.now(self.tz_est)
        elif current_time.tzinfo is None:
            current_time = self.tz_est.localize(current_time)
            
        t = current_time.time()
        
        for kill_zone, (start, end) in self.KILL_ZONES.items():
            if kill_zone == KillZone.ASIAN:
                # Asian crosses midnight
                if t >= start or t <= end:
                    return kill_zone
            else:
                if start <= t <= end:
                    return kill_zone
                    
        return None
    
    def is_macro_time(self, current_time: Optional[datetime] = None) -> Tuple[bool, Optional[MacroTimeWindow]]:
        """
        Check if current time is within an ICT macro time window.
        
        ICT Quote: "This candlestick comes in at 3:08. So, 8 minutes after 
        3:00 Eastern time... we see it trade down into the macro of 2:50 p.m. 
        3:10 Eastern time."
        
        Args:
            current_time: Time to check
            
        Returns:
            Tuple of (is_macro_time, macro_window if active)
        """
        if current_time is None:
            current_time = datetime.now(self.tz_est)
        elif current_time.tzinfo is None:
            current_time = self.tz_est.localize(current_time)
            
        t = current_time.time()
        
        for macro in self.MACRO_TIMES:
            if macro.check_active(t):
                return True, macro
                
        return False, None
    
    def get_time_context(self, current_time: Optional[datetime] = None) -> Dict:
        """
        Get comprehensive ICT time context.
        
        Returns:
            Dictionary with session, kill zone, macro time info
        """
        if current_time is None:
            current_time = datetime.now(self.tz_est)
            
        is_macro, macro_window = self.is_macro_time(current_time)
        
        return {
            'timestamp': current_time,
            'session': self.get_current_session(current_time),
            'kill_zone': self.get_current_kill_zone(current_time),
            'is_macro_time': is_macro,
            'macro_window': macro_window.name if macro_window else None,
            'is_opening_range': self._is_in_opening_range(current_time),
            'is_lunch_hour': self._is_lunch_hour(current_time),
            'trading_recommendation': self._get_time_based_recommendation(current_time)
        }
    
    def _is_in_opening_range(self, current_time: datetime) -> bool:
        """Check if in 9:30-10:00 opening range manipulation period"""
        t = current_time.time()
        return time(9, 30) <= t <= time(10, 0)
    
    def _is_lunch_hour(self, current_time: datetime) -> bool:
        """Check if in lunch hour (avoid trading)"""
        t = current_time.time()
        return time(12, 0) <= t <= time(13, 30)
    
    def _get_time_based_recommendation(self, current_time: datetime) -> str:
        """Get ICT-based trading recommendation for current time"""
        t = current_time.time()
        
        if time(9, 30) <= t <= time(10, 0):
            return "CAUTION: Opening range manipulation period. First move often FALSE."
        elif time(10, 0) <= t <= time(10, 30):
            return "ALERT: True move typically emerges 10:00-10:30. Watch for reversal."
        elif time(10, 30) <= t <= time(12, 0):
            return "OPTIMAL: Prime trading window. Execute confirmed setups."
        elif time(12, 0) <= t <= time(13, 30):
            return "AVOID: Lunch hour - low probability, avoid new positions."
        elif time(13, 30) <= t <= time(15, 0):
            return "SCALPING: PM session - cookie cutter shark approach."
        elif time(15, 0) <= t <= time(16, 0):
            return "CAUTION: Last hour - take profits, don't expect continuation."
        elif time(7, 0) <= t <= time(9, 30):
            return "WATCH: Pre-market - identify bias and liquidity targets."
        else:
            return "MONITOR: Outside primary trading hours."
    
    # =========================================================================
    # OPENING PRICE TRACKING
    # =========================================================================
    
    def track_opening_prices(self, symbol: str, 
                            data: List[MarketData]) -> ICTOpeningPrices:
        """
        Extract and track ICT key opening prices from data.
        
        ICT Quote: "You're going to see a lot of things by having 20 days 
        worth of regular trading hours opening prices on your charts."
        
        Args:
            symbol: Market symbol
            data: List of MarketData for the day
            
        Returns:
            ICTOpeningPrices object
        """
        today = datetime.now(self.tz_est).date()
        
        opening_prices = ICTOpeningPrices(
            date=datetime.combine(today, time(0, 0)),
            symbol=symbol
        )
        
        # Extract prices at key times
        for candle in data:
            candle_time = candle.timestamp.time()
            
            # Midnight open (True Day start)
            if time(0, 0) <= candle_time < time(0, 5):
                opening_prices.midnight_open = candle.open
                opening_prices.asian_open = candle.open
                
            # London open
            elif time(3, 0) <= candle_time < time(3, 5):
                opening_prices.london_open = candle.open
                
            # NY Open (7am - algorithms fire)
            elif time(7, 0) <= candle_time < time(7, 5):
                opening_prices.ny_open_7am = candle.open
                
            # Equities open (9:30)
            elif time(9, 30) <= candle_time < time(9, 35):
                opening_prices.equities_open = candle.open
                
            # PM session
            elif time(13, 30) <= candle_time < time(13, 35):
                opening_prices.pm_session_open = candle.open
        
        # Track session highs/lows
        asian_candles = [c for c in data if time(0, 0) <= c.timestamp.time() < time(5, 0)]
        if asian_candles:
            opening_prices.asian_high = max(c.high for c in asian_candles)
            opening_prices.asian_low = min(c.low for c in asian_candles)
            
        london_candles = [c for c in data if time(3, 0) <= c.timestamp.time() < time(12, 0)]
        if london_candles:
            opening_prices.london_high = max(c.high for c in london_candles)
            opening_prices.london_low = min(c.low for c in london_candles)
            
        ny_am_candles = [c for c in data if time(9, 30) <= c.timestamp.time() < time(12, 0)]
        if ny_am_candles:
            opening_prices.ny_am_high = max(c.high for c in ny_am_candles)
            opening_prices.ny_am_low = min(c.low for c in ny_am_candles)
        
        # Calculate gaps
        opening_prices.calculate_gaps()
        
        # Store
        self.opening_prices[symbol] = opening_prices
        
        return opening_prices
    
    def track_opening_range(self, symbol: str, 
                           data: List[MarketData]) -> Optional[OpeningRangeData]:
        """
        Track opening range (9:30-10:00 AM) for manipulation analysis.
        
        ICT Quote: "9:30-10:00 = manipulation period. First move often FALSE."
        
        Args:
            symbol: Market symbol
            data: List of MarketData
            
        Returns:
            OpeningRangeData or None
        """
        # Filter for opening range candles (9:30-10:00)
        opening_range_candles = [
            c for c in data 
            if time(9, 30) <= c.timestamp.time() < time(10, 0)
        ]
        
        if not opening_range_candles:
            return None
            
        # Calculate opening range OHLC
        range_open = opening_range_candles[0].open
        range_close = opening_range_candles[-1].close
        range_high = max(c.high for c in opening_range_candles)
        range_low = min(c.low for c in opening_range_candles)
        
        opening_range = OpeningRangeData(
            date=datetime.now(self.tz_est),
            symbol=symbol,
            range_high=range_high,
            range_low=range_low,
            range_open=range_open,
            range_close=range_close
        )
        
        # Check for post-range break (after 10:00)
        post_range_candles = [
            c for c in data 
            if c.timestamp.time() >= time(10, 0)
        ]
        
        if post_range_candles:
            opening_range.post_range_high = max(c.high for c in post_range_candles)
            opening_range.post_range_low = min(c.low for c in post_range_candles)
            
            if opening_range.post_range_high > range_high:
                opening_range.range_broken = 'upside'
            elif opening_range.post_range_low < range_low:
                opening_range.range_broken = 'downside'
        
        self.opening_ranges[symbol] = opening_range
        
        return opening_range
    
    # =========================================================================
    # INTERMARKET ANALYSIS
    # =========================================================================
    
    def analyze_intermarket(self) -> IntermarketCorrelation:
        """
        Analyze intermarket correlations for risk on/off determination.
        
        ICT Quote: "Higher dollar is what we were looking for... Dollar was 
        going higher. It moved above that range... sending euro and cable lower."
        
        Returns:
            IntermarketCorrelation object
        """
        correlation = IntermarketCorrelation(timestamp=datetime.now(self.tz_est))
        
        # Fetch DXY data
        dxy_data = self.get_forex_data('DXY')
        if dxy_data:
            correlation.dxy_price = dxy_data.close
            # Simple direction determination
            # In a real implementation, compare to previous day
            correlation.dxy_direction = 'neutral'  # Would need historical comparison
        
        # Fetch correlated pairs
        eur_data = self.get_forex_data('EURUSD')
        gbp_data = self.get_forex_data('GBPUSD')
        
        # In a full implementation, calculate actual correlations
        # from historical data
        
        correlation.determine_risk_sentiment()
        
        self.intermarket = correlation
        
        return correlation
    
    def get_dxy_bias(self) -> Dict:
        """
        Get DXY bias and implications for other markets.
        
        ICT Quote: "I was bullish on dollar. You can check that, go back 
        and look at the recordings. It's not ambiguous. It's rather one-sided."
        
        Returns:
            Dictionary with DXY bias and market implications
        """
        dxy_data = self.get_forex_data('DXY')
        
        if not dxy_data:
            return {'error': 'Unable to fetch DXY data'}
        
        # In a full implementation, compare to previous levels
        # For now, provide framework
        return {
            'dxy_price': dxy_data.close,
            'dxy_bias': 'Determine from higher timeframe analysis',
            'implications': {
                'eurusd': 'Inverse to DXY - DXY up = EUR down',
                'gbpusd': 'Inverse to DXY - DXY up = GBP down',
                'usdjpy': 'Correlated to DXY - DXY up = JPY up',
                'gold': 'Generally inverse to DXY',
                'indices': 'Risk off when DXY strong'
            },
            'ict_note': "Higher dollar = risk off. Lower dollar = risk on."
        }
    
    # =========================================================================
    # BASIC DATA FETCHING (from original)
    # =========================================================================
    
    def get_forex_data(self, pair: str, 
                      source: DataSource = DataSource.YAHOO_FINANCE,
                      use_cache: bool = True) -> Optional[MarketData]:
        """
        Fetch forex pair data with ICT session context.
        
        Args:
            pair: Forex pair symbol (e.g., 'EURUSD')
            source: Data source to use
            use_cache: Whether to use cached data
            
        Returns:
            MarketData object or None
        """
        cache_key = f"forex_{pair}_{source.value}"
        
        if use_cache and cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if time_module.time() - cached_time < self.cache_duration:
                return cached_data
        
        if pair not in self.FOREX_PAIRS:
            raise ValueError(f"Unknown forex pair: {pair}")
        
        data = None
        if source == DataSource.YAHOO_FINANCE:
            data = self._fetch_yahoo_finance(pair, MarketType.FOREX)
        
        if data:
            # Add ICT context
            data.session = self.get_current_session()
            data.kill_zone = self.get_current_kill_zone()
            data.is_macro_time, _ = self.is_macro_time()
            
            self.cache[cache_key] = (data, time_module.time())
        
        return data
    
    def get_index_data(self, index: str,
                      source: DataSource = DataSource.YAHOO_FINANCE,
                      use_cache: bool = True) -> Optional[MarketData]:
        """
        Fetch index data with ICT session context.
        
        Args:
            index: Index symbol (e.g., 'NQ', 'ES')
            source: Data source to use
            use_cache: Whether to use cached data
            
        Returns:
            MarketData object or None
        """
        cache_key = f"index_{index}_{source.value}"
        
        if use_cache and cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if time_module.time() - cached_time < self.cache_duration:
                return cached_data
        
        if index not in self.INDEXES:
            raise ValueError(f"Unknown index: {index}")
        
        data = None
        if source == DataSource.YAHOO_FINANCE:
            data = self._fetch_yahoo_finance(index, MarketType.INDEX)
        
        if data:
            data.session = self.get_current_session()
            data.kill_zone = self.get_current_kill_zone()
            data.is_macro_time, _ = self.is_macro_time()
            
            self.cache[cache_key] = (data, time_module.time())
        
        return data
    
    def _fetch_yahoo_finance(self, symbol: str, 
                            market_type: MarketType) -> Optional[MarketData]:
        """Fetch data from Yahoo Finance"""
        try:
            if market_type == MarketType.FOREX:
                yahoo_symbol = self.FOREX_PAIRS[symbol]['yahoo']
            elif market_type == MarketType.INDEX:
                yahoo_symbol = self.INDEXES[symbol]['yahoo']
            elif market_type == MarketType.COMMODITY:
                yahoo_symbol = self.COMMODITIES[symbol]['yahoo']
            else:
                return None
            
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
            params = {
                'interval': '1d',
                'range': '5d'
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'chart' not in data or 'result' not in data['chart']:
                return None
            
            result = data['chart']['result'][0]
            meta = result['meta']
            indicators = result['indicators']['quote'][0]
            
            # Get latest data
            idx = -1
            while indicators['close'][idx] is None and abs(idx) < len(indicators['close']):
                idx -= 1
            
            market_data = MarketData(
                symbol=symbol,
                market_type=market_type,
                timestamp=datetime.now(self.tz_est),
                open=indicators['open'][idx] or 0,
                high=indicators['high'][idx] or 0,
                low=indicators['low'][idx] or 0,
                close=indicators['close'][idx] or 0,
                volume=int(indicators['volume'][idx]) if indicators.get('volume') and indicators['volume'][idx] else None,
                source='yahoo_finance'
            )
            
            return market_data
            
        except Exception as e:
            print(f"Yahoo Finance fetch error for {symbol}: {e}")
            return None
    
    def get_historical_data(self, symbol: str, 
                           market_type: MarketType,
                           days: int = 30,
                           interval: TimeframeInterval = TimeframeInterval.D1) -> List[MarketData]:
        """
        Fetch historical data with specified interval.
        
        Args:
            symbol: Market symbol
            market_type: Type of market
            days: Number of days
            interval: Timeframe interval
            
        Returns:
            List of MarketData objects
        """
        try:
            if market_type == MarketType.FOREX:
                yahoo_symbol = self.FOREX_PAIRS[symbol]['yahoo']
            elif market_type == MarketType.INDEX:
                yahoo_symbol = self.INDEXES[symbol]['yahoo']
            else:
                return []
            
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)
            
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_symbol}"
            params = {
                'interval': interval.value,
                'period1': int(start_date.timestamp()),
                'period2': int(end_date.timestamp())
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'chart' not in data or 'result' not in data['chart']:
                return []
            
            result = data['chart']['result'][0]
            timestamps = result['timestamp']
            indicators = result['indicators']['quote'][0]
            
            historical_data = []
            for i in range(len(timestamps)):
                if indicators['close'][i] is not None:
                    candle_time = datetime.fromtimestamp(timestamps[i], tz=self.tz_est)
                    
                    market_data = MarketData(
                        symbol=symbol,
                        market_type=market_type,
                        timestamp=candle_time,
                        open=indicators['open'][i] or 0,
                        high=indicators['high'][i] or 0,
                        low=indicators['low'][i] or 0,
                        close=indicators['close'][i] or 0,
                        volume=int(indicators['volume'][i]) if indicators.get('volume') and indicators['volume'][i] else None,
                        source='yahoo_finance',
                        session=self.get_current_session(candle_time),
                        kill_zone=self.get_current_kill_zone(candle_time)
                    )
                    historical_data.append(market_data)
            
            return historical_data
            
        except Exception as e:
            print(f"Historical data fetch error for {symbol}: {e}")
            return []
    
    # =========================================================================
    # UTILITY METHODS
    # =========================================================================
    
    def clear_cache(self):
        """Clear all data caches"""
        self.cache.clear()
    
    def get_available_forex_pairs(self) -> Dict[str, str]:
        """Get list of available forex pairs"""
        return {k: v['name'] for k, v in self.FOREX_PAIRS.items()}
    
    def get_available_indexes(self) -> Dict[str, str]:
        """Get list of available indexes"""
        return {k: v['name'] for k, v in self.INDEXES.items()}
    
    def print_time_analysis(self):
        """Print current ICT time analysis"""
        context = self.get_time_context()
        
        print(f"\n{'='*60}")
        print(f"ICT TIME ANALYSIS")
        print(f"{'='*60}")
        print(f"Current Time (EST): {context['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Session: {context['session'].value.upper()}")
        print(f"Kill Zone: {context['kill_zone'].value.upper() if context['kill_zone'] else 'None'}")
        print(f"Macro Time: {'YES - ' + context['macro_window'] if context['is_macro_time'] else 'No'}")
        print(f"Opening Range: {'YES (9:30-10:00)' if context['is_opening_range'] else 'No'}")
        print(f"Lunch Hour: {'YES (AVOID)' if context['is_lunch_hour'] else 'No'}")
        print(f"\nRecommendation: {context['trading_recommendation']}")
        print(f"{'='*60}\n")
    
    @staticmethod
    def get_ict_time_rules() -> Dict[str, List[str]]:
        """Return comprehensive ICT time-based trading rules"""
        return {
            "Key_Opening_Prices": [
                "Midnight (00:00 EST) - True Day start for institutions",
                "London Open (03:00 EST) - Major Forex session start",
                "NY Pre-Market (07:00 EST) - All algorithms fire up",
                "Equities Open (09:30 EST) - Regular trading hours",
                "PM Session (13:30 EST) - Afternoon trading begins",
                "Close (16:00 EST) - Regular hours end",
                "Track last 20 days of opening prices on charts"
            ],
            
            "Kill_Zones": [
                "London Open (01:00-05:00 EST) - Prime Forex trading",
                "NY Open (07:00-10:00 EST) - All algorithms fire",
                "NY AM (09:30-12:00 EST) - Best for indices",
                "NY PM (13:30-16:00 EST) - Scalping environment",
                "Last Hour (15:00-16:00 EST) - Cookie cutter approach"
            ],
            
            "Opening_Range_Rules": [
                "9:30-10:00 AM is manipulation period",
                "First move often FALSE",
                "True direction emerges 10:00-10:30",
                "Don't chase the gap open",
                "Wait for reversal after manipulation"
            ],
            
            "Session_Characteristics": [
                "Asian (00:00-05:00): Builds liquidity for London",
                "London (03:00-12:00): Most volatile for Forex",
                "NY AM (09:30-12:00): Best trading window",
                "Lunch (12:00-13:30): AVOID - low probability",
                "NY PM (13:30-16:00): Scalping only"
            ],
            
            "Macro_Times": [
                "2:33-3:00 AM - Pre-London",
                "4:03-4:30 AM - London Session",
                "8:50-9:10 AM - Pre-Market",
                "9:50-10:10 AM - Post-Open (CE fill window)",
                "10:50-11:10 AM - Mid-Morning",
                "11:50-12:10 PM - Pre-Lunch",
                "13:10-13:40 PM - Post-Lunch",
                "14:50-15:10 PM - Afternoon",
                "15:15-15:45 PM - Power Hour"
            ]
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def format_price(price: float, decimals: int = 5) -> str:
    """Format price with appropriate decimal places"""
    return f"{price:.{decimals}f}"


def calculate_pips(price1: float, price2: float, pip_value: float = 0.0001) -> float:
    """Calculate pips between two prices"""
    return abs(price1 - price2) / pip_value


def calculate_points(price1: float, price2: float) -> float:
    """Calculate points between two prices (for indices)"""
    return abs(price1 - price2)


# =============================================================================
# USAGE EXAMPLE
# =============================================================================

if __name__ == "__main__":
    print("ICT Market Data Engine - Comprehensive Implementation")
    print("=" * 60)
    
    # Initialize engine
    engine = MarketDataEngine()
    
    # Print current time analysis
    engine.print_time_analysis()
    
    # Print ICT time rules
    print("\nICT TIME-BASED TRADING RULES:")
    rules = engine.get_ict_time_rules()
    for category, category_rules in rules.items():
        print(f"\n{category.replace('_', ' ')}:")
        for rule in category_rules:
            print(f"  • {rule}")
    
    # Fetch sample data
    print("\n" + "=" * 60)
    print("SAMPLE DATA FETCH:")
    print("=" * 60)
    
    # Fetch EURUSD
    eur_data = engine.get_forex_data('EURUSD')
    if eur_data:
        print(f"\nEURUSD:")
        print(f"  Close: {format_price(eur_data.close)}")
        print(f"  Session: {eur_data.session.value if eur_data.session else 'N/A'}")
        print(f"  Kill Zone: {eur_data.kill_zone.value if eur_data.kill_zone else 'None'}")
        print(f"  Macro Time: {eur_data.is_macro_time}")
    
    # Fetch NQ
    nq_data = engine.get_index_data('NQ')
    if nq_data:
        print(f"\nNQ (NASDAQ Futures):")
        print(f"  Close: {format_price(nq_data.close, 2)}")
        print(f"  Session: {nq_data.session.value if nq_data.session else 'N/A'}")
        print(f"  Kill Zone: {nq_data.kill_zone.value if nq_data.kill_zone else 'None'}")
    
    # DXY bias
    print("\n" + "=" * 60)
    print("DXY INTERMARKET ANALYSIS:")
    print("=" * 60)
    dxy_bias = engine.get_dxy_bias()
    for key, value in dxy_bias.items():
        if isinstance(value, dict):
            print(f"\n{key}:")
            for k, v in value.items():
                print(f"  {k}: {v}")
        else:
            print(f"{key}: {value}")
    
    print("\n" + "=" * 60)
    print("Engine ready for ICT trading analysis!")
