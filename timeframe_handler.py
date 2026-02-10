"""
Comprehensive ICT Timeframe Handler
Based on Inner Circle Trader methodology and transcript teachings

Key ICT Time Concepts:
- Kill Zones: Specific time windows for high-probability setups
- Macro Times: 2:33-3:00, 4:03-4:30, etc.
- Silver Bullet: First FVG in specific hour windows
- Opening Range: 9:30-10:00 manipulation, true move 10:00-10:30
- Sessions: Asian (builds liquidity), London (sweeps), NY (continuation/reversal)
- True Day Start: Midnight EST

ICT Quotes from transcripts:
- "If you're trading London, 1:00 AM to 5:00 AM - that's my entire universe"
- "Opening range for London is 1:30 AM to 2:00 AM"
- "First presented fair value gap - algorithmically this is what price is looking at"
- "Silver bullet is the first FVG that forms after 10:00 inside 10:00-11:00 hour"
- "7:00 AM is when all algorithms fire up"
- "9:30-10:00 = manipulation period, first move often FALSE"
- "True move starts 10:00-10:30"
- "New York open kill zone is 7:00 AM to 9:00/10:00 AM"
- "PM session 1:30-4:00 - scalping environment"
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from datetime import datetime, time, timedelta
import pandas as pd


# ==================== ENUMS ====================

class Session(Enum):
    """Trading sessions"""
    ASIAN = "asian"
    LONDON = "london"
    NEW_YORK = "new_york"
    PM = "pm"
    OVERNIGHT = "overnight"


class KillZone(Enum):
    """ICT Kill Zones - high probability time windows"""
    ASIAN_KZ = "asian_kz"                    # 00:00-05:00 EST
    LONDON_OPEN_KZ = "london_open_kz"        # 01:00-05:00 EST (ICT: "1 AM to 5 AM")
    LONDON_KZ = "london_kz"                  # 02:00-12:00 EST
    NEW_YORK_OPEN_KZ = "new_york_open_kz"    # 07:00-10:00 EST
    NEW_YORK_AM_KZ = "new_york_am_kz"        # 09:30-12:00 EST
    NEW_YORK_LUNCH = "new_york_lunch"        # 12:00-13:30 EST (avoid)
    NEW_YORK_PM_KZ = "new_york_pm_kz"        # 13:30-16:00 EST


class SilverBulletWindow(Enum):
    """Silver Bullet time windows"""
    ASIAN_SB = "asian_sb"       # 20:00-21:00 EST
    LONDON_SB = "london_sb"     # 03:00-04:00 EST
    AM_SB = "am_sb"             # 10:00-11:00 EST (most popular)
    PM_SB = "pm_sb"             # 14:00-15:00 EST


class MacroTime(Enum):
    """ICT Macro time windows"""
    MACRO_1 = "02:33-03:00"
    MACRO_2 = "04:03-04:30"
    MACRO_3 = "08:50-09:10"
    MACRO_4 = "09:50-10:10"
    MACRO_5 = "10:50-11:10"
    MACRO_6 = "11:50-12:10"
    MACRO_7 = "13:50-14:10"
    MACRO_8 = "14:50-15:10"
    MACRO_9 = "15:15-15:45"


class TimeframePurpose(Enum):
    """Purpose of each timeframe"""
    BIAS = "bias"                  # HTF bias (Daily, Weekly)
    STRUCTURE = "structure"        # Market structure (4H, 1H)
    ENTRY = "entry"                # Entry timeframe (15m, 5m)
    PRECISION = "precision"        # Precision entries (1m)


# ==================== DATA CLASSES ====================

@dataclass
class TimeWindow:
    """Time window with start/end times"""
    name: str
    start_time: time
    end_time: time
    timezone: str = "EST"
    
    def is_active(self, current_time: time) -> bool:
        """Check if current time is within window"""
        if self.start_time <= self.end_time:
            return self.start_time <= current_time <= self.end_time
        else:
            # Handles overnight windows
            return current_time >= self.start_time or current_time <= self.end_time
    
    def minutes_until_start(self, current_time: time) -> int:
        """Calculate minutes until window starts"""
        current_mins = current_time.hour * 60 + current_time.minute
        start_mins = self.start_time.hour * 60 + self.start_time.minute
        
        if start_mins > current_mins:
            return start_mins - current_mins
        else:
            return (24 * 60 - current_mins) + start_mins
    
    def minutes_until_end(self, current_time: time) -> int:
        """Calculate minutes until window ends"""
        current_mins = current_time.hour * 60 + current_time.minute
        end_mins = self.end_time.hour * 60 + self.end_time.minute
        
        if end_mins > current_mins:
            return end_mins - current_mins
        else:
            return (24 * 60 - current_mins) + end_mins


@dataclass
class KillZoneConfig:
    """Kill Zone configuration"""
    killzone: KillZone
    window: TimeWindow
    best_for: List[str]
    typical_behavior: str
    what_to_look_for: List[str]
    avoid: List[str]


@dataclass
class SessionConfig:
    """Session configuration"""
    session: Session
    window: TimeWindow
    characteristics: List[str]
    best_pairs: List[str]
    volatility: str  # LOW, MEDIUM, HIGH
    role: str  # What this session does


@dataclass
class TimeframeConfig:
    """Configuration for a specific timeframe"""
    timeframe: str
    minutes: int
    purpose: TimeframePurpose
    what_to_look_for: List[str]
    typical_use: str


@dataclass
class TimeAnalysis:
    """Complete time analysis result"""
    current_time: time
    current_datetime: Optional[datetime] = None
    
    # Session info
    current_session: Optional[Session] = None
    session_progress_pct: float = 0.0
    
    # Kill zone info
    active_killzone: Optional[KillZone] = None
    in_killzone: bool = False
    killzone_minutes_remaining: int = 0
    
    # Silver bullet
    in_silver_bullet: bool = False
    silver_bullet_window: Optional[SilverBulletWindow] = None
    sb_minutes_remaining: int = 0
    
    # Macro time
    in_macro: bool = False
    current_macro: Optional[MacroTime] = None
    next_macro: Optional[MacroTime] = None
    minutes_to_next_macro: int = 0
    
    # Special times
    is_opening_range: bool = False          # 9:30-10:00 manipulation
    is_true_move_window: bool = False       # 10:00-10:30
    is_lunch_hour: bool = False             # 12:00-13:30 (avoid)
    is_power_hour: bool = False             # 15:00-16:00
    
    # Recommendations
    trading_recommended: bool = True
    avoid_reason: Optional[str] = None
    what_to_look_for: List[str] = field(default_factory=list)


# ==================== MAIN HANDLER ====================

class TimeframeHandler:
    """
    Comprehensive ICT Timeframe Handler
    
    ICT Rules:
    - London: 1:00 AM to 5:00 AM EST is the primary window
    - NY Open KZ: 7:00 AM to 10:00 AM EST
    - 9:30-10:00 = manipulation, first move often FALSE
    - True move starts 10:00-10:30
    - Silver Bullet: First FVG in specific hour windows
    - Avoid lunch hour 12:00-13:30
    - PM session: 13:30-16:00 for scalping
    """
    
    def __init__(self):
        self.killzones = self._init_killzones()
        self.sessions = self._init_sessions()
        self.silver_bullets = self._init_silver_bullets()
        self.macro_times = self._init_macro_times()
        self.timeframes = self._init_timeframes()
    
    def _init_killzones(self) -> Dict[KillZone, KillZoneConfig]:
        """Initialize kill zone configurations from ICT teachings"""
        return {
            KillZone.ASIAN_KZ: KillZoneConfig(
                killzone=KillZone.ASIAN_KZ,
                window=TimeWindow("Asian KZ", time(0, 0), time(5, 0)),
                best_for=["Asian pairs", "Range trading", "Liquidity building"],
                typical_behavior="Range-bound, builds liquidity for London",
                what_to_look_for=[
                    "Range formation", "Equal highs/lows forming",
                    "Consolidation for London to raid"
                ],
                avoid=["Major breakout attempts", "Large positions"]
            ),
            
            KillZone.LONDON_OPEN_KZ: KillZoneConfig(
                killzone=KillZone.LONDON_OPEN_KZ,
                window=TimeWindow("London Open KZ", time(1, 0), time(5, 0)),
                best_for=["EUR/GBP pairs", "Initial daily moves", "Liquidity sweeps"],
                typical_behavior="ICT: '1 AM to 5 AM - my entire universe of London trading'",
                what_to_look_for=[
                    "Asian high/low sweeps", "First Presented FVG",
                    "Displacement", "Market structure shift",
                    "Opening range 1:30-2:00 AM"
                ],
                avoid=["Fading strong moves without structure shift"]
            ),
            
            KillZone.NEW_YORK_OPEN_KZ: KillZoneConfig(
                killzone=KillZone.NEW_YORK_OPEN_KZ,
                window=TimeWindow("NY Open KZ", time(7, 0), time(10, 0)),
                best_for=["Indices (NQ/ES)", "USD pairs", "Continuation/Reversal"],
                typical_behavior="ICT: '7:00 AM is when all algorithms fire up'",
                what_to_look_for=[
                    "Pre-market high/low runs", "Buy/sell side taken at 6:30",
                    "Market structure shift", "First Presented FVG after open"
                ],
                avoid=["Trading against London bias without clear reversal"]
            ),
            
            KillZone.NEW_YORK_AM_KZ: KillZoneConfig(
                killzone=KillZone.NEW_YORK_AM_KZ,
                window=TimeWindow("NY AM KZ", time(9, 30), time(12, 0)),
                best_for=["Indices", "Opening range plays", "Silver Bullet"],
                typical_behavior="9:30-10:00 = manipulation. True move 10:00-10:30",
                what_to_look_for=[
                    "Opening range gap (9:30)", "First 30 min manipulation",
                    "Silver Bullet 10:00-11:00", "True direction after 10:00"
                ],
                avoid=["Trading 9:30-10:00 without clear bias", "Chasing first move"]
            ),
            
            KillZone.NEW_YORK_LUNCH: KillZoneConfig(
                killzone=KillZone.NEW_YORK_LUNCH,
                window=TimeWindow("NY Lunch", time(12, 0), time(13, 30)),
                best_for=["Nothing - AVOID"],
                typical_behavior="Low volume, consolidation, traps",
                what_to_look_for=["Setup for PM session"],
                avoid=["Trading during lunch - low probability"]
            ),
            
            KillZone.NEW_YORK_PM_KZ: KillZoneConfig(
                killzone=KillZone.NEW_YORK_PM_KZ,
                window=TimeWindow("NY PM KZ", time(13, 30), time(16, 0)),
                best_for=["Scalping", "Continuation", "End of day moves"],
                typical_behavior="ICT: 'PM session - scalping environment'",
                what_to_look_for=[
                    "Macro times (2:33-3:00, etc.)", "Continuation of AM move",
                    "Final push to daily target", "Position closing"
                ],
                avoid=["Large swings", "Holding overnight without reason"]
            )
        }
    
    def _init_sessions(self) -> Dict[Session, SessionConfig]:
        """Initialize session configurations"""
        return {
            Session.ASIAN: SessionConfig(
                session=Session.ASIAN,
                window=TimeWindow("Asian", time(19, 0), time(4, 0)),  # 7 PM - 4 AM EST
                characteristics=[
                    "Quieter, range-bound",
                    "Builds liquidity pools for London",
                    "Lower volume"
                ],
                best_pairs=["USDJPY", "AUDUSD", "NZDUSD", "EURJPY"],
                volatility="LOW",
                role="Builds liquidity for London to raid"
            ),
            
            Session.LONDON: SessionConfig(
                session=Session.LONDON,
                window=TimeWindow("London", time(3, 0), time(12, 0)),
                characteristics=[
                    "Highest volume session",
                    "Runs overnight highs/lows",
                    "Sets daily bias",
                    "Strong directional moves"
                ],
                best_pairs=["EURUSD", "GBPUSD", "EURGBP", "EURJPY", "GBPJPY"],
                volatility="HIGH",
                role="Sweeps Asian liquidity, establishes trend"
            ),
            
            Session.NEW_YORK: SessionConfig(
                session=Session.NEW_YORK,
                window=TimeWindow("New York", time(8, 0), time(17, 0)),
                characteristics=[
                    "High volatility at open",
                    "Can reverse or continue London",
                    "Indices very active",
                    "Economic news impact"
                ],
                best_pairs=["EURUSD", "GBPUSD", "USDCAD", "USDJPY", "NQ", "ES"],
                volatility="HIGH",
                role="Can reverse or continue London trend"
            ),
            
            Session.PM: SessionConfig(
                session=Session.PM,
                window=TimeWindow("PM Session", time(13, 30), time(16, 0)),
                characteristics=[
                    "Scalping environment",
                    "Lower than morning volume",
                    "Final daily moves"
                ],
                best_pairs=["NQ", "ES", "EURUSD"],
                volatility="MEDIUM",
                role="Scalping, position closing"
            )
        }
    
    def _init_silver_bullets(self) -> Dict[SilverBulletWindow, TimeWindow]:
        """
        Initialize Silver Bullet windows
        ICT: "First FVG that forms after 10:00 inside 10:00-11:00 hour"
        """
        return {
            SilverBulletWindow.ASIAN_SB: TimeWindow("Asian SB", time(20, 0), time(21, 0)),
            SilverBulletWindow.LONDON_SB: TimeWindow("London SB", time(3, 0), time(4, 0)),
            SilverBulletWindow.AM_SB: TimeWindow("AM SB", time(10, 0), time(11, 0)),
            SilverBulletWindow.PM_SB: TimeWindow("PM SB", time(14, 0), time(15, 0))
        }
    
    def _init_macro_times(self) -> Dict[MacroTime, TimeWindow]:
        """Initialize macro time windows"""
        return {
            MacroTime.MACRO_1: TimeWindow("Macro 1", time(2, 33), time(3, 0)),
            MacroTime.MACRO_2: TimeWindow("Macro 2", time(4, 3), time(4, 30)),
            MacroTime.MACRO_3: TimeWindow("Macro 3", time(8, 50), time(9, 10)),
            MacroTime.MACRO_4: TimeWindow("Macro 4", time(9, 50), time(10, 10)),
            MacroTime.MACRO_5: TimeWindow("Macro 5", time(10, 50), time(11, 10)),
            MacroTime.MACRO_6: TimeWindow("Macro 6", time(11, 50), time(12, 10)),
            MacroTime.MACRO_7: TimeWindow("Macro 7", time(13, 50), time(14, 10)),
            MacroTime.MACRO_8: TimeWindow("Macro 8", time(14, 50), time(15, 10)),
            MacroTime.MACRO_9: TimeWindow("Macro 9", time(15, 15), time(15, 45))
        }
    
    def _init_timeframes(self) -> Dict[str, TimeframeConfig]:
        """Initialize timeframe configurations"""
        return {
            "1W": TimeframeConfig("1W", 10080, TimeframePurpose.BIAS,
                                  ["Weekly structure", "Major liquidity pools", "Weekly FVGs"],
                                  "Establish weekly bias"),
            "1D": TimeframeConfig("1D", 1440, TimeframePurpose.BIAS,
                                  ["Daily FVGs", "Daily order blocks", "Premium/Discount wicks",
                                   "NDOG", "Daily high/low liquidity"],
                                  "PRIMARY BIAS - Trade in this direction"),
            "4H": TimeframeConfig("4H", 240, TimeframePurpose.STRUCTURE,
                                  ["Session highs/lows", "4H FVGs", "Intraday structure"],
                                  "Refine daily bias for session trading"),
            "1H": TimeframeConfig("1H", 60, TimeframePurpose.STRUCTURE,
                                  ["Hourly FVGs", "Market maker models", "Breaker blocks"],
                                  "Identify potential setups"),
            "15m": TimeframeConfig("15m", 15, TimeframePurpose.ENTRY,
                                   ["Kill zone FVGs", "Opening range manipulation",
                                    "Silver Bullet setups", "First Presented FVG"],
                                   "PRIMARY ENTRY TIMEFRAME"),
            "5m": TimeframeConfig("5m", 5, TimeframePurpose.ENTRY,
                                  ["Exact FVG entries", "Stop hunts", "Displacement"],
                                  "Precision entries"),
            "1m": TimeframeConfig("1m", 1, TimeframePurpose.PRECISION,
                                  ["Exact entry timing", "Stop placement", "Scalping"],
                                  "Precision timing in kill zones")
        }
    
    # ==================== ANALYSIS ====================
    
    def analyze(self, current_time: time, 
                current_datetime: Optional[datetime] = None) -> TimeAnalysis:
        """
        Comprehensive time analysis
        """
        analysis = TimeAnalysis(
            current_time=current_time,
            current_datetime=current_datetime
        )
        
        # Check current session
        for session, config in self.sessions.items():
            if config.window.is_active(current_time):
                analysis.current_session = session
                break
        
        # Check kill zones
        for kz, config in self.killzones.items():
            if config.window.is_active(current_time):
                analysis.active_killzone = kz
                analysis.in_killzone = True
                analysis.killzone_minutes_remaining = config.window.minutes_until_end(current_time)
                analysis.what_to_look_for.extend(config.what_to_look_for)
                break
        
        # Check silver bullet windows
        for sb, window in self.silver_bullets.items():
            if window.is_active(current_time):
                analysis.in_silver_bullet = True
                analysis.silver_bullet_window = sb
                analysis.sb_minutes_remaining = window.minutes_until_end(current_time)
                analysis.what_to_look_for.append("First Presented FVG for Silver Bullet")
                break
        
        # Check macro times
        for macro, window in self.macro_times.items():
            if window.is_active(current_time):
                analysis.in_macro = True
                analysis.current_macro = macro
                break
        
        # Find next macro
        next_macro = None
        min_minutes = float('inf')
        for macro, window in self.macro_times.items():
            if not window.is_active(current_time):
                mins = window.minutes_until_start(current_time)
                if mins < min_minutes:
                    min_minutes = mins
                    next_macro = macro
        analysis.next_macro = next_macro
        analysis.minutes_to_next_macro = int(min_minutes) if min_minutes != float('inf') else 0
        
        # Special time checks
        analysis.is_opening_range = time(9, 30) <= current_time <= time(10, 0)
        analysis.is_true_move_window = time(10, 0) <= current_time <= time(10, 30)
        analysis.is_lunch_hour = time(12, 0) <= current_time <= time(13, 30)
        analysis.is_power_hour = time(15, 0) <= current_time <= time(16, 0)
        
        # Trading recommendations
        if analysis.is_lunch_hour:
            analysis.trading_recommended = False
            analysis.avoid_reason = "Lunch hour - low probability"
        elif analysis.is_opening_range:
            analysis.what_to_look_for.append("MANIPULATION PERIOD - First move often FALSE")
            analysis.what_to_look_for.append("Wait for 10:00 for true direction")
        elif analysis.is_true_move_window:
            analysis.what_to_look_for.append("TRUE MOVE WINDOW - Direction should be clear now")
        
        return analysis
    
    def get_current_killzone(self, current_time: time) -> Optional[KillZoneConfig]:
        """Get current active kill zone configuration"""
        for kz, config in self.killzones.items():
            if config.window.is_active(current_time):
                return config
        return None
    
    def get_current_session(self, current_time: time) -> Optional[SessionConfig]:
        """Get current session configuration"""
        for session, config in self.sessions.items():
            if config.window.is_active(current_time):
                return config
        return None
    
    def is_in_silver_bullet(self, current_time: time) -> Tuple[bool, Optional[SilverBulletWindow]]:
        """Check if currently in a silver bullet window"""
        for sb, window in self.silver_bullets.items():
            if window.is_active(current_time):
                return True, sb
        return False, None
    
    def get_optimal_timeframes(self, trading_style: str = "intraday") -> Dict[str, List[str]]:
        """
        Get optimal timeframe combinations
        """
        if trading_style == "scalping":
            return {
                "bias": ["1H", "15m"],
                "structure": ["5m"],
                "entry": ["1m"]
            }
        elif trading_style == "intraday":
            return {
                "bias": ["1D", "4H"],
                "structure": ["1H", "15m"],
                "entry": ["5m", "1m"]
            }
        elif trading_style == "swing":
            return {
                "bias": ["1W", "1D"],
                "structure": ["4H", "1H"],
                "entry": ["15m", "5m"]
            }
        return {}
    
    # ==================== TRADING RULES ====================
    
    def get_time_trading_rules(self) -> Dict[str, List[str]]:
        """Get ICT time-based trading rules"""
        return {
            "London_Rules": [
                "1:00 AM to 5:00 AM EST - primary London window",
                "Opening range: 1:30 AM to 2:00 AM",
                "Look for Asian high/low sweeps",
                "First Presented FVG after open"
            ],
            "NY_Open_Rules": [
                "7:00 AM - all algorithms fire up",
                "Check what happened at 6:30 (buy/sell side taken)",
                "Pre-market high/low runs"
            ],
            "Opening_Range_Rules": [
                "9:30-10:00 = MANIPULATION PERIOD",
                "First move often FALSE",
                "Wait for 10:00-10:30 for true direction",
                "Don't chase the first move"
            ],
            "Silver_Bullet_Rules": [
                "First FVG that forms in the window",
                "AM: 10:00-11:00 (most popular)",
                "PM: 14:00-15:00",
                "London: 03:00-04:00",
                "Asian: 20:00-21:00"
            ],
            "Avoid_Rules": [
                "Lunch hour: 12:00-13:30 (low probability)",
                "Don't trade between quadrants",
                "Don't trade without kill zone alignment"
            ],
            "PM_Session_Rules": [
                "13:30-16:00 = scalping environment",
                "Use macro times for entries",
                "Smaller position sizes",
                "Trail stops, take partials"
            ]
        }
    
    def get_summary(self, current_time: time) -> str:
        """Generate text summary"""
        analysis = self.analyze(current_time)
        
        lines = ["", "=" * 70, f"ICT TIME ANALYSIS - {current_time.strftime('%H:%M')} EST", "=" * 70]
        
        lines.append(f"\nSession: {analysis.current_session.value if analysis.current_session else 'None'}")
        
        if analysis.in_killzone:
            lines.append(f"Kill Zone: {analysis.active_killzone.value} ({analysis.killzone_minutes_remaining} min remaining)")
        else:
            lines.append("Kill Zone: None active")
        
        if analysis.in_silver_bullet:
            lines.append(f"Silver Bullet: {analysis.silver_bullet_window.value} ACTIVE!")
        
        if analysis.in_macro:
            lines.append(f"Macro Time: {analysis.current_macro.value} ACTIVE!")
        else:
            lines.append(f"Next Macro: {analysis.next_macro.value if analysis.next_macro else 'N/A'} in {analysis.minutes_to_next_macro} min")
        
        # Special flags
        if analysis.is_opening_range:
            lines.append("\n*** OPENING RANGE - MANIPULATION PERIOD ***")
        if analysis.is_true_move_window:
            lines.append("\n*** TRUE MOVE WINDOW ***")
        if analysis.is_lunch_hour:
            lines.append("\n*** LUNCH HOUR - AVOID TRADING ***")
        
        if analysis.what_to_look_for:
            lines.append("\nWhat to look for:")
            for item in analysis.what_to_look_for:
                lines.append(f"  • {item}")
        
        if not analysis.trading_recommended:
            lines.append(f"\n*** TRADING NOT RECOMMENDED: {analysis.avoid_reason} ***")
        
        lines.append("=" * 70)
        return "\n".join(lines)


if __name__ == "__main__":
    print("ICT Timeframe Handler - Comprehensive")
    print("=" * 50)
    print("\nKey ICT Time Concepts:")
    print("  • London: 1:00 AM to 5:00 AM EST")
    print("  • NY Open: 7:00 AM - algorithms fire up")
    print("  • 9:30-10:00: Manipulation (first move FALSE)")
    print("  • 10:00-10:30: True move emerges")
    print("  • Silver Bullet: First FVG in hour window")
    print("  • Avoid: Lunch hour 12:00-13:30")
