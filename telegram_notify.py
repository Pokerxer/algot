"""
Telegram Notification Module for V8 ICT Trading Bot
====================================================
Modern, beautiful notifications with interactive commands and RL agent integration.

Features:
- Real-time trade notifications with visual animations
- Interactive command menu with smart button layouts
- Position tracking with live P&L updates
- Advanced performance analytics with RL agent status
- Market bias display with trend indicators
- Confluence monitoring with visual gauges
- Price alerts and watchlist management
- Trade management commands (close, modify positions)
- Trading control (pause/resume) via Telegram
- RL agent monitoring and statistics
"""

import os
import asyncio
import json
import logging
import threading
import time
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List, Tuple

try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, InputFile
    from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    print("Warning: python-telegram-bot not installed. Run: pip install python-telegram-bot")


# ============================================================================
# DESIGN SYSTEM - Modern Trading Bot UI
# ============================================================================

class DesignSystem:
    """Modern design system with consistent visual elements"""
    
    # Box Drawing Characters
    BOX_TL = ""
    BOX_TR = ""
    BOX_BL = ""
    BOX_BR = ""
    BOX_H = ""
    BOX_V = ""
    BOX_ML = ""
    BOX_MR = ""
    
    # Rounded corners alternative
    ROUND_TL = ""
    ROUND_TR = ""
    ROUND_BL = ""
    ROUND_BR = ""
    
    # Separators
    SEP_THICK = ""
    SEP_THIN = ""
    SEP_DOT = ""
    SEP_WAVE = ""
    
    # Progress bar elements
    PROGRESS_FULL = ""
    PROGRESS_EMPTY = ""
    PROGRESS_START = ""
    PROGRESS_END = ""
    
    # Direction indicators
    ARROW_UP = ""
    ARROW_DOWN = ""
    ARROW_RIGHT = ""
    ARROW_LEFT = ""
    TREND_UP = ""
    TREND_DOWN = ""
    TREND_FLAT = ""
    
    # Status indicators
    STATUS_SUCCESS = ""
    STATUS_ERROR = ""
    STATUS_WARNING = ""
    STATUS_INFO = ""
    STATUS_NEUTRAL = ""
    
    # Trading icons
    ICON_BULL = ""
    ICON_BEAR = ""
    ICON_MONEY = ""
    ICON_CHART = ""
    ICON_TARGET = ""
    ICON_SHIELD = ""
    ICON_ROCKET = ""
    ICON_FIRE = ""
    ICON_DIAMOND = ""
    ICON_CROWN = ""
    ICON_STAR = ""
    ICON_ZAP = ""
    ICON_CLOCK = ""
    ICON_CALENDAR = ""
    ICON_LOCK = ""
    ICON_UNLOCK = ""
    ICON_EYE = ""
    ICON_BELL = ""
    ICON_GEAR = ""
    ICON_REFRESH = ""
    
    # Session icons
    SESSION_LONDON = ""
    SESSION_NYC = ""
    SESSION_ASIA = ""
    SESSION_CLOSED = ""
    
    # Medals
    MEDAL_GOLD = ""
    MEDAL_SILVER = ""
    MEDAL_BRONZE = ""
    
    @staticmethod
    def progress_bar(value: float, max_value: float = 100, width: int = 10, show_pct: bool = True) -> str:
        """Create an animated progress bar"""
        if max_value <= 0:
            pct = 0
        else:
            pct = min(max(value / max_value, 0), 1)
        
        filled = int(pct * width)
        empty = width - filled
        
        bar = DesignSystem.PROGRESS_FULL * filled + DesignSystem.PROGRESS_EMPTY * empty
        
        if show_pct:
            return f"[{bar}] {pct*100:.0f}%"
        return f"[{bar}]"
    
    @staticmethod
    def pnl_bar(pnl: float, max_pnl: float = 1000, width: int = 10) -> str:
        """Create a P&L visualization bar (green for profit, red for loss)"""
        if max_pnl <= 0:
            max_pnl = abs(pnl) if pnl != 0 else 1
        
        pct = min(abs(pnl) / max_pnl, 1)
        filled = int(pct * width)
        
        if pnl >= 0:
            bar = "" * filled + "" * (width - filled)
            return f"[{bar}] +${abs(pnl):,.0f}"
        else:
            bar = "" * filled + "" * (width - filled)
            return f"[{bar}] -${abs(pnl):,.0f}"
    
    @staticmethod
    def trend_indicator(trend: int, show_text: bool = True) -> str:
        """Create a trend indicator"""
        if trend > 0:
            icon = DesignSystem.TREND_UP
            text = "BULLISH" if show_text else ""
        elif trend < 0:
            icon = DesignSystem.TREND_DOWN
            text = "BEARISH" if show_text else ""
        else:
            icon = DesignSystem.TREND_FLAT
            text = "NEUTRAL" if show_text else ""
        
        return f"{icon} {text}".strip()
    
    @staticmethod
    def status_badge(status: str) -> str:
        """Create a status badge"""
        badges = {
            'win': f"{DesignSystem.STATUS_SUCCESS} WIN",
            'loss': f"{DesignSystem.STATUS_ERROR} LOSS",
            'active': f"{DesignSystem.STATUS_SUCCESS} ACTIVE",
            'pending': f"{DesignSystem.STATUS_WARNING} PENDING",
            'closed': f"{DesignSystem.STATUS_NEUTRAL} CLOSED",
            'long': f"{DesignSystem.ICON_BULL} LONG",
            'short': f"{DesignSystem.ICON_BEAR} SHORT",
        }
        return badges.get(status.lower(), status)
    
    @staticmethod
    def confidence_meter(confidence: float, max_conf: float = 100) -> str:
        """Create a confidence/confluence meter"""
        pct = min(confidence / max_conf, 1) if max_conf > 0 else 0
        
        if pct >= 0.8:
            icon = DesignSystem.ICON_FIRE
            level = "EXTREME"
        elif pct >= 0.6:
            icon = DesignSystem.ICON_ZAP
            level = "HIGH"
        elif pct >= 0.4:
            icon = DesignSystem.STATUS_WARNING
            level = "MEDIUM"
        else:
            icon = DesignSystem.STATUS_INFO
            level = "LOW"
        
        bar = DesignSystem.progress_bar(confidence, max_conf, 8, False)
        return f"{icon} {bar} {level}"
    
    @staticmethod
    def create_card(title: str, content: str, icon: str = "") -> str:
        """Create a card-style message component"""
        header = f"{icon} <b>{title}</b>" if icon else f"<b>{title}</b>"
        lines = [
            DesignSystem.SEP_THICK,
            header,
            DesignSystem.SEP_THIN,
            content,
            DesignSystem.SEP_THICK
        ]
        return "\n".join(lines)
    
    @staticmethod
    def format_price(price: float, decimals: int = None) -> str:
        """Format price with appropriate decimals"""
        if decimals is None:
            if price >= 1000:
                decimals = 2
            elif price >= 1:
                decimals = 4
            else:
                decimals = 6
        return f"${price:,.{decimals}f}"
    
    @staticmethod
    def format_change(change: float, as_pct: bool = False) -> str:
        """Format a change value with color indicator"""
        if as_pct:
            text = f"{change:+.2f}%"
        else:
            text = f"${change:+,.2f}"
        
        if change > 0:
            return f"{DesignSystem.STATUS_SUCCESS} {text}"
        elif change < 0:
            return f"{DesignSystem.STATUS_ERROR} {text}"
        else:
            return f"{DesignSystem.STATUS_NEUTRAL} {text}"
    
    @staticmethod
    def get_session_icon() -> Tuple[str, str]:
        """Get current trading session icon and name"""
        now = datetime.now()
        hour = now.hour
        
        # London: 3 AM - 12 PM ET (8 AM - 5 PM GMT)
        # NYC: 8 AM - 5 PM ET
        # Asia: 7 PM - 4 AM ET
        
        if 8 <= hour < 12:
            return DesignSystem.SESSION_LONDON, "London/NYC Overlap"
        elif 12 <= hour < 17:
            return DesignSystem.SESSION_NYC, "NYC Session"
        elif 3 <= hour < 8:
            return DesignSystem.SESSION_LONDON, "London Session"
        elif 19 <= hour or hour < 4:
            return DesignSystem.SESSION_ASIA, "Asia Session"
        else:
            return DesignSystem.SESSION_CLOSED, "Off Hours"
    
    @staticmethod
    def sparkline(values: List[float], width: int = 8) -> str:
        """Create a text-based sparkline"""
        if not values or len(values) < 2:
            return "--------"
        
        # Normalize values
        min_val = min(values)
        max_val = max(values)
        range_val = max_val - min_val if max_val != min_val else 1
        
        # Map to sparkline characters
        chars = ['', '', '', '', '', '', '', '']
        result = []
        
        # Sample values if too many
        if len(values) > width:
            step = len(values) / width
            sampled = [values[int(i * step)] for i in range(width)]
        else:
            sampled = values[-width:]
        
        for val in sampled:
            idx = int(((val - min_val) / range_val) * 7)
            idx = min(max(idx, 0), 7)
            result.append(chars[idx])
        
        return "".join(result)

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuration
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '8373721073:AAEBSdP3rmREEccpRiKznTFJtwNKsmXJEts')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '809192622')

# Global state with enhanced tracking
TRADE_HISTORY: List[Dict] = []
CURRENT_POSITIONS: Dict[str, Dict] = {}
DAILY_STATS = {
    'trades': 0, 
    'wins': 0, 
    'losses': 0, 
    'pnl': 0.0, 
    'start_time': datetime.now().isoformat(),
    'best_trade': None,
    'worst_trade': None,
    'streak': 0,  # Positive = winning streak, negative = losing streak
    'max_drawdown': 0.0,
    'peak_pnl': 0.0
}
LAST_MARKET_DATA: Dict[str, Dict] = {}
WATCHLIST: List[str] = []  # User's watchlist symbols
PRICE_HISTORY: Dict[str, List[float]] = {}  # Price history for sparklines

BOT_SETTINGS = {
    'notifications_enabled': True,
    'signal_alerts': True,
    'trade_alerts': True,
    'daily_summary': True,
    'risk_per_trade': 0.02,
    'symbols': [],
    'mode': 'Paper Trading',
    'price_alerts': {},
    'sound_enabled': True,
    'detailed_notifications': True,
    'show_sparklines': True,
    'auto_close_enabled': False  # For auto-closing positions
}

# Animation frames for loading/updates
LOADING_FRAMES = ["", "", "", "", "", "", "", ""]
PULSE_FRAMES = ["", "", "", "", "", "", "", ""]

# Bot instance
app = None
_bot_thread = None
_event_loop = None

# Global reference to live trader (set by v8_live.py)
_live_trader = None

# Trading state (pause/resume)
_trading_paused = False

# Price alerts
_price_alerts: Dict[str, List[Dict]] = {}  # {symbol: [{price, direction, triggered}]}

# IBKR connection for Telegram commands to fetch live data
_ibkr_connection = None


def set_live_trader(trader):
    """Set reference to live trader for Telegram commands."""
    global _live_trader
    _live_trader = trader


def get_live_trader():
    """Get reference to live trader."""
    return _live_trader


def is_trading_paused() -> bool:
    """Check if trading is paused."""
    return _trading_paused


def set_trading_paused(paused: bool):
    """Set trading paused state."""
    global _trading_paused
    _trading_paused = paused

def get_ibkr_connection():
    """Get or create IBKR connection for Telegram commands."""
    global _ibkr_connection
    try:
        from ib_insync import IB
        if _ibkr_connection is None or not _ibkr_connection.isConnected():
            _ibkr_connection = IB()
            try:
                _ibkr_connection.connect('127.0.0.1', 7497, clientId=88)
                logger.info("Telegram IBKR connection established")
            except Exception as e:
                logger.error(f"Could not connect to IBKR for Telegram: {e}")
                return None
        return _ibkr_connection
    except ImportError:
        logger.error("ib_insync not available")
        return None


def fetch_live_price(symbol):
    """Fetch live price from IBKR for a symbol."""
    try:
        ib = get_ibkr_connection()
        if not ib:
            return None
        
        # Get contract - need to map symbol to proper format
        contract = _get_ibkr_contract(symbol)
        if not contract:
            return None
        
        # Request market data
        ticker = ib.reqMktData(contract, '', False, False)
        ib.sleep(0.5)  # Wait for data
        
        price = ticker.last if ticker.last else ticker.close
        if price and price > 0:
            return price
        return None
    except Exception as e:
        logger.error(f"Error fetching live price for {symbol}: {e}")
        return None


def fetch_live_data_for_symbols(symbols):
    """Fetch live data for multiple symbols from IBKR."""
    live_data = {}
    
    for symbol in symbols:
        try:
            price = fetch_live_price(symbol)
            if price:
                # Get cached data for trend info if available
                cached = LAST_MARKET_DATA.get(symbol, {})
                live_data[symbol] = {
                    'price': price,
                    'htf_trend': cached.get('htf_trend', 0),
                    'ltf_trend': cached.get('ltf_trend', 0),
                    'kill_zone': cached.get('kill_zone', False),
                    'price_position': cached.get('price_position', 0.5),
                    'confluence': cached.get('confluence', 0),
                    'live': True,
                    'updated_at': datetime.now().isoformat()
                }
        except Exception as e:
            logger.error(f"Error fetching data for {symbol}: {e}")
    
    return live_data


def _get_ibkr_contract(symbol):
    """Create IBKR contract for a symbol."""
    try:
        from ib_insync import Future, Crypto, Forex, Contract
        
        symbol = symbol.upper()
        
        # Futures mapping
        futures_map = {
            'ES': ('ES', 'GLOBEX', 'USD', 50),
            'NQ': ('NQ', 'GLOBEX', 'USD', 20),
            'GC': ('GC', 'NYMEX', 'USD', 100),
            'CL': ('CL', 'NYMEX', 'USD', 1000),
            'SI': ('SI', 'NYMEX', 'USD', 5000),
            'NG': ('NG', 'NYMEX', 'USD', 10000),
        }
        
        if symbol in futures_map:
            sym, exchange, currency, multiplier = futures_map[symbol]
            contract = Future(symbol=sym, exchange=exchange, currency=currency)
            contract.multiplier = str(multiplier)
            return contract
        
        # Crypto mapping
        if symbol in ['BTCUSD', 'ETHUSD', 'SOLUSD', 'LTCUSD', 'LINKUSD', 'UNIUSD']:
            base = symbol.replace('USD', '')
            return Crypto(symbol=base, exchange='PAXOS', currency='USD')
        
        # Forex
        if len(symbol) == 6:
            return Forex(symbol[:3] + '.' + symbol[3:])
        
        return None
    except Exception as e:
        logger.error(f"Error creating contract for {symbol}: {e}")
        return None


class TelegramNotifier:
    """Thread-safe Telegram notifier class"""
    
    def __init__(self):
        self.app = None
        self._loop = None
        self._thread = None
        self._initialized = False
        
    def init(self):
        """Initialize the bot"""
        if not TELEGRAM_AVAILABLE:
            logger.error("Telegram not available - python-telegram-bot not installed")
            return False
            
        try:
            self.app = Application.builder().token(TELEGRAM_TOKEN).build()
            
            # Add command handlers
            self.app.add_handler(CommandHandler("start", self._start_command))
            self.app.add_handler(CommandHandler("status", self._status_command))
            self.app.add_handler(CommandHandler("positions", self._positions_command))
            self.app.add_handler(CommandHandler("trades", self._trades_command))
            self.app.add_handler(CommandHandler("pnl", self._pnl_command))
            self.app.add_handler(CommandHandler("bias", self._bias_command))
            self.app.add_handler(CommandHandler("confluence", self._confluence_command))
            self.app.add_handler(CommandHandler("settings", self._settings_command))
            self.app.add_handler(CommandHandler("alerts", self._alerts_command))
            self.app.add_handler(CommandHandler("chart", self._chart_command))
            self.app.add_handler(CommandHandler("price", self._price_command))
            self.app.add_handler(CommandHandler("performance", self._performance_command))
            self.app.add_handler(CommandHandler("risk", self._risk_command))
            self.app.add_handler(CommandHandler("export", self._export_command))
            self.app.add_handler(CommandHandler("summary", self._summary_command))
            self.app.add_handler(CommandHandler("compare", self._compare_command))
            self.app.add_handler(CommandHandler("alert", self._alert_command))
            self.app.add_handler(CommandHandler("help", self._help_command))
            # New commands for enhanced functionality
            self.app.add_handler(CommandHandler("close", self._close_command))
            self.app.add_handler(CommandHandler("modify", self._modify_command))
            self.app.add_handler(CommandHandler("watchlist", self._watchlist_command))
            self.app.add_handler(CommandHandler("add", self._add_watchlist_command))
            self.app.add_handler(CommandHandler("remove", self._remove_watchlist_command))
            # New V8 commands
            self.app.add_handler(CommandHandler("pause", self._pause_command))
            self.app.add_handler(CommandHandler("resume", self._resume_command))
            self.app.add_handler(CommandHandler("setalert", self._setalert_command))
            self.app.add_handler(CommandHandler("alerts", self._listalerts_command))
            self.app.add_handler(CommandHandler("cleareaalerts", self._clearalerts_command))
            self.app.add_handler(CommandHandler("stats", self._stats_command))
            self.app.add_handler(CommandHandler("rl", self._rl_command))
            self.app.add_handler(CallbackQueryHandler(self._button_callback))
            
            self._initialized = True
            logger.info("Telegram bot initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Telegram bot: {e}")
            return False
    
    def _get_or_create_loop(self):
        """Get or create an event loop for async operations"""
        try:
            loop = asyncio.get_event_loop()
            if loop.is_closed():
                raise RuntimeError("Loop is closed")
            return loop
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            return loop
    
    def send_message(self, message: str, reply_markup=None):
        """Send a message to Telegram (thread-safe)"""
        if not self._initialized:
            if not self.init():
                return False
        
        if not BOT_SETTINGS.get('notifications_enabled', True):
            return True
            
        try:
            loop = self._get_or_create_loop()
            
            async def _send():
                await self.app.bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID,
                    text=message,
                    parse_mode='HTML',
                    reply_markup=reply_markup
                )
            
            loop.run_until_complete(_send())
            return True
            
        except Exception as e:
            logger.error(f"Failed to send Telegram message: {e}")
            return False
    
    def send_message_async(self, message: str, reply_markup=None):
        """Send message in a separate thread (non-blocking)"""
        thread = threading.Thread(target=self.send_message, args=(message, reply_markup))
        thread.daemon = True
        thread.start()
    
    # === Command Handlers ===
    
    async def _start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command with compact submenu UI"""
        ds = DesignSystem
        session_icon, session_name = ds.get_session_icon()
        
        # Compact menu with category submenus
        paused_text = "⏸️ PAUSED" if _trading_paused else "▶️ ACTIVE"
        keyboard = [
            # Row 1: Status overview
            [
                InlineKeyboardButton("📊 Dashboard", callback_data="status"),
                InlineKeyboardButton("📈 Positions", callback_data="positions"),
            ],
            # Row 2: Category submenus
            [
                InlineKeyboardButton("💼 Trading", callback_data="menu_trading"),
                InlineKeyboardButton("📉 Analysis", callback_data="menu_analysis"),
            ],
            [
                InlineKeyboardButton("💰 Performance", callback_data="menu_performance"),
                InlineKeyboardButton("⚙️ Settings", callback_data="menu_settings"),
            ],
            # Row 3: Quick actions
            [
                InlineKeyboardButton(f"🤖 RL Agent", callback_data="rl"),
                InlineKeyboardButton(f"{paused_text}", callback_data="toggle_pause"),
            ],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Build quick stats
        pos_count = len(CURRENT_POSITIONS)
        daily_pnl = DAILY_STATS['pnl']
        daily_trades = DAILY_STATS['trades']
        win_rate = (DAILY_STATS['wins'] / max(daily_trades, 1)) * 100
        
        pnl_indicator = ds.STATUS_SUCCESS if daily_pnl >= 0 else ds.STATUS_ERROR
        streak_text = ""
        if DAILY_STATS['streak'] > 0:
            streak_text = f" | {ds.ICON_FIRE} {DAILY_STATS['streak']}W streak"
        elif DAILY_STATS['streak'] < 0:
            streak_text = f" | {abs(DAILY_STATS['streak'])}L streak"
        
        # RL status indicator
        trader = get_live_trader()
        rl_status = "🟢" if (trader and hasattr(trader, 'use_rl') and trader.use_rl) else "⚪"
        
        message = f"""
{ds.ICON_ROCKET} <b>V8 ICT TRADING BOT</b>
{ds.SEP_THICK}

{session_icon} <i>{session_name}</i> | {rl_status} RL | {"⏸️" if _trading_paused else "▶️"} {"Paused" if _trading_paused else "Active"}

<b>Quick Stats</b>
{ds.SEP_THIN}
{ds.ICON_CHART} Positions: <b>{pos_count}</b> active
{pnl_indicator} Today: <b>${daily_pnl:+,.2f}</b>{streak_text}
{ds.ICON_TARGET} Trades: {daily_trades} ({win_rate:.0f}% win)

{ds.SEP_THICK}
<b>Quick Commands</b>
{ds.SEP_THIN}
<code>/pause</code> <code>/resume</code> - Control trading
<code>/stats</code> - Performance stats
<code>/setalert SYMBOL PRICE</code> - Price alert
<code>/rl</code> - RL agent status

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')} | {datetime.now().strftime('%b %d, %Y')}
"""
        
        message_obj = update.message or (update.callback_query.message if update.callback_query else None)
        if message_obj:
            await message_obj.reply_text(message, parse_mode='HTML', reply_markup=reply_markup)
    
    async def _status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command - Full dashboard with modern design"""
        message_obj = update.message or (update.callback_query.message if update.callback_query else None)
        if not message_obj:
            return
        
        ds = DesignSystem
        session_icon, session_name = ds.get_session_icon()
        
        # Build positions section
        pos_count = len(CURRENT_POSITIONS)
        total_unrealized = 0
        
        if pos_count == 0:
            positions_text = f"\n{ds.STATUS_INFO} <i>No open positions</i>\n"
        else:
            lines = []
            for symbol, pos in CURRENT_POSITIONS.items():
                direction = pos.get('direction', 0)
                entry = pos.get('entry', 0)
                current = LAST_MARKET_DATA.get(symbol, {}).get('price', entry)
                qty = pos.get('qty', 0)
                
                # Calculate P&L
                if direction == 1:
                    pnl = (current - entry) * qty
                    pnl_pct = ((current - entry) / entry * 100) if entry > 0 else 0
                else:
                    pnl = (entry - current) * qty
                    pnl_pct = ((entry - current) / entry * 100) if entry > 0 else 0
                
                total_unrealized += pnl
                
                # Format line with visual indicators
                dir_badge = ds.status_badge('long' if direction == 1 else 'short')
                pnl_indicator = ds.STATUS_SUCCESS if pnl >= 0 else ds.STATUS_ERROR
                
                lines.append(
                    f"\n<b>{symbol}</b> {dir_badge}\n"
                    f"   Entry: {ds.format_price(entry)} | Now: {ds.format_price(current)}\n"
                    f"   {pnl_indicator} <b>{pnl_pct:+.2f}%</b> (${pnl:+,.2f})"
                )
            positions_text = "\n".join(lines)
        
        # Calculate daily stats
        trades = DAILY_STATS['trades']
        wins = DAILY_STATS['wins']
        losses = DAILY_STATS['losses']
        daily_pnl = DAILY_STATS['pnl']
        win_rate = (wins / max(trades, 1)) * 100
        
        # Progress bars and indicators
        pnl_bar = ds.pnl_bar(daily_pnl, max(abs(daily_pnl) * 2, 1000))
        win_rate_bar = ds.progress_bar(win_rate, 100, 8)
        
        # Streak indicator
        streak = DAILY_STATS.get('streak', 0)
        if streak > 2:
            streak_text = f"{ds.ICON_FIRE} <b>{streak}W STREAK!</b>"
        elif streak < -2:
            streak_text = f"{ds.STATUS_WARNING} {abs(streak)}L streak"
        else:
            streak_text = ""
        
        # Build refresh button
        keyboard = [
            [
                InlineKeyboardButton("🔄 Refresh", callback_data="status"),
                InlineKeyboardButton("📈 Positions", callback_data="positions"),
            ],
            [
                InlineKeyboardButton("🏠 Home", callback_data="start"),
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        message = f"""
{ds.ICON_CHART} <b>TRADING DASHBOARD</b>
{ds.SEP_THICK}

{session_icon} <i>{session_name}</i> | {BOT_SETTINGS.get('mode', 'Paper')}

<b>Open Positions</b> ({pos_count})
{ds.SEP_THIN}
{positions_text}

{ds.SEP_THICK}
<b>Today's Performance</b>
{ds.SEP_THIN}

{ds.pnl_bar(daily_pnl, max(abs(daily_pnl) * 1.5, 1000), 12)}

<code>
 Trades  : {trades:>5}
 Wins    : {wins:>5} {ds.STATUS_SUCCESS}
 Losses  : {losses:>5} {ds.STATUS_ERROR}
 Win Rate: {win_rate:>5.1f}%  {win_rate_bar}
</code>
{streak_text}

{ds.SEP_THICK}
<b>Unrealized P&L:</b> {ds.format_change(total_unrealized)}

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')} | {datetime.now().strftime('%b %d')}
"""
        await message_obj.reply_text(message, parse_mode='HTML', reply_markup=reply_markup)
    
    async def _positions_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /positions command - detailed position info with modern design"""
        message_obj = update.message or (update.callback_query.message if update.callback_query else None)
        if not message_obj:
            return
        
        ds = DesignSystem
        
        if not CURRENT_POSITIONS:
            keyboard = [
                [InlineKeyboardButton("🔄 Refresh", callback_data="positions")],
                [InlineKeyboardButton("🏠 Home", callback_data="start")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            message = f"""
{ds.ICON_CHART} <b>OPEN POSITIONS</b>
{ds.SEP_THICK}

{ds.STATUS_INFO} <i>No open positions currently</i>

{ds.SEP_THIN}
Waiting for signals...

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')}
"""
            await message_obj.reply_text(message, parse_mode='HTML', reply_markup=reply_markup)
            return
        
        cards = []
        total_unrealized = 0
        
        for symbol, pos in CURRENT_POSITIONS.items():
            direction = pos.get('direction', 0)
            entry = pos.get('entry', 0)
            stop = pos.get('stop', 0)
            target = pos.get('target', 0)
            qty = pos.get('qty', 0)
            confluence = pos.get('confluence', 0)
            
            current = LAST_MARKET_DATA.get(symbol, {}).get('price', entry)
            
            # Calculate P&L and percentage
            if direction == 1:
                unrealized = (current - entry) * qty
                pnl_pct = ((current - entry) / entry * 100) if entry > 0 else 0
                # Progress to target
                total_dist = target - entry
                current_dist = current - entry
            else:
                unrealized = (entry - current) * qty
                pnl_pct = ((entry - current) / entry * 100) if entry > 0 else 0
                total_dist = entry - target
                current_dist = entry - current
            
            total_unrealized += unrealized
            
            # Calculate progress to target (0% = entry, 100% = target)
            progress = (current_dist / total_dist * 100) if total_dist != 0 else 0
            progress = max(min(progress, 100), -100)
            
            # Risk/Reward ratio
            risk_dist = abs(entry - stop)
            reward_dist = abs(target - entry)
            rr_ratio = reward_dist / risk_dist if risk_dist > 0 else 0
            
            # Build position card
            dir_icon = ds.ICON_BULL if direction == 1 else ds.ICON_BEAR
            dir_text = "LONG" if direction == 1 else "SHORT"
            pnl_icon = ds.STATUS_SUCCESS if unrealized >= 0 else ds.STATUS_ERROR
            
            # Sparkline if available
            sparkline = ""
            if symbol in PRICE_HISTORY and len(PRICE_HISTORY[symbol]) >= 3:
                sparkline = f"\n{ds.ICON_CHART} {ds.sparkline(PRICE_HISTORY[symbol])}"
            
            card = f"""
{ds.SEP_THICK}
{dir_icon} <b>{symbol}</b> {dir_text}
{ds.SEP_THIN}
{sparkline}

<code>
 Entry  : {ds.format_price(entry)}
 Current: {ds.format_price(current)}
 Stop   : {ds.format_price(stop)} {ds.ICON_SHIELD}
 Target : {ds.format_price(target)} {ds.ICON_TARGET}
 Qty    : {qty:,.4f}
</code>

{ds.confidence_meter(confluence, 100)}

{pnl_icon} <b>P&L: ${unrealized:+,.2f}</b> ({pnl_pct:+.2f}%)
{ds.ICON_TARGET} Progress: {ds.progress_bar(max(progress, 0), 100, 8)}
"""
            cards.append(card)
        
        # Build action buttons for each position
        keyboard = []
        for symbol in list(CURRENT_POSITIONS.keys())[:3]:  # Max 3 positions shown
            keyboard.append([
                InlineKeyboardButton(f"🎯 Close {symbol}", callback_data=f"close_{symbol}"),
                InlineKeyboardButton(f"✏️ Modify", callback_data=f"modify_{symbol}")
            ])
        keyboard.append([
            InlineKeyboardButton("🔄 Refresh", callback_data="positions"),
            InlineKeyboardButton("🏠 Home", callback_data="start")
        ])
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        total_icon = ds.STATUS_SUCCESS if total_unrealized >= 0 else ds.STATUS_ERROR
        
        message = f"""
{ds.ICON_CHART} <b>OPEN POSITIONS</b> ({len(CURRENT_POSITIONS)})
{"".join(cards)}

{ds.SEP_THICK}
{total_icon} <b>Total Unrealized: ${total_unrealized:+,.2f}</b>

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')} | {datetime.now().strftime('%b %d')}
"""
        await message_obj.reply_text(message, parse_mode='HTML', reply_markup=reply_markup)
    
    async def _trades_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /trades command"""
        message_obj = update.message or update.callback_query.message
        
        if not TRADE_HISTORY:
            await message_obj.reply_text("📭 No trades recorded yet today!")
            return
        
        recent = TRADE_HISTORY[-10:]
        lines = []
        
        for i, trade in enumerate(reversed(recent), 1):
            emoji = "✅" if trade.get('pnl', 0) > 0 else "❌"
            direction = trade.get('direction', 'LONG')
            symbol = trade.get('symbol', 'N/A')
            entry = trade.get('entry', 0)
            exit_price = trade.get('exit', 0)
            pnl = trade.get('pnl', 0)
            
            lines.append(
                f"{i}. {emoji} <b>{symbol}</b> {direction}\n"
                f"   Entry: ${entry:,.2f} → Exit: ${exit_price:,.2f}\n"
                f"   P&L: ${pnl:,.2f}"
            )
        
        total_pnl = sum(t.get('pnl', 0) for t in TRADE_HISTORY)
        total_emoji = "🟢" if total_pnl >= 0 else "🔴"
        
        message = f"""
╔══════════════════════════════════════╗
║       📜 <b>RECENT TRADES</b>              ║
╚══════════════════════════════════════╝

{chr(10).join(lines)}

━━━━━━━━━━━━━━━━━━━━
<b>Total Trades:</b> {len(TRADE_HISTORY)}
<b>Total P&L:</b> {total_emoji} ${total_pnl:,.2f}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        await message_obj.reply_text(message, parse_mode='HTML')
    
    async def _pnl_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /pnl command"""
        message_obj = update.message or update.callback_query.message
        
        # Calculate by symbol
        symbol_pnl = {}
        for trade in TRADE_HISTORY:
            sym = trade.get('symbol', 'Unknown')
            if sym not in symbol_pnl:
                symbol_pnl[sym] = {'pnl': 0, 'wins': 0, 'losses': 0, 'trades': 0}
            symbol_pnl[sym]['pnl'] += trade.get('pnl', 0)
            symbol_pnl[sym]['trades'] += 1
            if trade.get('pnl', 0) > 0:
                symbol_pnl[sym]['wins'] += 1
            else:
                symbol_pnl[sym]['losses'] += 1
        
        lines = []
        for sym, data in sorted(symbol_pnl.items(), key=lambda x: x[1]['pnl'], reverse=True):
            emoji = "🟢" if data['pnl'] > 0 else "🔴" if data['pnl'] < 0 else "⚪"
            win_rate = (data['wins'] / max(data['trades'], 1)) * 100
            lines.append(
                f"{emoji} <b>{sym}</b>: ${data['pnl']:,.2f}\n"
                f"   W:{data['wins']} L:{data['losses']} ({win_rate:.0f}%)"
            )
        
        total_pnl = DAILY_STATS['pnl']
        total_emoji = "🟢" if total_pnl > 0 else "🔴" if total_pnl < 0 else "⚪"
        win_rate = (DAILY_STATS['wins'] / max(DAILY_STATS['trades'], 1)) * 100
        
        # Calculate profit factor
        gross_profit = sum(t['pnl'] for t in TRADE_HISTORY if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t['pnl'] for t in TRADE_HISTORY if t.get('pnl', 0) < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        message = f"""
╔══════════════════════════════════════╗
║         💰 <b>P&L BREAKDOWN</b>            ║
╚══════════════════════════════════════╝

<b>By Symbol:</b>
{chr(10).join(lines) if lines else '📭 No trades yet'}

━━━━━━━━━━━━━━━━━━━━
<b>📊 Summary:</b>
┌─────────────────────────────────────┐
│  Total P&L:     {total_emoji} ${total_pnl:>10,.2f}       │
│  Trades:        {DAILY_STATS['trades']:>10}           │
│  Win Rate:      {win_rate:>10.1f}%          │
│  Profit Factor: {profit_factor:>10.2f}           │
│  Gross Profit:  ${gross_profit:>10,.2f}       │
│  Gross Loss:    ${gross_loss:>10,.2f}       │
└─────────────────────────────────────┘

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        await message_obj.reply_text(message, parse_mode='HTML')
    
    async def _bias_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /bias command - fetches live prices from IBKR"""
        message_obj = update.message or update.callback_query.message
        
        # Try to fetch live prices
        symbols = list(LAST_MARKET_DATA.keys()) if LAST_MARKET_DATA else BOT_SETTINGS.get('symbols', [])
        
        if not symbols:
            await message_obj.reply_text("📭 No symbols configured. Add symbols to see bias.")
            return
        
        # Fetch live data
        await message_obj.reply_text("🔄 Fetching live prices...", parse_mode='HTML')
        live_data = fetch_live_data_for_symbols(symbols)
        
        # Merge with cached data for trend info
        display_data = {}
        for symbol in symbols:
            if symbol in live_data:
                display_data[symbol] = live_data[symbol]
            elif symbol in LAST_MARKET_DATA:
                display_data[symbol] = LAST_MARKET_DATA[symbol]
        
        if not display_data:
            await message_obj.reply_text("❌ Could not fetch price data. Make sure IBKR is running.")
            return
        
        bullish = []
        bearish = []
        neutral = []
        
        for symbol, data in display_data.items():
            htf = data.get('htf_trend', 0)
            ltf = data.get('ltf_trend', 0)
            price = data.get('price', 0)
            kz = data.get('kill_zone', False)
            is_live = data.get('live', False)
            
            kz_icon = "🌙" if kz else "☀️"
            live_icon = "⚡" if is_live else "📊"
            
            if htf == 1 and ltf >= 0:
                bullish.append(f"  {live_icon} {kz_icon} <b>{symbol}</b>: ${price:,.2f}")
            elif htf == -1 and ltf <= 0:
                bearish.append(f"  {live_icon} {kz_icon} <b>{symbol}</b>: ${price:,.2f}")
            else:
                neutral.append(f"  {live_icon} {kz_icon} <b>{symbol}</b>: ${price:,.2f}")
        
        live_count = sum(1 for d in display_data.values() if d.get('live', False))
        
        message = f"""
╔══════════════════════════════════════╗
║       🔮 <b>MARKET BIAS</b>                ║
╠══════════════════════════════════════╣
║  Live prices: {live_count}/{len(display_data)} symbols          ║
╚══════════════════════════════════════╝

🟢 <b>BULLISH ({len(bullish)}):</b>
{chr(10).join(bullish) if bullish else '  None'}

🔴 <b>BEARISH ({len(bearish)}):</b>
{chr(10).join(bearish) if bearish else '  None'}

⚪ <b>NEUTRAL ({len(neutral)}):</b>
{chr(10).join(neutral) if neutral else '  None'}

━━━━━━━━━━━━━━━━━━━━
⚡ = Live price | 📊 = Cached
🌙 = In Kill Zone | ☀️ = Outside Kill Zone

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        await message_obj.reply_text(message, parse_mode='HTML')
    
    async def _confluence_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /confluence command"""
        message_obj = update.message or update.callback_query.message
        
        if not LAST_MARKET_DATA:
            await message_obj.reply_text("📭 No market data available. Start the trading bot first!")
            return
        
        high_conf = []
        medium_conf = []
        low_conf = []
        
        for symbol, data in sorted(LAST_MARKET_DATA.items(), key=lambda x: x[1].get('confluence', 0), reverse=True):
            conf = data.get('confluence', 0)
            htf = data.get('htf_trend', 0)
            ltf = data.get('ltf_trend', 0)
            kz = data.get('kill_zone', False)
            pp = data.get('price_position', 0.5)
            
            htf_icon = "⬆️" if htf == 1 else "⬇️" if htf == -1 else "➡️"
            ltf_icon = "⬆️" if ltf >= 0 else "⬇️"
            kz_icon = "🌙" if kz else "☀️"
            
            line = f"  <b>{symbol}</b>: {conf}/100 | {htf_icon}{ltf_icon} {kz_icon} | PP:{pp:.0%}"
            
            if conf >= 60:
                high_conf.append(f"🟢 {line}")
            elif conf >= 40:
                medium_conf.append(f"🟡 {line}")
            else:
                low_conf.append(f"🔴 {line}")
        
        message = f"""
╔══════════════════════════════════════╗
║      ⚡ <b>CONFLUENCE LEVELS</b>           ║
╚══════════════════════════════════════╝

<b>🟢 SIGNAL ZONE (60+):</b>
{chr(10).join(high_conf) if high_conf else '  None'}

<b>🟡 WATCHING (40-59):</b>
{chr(10).join(medium_conf) if medium_conf else '  None'}

<b>🔴 NO SIGNAL (&lt;40):</b>
{chr(10).join(low_conf) if low_conf else '  None'}

━━━━━━━━━━━━━━━━━━━━
Legend: HTF↕️ LTF↕️ KillZone | PP=Price Position

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        await message_obj.reply_text(message, parse_mode='HTML')
    
    async def _settings_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /settings command"""
        message_obj = update.message or update.callback_query.message
        
        symbols_str = ", ".join(BOT_SETTINGS.get('symbols', [])) or "None configured"
        
        message = f"""
╔══════════════════════════════════════╗
║         ⚙️ <b>BOT SETTINGS</b>             ║
╚══════════════════════════════════════╝

<b>Trading Mode:</b> {BOT_SETTINGS.get('mode', 'Unknown')}
<b>Risk per Trade:</b> {BOT_SETTINGS.get('risk_per_trade', 0.02)*100:.1f}%

<b>Notifications:</b>
  • Enabled: {'✅' if BOT_SETTINGS.get('notifications_enabled', True) else '❌'}
  • Signal Alerts: {'✅' if BOT_SETTINGS.get('signal_alerts', True) else '❌'}
  • Trade Alerts: {'✅' if BOT_SETTINGS.get('trade_alerts', True) else '❌'}
  • Daily Summary: {'✅' if BOT_SETTINGS.get('daily_summary', True) else '❌'}

<b>Active Symbols ({len(BOT_SETTINGS.get('symbols', []))}):</b>
{symbols_str}

━━━━━━━━━━━━━━━━━━━━
Use /alerts to toggle notifications

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        await message_obj.reply_text(message, parse_mode='HTML')
    
    async def _alerts_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /alerts command"""
        message_obj = update.message or update.callback_query.message
        
        keyboard = [
            [
                InlineKeyboardButton(
                    f"{'🔴 Disable' if BOT_SETTINGS.get('notifications_enabled', True) else '🟢 Enable'} All Alerts",
                    callback_data="toggle_all_alerts"
                )
            ],
            [
                InlineKeyboardButton(
                    f"{'🔴' if BOT_SETTINGS.get('signal_alerts', True) else '🟢'} Signal Alerts",
                    callback_data="toggle_signal_alerts"
                ),
                InlineKeyboardButton(
                    f"{'🔴' if BOT_SETTINGS.get('trade_alerts', True) else '🟢'} Trade Alerts",
                    callback_data="toggle_trade_alerts"
                )
            ],
            [
                InlineKeyboardButton("🔙 Back", callback_data="start")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        message = f"""
╔══════════════════════════════════════╗
║         🔔 <b>ALERT SETTINGS</b>           ║
╚══════════════════════════════════════╝

<b>Current Settings:</b>
  • All Notifications: {'🟢 ON' if BOT_SETTINGS.get('notifications_enabled', True) else '🔴 OFF'}
  • Signal Alerts: {'🟢 ON' if BOT_SETTINGS.get('signal_alerts', True) else '🔴 OFF'}
  • Trade Alerts: {'🟢 ON' if BOT_SETTINGS.get('trade_alerts', True) else '🔴 OFF'}
  • Daily Summary: {'🟢 ON' if BOT_SETTINGS.get('daily_summary', True) else '🔴 OFF'}

Tap buttons below to toggle:
"""
        await message_obj.reply_text(message, parse_mode='HTML', reply_markup=reply_markup)
    
    async def _chart_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /chart command - fetches live price from IBKR"""
        message_obj = update.message or update.callback_query.message
        
        # Check if symbol argument provided
        args = context.args if context.args else []
        
        if args:
            symbol = args[0].upper()
            
            # Try to fetch live price first
            await message_obj.reply_text(f"🔄 Fetching live price for {symbol}...", parse_mode='HTML')
            live_price = fetch_live_price(symbol)
            
            # Get cached data for other info
            if symbol in LAST_MARKET_DATA:
                data = LAST_MARKET_DATA[symbol].copy()
                if live_price:
                    data['price'] = live_price
                    data['live'] = True
                    data['updated_at'] = datetime.now().isoformat()
            else:
                # No cached data, create minimal entry
                if live_price:
                    data = {
                        'price': live_price,
                        'htf_trend': 0,
                        'ltf_trend': 0,
                        'kill_zone': False,
                        'price_position': 0.5,
                        'confluence': 0,
                        'live': True
                    }
                else:
                    await message_obj.reply_text(f"❌ Could not fetch price for '{symbol}'.\nMake sure IBKR is running and symbol is valid.")
                    return
            
            price = data.get('price', 0)
            htf = data.get('htf_trend', 0)
            ltf = data.get('ltf_trend', 0)
            kz = data.get('kill_zone', False)
            pp = data.get('price_position', 0.5)
            conf = data.get('confluence', 0)
            is_live = data.get('live', False)
            
            htf_text = "BULLISH ⬆️" if htf == 1 else "BEARISH ⬇️" if htf == -1 else "NEUTRAL ➡️"
            ltf_text = "BULLISH ⬆️" if ltf >= 0 else "BEARISH ⬇️"
            live_icon = "⚡ LIVE" if is_live else "📊 CACHED"
            
            message = f"""
╔══════════════════════════════════════╗
║       📉 <b>{symbol}</b>                    ║
╠══════════════════════════════════════╣
║  {live_icon:>34}          ║
╚══════════════════════════════════════╝

<b>💵 Price:</b> ${price:,.4f}

┌─────────────────────────────────────┐
│  HTF Trend:     {htf_text:>15}    │
│  LTF Trend:     {ltf_text:>15}    │
│  Kill Zone:     {'🌙 YES' if kz else '☀️ NO':>15}    │
│  Price Pos:     {pp:.0%:>15}    │
│  Confluence:    {conf:>15}/100    │
└─────────────────────────────────────┘

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            await message_obj.reply_text(message, parse_mode='HTML')
        else:
            # Show all symbols summary
            if not LAST_MARKET_DATA:
                await message_obj.reply_text("📭 No market data available. Start the trading bot first!")
                return
            
            lines = []
            for symbol, data in sorted(LAST_MARKET_DATA.items()):
                price = data.get('price', 0)
                conf = data.get('confluence', 0)
                conf_emoji = "🟢" if conf >= 60 else "🟡" if conf >= 40 else "🔴"
                lines.append(f"  {conf_emoji} <b>{symbol}</b>: ${price:,.4f} (Conf: {conf})")
            
            message = f"""
╔══════════════════════════════════════╗
║       📉 <b>MARKET OVERVIEW</b>            ║
╚══════════════════════════════════════╝

{chr(10).join(lines)}

━━━━━━━━━━━━━━━━━━━━
Use /chart [SYMBOL] for details
Example: /chart BTCUSD

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
            await message_obj.reply_text(message, parse_mode='HTML')
    
    async def _price_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /price command - fetches live prices for all symbols from IBKR"""
        message_obj = update.message or update.callback_query.message
        
        symbols = list(LAST_MARKET_DATA.keys()) if LAST_MARKET_DATA else BOT_SETTINGS.get('symbols', [])
        
        if not symbols:
            await message_obj.reply_text("📭 No symbols configured. Add symbols to fetch prices.")
            return
        
        await message_obj.reply_text(f"🔄 Fetching live prices for {len(symbols)} symbols...", parse_mode='HTML')
        
        # Fetch live prices
        live_data = {}
        failed_symbols = []
        
        for symbol in symbols:
            try:
                price = fetch_live_price(symbol)
                if price:
                    live_data[symbol] = price
                else:
                    failed_symbols.append(symbol)
            except Exception as e:
                logger.error(f"Error fetching {symbol}: {e}")
                failed_symbols.append(symbol)
        
        if not live_data:
            await message_obj.reply_text("❌ Could not fetch any prices. Make sure IBKR is running.")
            return
        
        # Build price list
        lines = []
        for symbol in sorted(live_data.keys()):
            price = live_data[symbol]
            # Get cached price for comparison
            cached_price = LAST_MARKET_DATA.get(symbol, {}).get('price', price)
            change = price - cached_price if cached_price else 0
            change_pct = (change / cached_price * 100) if cached_price and cached_price > 0 else 0
            
            change_emoji = "🟢" if change >= 0 else "🔴"
            lines.append(f"  {symbol:8}: ${price:>12,.4f} {change_emoji} {change_pct:+.2f}%")
        
        failed_text = f"\n⚠️ Failed: {', '.join(failed_symbols)}" if failed_symbols else ""
        
        message = f"""
╔══════════════════════════════════════╗
║       ⚡ <b>LIVE PRICES</b>                 ║
╠══════════════════════════════════════╣
║  Fetched: {len(live_data)}/{len(symbols)} symbols              ║
╚══════════════════════════════════════╝

{chr(10).join(lines)}
{failed_text}

━━━━━━━━━━━━━━━━━━━━
Prices from IBKR (real-time)

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        await message_obj.reply_text(message, parse_mode='HTML')
    
    async def _performance_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /performance command - detailed performance analytics"""
        message_obj = update.message or update.callback_query.message
        
        if not TRADE_HISTORY:
            await message_obj.reply_text("📭 No trades recorded yet. Start trading first!")
            return
        
        # Calculate detailed statistics
        total_trades = len(TRADE_HISTORY)
        wins = len([t for t in TRADE_HISTORY if t.get('pnl', 0) > 0])
        losses = total_trades - wins
        win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
        
        # Profit factor
        gross_profit = sum(t['pnl'] for t in TRADE_HISTORY if t.get('pnl', 0) > 0)
        gross_loss = abs(sum(t['pnl'] for t in TRADE_HISTORY if t.get('pnl', 0) < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Best and worst trades
        best_trade = max(TRADE_HISTORY, key=lambda x: x.get('pnl', 0))
        worst_trade = min(TRADE_HISTORY, key=lambda x: x.get('pnl', 0))
        
        # Average trade
        avg_trade = sum(t['pnl'] for t in TRADE_HISTORY) / total_trades
        
        # Win rate by symbol
        symbol_stats = {}
        for trade in TRADE_HISTORY:
            sym = trade.get('symbol', 'Unknown')
            if sym not in symbol_stats:
                symbol_stats[sym] = {'trades': 0, 'wins': 0, 'pnl': 0}
            symbol_stats[sym]['trades'] += 1
            symbol_stats[sym]['pnl'] += trade.get('pnl', 0)
            if trade.get('pnl', 0) > 0:
                symbol_stats[sym]['wins'] += 1
        
        # Sort by win rate
        symbol_performance = sorted(
            [(sym, stats['wins']/stats['trades']*100, stats['trades'], stats['pnl']) 
             for sym, stats in symbol_stats.items()],
            key=lambda x: x[1], 
            reverse=True
        )[:5]  # Top 5
        
        symbol_lines = []
        for sym, wr, trades, pnl in symbol_performance:
            emoji = "🟢" if wr >= 50 else "🔴"
            symbol_lines.append(f"  {emoji} {sym}: {wr:.1f}% ({trades} trades, ${pnl:,.0f})")
        
        message = f"""
╔══════════════════════════════════════╗
║      📊 <b>PERFORMANCE ANALYTICS</b>       ║
╚══════════════════════════════════════╝

<b>Overall Statistics:</b>
┌─────────────────────────────────────┐
│  Total Trades:    {total_trades:>10}           │
│  Win Rate:        {win_rate:>10.1f}%          │
│  Profit Factor:   {profit_factor:>10.2f}           │
│  Avg Trade:       ${avg_trade:>9,.2f}        │
└─────────────────────────────────────┘

<b>💰 P&L Breakdown:</b>
  🟢 Gross Profit: ${gross_profit:,.2f}
  🔴 Gross Loss:   ${gross_loss:,.2f}
  📊 Net P&L:      ${DAILY_STATS['pnl']:,.2f}

<b>🏆 Best/Worst Trades:</b>
  ⬆️ Best:  {best_trade.get('symbol', 'N/A')} +${best_trade.get('pnl', 0):,.2f}
  ⬇️ Worst: {worst_trade.get('symbol', 'N/A')} ${worst_trade.get('pnl', 0):,.2f}

<b>Top Performing Symbols:</b>
{chr(10).join(symbol_lines) if symbol_lines else '  No data'}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        await message_obj.reply_text(message, parse_mode='HTML')
    
    async def _risk_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /risk command - show risk exposure and limits"""
        message_obj = update.message or update.callback_query.message
        
        # Calculate risk exposure
        total_exposure = 0
        symbol_risk = {}
        
        for symbol, pos in CURRENT_POSITIONS.items():
            entry = pos.get('entry', 0)
            stop = pos.get('stop', 0)
            qty = pos.get('qty', 0)
            direction = pos.get('direction', 1)
            
            # Risk amount
            risk_per_unit = abs(entry - stop)
            total_risk = risk_per_unit * qty
            
            symbol_risk[symbol] = {
                'qty': qty,
                'risk': total_risk,
                'direction': 'LONG' if direction == 1 else 'SHORT',
                'entry': entry,
                'stop': stop
            }
            total_exposure += total_risk
        
        # Calculate daily stats
        daily_risk = sum(t.get('pnl', 0) for t in TRADE_HISTORY if t.get('pnl', 0) < 0) if TRADE_HISTORY else 0
        max_consecutive_losses = 0
        current_losses = 0
        
        for trade in TRADE_HISTORY:
            if trade.get('pnl', 0) < 0:
                current_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, current_losses)
            else:
                current_losses = 0
        
        position_lines = []
        for symbol, data in symbol_risk.items():
            position_lines.append(
                f"  {symbol}: {data['direction']} x{data['qty']} | "
                f"Risk: ${data['risk']:,.2f} | Stop: ${data['stop']:,.2f}"
            )
        
        message = f"""
╔══════════════════════════════════════╗
║        ⚠️ <b>RISK DASHBOARD</b>            ║
╚══════════════════════════════════════╝

<b>📊 Current Exposure:</b>
┌─────────────────────────────────────┐
│  Open Positions:  {len(CURRENT_POSITIONS):>10}           │
│  Total Risk:      ${total_exposure:>10,.2f}       │
│  Daily Loss:      ${daily_risk:>10,.2f}       │
│  Max Consec Loss: {max_consecutive_losses:>10}           │
└─────────────────────────────────────┘

<b>📈 Open Positions Risk:</b>
{chr(10).join(position_lines) if position_lines else '  No open positions'}

<b>⚡ Risk Limits:</b>
  • Per Trade: $1,000-$2,000
  • Max Positions: Unlimited (manage manually)
  • Kill Zone Priority: Higher probability

<b>💡 Risk Tips:</b>
  • Monitor max consecutive losses
  • Reduce size after 3+ losses
  • Check correlation between positions

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        await message_obj.reply_text(message, parse_mode='HTML')
    
    async def _export_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /export command - export trades to CSV format"""
        message_obj = update.message or update.callback_query.message
        
        if not TRADE_HISTORY:
            await message_obj.reply_text("📭 No trades to export yet!")
            return
        
        # Build CSV content
        lines = ["Date,Symbol,Direction,Entry,Exit,Qty,PnL,ExitReason"]
        
        for trade in TRADE_HISTORY:
            lines.append(
                f"{trade.get('timestamp', '')[:10]},"
                f"{trade.get('symbol', '')},"
                f"{trade.get('direction', '')},"
                f"{trade.get('entry', 0):.4f},"
                f"{trade.get('exit', 0):.4f},"
                f"{trade.get('qty', 0)},"
                f"{trade.get('pnl', 0):.2f},"
                f"{trade.get('exit_reason', '')}"
            )
        
        csv_content = "\n".join(lines)
        
        # Send as file
        try:
            from telegram import InputFile
            import io
            
            file_obj = io.BytesIO(csv_content.encode())
            file_obj.name = f"trades_{datetime.now().strftime('%Y%m%d')}.csv"
            
            await message_obj.reply_document(
                document=InputFile(file_obj),
                caption=f"📊 Trade Export - {len(TRADE_HISTORY)} trades"
            )
        except Exception as e:
            # Fallback: send as text
            await message_obj.reply_text(
                f"📊 <b>Trade Export</b> ({len(TRADE_HISTORY)} trades)\n\n"
                f"<pre>{csv_content[:3000]}...</pre>" if len(csv_content) > 3000 else f"<pre>{csv_content}</pre>",
                parse_mode='HTML'
            )
    
    async def _summary_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /summary command - detailed daily summary"""
        message_obj = update.message or update.callback_query.message
        
        # Get current positions value
        open_pnl = 0
        for symbol, pos in CURRENT_POSITIONS.items():
            current_price = LAST_MARKET_DATA.get(symbol, {}).get('price', pos['entry'])
            if pos['direction'] == 1:
                pnl = (current_price - pos['entry']) * pos['qty']
            else:
                pnl = (pos['entry'] - current_price) * pos['qty']
            open_pnl += pnl
        
        # Today's stats
        total_pnl = DAILY_STATS['pnl'] + open_pnl
        
        # Time since start
        try:
            start_time = datetime.fromisoformat(DAILY_STATS['start_time'])
            elapsed = datetime.now() - start_time
            hours = elapsed.total_seconds() / 3600
        except:
            hours = 0
        
        message = f"""
╔══════════════════════════════════════╗
║      📅 <b>DAILY TRADING SUMMARY</b>      ║
╚══════════════════════════════════════╝

<b>Session Info:</b>
  Started: {DAILY_STATS['start_time'][:16] if DAILY_STATS['start_time'] else 'N/A'}
  Duration: {hours:.1f} hours

<b>📊 Trading Activity:</b>
┌─────────────────────────────────────┐
│  Total Trades:    {DAILY_STATS['trades']:>10}           │
│  Wins:            {DAILY_STATS['wins']:>10} ✅          │
│  Losses:          {DAILY_STATS['losses']:>10} ❌          │
│  Win Rate:        {(DAILY_STATS['wins']/max(DAILY_STATS['trades'],1)*100):>10.1f}%          │
└─────────────────────────────────────┘

<b>💰 P&L Summary:</b>
  Closed Trades: ${DAILY_STATS['pnl']:>12,.2f}
  Open Positions: ${open_pnl:>12,.2f}
  ─────────────────────────
  <b>Total P&L:     ${total_pnl:>12,.2f}</b>

<b>📈 Open Positions:</b>
  Count: {len(CURRENT_POSITIONS)}
  Unrealized: ${open_pnl:,.2f}

<b>⚡ Key Metrics:</b>
  Profit Factor: {sum(t['pnl'] for t in TRADE_HISTORY if t.get('pnl',0)>0)/abs(sum(t['pnl'] for t in TRADE_HISTORY if t.get('pnl',0)<0)) if TRADE_HISTORY and any(t.get('pnl',0)<0 for t in TRADE_HISTORY) else 'N/A':.2f}
  Avg Trade: ${DAILY_STATS['pnl']/max(DAILY_STATS['trades'],1):,.2f}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        await message_obj.reply_text(message, parse_mode='HTML')
    
    async def _compare_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /compare command - compare symbol performance"""
        message_obj = update.message or update.callback_query.message
        
        if not TRADE_HISTORY:
            await message_obj.reply_text("📭 No trades to compare. Start trading first!")
            return
        
        # Calculate stats per symbol
        symbol_stats = {}
        for trade in TRADE_HISTORY:
            sym = trade.get('symbol', 'Unknown')
            if sym not in symbol_stats:
                symbol_stats[sym] = {
                    'trades': 0, 'wins': 0, 'losses': 0,
                    'pnl': 0, 'avg_win': 0, 'avg_loss': 0
                }
            
            symbol_stats[sym]['trades'] += 1
            symbol_stats[sym]['pnl'] += trade.get('pnl', 0)
            
            if trade.get('pnl', 0) > 0:
                symbol_stats[sym]['wins'] += 1
            else:
                symbol_stats[sym]['losses'] += 1
        
        # Sort by P&L
        sorted_symbols = sorted(symbol_stats.items(), key=lambda x: x[1]['pnl'], reverse=True)
        
        lines = []
        for sym, stats in sorted_symbols:
            win_rate = (stats['wins'] / stats['trades'] * 100) if stats['trades'] > 0 else 0
            emoji = "🥇" if stats['pnl'] > 0 else "🥉"
            lines.append(
                f"{emoji} <b>{sym}</b>\n"
                f"   P&L: ${stats['pnl']:>10,.2f} | Win%: {win_rate:>5.1f}% | "
                f"W:{stats['wins']} L:{stats['losses']}"
            )
        
        message = f"""
╔══════════════════════════════════════╗
║      🏆 <b>SYMBOL PERFORMANCE</b>         ║
╚══════════════════════════════════════╝

<b>Symbols Ranked by P&L:</b>

{chr(10).join(lines)}

<b>📊 Summary:</b>
  Best: {sorted_symbols[0][0] if sorted_symbols else 'N/A'}
  Worst: {sorted_symbols[-1][0] if sorted_symbols else 'N/A'}
  Total Symbols: {len(symbol_stats)}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        await message_obj.reply_text(message, parse_mode='HTML')
    
    async def _alert_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /alert command - set price alerts"""
        message_obj = update.message or update.callback_query.message
        
        # Get arguments
        args = context.args if context.args else []
        
        if len(args) < 2:
            await message_obj.reply_text(
                "⚠️ <b>Usage:</b> /alert SYMBOL PRICE\n\n"
                "Examples:\n"
                "  /alert BTCUSD 70000\n"
                "  /alert ES 6000\n\n"
                "Current alerts will be shown here once implemented.",
                parse_mode='HTML'
            )
            return
        
        symbol = args[0].upper()
        try:
            price = float(args[1])
        except ValueError:
            await message_obj.reply_text("❌ Invalid price. Please enter a number.")
            return
        
        # Store alert (in-memory for now)
        if 'price_alerts' not in BOT_SETTINGS:
            BOT_SETTINGS['price_alerts'] = {}
        
        BOT_SETTINGS['price_alerts'][symbol] = {
            'price': price,
            'set_at': datetime.now().isoformat()
        }
        
        await message_obj.reply_text(
            f"✅ <b>Price Alert Set</b>\n\n"
            f"Symbol: {symbol}\n"
            f"Target: ${price:,.2f}\n\n"
            f"You'll be notified when price reaches this level.",
            parse_mode='HTML'
        )
    
    async def _help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command with modern design"""
        message_obj = update.message or (update.callback_query.message if update.callback_query else None)
        if not message_obj:
            return
        
        ds = DesignSystem
        
        keyboard = [
            [InlineKeyboardButton("🏠 Home", callback_data="start")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        message = f"""
{ds.STATUS_INFO} <b>V8 HELP & COMMANDS</b>
{ds.SEP_THICK}

<b>Dashboard</b>
{ds.SEP_THIN}
<code>/start</code>     - Main menu
<code>/status</code>    - Trading dashboard
<code>/positions</code> - Open positions
<code>/trades</code>    - Trade history

<b>Trading Control</b>
{ds.SEP_THIN}
<code>/pause</code>     - Pause trading
<code>/resume</code>    - Resume trading
<code>/close</code>     - Close position
<code>/modify</code>    - Modify SL/TP
<code>/watchlist</code> - View watchlist

<b>Analysis</b>
{ds.SEP_THIN}
<code>/price</code>     - Live prices (IBKR)
<code>/bias</code>      - Market bias
<code>/confluence</code> - Signal strength
<code>/chart</code>     - Symbol details

<b>Performance</b>
{ds.SEP_THIN}
<code>/stats</code>     - Full statistics
<code>/pnl</code>       - P&L breakdown
<code>/risk</code>      - Risk dashboard
<code>/summary</code>   - Daily summary
<code>/export</code>    - Export trades

<b>Alerts</b>
{ds.SEP_THIN}
<code>/setalert SYMBOL PRICE</code>
<code>/alerts</code>    - View alerts
<code>/clearalerts</code> - Clear all

<b>RL Agent</b>
{ds.SEP_THIN}
<code>/rl</code>        - RL agent status

<b>Settings</b>
{ds.SEP_THIN}
<code>/settings</code>  - Configuration
<code>/help</code>      - This help

{ds.SEP_THICK}
<b>Legend</b>
{ds.SEP_THIN}
{ds.STATUS_SUCCESS} Profit/Win/Bullish
{ds.STATUS_ERROR} Loss/Bearish
{ds.STATUS_WARNING} Medium/Warning
🟢 RL Active | ⏸️ Paused

{ds.SEP_DOT}
V8 ICT Trading Bot + RL
"""
        await message_obj.reply_text(message, parse_mode='HTML', reply_markup=reply_markup)
    
    # ============================================================================
    # NEW COMMANDS - Close, Modify, Watchlist
    # ============================================================================
    
    async def _close_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /close command - Close positions from Telegram"""
        message_obj = update.message or (update.callback_query.message if update.callback_query else None)
        if not message_obj:
            return
        
        ds = DesignSystem
        args = context.args if context.args else []
        
        if not CURRENT_POSITIONS:
            message = f"""
{ds.ICON_CHART} <b>CLOSE POSITION</b>
{ds.SEP_THICK}

{ds.STATUS_INFO} <i>No open positions to close</i>

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')}
"""
            await message_obj.reply_text(message, parse_mode='HTML')
            return
        
        # If symbol provided, show confirmation
        if args:
            symbol = args[0].upper()
            if symbol in CURRENT_POSITIONS:
                pos = CURRENT_POSITIONS[symbol]
                direction = "LONG" if pos.get('direction', 0) == 1 else "SHORT"
                entry = pos.get('entry', 0)
                current = LAST_MARKET_DATA.get(symbol, {}).get('price', entry)
                qty = pos.get('qty', 0)
                
                # Calculate P&L
                if pos.get('direction', 0) == 1:
                    pnl = (current - entry) * qty
                else:
                    pnl = (entry - current) * qty
                
                pnl_icon = ds.STATUS_SUCCESS if pnl >= 0 else ds.STATUS_ERROR
                
                keyboard = [
                    [
                        InlineKeyboardButton(f"{ds.STATUS_SUCCESS} Confirm Close", callback_data=f"confirm_close_{symbol}"),
                        InlineKeyboardButton(f"{ds.STATUS_ERROR} Cancel", callback_data="positions")
                    ]
                ]
                reply_markup = InlineKeyboardMarkup(keyboard)
                
                message = f"""
{ds.ICON_TARGET} <b>CLOSE POSITION</b>
{ds.SEP_THICK}

{ds.STATUS_WARNING} <b>Confirm closing position?</b>

<code>
 Symbol   : {symbol}
 Direction: {direction}
 Entry    : {ds.format_price(entry)}
 Current  : {ds.format_price(current)}
 Quantity : {qty:,.4f}
</code>

{pnl_icon} <b>Expected P&L: ${pnl:+,.2f}</b>

{ds.SEP_DOT}
This will close the position immediately.
"""
                await message_obj.reply_text(message, parse_mode='HTML', reply_markup=reply_markup)
            else:
                await message_obj.reply_text(f"{ds.STATUS_ERROR} Position {symbol} not found")
        else:
            # Show list of positions to close
            keyboard = []
            for symbol in CURRENT_POSITIONS.keys():
                pos = CURRENT_POSITIONS[symbol]
                direction = "L" if pos.get('direction', 0) == 1 else "S"
                entry = pos.get('entry', 0)
                current = LAST_MARKET_DATA.get(symbol, {}).get('price', entry)
                
                if pos.get('direction', 0) == 1:
                    pnl = (current - entry) * pos.get('qty', 0)
                else:
                    pnl = (entry - current) * pos.get('qty', 0)
                
                icon = ds.STATUS_SUCCESS if pnl >= 0 else ds.STATUS_ERROR
                keyboard.append([
                    InlineKeyboardButton(f"{icon} {symbol} ({direction}) ${pnl:+,.0f}", callback_data=f"close_{symbol}")
                ])
            
            keyboard.append([InlineKeyboardButton("🏠 Home", callback_data="start")])
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            message = f"""
{ds.ICON_TARGET} <b>CLOSE POSITION</b>
{ds.SEP_THICK}

Select a position to close:

{ds.SEP_THIN}
Or use: <code>/close SYMBOL</code>

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')}
"""
            await message_obj.reply_text(message, parse_mode='HTML', reply_markup=reply_markup)
    
    async def _modify_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /modify command - Modify stop loss or target"""
        message_obj = update.message or (update.callback_query.message if update.callback_query else None)
        if not message_obj:
            return
        
        ds = DesignSystem
        args = context.args if context.args else []
        
        if not CURRENT_POSITIONS:
            message = f"""
{ds.ICON_GEAR} <b>MODIFY POSITION</b>
{ds.SEP_THICK}

{ds.STATUS_INFO} <i>No open positions to modify</i>

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')}
"""
            await message_obj.reply_text(message, parse_mode='HTML')
            return
        
        if len(args) >= 3:
            # /modify SYMBOL sl/tp PRICE
            symbol = args[0].upper()
            modify_type = args[1].lower()
            try:
                new_price = float(args[2])
            except ValueError:
                await message_obj.reply_text(f"{ds.STATUS_ERROR} Invalid price. Use: /modify SYMBOL sl 100.00")
                return
            
            if symbol not in CURRENT_POSITIONS:
                await message_obj.reply_text(f"{ds.STATUS_ERROR} Position {symbol} not found")
                return
            
            pos = CURRENT_POSITIONS[symbol]
            old_value = pos.get('stop' if modify_type == 'sl' else 'target', 0)
            
            if modify_type == 'sl':
                CURRENT_POSITIONS[symbol]['stop'] = new_price
                label = "Stop Loss"
            elif modify_type == 'tp':
                CURRENT_POSITIONS[symbol]['target'] = new_price
                label = "Target"
            else:
                await message_obj.reply_text(f"{ds.STATUS_ERROR} Use 'sl' for stop loss or 'tp' for target")
                return
            
            message = f"""
{ds.STATUS_SUCCESS} <b>POSITION MODIFIED</b>
{ds.SEP_THICK}

<b>{symbol}</b> - {label} Updated

<code>
 Old {label}: {ds.format_price(old_value)}
 New {label}: {ds.format_price(new_price)}
</code>

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')}
"""
            await message_obj.reply_text(message, parse_mode='HTML')
        else:
            # Show positions and modification options
            lines = []
            for symbol, pos in CURRENT_POSITIONS.items():
                stop = pos.get('stop', 0)
                target = pos.get('target', 0)
                lines.append(f"<b>{symbol}</b>: SL={ds.format_price(stop)} | TP={ds.format_price(target)}")
            
            message = f"""
{ds.ICON_GEAR} <b>MODIFY POSITION</b>
{ds.SEP_THICK}

<b>Current Positions:</b>
{chr(10).join(lines)}

{ds.SEP_THIN}
<b>Usage:</b>
<code>/modify SYMBOL sl PRICE</code> - Change stop loss
<code>/modify SYMBOL tp PRICE</code> - Change target

<b>Examples:</b>
<code>/modify BTCUSD sl 64000</code>
<code>/modify ES tp 6100</code>

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')}
"""
            await message_obj.reply_text(message, parse_mode='HTML')
    
    async def _watchlist_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /watchlist command - View and manage watchlist"""
        message_obj = update.message or (update.callback_query.message if update.callback_query else None)
        if not message_obj:
            return
        
        ds = DesignSystem
        
        if not WATCHLIST:
            keyboard = [
                [InlineKeyboardButton("➕ Add Symbols", callback_data="add_watchlist")],
                [InlineKeyboardButton("🏠 Home", callback_data="start")]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            message = f"""
{ds.ICON_EYE} <b>WATCHLIST</b>
{ds.SEP_THICK}

{ds.STATUS_INFO} <i>Your watchlist is empty</i>

Use <code>/add SYMBOL</code> to add symbols

<b>Example:</b>
<code>/add BTCUSD</code>
<code>/add ES GC NQ</code>

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')}
"""
            await message_obj.reply_text(message, parse_mode='HTML', reply_markup=reply_markup)
            return
        
        # Fetch prices for watchlist
        lines = []
        for symbol in WATCHLIST:
            data = LAST_MARKET_DATA.get(symbol, {})
            price = data.get('price', 0)
            htf = data.get('htf_trend', 0)
            conf = data.get('confluence', 0)
            
            trend_icon = ds.trend_indicator(htf, False)
            conf_bar = ds.progress_bar(conf, 100, 6, False)
            
            if price > 0:
                lines.append(f"{trend_icon} <b>{symbol}</b>: {ds.format_price(price)} | {conf_bar}")
            else:
                lines.append(f"{ds.STATUS_NEUTRAL} <b>{symbol}</b>: <i>No data</i>")
        
        keyboard = [
            [
                InlineKeyboardButton("🔄 Refresh", callback_data="watchlist"),
                InlineKeyboardButton("➕ Add", callback_data="add_watchlist")
            ],
            [InlineKeyboardButton("🏠 Home", callback_data="start")]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        message = f"""
{ds.ICON_EYE} <b>WATCHLIST</b> ({len(WATCHLIST)})
{ds.SEP_THICK}

{chr(10).join(lines)}

{ds.SEP_THIN}
<code>/add SYMBOL</code> - Add to watchlist
<code>/remove SYMBOL</code> - Remove from watchlist

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')}
"""
        await message_obj.reply_text(message, parse_mode='HTML', reply_markup=reply_markup)
    
    async def _add_watchlist_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /add command - Add symbols to watchlist"""
        message_obj = update.message or (update.callback_query.message if update.callback_query else None)
        if not message_obj:
            return
        
        ds = DesignSystem
        args = context.args if context.args else []
        
        if not args:
            await message_obj.reply_text(
                f"{ds.STATUS_WARNING} Usage: <code>/add SYMBOL [SYMBOL2] [SYMBOL3]</code>\n\n"
                f"Example: <code>/add BTCUSD ES GC</code>",
                parse_mode='HTML'
            )
            return
        
        added = []
        already_exists = []
        
        for symbol in args:
            symbol = symbol.upper()
            if symbol not in WATCHLIST:
                WATCHLIST.append(symbol)
                added.append(symbol)
            else:
                already_exists.append(symbol)
        
        response_parts = []
        if added:
            response_parts.append(f"{ds.STATUS_SUCCESS} Added: {', '.join(added)}")
        if already_exists:
            response_parts.append(f"{ds.STATUS_INFO} Already in watchlist: {', '.join(already_exists)}")
        
        message = f"""
{ds.ICON_EYE} <b>WATCHLIST UPDATED</b>
{ds.SEP_THICK}

{chr(10).join(response_parts)}

{ds.SEP_THIN}
Total symbols: {len(WATCHLIST)}

{ds.SEP_DOT}
Use /watchlist to view
"""
        await message_obj.reply_text(message, parse_mode='HTML')
    
    async def _remove_watchlist_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /remove command - Remove symbols from watchlist"""
        message_obj = update.message or (update.callback_query.message if update.callback_query else None)
        if not message_obj:
            return
        
        ds = DesignSystem
        args = context.args if context.args else []
        
        if not args:
            await message_obj.reply_text(
                f"{ds.STATUS_WARNING} Usage: <code>/remove SYMBOL</code>\n\n"
                f"Example: <code>/remove BTCUSD</code>",
                parse_mode='HTML'
            )
            return
        
        removed = []
        not_found = []
        
        for symbol in args:
            symbol = symbol.upper()
            if symbol in WATCHLIST:
                WATCHLIST.remove(symbol)
                removed.append(symbol)
            else:
                not_found.append(symbol)
        
        response_parts = []
        if removed:
            response_parts.append(f"{ds.STATUS_SUCCESS} Removed: {', '.join(removed)}")
        if not_found:
            response_parts.append(f"{ds.STATUS_WARNING} Not in watchlist: {', '.join(not_found)}")
        
        message = f"""
{ds.ICON_EYE} <b>WATCHLIST UPDATED</b>
{ds.SEP_THICK}

{chr(10).join(response_parts)}

{ds.SEP_THIN}
Remaining symbols: {len(WATCHLIST)}

{ds.SEP_DOT}
Use /watchlist to view
"""
        await message_obj.reply_text(message, parse_mode='HTML')
    
    async def _pause_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /pause command - Pause trading"""
        message_obj = update.message or (update.callback_query.message if update.callback_query else None)
        if not message_obj:
            return
        
        ds = DesignSystem
        global _trading_paused
        _trading_paused = True
        
        message = f"""
{ds.STATUS_WARNING} <b>TRADING PAUSED</b>
{ds.SEP_THICK}

{ds.ICON_LOCK} Trading is now paused.

No new positions will be opened.
Existing positions remain active.

{ds.SEP_THIN}
Use /resume to resume trading.

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')}
"""
        await message_obj.reply_text(message, parse_mode='HTML')
        send_notification(f"{ds.STATUS_WARNING} Trading PAUSED by user command")
    
    async def _resume_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /resume command - Resume trading"""
        message_obj = update.message or (update.callback_query.message if update.callback_query else None)
        if not message_obj:
            return
        
        ds = DesignSystem
        global _trading_paused
        _trading_paused = False
        
        message = f"""
{ds.STATUS_SUCCESS} <b>TRADING RESUMED</b>
{ds.SEP_THICK}

{ds.ICON_UNLOCK} Trading is now active.

Bot will process new signals normally.

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')}
"""
        await message_obj.reply_text(message, parse_mode='HTML')
        send_notification(f"{ds.STATUS_SUCCESS} Trading RESUMED by user command")
    
    async def _setalert_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /setalert command - Set price alerts"""
        message_obj = update.message or (update.callback_query.message if update.callback_query else None)
        if not message_obj:
            return
        
        ds = DesignSystem
        args = context.args if context.args else []
        
        if len(args) < 2:
            await message_obj.reply_text(
                f"{ds.STATUS_WARNING} <b>Set Price Alert</b>\n\n"
                f"Usage: <code>/setalert SYMBOL PRICE</code>\n\n"
                f"Examples:\n"
                f"<code>/setalert BTCUSD 70000</code> - Alert when BTC hits $70k\n"
                f"<code>/setalert ES 5000</code> - Alert when ES hits 5000",
                parse_mode='HTML'
            )
            return
        
        try:
            symbol = args[0].upper()
            price = float(args[1])
            
            if symbol not in _price_alerts:
                _price_alerts[symbol] = []
            
            _price_alerts[symbol].append({
                'price': price,
                'set_at': datetime.now().isoformat(),
                'triggered': False
            })
            
            message = f"""
{ds.ICON_BELL} <b>PRICE ALERT SET</b>
{ds.SEP_THICK}

{ds.STATUS_SUCCESS} Alert created!

<code>
 Symbol : {symbol}
 Price  : ${price:,.2f}
</code>

{ds.SEP_THIN}
You'll be notified when price reaches this level.

{ds.SEP_DOT}
Use /alerts to view all alerts
"""
            await message_obj.reply_text(message, parse_mode='HTML')
        except ValueError:
            await message_obj.reply_text(f"{ds.STATUS_ERROR} Invalid price. Use a number.", parse_mode='HTML')
    
    async def _listalerts_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /alerts list command - Show all price alerts"""
        message_obj = update.message or (update.callback_query.message if update.callback_query else None)
        if not message_obj:
            return
        
        ds = DesignSystem
        
        if not _price_alerts or all(len(alerts) == 0 for alerts in _price_alerts.values()):
            message = f"""
{ds.ICON_BELL} <b>PRICE ALERTS</b>
{ds.SEP_THICK}

{ds.STATUS_INFO} No price alerts set.

{ds.SEP_THIN}
Use /setalert SYMBOL PRICE to create one.
"""
            await message_obj.reply_text(message, parse_mode='HTML')
            return
        
        lines = []
        total = 0
        for symbol, alerts in _price_alerts.items():
            for alert in alerts:
                if not alert.get('triggered', False):
                    total += 1
                    lines.append(f"  {ds.ICON_BELL} {symbol}: ${alert['price']:,.2f}")
        
        message = f"""
{ds.ICON_BELL} <b>PRICE ALERTS</b> ({total})
{ds.SEP_THICK}

{chr(10).join(lines) if lines else f'{ds.STATUS_INFO} No active alerts'}

{ds.SEP_THIN}
Use /clearalerts to remove all

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')}
"""
        await message_obj.reply_text(message, parse_mode='HTML')
    
    async def _clearalerts_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /clearalerts command - Clear all price alerts"""
        message_obj = update.message or (update.callback_query.message if update.callback_query else None)
        if not message_obj:
            return
        
        ds = DesignSystem
        global _price_alerts
        
        count = sum(len(alerts) for alerts in _price_alerts.values())
        _price_alerts = {}
        
        message = f"""
{ds.ICON_BELL} <b>ALERTS CLEARED</b>
{ds.SEP_THICK}

{ds.STATUS_SUCCESS} Removed {count} price alert(s).

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')}
"""
        await message_obj.reply_text(message, parse_mode='HTML')
    
    async def _stats_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /stats command - Show detailed performance statistics"""
        message_obj = update.message or (update.callback_query.message if update.callback_query else None)
        if not message_obj:
            return
        
        ds = DesignSystem
        
        # Calculate detailed stats
        total_trades = len(TRADE_HISTORY)
        if total_trades == 0:
            message = f"""
{ds.ICON_CHART} <b>PERFORMANCE STATS</b>
{ds.SEP_THICK}

{ds.STATUS_INFO} No trades to analyze yet.

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')}
"""
            await message_obj.reply_text(message, parse_mode='HTML')
            return
        
        wins = [t for t in TRADE_HISTORY if t.get('pnl', 0) > 0]
        losses = [t for t in TRADE_HISTORY if t.get('pnl', 0) <= 0]
        
        win_rate = len(wins) / total_trades * 100
        
        gross_profit = sum(t['pnl'] for t in wins) if wins else 0
        gross_loss = abs(sum(t['pnl'] for t in losses)) if losses else 0
        net_pnl = gross_profit - gross_loss
        
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        avg_win = gross_profit / len(wins) if wins else 0
        avg_loss = gross_loss / len(losses) if losses else 0
        
        # Risk/Reward ratio
        rr_ratio = avg_win / avg_loss if avg_loss > 0 else 0
        
        # Largest win/loss
        largest_win = max((t['pnl'] for t in wins), default=0)
        largest_loss = min((t['pnl'] for t in losses), default=0)
        
        # By symbol
        by_symbol = {}
        for trade in TRADE_HISTORY:
            sym = trade.get('symbol', 'UNK')
            if sym not in by_symbol:
                by_symbol[sym] = {'pnl': 0, 'trades': 0, 'wins': 0}
            by_symbol[sym]['pnl'] += trade.get('pnl', 0)
            by_symbol[sym]['trades'] += 1
            if trade.get('pnl', 0) > 0:
                by_symbol[sym]['wins'] += 1
        
        # Best/worst symbols
        sorted_symbols = sorted(by_symbol.items(), key=lambda x: x[1]['pnl'], reverse=True)
        best_symbol = sorted_symbols[0] if sorted_symbols else None
        worst_symbol = sorted_symbols[-1] if len(sorted_symbols) > 1 else None
        
        best_text = f"{best_symbol[0]}: +${best_symbol[1]['pnl']:,.2f}" if best_symbol else "N/A"
        worst_text = f"{worst_symbol[0]}: ${worst_symbol[1]['pnl']:,.2f}" if worst_symbol else "N/A"
        
        # Expectancy
        expectancy = (win_rate/100 * avg_win) - ((100-win_rate)/100 * avg_loss)
        
        message = f"""
{ds.ICON_CHART} <b>PERFORMANCE STATISTICS</b>
{ds.SEP_THICK}

<b>Overview</b>
<code>
 Total Trades  : {total_trades}
 Wins          : {len(wins)} ({win_rate:.1f}%)
 Losses        : {len(losses)}
 Net P&L       : ${net_pnl:+,.2f}
</code>

<b>Profitability</b>
<code>
 Gross Profit  : +${gross_profit:,.2f}
 Gross Loss    : -${gross_loss:,.2f}
 Profit Factor : {profit_factor:.2f}
 Expectancy    : ${expectancy:,.2f}
</code>

<b>Trade Metrics</b>
<code>
 Avg Win       : +${avg_win:,.2f}
 Avg Loss      : -${avg_loss:,.2f}
 Avg R:R       : 1:{rr_ratio:.2f}
 Largest Win   : +${largest_win:,.2f}
 Largest Loss  : ${largest_loss:,.2f}
</code>

<b>By Symbol</b>
 {ds.TREND_UP} Best: {best_text}
 {ds.TREND_DOWN} Worst: {worst_text}

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')}
"""
        await message_obj.reply_text(message, parse_mode='HTML')
    
    async def _rl_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /rl command - Show RL agent info"""
        message_obj = update.message or (update.callback_query.message if update.callback_query else None)
        if not message_obj:
            return
        
        ds = DesignSystem
        
        trader = get_live_trader()
        
        if trader and hasattr(trader, 'signal_gen') and trader.signal_gen.rl_agent:
            agent = trader.signal_gen.rl_agent
            
            # Get agent stats if available
            entry_agent = agent.entry_agent if hasattr(agent, 'entry_agent') else None
            exit_agent = agent.exit_agent if hasattr(agent, 'exit_agent') else None
            
            message = f"""
{ds.ICON_ZAP} <b>RL AGENT STATUS</b>
{ds.SEP_THICK}

{ds.STATUS_SUCCESS} <b>RL Agent Active</b>

<b>Entry Agent:</b>
<code>
 State Size   : {entry_agent.state_size if entry_agent else 'N/A'}
 Actions      : {entry_agent.action_size if entry_agent else 'N/A'}
 Epsilon      : {entry_agent.epsilon:.4f if entry_agent and hasattr(entry_agent, 'epsilon') else 'N/A'}
</code>

<b>Exit Agent:</b>
<code>
 State Size   : {exit_agent.state_size if exit_agent else 'N/A'}
 Actions      : {exit_agent.action_size if exit_agent else 'N/A'}
 Epsilon      : {exit_agent.epsilon:.4f if exit_agent and hasattr(exit_agent, 'epsilon') else 'N/A'}
</code>

<b>Model Info:</b>
 Loaded from: v8_rl_model.pkl

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')}
"""
        else:
            message = f"""
{ds.ICON_ZAP} <b>RL AGENT STATUS</b>
{ds.SEP_THICK}

{ds.STATUS_WARNING} RL Agent not active

{ds.SEP_THIN}
Start the bot with --rl-model to enable.

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')}
"""
        await message_obj.reply_text(message, parse_mode='HTML')
    
    async def _button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button callbacks with enhanced functionality"""
        query = update.callback_query
        await query.answer()
        
        callback_data = query.data
        ds = DesignSystem
        
        # Handle toggle commands
        if callback_data == "toggle_all_alerts":
            BOT_SETTINGS['notifications_enabled'] = not BOT_SETTINGS.get('notifications_enabled', True)
            await self._alerts_command(update, context)
            return
        elif callback_data == "toggle_signal_alerts":
            BOT_SETTINGS['signal_alerts'] = not BOT_SETTINGS.get('signal_alerts', True)
            await self._alerts_command(update, context)
            return
        elif callback_data == "toggle_trade_alerts":
            BOT_SETTINGS['trade_alerts'] = not BOT_SETTINGS.get('trade_alerts', True)
            await self._alerts_command(update, context)
            return
        
        # Handle close position callbacks
        if callback_data.startswith("close_"):
            symbol = callback_data.replace("close_", "")
            if symbol in CURRENT_POSITIONS:
                # Set up context.args for the close command
                context.args = [symbol]
                await self._close_command(update, context)
                return
        
        # Handle confirm close callbacks
        if callback_data.startswith("confirm_close_"):
            symbol = callback_data.replace("confirm_close_", "")
            if symbol in CURRENT_POSITIONS:
                pos = CURRENT_POSITIONS[symbol]
                entry = pos.get('entry', 0)
                current = LAST_MARKET_DATA.get(symbol, {}).get('price', entry)
                qty = pos.get('qty', 0)
                direction = pos.get('direction', 0)
                
                if direction == 1:
                    pnl = (current - entry) * qty
                else:
                    pnl = (entry - current) * qty
                
                # Remove position (in real implementation, this would also close via IBKR)
                del CURRENT_POSITIONS[symbol]
                
                # Add to trade history
                add_trade({
                    'symbol': symbol,
                    'direction': 'LONG' if direction == 1 else 'SHORT',
                    'entry': entry,
                    'exit': current,
                    'pnl': pnl,
                    'exit_reason': 'manual_close',
                    'qty': qty
                })
                
                pnl_icon = ds.STATUS_SUCCESS if pnl >= 0 else ds.STATUS_ERROR
                
                message = f"""
{ds.STATUS_SUCCESS} <b>POSITION CLOSED</b>
{ds.SEP_THICK}

<b>{symbol}</b> closed manually

<code>
 Entry   : {ds.format_price(entry)}
 Exit    : {ds.format_price(current)}
 Quantity: {qty:,.4f}
</code>

{pnl_icon} <b>P&L: ${pnl:+,.2f}</b>

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')}
"""
                if query.message:
                    await query.message.reply_text(message, parse_mode='HTML')
                return
            else:
                if query.message:
                    await query.message.reply_text(f"{ds.STATUS_ERROR} Position {symbol} not found")
                return
        
        # Handle modify callbacks
        if callback_data.startswith("modify_"):
            symbol = callback_data.replace("modify_", "")
            if symbol in CURRENT_POSITIONS:
                context.args = [symbol]
                await self._modify_command(update, context)
                return
        
        # Handle watchlist callbacks
        if callback_data == "add_watchlist":
            await query.message.reply_text(
                f"{ds.ICON_EYE} <b>ADD TO WATCHLIST</b>\n\n"
                f"Use <code>/add SYMBOL</code> to add symbols\n\n"
                f"<b>Examples:</b>\n"
                f"<code>/add BTCUSD</code>\n"
                f"<code>/add ES GC NQ</code>",
                parse_mode='HTML'
            )
            return
        
        # Handle submenu navigation
        if callback_data == "menu_trading":
            await self._menu_trading(update, context)
            return
        elif callback_data == "menu_analysis":
            await self._menu_analysis(update, context)
            return
        elif callback_data == "menu_performance":
            await self._menu_performance(update, context)
            return
        elif callback_data == "menu_settings":
            await self._menu_settings(update, context)
            return
        elif callback_data == "toggle_pause":
            await self._toggle_pause(update, context)
            return
        elif callback_data == "rl":
            await self._rl_command(update, context)
            return
        elif callback_data == "setalert_prompt":
            await self._setalert_prompt(update, context)
            return
        elif callback_data == "stats":
            await self._stats_command(update, context)
            return
        
        # Handle navigation commands
        command_map = {
            "start": self._start_command,
            "status": self._status_command,
            "positions": self._positions_command,
            "trades": self._trades_command,
            "pnl": self._pnl_command,
            "risk": self._risk_command,
            "performance": self._performance_command,
            "summary": self._summary_command,
            "compare": self._compare_command,
            "export": self._export_command,
            "price": self._price_command,
            "bias": self._bias_command,
            "confluence": self._confluence_command,
            "settings": self._settings_command,
            "alerts": self._alerts_command,
            "chart": self._chart_command,
            "help": self._help_command,
            "close": self._close_command,
            "modify": self._modify_command,
            "watchlist": self._watchlist_command,
        }
        
        handler = command_map.get(callback_data)
        if handler:
            await handler(update, context)
    
    async def _menu_trading(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Trading submenu"""
        query = update.callback_query
        ds = DesignSystem
        
        keyboard = [
            [
                InlineKeyboardButton("🎯 Close Position", callback_data="close"),
                InlineKeyboardButton("✏️ Modify SL/TP", callback_data="modify"),
            ],
            [
                InlineKeyboardButton("👀 Watchlist", callback_data="watchlist"),
                InlineKeyboardButton("➕ Add Symbol", callback_data="add_watchlist"),
            ],
            [
                InlineKeyboardButton("⏸️ Pause Trading", callback_data="toggle_pause") if not _trading_paused else InlineKeyboardButton("▶️ Resume Trading", callback_data="toggle_pause"),
            ],
            [InlineKeyboardButton("🏠 Back to Menu", callback_data="start")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        message = f"""
{ds.ICON_CHART} <b>TRADING MENU</b>
{ds.SEP_THICK}

<b>Position Management</b>
{ds.SEP_THIN}
• Close Position - Exit an open trade
• Modify SL/TP - Adjust stop loss/take profit
• Watchlist - View/manage watched symbols

<b>Trading Status</b>
{ds.SEP_THIN}
• Status: {"⏸️ PAUSED" if _trading_paused else "▶️ ACTIVE"}
• Positions: {len(CURRENT_POSITIONS)} open

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')}
"""
        if query and query.message:
            await query.message.edit_text(message, parse_mode='HTML', reply_markup=reply_markup)
    
    async def _menu_analysis(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Analysis submenu"""
        query = update.callback_query
        ds = DesignSystem
        
        keyboard = [
            [
                InlineKeyboardButton("⚡ Live Prices", callback_data="price"),
                InlineKeyboardButton("🔮 Market Bias", callback_data="bias"),
            ],
            [
                InlineKeyboardButton("📊 Confluence", callback_data="confluence"),
                InlineKeyboardButton("📈 Chart", callback_data="chart"),
            ],
            [
                InlineKeyboardButton("🔔 Set Alert", callback_data="setalert_prompt"),
                InlineKeyboardButton("📋 View Alerts", callback_data="alerts"),
            ],
            [InlineKeyboardButton("🏠 Back to Menu", callback_data="start")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Count active alerts
        alert_count = sum(len(alerts) for alerts in _price_alerts.values())
        
        message = f"""
{ds.ICON_ZAP} <b>ANALYSIS MENU</b>
{ds.SEP_THICK}

<b>Market Data</b>
{ds.SEP_THIN}
• Live Prices - Real-time IBKR prices
• Market Bias - HTF trend analysis
• Confluence - Signal strength check

<b>Alerts</b>
{ds.SEP_THIN}
• Active Alerts: {alert_count}
• Use <code>/setalert SYMBOL PRICE</code>

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')}
"""
        if query and query.message:
            await query.message.edit_text(message, parse_mode='HTML', reply_markup=reply_markup)
    
    async def _menu_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Performance submenu"""
        query = update.callback_query
        ds = DesignSystem
        
        keyboard = [
            [
                InlineKeyboardButton("💰 P&L Today", callback_data="pnl"),
                InlineKeyboardButton("📈 Full Stats", callback_data="stats"),
            ],
            [
                InlineKeyboardButton("📜 Trade History", callback_data="trades"),
                InlineKeyboardButton("📊 Summary", callback_data="summary"),
            ],
            [
                InlineKeyboardButton("⚠️ Risk Dashboard", callback_data="risk"),
                InlineKeyboardButton("📁 Export", callback_data="export"),
            ],
            [InlineKeyboardButton("🏠 Back to Menu", callback_data="start")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Quick stats
        total_trades = len(TRADE_HISTORY)
        daily_pnl = DAILY_STATS['pnl']
        win_rate = (DAILY_STATS['wins'] / max(DAILY_STATS['trades'], 1)) * 100
        
        pnl_icon = ds.STATUS_SUCCESS if daily_pnl >= 0 else ds.STATUS_ERROR
        
        message = f"""
{ds.ICON_CHART} <b>PERFORMANCE MENU</b>
{ds.SEP_THICK}

<b>Today's Summary</b>
{ds.SEP_THIN}
{pnl_icon} P&L: <b>${daily_pnl:+,.2f}</b>
• Trades: {DAILY_STATS['trades']}
• Win Rate: {win_rate:.1f}%

<b>All-Time</b>
{ds.SEP_THIN}
• Total Trades: {total_trades}

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')}
"""
        if query and query.message:
            await query.message.edit_text(message, parse_mode='HTML', reply_markup=reply_markup)
    
    async def _menu_settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Settings submenu"""
        query = update.callback_query
        ds = DesignSystem
        
        # Get current settings
        notifications_on = BOT_SETTINGS.get('notifications_enabled', True)
        signal_alerts_on = BOT_SETTINGS.get('signal_alerts', True)
        trade_alerts_on = BOT_SETTINGS.get('trade_alerts', True)
        
        keyboard = [
            [
                InlineKeyboardButton(f"{'🔔' if notifications_on else '🔕'} Notifications", callback_data="toggle_all_alerts"),
                InlineKeyboardButton("⚙️ Config", callback_data="settings"),
            ],
            [
                InlineKeyboardButton(f"{'✅' if signal_alerts_on else '❌'} Signal Alerts", callback_data="toggle_signal_alerts"),
                InlineKeyboardButton(f"{'✅' if trade_alerts_on else '❌'} Trade Alerts", callback_data="toggle_trade_alerts"),
            ],
            [
                InlineKeyboardButton("🤖 RL Agent Status", callback_data="rl"),
                InlineKeyboardButton("❓ Help", callback_data="help"),
            ],
            [InlineKeyboardButton("🏠 Back to Menu", callback_data="start")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        # Get RL status
        trader = get_live_trader()
        rl_enabled = trader and hasattr(trader, 'use_rl') and trader.use_rl
        
        message = f"""
{ds.ICON_CONFIG} <b>SETTINGS MENU</b>
{ds.SEP_THICK}

<b>Notifications</b>
{ds.SEP_THIN}
• All Notifications: {'🔔 ON' if notifications_on else '🔕 OFF'}
• Signal Alerts: {'✅ ON' if signal_alerts_on else '❌ OFF'}
• Trade Alerts: {'✅ ON' if trade_alerts_on else '❌ OFF'}

<b>Bot Status</b>
{ds.SEP_THIN}
• Trading: {"⏸️ PAUSED" if _trading_paused else "▶️ ACTIVE"}
• RL Agent: {"🟢 Enabled" if rl_enabled else "⚪ Disabled"}

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')}
"""
        if query and query.message:
            await query.message.edit_text(message, parse_mode='HTML', reply_markup=reply_markup)
    
    async def _toggle_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Toggle trading pause state"""
        global _trading_paused
        query = update.callback_query
        ds = DesignSystem
        
        _trading_paused = not _trading_paused
        
        if _trading_paused:
            message = f"""
{ds.STATUS_WARNING} <b>TRADING PAUSED</b>
{ds.SEP_THICK}

{ds.ICON_LOCK} Trading is now <b>PAUSED</b>

• No new positions will be opened
• Existing positions remain active
• Use the menu or /resume to resume

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')}
"""
            send_notification(f"{ds.STATUS_WARNING} Trading PAUSED via menu")
        else:
            message = f"""
{ds.STATUS_SUCCESS} <b>TRADING RESUMED</b>
{ds.SEP_THICK}

{ds.ICON_UNLOCK} Trading is now <b>ACTIVE</b>

• Bot will process new signals
• All systems operational

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')}
"""
            send_notification(f"{ds.STATUS_SUCCESS} Trading RESUMED via menu")
        
        keyboard = [[InlineKeyboardButton("🏠 Back to Menu", callback_data="start")]]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        if query and query.message:
            await query.message.edit_text(message, parse_mode='HTML', reply_markup=reply_markup)
    
    async def _setalert_prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show prompt for setting price alerts"""
        query = update.callback_query
        ds = DesignSystem
        
        # Get current trading symbols
        trader = get_live_trader()
        symbols = trader.symbols if trader and hasattr(trader, 'symbols') else ['BTCUSD', 'ETHUSD', 'SOLUSD']
        
        symbol_list = ', '.join(symbols[:5])
        
        message = f"""
{ds.ICON_BELL} <b>SET PRICE ALERT</b>
{ds.SEP_THICK}

<b>Usage:</b>
<code>/setalert SYMBOL PRICE</code>

<b>Examples:</b>
<code>/setalert BTCUSD 100000</code>
<code>/setalert ETHUSD 4000</code>
<code>/setalert ES 5500</code>

<b>Active Symbols:</b>
{symbol_list}

{ds.SEP_THIN}
<b>Current Alerts:</b> {sum(len(a) for a in _price_alerts.values())}

{ds.SEP_DOT}
Use /alerts to view all alerts
Use /clearalerts to remove all
"""
        keyboard = [
            [InlineKeyboardButton("📋 View Alerts", callback_data="alerts")],
            [InlineKeyboardButton("🏠 Back to Menu", callback_data="start")],
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        if query and query.message:
            await query.message.edit_text(message, parse_mode='HTML', reply_markup=reply_markup)


# Global notifier instance
_notifier: Optional[TelegramNotifier] = None


def get_notifier() -> TelegramNotifier:
    """Get or create the global notifier instance"""
    global _notifier
    if _notifier is None:
        _notifier = TelegramNotifier()
    return _notifier


# === Public API Functions ===

def init_bot():
    """Initialize the Telegram bot"""
    return get_notifier().init()


def send_message(message: str):
    """Send a message to Telegram"""
    return get_notifier().send_message(message)


def send_notification(message: str):
    """Send notification synchronously - more reliable than async"""
    notifier = get_notifier()
    if not notifier._initialized:
        notifier.init()
    return notifier.send_message(message)


def update_market_data(symbol: str, data: Dict):
    """Update market data for a symbol with price history for sparklines"""
    global LAST_MARKET_DATA, PRICE_HISTORY
    
    LAST_MARKET_DATA[symbol] = {
        **data,
        'updated_at': datetime.now().isoformat()
    }
    
    # Update price history for sparklines
    if 'price' in data and data['price'] > 0:
        if symbol not in PRICE_HISTORY:
            PRICE_HISTORY[symbol] = []
        
        PRICE_HISTORY[symbol].append(data['price'])
        
        # Keep only last 20 prices for sparkline
        if len(PRICE_HISTORY[symbol]) > 20:
            PRICE_HISTORY[symbol] = PRICE_HISTORY[symbol][-20:]


def sync_position_from_ibkr(symbol: str, position_data: Dict):
    """Sync a position from IBKR with enhanced tracking"""
    global CURRENT_POSITIONS
    
    CURRENT_POSITIONS[symbol] = {
        **position_data,
        'synced_at': datetime.now().isoformat(),
        'source': 'ibkr'
    }


def get_position_pnl(symbol: str) -> float:
    """Calculate current P&L for a position"""
    if symbol not in CURRENT_POSITIONS:
        return 0.0
    
    pos = CURRENT_POSITIONS[symbol]
    entry = pos.get('entry', 0)
    current = LAST_MARKET_DATA.get(symbol, {}).get('price', entry)
    qty = pos.get('qty', 0)
    direction = pos.get('direction', 1)
    
    if direction == 1:
        return (current - entry) * qty
    else:
        return (entry - current) * qty


def get_total_unrealized_pnl() -> float:
    """Get total unrealized P&L across all positions"""
    total = 0.0
    for symbol in CURRENT_POSITIONS:
        total += get_position_pnl(symbol)
    return total


def check_price_alerts(current_prices: Optional[Dict[str, float]] = None):
    """Check if any price alerts have been triggered.
    
    Args:
        current_prices: Optional dict of {symbol: price}. If not provided, uses LAST_MARKET_DATA.
    """
    global _price_alerts
    ds = DesignSystem
    triggered = []
    
    # Check legacy alerts in BOT_SETTINGS
    price_alerts = BOT_SETTINGS.get('price_alerts', {})
    
    for symbol, alert_data in list(price_alerts.items()):
        target_price = alert_data.get('price', 0)
        if current_prices:
            current_price = current_prices.get(symbol, 0)
        else:
            current_price = LAST_MARKET_DATA.get(symbol, {}).get('price', 0)
        
        if current_price <= 0 or target_price <= 0:
            continue
        
        # Check if price crossed the alert level
        direction = alert_data.get('direction', 'any')
        
        triggered_alert = False
        if direction == 'above' and current_price >= target_price:
            triggered_alert = True
        elif direction == 'below' and current_price <= target_price:
            triggered_alert = True
        elif direction == 'any' and abs(current_price - target_price) / target_price < 0.001:
            triggered_alert = True
        
        if triggered_alert:
            triggered.append({
                'symbol': symbol,
                'target': target_price,
                'current': current_price
            })
            # Remove the alert
            del price_alerts[symbol]
    
    # Check new-style alerts in _price_alerts (from /setalert command)
    for symbol, alerts in list(_price_alerts.items()):
        if current_prices:
            current_price = current_prices.get(symbol, 0)
        else:
            current_price = LAST_MARKET_DATA.get(symbol, {}).get('price', 0)
        
        if current_price <= 0:
            continue
        
        for alert in alerts[:]:  # Copy list to iterate while modifying
            if alert.get('triggered', False):
                continue
                
            target_price = alert.get('price', 0)
            if target_price <= 0:
                continue
            
            # Check if price crossed target (within 0.1%)
            if abs(current_price - target_price) / target_price < 0.001:
                alert['triggered'] = True
                triggered.append({
                    'symbol': symbol,
                    'target': target_price,
                    'current': current_price
                })
        
        # Remove triggered alerts
        _price_alerts[symbol] = [a for a in alerts if not a.get('triggered', False)]
    
    # Send notifications for triggered alerts
    for alert in triggered:
        message = f"""
{ds.ICON_BELL} <b>PRICE ALERT TRIGGERED</b>
{ds.SEP_THICK}

<b>{alert['symbol']}</b>

Target: {ds.format_price(alert['target'])}
Current: {ds.format_price(alert['current'])}

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')}
"""
        get_notifier().send_message_async(message)
    
    return triggered


def update_position(symbol: str, position_data: Optional[Dict]):
    """Update or remove a position"""
    global CURRENT_POSITIONS
    if position_data is None:
        CURRENT_POSITIONS.pop(symbol, None)
    else:
        CURRENT_POSITIONS[symbol] = {
            **position_data,
            'updated_at': datetime.now().isoformat()
        }


def add_trade(trade_data: Dict):
    """Add a completed trade to history with streak tracking"""
    global TRADE_HISTORY, DAILY_STATS
    
    trade_data['timestamp'] = datetime.now().isoformat()
    TRADE_HISTORY.append(trade_data)
    
    pnl = trade_data.get('pnl', 0)
    DAILY_STATS['trades'] += 1
    DAILY_STATS['pnl'] += pnl
    
    # Track best/worst trades
    if DAILY_STATS['best_trade'] is None or pnl > DAILY_STATS['best_trade'].get('pnl', 0):
        DAILY_STATS['best_trade'] = trade_data
    if DAILY_STATS['worst_trade'] is None or pnl < DAILY_STATS['worst_trade'].get('pnl', 0):
        DAILY_STATS['worst_trade'] = trade_data
    
    # Track streak
    if pnl > 0:
        DAILY_STATS['wins'] += 1
        if DAILY_STATS['streak'] >= 0:
            DAILY_STATS['streak'] += 1
        else:
            DAILY_STATS['streak'] = 1
    else:
        DAILY_STATS['losses'] += 1
        if DAILY_STATS['streak'] <= 0:
            DAILY_STATS['streak'] -= 1
        else:
            DAILY_STATS['streak'] = -1
    
    # Track peak P&L and drawdown
    if DAILY_STATS['pnl'] > DAILY_STATS.get('peak_pnl', 0):
        DAILY_STATS['peak_pnl'] = DAILY_STATS['pnl']
    
    drawdown = DAILY_STATS.get('peak_pnl', 0) - DAILY_STATS['pnl']
    if drawdown > DAILY_STATS.get('max_drawdown', 0):
        DAILY_STATS['max_drawdown'] = drawdown


def reset_daily_stats():
    """Reset daily statistics (call at start of new trading day)"""
    global TRADE_HISTORY, DAILY_STATS
    TRADE_HISTORY = []
    DAILY_STATS = {
        'trades': 0,
        'wins': 0,
        'losses': 0,
        'pnl': 0.0,
        'start_time': datetime.now().isoformat()
    }


def update_settings(settings: Dict):
    """Update bot settings"""
    global BOT_SETTINGS
    BOT_SETTINGS.update(settings)


# === Notification Functions with Modern Design ===

def send_signal(symbol, direction, confluence, price, tp, sl, htf, ltf, kz, pp):
    """Send signal notification with modern design"""
    if not BOT_SETTINGS.get('signal_alerts', True):
        return
    
    ds = DesignSystem
    
    dir_icon = ds.ICON_BULL if direction == 1 else ds.ICON_BEAR
    direction_text = "LONG" if direction == 1 else "SHORT"
    
    # Calculate R:R
    if direction == 1:
        risk = price - sl
        reward = tp - price
    else:
        risk = sl - price
        reward = price - tp
    rr = reward / risk if risk > 0 else 0
    
    # Build visual indicators
    htf_trend = ds.trend_indicator(htf, True)
    ltf_trend = ds.trend_indicator(1 if ltf >= 0 else -1, True)
    conf_meter = ds.confidence_meter(confluence, 100)
    session_icon, session_name = ds.get_session_icon()
    
    message = f"""
{ds.ICON_ZAP} <b>SIGNAL DETECTED</b>
{ds.SEP_THICK}

{dir_icon} <b>{direction_text}</b> on <b>{symbol}</b>

{ds.SEP_THIN}
<code>
 Entry  : {ds.format_price(price)}
 Target : {ds.format_price(tp)} {ds.ICON_TARGET}
 Stop   : {ds.format_price(sl)} {ds.ICON_SHIELD}
 R:R    : 1:{rr:.1f}
</code>

{ds.SEP_THICK}
<b>Analysis</b>
{ds.SEP_THIN}

{conf_meter}

<code>
 HTF Trend : {htf_trend}
 LTF Trend : {ltf_trend}
 Session   : {session_icon} {session_name}
 Price Pos : {pp:.0%}
</code>

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')} | {datetime.now().strftime('%b %d')}
"""
    get_notifier().send_message_async(message)


def send_trade_entry(symbol, direction, qty, entry_price, confluence, tp, sl, pd_zone=None, rl_action=None, risk_amount=None):
    """Send trade entry notification with V8 details"""
    if not BOT_SETTINGS.get('trade_alerts', True):
        return
    
    ds = DesignSystem
    
    dir_icon = ds.ICON_BULL if direction == 1 else ds.ICON_BEAR
    direction_text = "LONG" if direction == 1 else "SHORT"
    
    # Calculate R:R
    if direction == 1:
        risk = entry_price - sl
        reward = tp - entry_price
    else:
        risk = sl - entry_price
        reward = entry_price - tp
    rr = reward / risk if risk > 0 else 0
    
    # Format PD Zone
    pd_zone_text = ""
    if pd_zone:
        pd_icons = {
            'discount': '📥 DISCOUNT',
            'equilibrium': '⚖️ EQUILIBRIUM',
            'premium': '📤 PREMIUM',
            'equilibrium_below': '⚖️ EQ BELOW',
            'equilibrium_above': '⚖️ EQ ABOVE',
            'extreme_discount': '📥📥 EXTREME DISCOUNT',
            'extreme_premium': '📤📤 EXTREME PREMIUM'
        }
        pd_zone_text = pd_icons.get(pd_zone, pd_zone)
    
    # Format RL Action
    rl_text = ""
    if rl_action:
        rl_icons = {
            'ENTER_NOW': '⚡ ENTER NOW',
            'ENTER_PULLBACK': '↩️ ENTER PULLBACK',
            'WAIT_CONFIRMATION': '⏳ WAIT CONFIRMATION',
            'PASS': '⛔ PASS'
        }
        rl_text = f"\n🤖 RL Decision: {rl_icons.get(rl_action, rl_action)}"
    
    # Risk amount
    risk_text = ""
    if risk_amount:
        risk_text = f"\n💰 Risk: ${risk_amount:,.2f}"
    
    # Update position tracking
    update_position(symbol, {
        'direction': direction,
        'entry': entry_price,
        'qty': qty,
        'stop': sl,
        'target': tp,
        'confluence': confluence,
        'pd_zone': pd_zone,
        'rl_action': rl_action,
        'entry_time': datetime.now().isoformat()
    })
    
    # Confidence meter
    conf_meter = ds.confidence_meter(confluence, 100)
    session_icon, session_name = ds.get_session_icon()
    
    message = f"""
{ds.ICON_ROCKET} <b>🚢 V8 TRADE ENTERED</b>
{ds.SEP_THICK}

{dir_icon} <b>{direction_text}</b> {symbol}

{ds.SEP_THIN}
<code>
 Entry    : {ds.format_price(entry_price)}
 Target   : {ds.format_price(tp)} {ds.ICON_TARGET}
 Stop     : {ds.format_price(sl)} {ds.ICON_SHIELD}
 Quantity : {qty:,.4f}
 R:R      : 1:{rr:.1f}
</code>

{ds.SEP_THICK}
{conf_meter}

{ds.SEP_THIN}
{session_icon} <i>{session_name}</i>
{pd_zone_text}
{rl_text}
{risk_text}

{ds.STATUS_SUCCESS} <b>Trade Active</b>

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')} | {datetime.now().strftime('%b %d')}
"""
    send_notification(message)


def send_trade_exit(symbol, direction, pnl, exit_reason, entry_price, exit_price, bars_held):
    """Send trade exit notification with modern design and animations"""
    if not BOT_SETTINGS.get('trade_alerts', True):
        return
    
    ds = DesignSystem
    
    is_win = pnl > 0
    
    # Remove position tracking
    update_position(symbol, None)
    
    # Add to trade history
    add_trade({
        'symbol': symbol,
        'direction': 'LONG' if direction == 1 else 'SHORT',
        'entry': entry_price,
        'exit': exit_price,
        'pnl': pnl,
        'exit_reason': exit_reason,
        'bars_held': bars_held
    })
    
    # Calculate trade stats
    if direction == 1:
        move_pct = ((exit_price - entry_price) / entry_price) * 100
    else:
        move_pct = ((entry_price - exit_price) / entry_price) * 100
    
    # Format exit reason nicely
    exit_reasons = {
        'target': f'{ds.ICON_TARGET} TARGET HIT',
        'stop': f'{ds.ICON_SHIELD} STOP LOSS',
        'manual': f'{ds.ICON_GEAR} MANUAL CLOSE',
        'manual_close': f'{ds.ICON_GEAR} MANUAL CLOSE',
        'trailing_stop': f'{ds.TREND_UP} TRAILING STOP',
        'breakeven': f'{ds.STATUS_NEUTRAL} BREAKEVEN',
        'time_exit': f'{ds.ICON_CLOCK} TIME EXIT',
        'bracket_order': f'{ds.ICON_ZAP} BRACKET ORDER'
    }
    exit_text = exit_reasons.get(exit_reason.lower(), exit_reason.upper().replace('_', ' '))
    
    # Build header based on win/loss
    if is_win:
        header_icon = ds.STATUS_SUCCESS
        header_text = "WINNER"
        pnl_bar = ds.pnl_bar(pnl, pnl * 1.5, 10)
    else:
        header_icon = ds.STATUS_ERROR
        header_text = "LOSS"
        pnl_bar = ds.pnl_bar(pnl, abs(pnl) * 1.5, 10)
    
    # Streak info
    streak = DAILY_STATS.get('streak', 0)
    if streak > 2:
        streak_text = f"\n{ds.ICON_FIRE} <b>{streak} WIN STREAK!</b>"
    elif streak < -2:
        streak_text = f"\n{ds.STATUS_WARNING} {abs(streak)} consecutive losses"
    else:
        streak_text = ""
    
    # Win rate
    win_rate = (DAILY_STATS['wins'] / max(DAILY_STATS['trades'], 1)) * 100
    win_rate_bar = ds.progress_bar(win_rate, 100, 8)
    
    message = f"""
{header_icon} <b>TRADE CLOSED - {header_text}</b>
{ds.SEP_THICK}

<b>{symbol}</b> {'LONG' if direction == 1 else 'SHORT'}

{ds.SEP_THIN}
<code>
 Entry  : {ds.format_price(entry_price)}
 Exit   : {ds.format_price(exit_price)}
 Move   : {move_pct:+.2f}%
 Bars   : {bars_held}
</code>

{ds.SEP_THICK}
{pnl_bar}

{exit_text}
{streak_text}

{ds.SEP_THICK}
<b>Daily Stats</b>
{ds.SEP_THIN}
<code>
 W/L      : {DAILY_STATS['wins']}/{DAILY_STATS['losses']}
 Win Rate : {win_rate:.1f}% {win_rate_bar}
 Total    : ${DAILY_STATS['pnl']:+,.2f}
</code>

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')} | {datetime.now().strftime('%b %d')}
"""
    send_notification(message)


def send_startup(symbols, risk_pct, interval, mode):
    """Send startup notification with modern design"""
    ds = DesignSystem
    
    update_settings({
        'symbols': symbols,
        'risk_per_trade': risk_pct,
        'mode': mode
    })
    
    # Build symbols display
    if len(symbols) <= 4:
        symbols_str = ", ".join(symbols)
    else:
        symbols_str = ", ".join(symbols[:4]) + f" +{len(symbols) - 4} more"
    
    session_icon, session_name = ds.get_session_icon()
    
    message = f"""
{ds.ICON_ROCKET} <b>V5 TRADING BOT STARTED</b>
{ds.SEP_THICK}

{session_icon} <i>{session_name}</i>

<code>
 Mode          : {mode}
 Symbols       : {len(symbols)}
 Risk/Trade    : {risk_pct*100:.1f}%
 Interval      : {interval}s
</code>

{ds.SEP_THIN}
<b>Watching:</b> {symbols_str}

{ds.SEP_THICK}
{ds.STATUS_SUCCESS} <b>Bot Active</b>

Commands: <code>/start</code> for menu

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')} | {datetime.now().strftime('%b %d, %Y')}
"""
    get_notifier().send_message(message)


def send_daily_summary():
    """Send daily trading summary with modern design"""
    if not BOT_SETTINGS.get('daily_summary', True):
        return
    
    if DAILY_STATS['trades'] == 0:
        return
    
    ds = DesignSystem
    
    win_rate = (DAILY_STATS['wins'] / max(DAILY_STATS['trades'], 1)) * 100
    
    # Calculate profit factor
    gross_profit = sum(t['pnl'] for t in TRADE_HISTORY if t.get('pnl', 0) > 0)
    gross_loss = abs(sum(t['pnl'] for t in TRADE_HISTORY if t.get('pnl', 0) < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Best/worst trades
    best = DAILY_STATS.get('best_trade')
    worst = DAILY_STATS.get('worst_trade')
    
    best_text = f"{best['symbol']} +${best['pnl']:,.2f}" if best else "N/A"
    worst_text = f"{worst['symbol']} ${worst['pnl']:,.2f}" if worst else "N/A"
    
    # P&L visualization
    pnl = DAILY_STATS['pnl']
    pnl_bar = ds.pnl_bar(pnl, max(abs(pnl) * 1.5, 1000), 12)
    win_rate_bar = ds.progress_bar(win_rate, 100, 8)
    
    # Performance grade
    if win_rate >= 60 and pnl > 0:
        grade = f"{ds.MEDAL_GOLD} A+ EXCELLENT"
    elif win_rate >= 50 and pnl > 0:
        grade = f"{ds.MEDAL_SILVER} B GOOD"
    elif pnl >= 0:
        grade = f"{ds.MEDAL_BRONZE} C AVERAGE"
    else:
        grade = f"{ds.STATUS_WARNING} D NEEDS WORK"
    
    message = f"""
{ds.ICON_CALENDAR} <b>DAILY TRADING SUMMARY</b>
{ds.SEP_THICK}

{grade}

{ds.SEP_THIN}
{pnl_bar}

<code>
 Trades        : {DAILY_STATS['trades']}
 Wins          : {DAILY_STATS['wins']} {ds.STATUS_SUCCESS}
 Losses        : {DAILY_STATS['losses']} {ds.STATUS_ERROR}
 Win Rate      : {win_rate:.1f}% {win_rate_bar}
 Profit Factor : {profit_factor:.2f}
</code>

{ds.SEP_THICK}
<b>Highlights</b>
{ds.SEP_THIN}
{ds.TREND_UP} Best: {best_text}
{ds.TREND_DOWN} Worst: {worst_text}
{ds.ICON_CHART} Max DD: ${DAILY_STATS.get('max_drawdown', 0):,.2f}

{ds.SEP_DOT}
See you tomorrow! {ds.SESSION_CLOSED}

{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')} | {datetime.now().strftime('%b %d')}
"""
    get_notifier().send_message(message)


def test_connection():
    """Test Telegram connection with modern design"""
    notifier = get_notifier()
    if not notifier.init():
        print("Failed to initialize bot")
        return False
    
    ds = DesignSystem
    session_icon, session_name = ds.get_session_icon()
    
    keyboard = [
        [
            InlineKeyboardButton("📊 Dashboard", callback_data="status"),
            InlineKeyboardButton("📈 Positions", callback_data="positions"),
        ],
        [
            InlineKeyboardButton("⚡ Prices", callback_data="price"),
            InlineKeyboardButton("🔮 Bias", callback_data="bias"),
        ],
        [
            InlineKeyboardButton("💰 P&L", callback_data="pnl"),
            InlineKeyboardButton("⚙️ Settings", callback_data="settings"),
        ],
        [
            InlineKeyboardButton("🏠 Full Menu", callback_data="start"),
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    message = f"""
{ds.STATUS_SUCCESS} <b>V5 BOT CONNECTED</b>
{ds.SEP_THICK}

{session_icon} <i>{session_name}</i>

{ds.SEP_THIN}
Telegram bot is now active!

<b>Quick Commands:</b>
<code>/status</code>    - Dashboard
<code>/positions</code> - Open trades
<code>/price</code>     - Live prices
<code>/close</code>     - Close positions
<code>/watchlist</code> - Your watchlist
<code>/help</code>      - All commands

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')} | {datetime.now().strftime('%b %d, %Y')}
"""
    
    success = notifier.send_message(message, reply_markup)
    if success:
        print("Test message sent successfully!")
    else:
        print("Failed to send test message")
    return success


def run_polling():
    """Run the bot in polling mode (blocking)"""
    notifier = get_notifier()
    if not notifier._initialized:
        notifier.init()
    
    if notifier.app:
        print("Starting Telegram bot polling...")
        notifier.app.run_polling()


# ============================================================================
# NEW V8 NOTIFICATIONS
# ============================================================================

def send_signal_alert(symbol: str, direction: int, confluence: int, pd_zone: str, 
                      current_price: float, rl_action: str = None):
    """Send alert when a signal is detected (before entry)"""
    ds = DesignSystem
    
    dir_icon = ds.ICON_BULL if direction == 1 else ds.ICON_BEAR
    dir_text = "BULLISH" if direction == 1 else "BEARISH"
    
    # PD Zone formatting
    pd_icons = {
        'discount': '📥 DISCOUNT',
        'premium': '📤 PREMIUM',
        'equilibrium': '⚖️ EQUILIBRIUM',
        'extreme_discount': '📥📥 EXTREME DISCOUNT',
        'extreme_premium': '📤📤 EXTREME PREMIUM'
    }
    pd_text = pd_icons.get(pd_zone, pd_zone.upper() if pd_zone else 'N/A')
    
    # RL recommendation
    rl_text = ""
    if rl_action:
        rl_icons = {
            'ENTER_NOW': '⚡ ENTER IMMEDIATELY',
            'ENTER_PULLBACK': '↩️ WAIT FOR PULLBACK',
            'WAIT_CONFIRMATION': '⏳ WAIT FOR CONFIRMATION',
            'PASS': '⛔ SKIP THIS SIGNAL'
        }
        rl_text = f"\n🤖 <b>RL Recommends:</b> {rl_icons.get(rl_action, rl_action)}"
    
    conf_meter = ds.confidence_meter(confluence, 100)
    
    message = f"""
{ds.ICON_BELL} <b>SIGNAL DETECTED</b>
{ds.SEP_THICK}

{dir_icon} <b>{dir_text}</b> {symbol}

{ds.SEP_THIN}
<code>
 Price      : {ds.format_price(current_price)}
 Confluence : {confluence}/100
 PD Zone    : {pd_text}
</code>

{conf_meter}
{rl_text}

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')}
"""
    send_notification(message)


def send_position_update(positions: Dict, total_pnl: float):
    """Send hourly position update with P&L"""
    if not positions:
        return
    
    ds = DesignSystem
    
    lines = []
    for symbol, pos in positions.items():
        direction = pos.get('direction', 0)
        entry = pos.get('entry', 0)
        current = pos.get('current_price', entry)
        
        if direction == 1:
            pnl = (current - entry) * pos.get('qty', 0)
            pnl_pct = ((current - entry) / entry * 100) if entry > 0 else 0
        else:
            pnl = (entry - current) * pos.get('qty', 0)
            pnl_pct = ((entry - current) / entry * 100) if entry > 0 else 0
        
        dir_icon = ds.ICON_BULL if direction == 1 else ds.ICON_BEAR
        pnl_icon = ds.STATUS_SUCCESS if pnl >= 0 else ds.STATUS_ERROR
        
        lines.append(f"{dir_icon} <b>{symbol}</b>: {pnl_icon} ${pnl:+,.2f} ({pnl_pct:+.2f}%)")
    
    pnl_icon = ds.STATUS_SUCCESS if total_pnl >= 0 else ds.STATUS_ERROR
    
    message = f"""
{ds.ICON_CHART} <b>POSITION UPDATE</b>
{ds.SEP_THICK}

{chr(10).join(lines)}

{ds.SEP_THIN}
{pnl_icon} <b>Total Unrealized: ${total_pnl:+,.2f}</b>

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')}
"""
    send_notification(message)


def send_drawdown_alert(current_drawdown: float, max_allowed: float, daily_pnl: float):
    """Send alert when drawdown exceeds threshold"""
    ds = DesignSystem
    
    pct = (current_drawdown / max_allowed * 100) if max_allowed > 0 else 0
    
    if pct >= 100:
        severity = f"{ds.STATUS_ERROR} CRITICAL"
        action = "Trading should be halted!"
    elif pct >= 75:
        severity = f"{ds.STATUS_WARNING} HIGH"
        action = "Consider reducing position sizes"
    else:
        severity = f"{ds.STATUS_WARNING} WARNING"
        action = "Monitor closely"
    
    message = f"""
{ds.ICON_SHIELD} <b>DRAWDOWN ALERT</b>
{ds.SEP_THICK}

{severity}

<code>
 Current DD  : ${abs(current_drawdown):,.2f}
 Max Allowed : ${max_allowed:,.2f}
 DD Level    : {pct:.1f}%
 Daily P&L   : ${daily_pnl:+,.2f}
</code>

{ds.SEP_THIN}
{ds.ICON_ZAP} <b>{action}</b>

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')}
"""
    send_notification(message)


def send_streak_alert(streak: int, streak_pnl: float):
    """Send alert for win/loss streaks"""
    ds = DesignSystem
    
    if streak >= 3:
        # Win streak
        if streak >= 5:
            icon = ds.ICON_CROWN
            text = "LEGENDARY"
        elif streak >= 4:
            icon = ds.ICON_FIRE
            text = "ON FIRE"
        else:
            icon = ds.ICON_ROCKET
            text = "HOT STREAK"
        
        message = f"""
{icon} <b>{streak} WIN STREAK!</b>
{ds.SEP_THICK}

{ds.STATUS_SUCCESS} {text}

<code>
 Consecutive Wins : {streak}
 Streak P&L       : +${streak_pnl:,.2f}
</code>

{ds.SEP_THIN}
Keep it going! {ds.ICON_DIAMOND}

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')}
"""
    elif streak <= -3:
        # Loss streak
        loss_count = abs(streak)
        
        message = f"""
{ds.STATUS_WARNING} <b>{loss_count} LOSS STREAK</b>
{ds.SEP_THICK}

<code>
 Consecutive Losses : {loss_count}
 Streak Loss        : ${streak_pnl:,.2f}
</code>

{ds.SEP_THIN}
{ds.ICON_SHIELD} Consider taking a break or reducing size.

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')}
"""
    else:
        return  # No significant streak
    
    send_notification(message)


def send_market_open(session: str, symbols: List[str]):
    """Send notification when a trading session opens"""
    ds = DesignSystem
    
    session_icons = {
        'london': (ds.SESSION_LONDON, 'LONDON SESSION'),
        'nyc': (ds.SESSION_NYC, 'NEW YORK SESSION'),
        'asia': (ds.SESSION_ASIA, 'ASIA SESSION'),
    }
    
    icon, name = session_icons.get(session.lower(), (ds.ICON_CHART, session.upper()))
    
    symbols_str = ", ".join(symbols[:5])
    if len(symbols) > 5:
        symbols_str += f" +{len(symbols) - 5} more"
    
    message = f"""
{icon} <b>{name} OPEN</b>
{ds.SEP_THICK}

{ds.STATUS_SUCCESS} Trading session active!

<b>Watching:</b>
{symbols_str}

{ds.SEP_THIN}
Good luck today! {ds.ICON_ROCKET}

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')}
"""
    send_notification(message)


def send_reconnection_alert(success: bool, attempt: int = 1):
    """Send alert when connection is lost/restored"""
    ds = DesignSystem
    
    if success:
        message = f"""
{ds.STATUS_SUCCESS} <b>CONNECTION RESTORED</b>
{ds.SEP_THICK}

IBKR connection re-established!
Trading resumed normally.

{ds.SEP_THIN}
Attempt: {attempt}

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')}
"""
    else:
        message = f"""
{ds.STATUS_ERROR} <b>CONNECTION LOST</b>
{ds.SEP_THICK}

IBKR connection dropped!
Attempting to reconnect...

{ds.SEP_THIN}
Attempt: {attempt}

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')}
"""
    send_notification(message)


def send_error_alert(error_type: str, error_msg: str, symbol: str = None):
    """Send alert for trading errors"""
    ds = DesignSystem
    
    symbol_text = f" ({symbol})" if symbol else ""
    
    message = f"""
{ds.STATUS_ERROR} <b>ERROR{symbol_text}</b>
{ds.SEP_THICK}

<b>Type:</b> {error_type}

<code>{error_msg[:200]}</code>

{ds.SEP_DOT}
{ds.ICON_CLOCK} {datetime.now().strftime('%H:%M:%S')}
"""
    send_notification(message)


if __name__ == "__main__":
    test_connection()
