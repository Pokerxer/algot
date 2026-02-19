"""
Telegram Notification Module for V5 Trading Bot
=============================================
Beautifully designed notifications with interactive commands.

Features:
- Real-time trade notifications
- Interactive command menu
- Position tracking
- P&L tracking
- Market bias display
- Confluence monitoring
- Alert settings
"""

import os
import asyncio
import json
import logging
import threading
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List

try:
    from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
    from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    print("Warning: python-telegram-bot not installed. Run: pip install python-telegram-bot")

# Setup logging
logging.basicConfig(
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Configuration
TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '8373721073:AAEBSdP3rmREEccpRiKznTFJtwNKsmXJEts')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '809192622')

# Global state
TRADE_HISTORY: List[Dict] = []
CURRENT_POSITIONS: Dict[str, Dict] = {}
DAILY_STATS = {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0, 'start_time': datetime.now().isoformat()}
LAST_MARKET_DATA: Dict[str, Dict] = {}
BOT_SETTINGS = {
    'notifications_enabled': True,
    'signal_alerts': True,
    'trade_alerts': True,
    'daily_summary': True,
    'risk_per_trade': 0.02,
    'symbols': [],
    'mode': 'Paper Trading'
}

# Bot instance
app = None
_bot_thread = None
_event_loop = None

# IBKR connection for Telegram commands to fetch live data
_ibkr_connection = None

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
        """Handle /start command"""
        keyboard = [
            [
                InlineKeyboardButton("📊 Status", callback_data="status"),
                InlineKeyboardButton("📈 Positions", callback_data="positions"),
                InlineKeyboardButton("📜 Trades", callback_data="trades")
            ],
            [
                InlineKeyboardButton("💰 P&L", callback_data="pnl"),
                InlineKeyboardButton("⚠️ Risk", callback_data="risk"),
                InlineKeyboardButton("📈 Performance", callback_data="performance")
            ],
            [
                InlineKeyboardButton("⚡ Prices", callback_data="price"),
                InlineKeyboardButton("🔮 Bias", callback_data="bias"),
                InlineKeyboardButton("🏆 Compare", callback_data="compare")
            ],
            [
                InlineKeyboardButton("📊 Summary", callback_data="summary"),
                InlineKeyboardButton("📤 Export", callback_data="export"),
                InlineKeyboardButton("⚙️ Settings", callback_data="settings")
            ],
            [
                InlineKeyboardButton("🔔 Alerts", callback_data="alerts"),
                InlineKeyboardButton("❓ Help", callback_data="help")
            ]
        ]
        reply_markup = InlineKeyboardMarkup(keyboard)
        
        message = """
╔══════════════════════════════════════╗
║    🚀 <b>V5 TRADING BOT</b>              ║
╚══════════════════════════════════════╝

Welcome! I'm your ICT trading assistant.

Use the buttons below or type commands:

<b>Quick Commands:</b>
/status - Current status
/positions - Open positions  
/trades - Recent trades
/pnl - P&L breakdown
/bias - Market bias
/confluence - Signal strength
/settings - Bot settings
/alerts - Toggle alerts
/chart [symbol] - Price info
/help - Full help

━━━━━━━━━━━━━━━━━━━━
⏰ """ + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        
        await update.message.reply_text(message, parse_mode='HTML', reply_markup=reply_markup)
    
    async def _status_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /status command"""
        # Get message object (works for both direct command and callback)
        message_obj = update.message or update.callback_query.message
        
        pos_count = len(CURRENT_POSITIONS)
        if pos_count == 0:
            positions_text = "📭 No open positions"
        else:
            lines = []
            for symbol, pos in CURRENT_POSITIONS.items():
                direction = "🟢 LONG" if pos.get('direction', 0) == 1 else "🔴 SHORT"
                entry = pos.get('entry', 0)
                current = LAST_MARKET_DATA.get(symbol, {}).get('price', entry)
                pnl_pct = ((current - entry) / entry * 100) if entry > 0 else 0
                if pos.get('direction', 0) == -1:
                    pnl_pct = -pnl_pct
                pnl_emoji = "📈" if pnl_pct >= 0 else "📉"
                lines.append(f"  • {symbol}: {direction} @ ${entry:,.2f} {pnl_emoji} {pnl_pct:+.2f}%")
            positions_text = "\n".join(lines)
        
        win_rate = (DAILY_STATS['wins'] / max(DAILY_STATS['trades'], 1)) * 100
        pnl_emoji = "🟢" if DAILY_STATS['pnl'] >= 0 else "🔴"
        
        message = f"""
╔══════════════════════════════════════╗
║         📊 <b>CURRENT STATUS</b>           ║
╚══════════════════════════════════════╝

<b>📈 Open Positions ({pos_count}):</b>
{positions_text}

<b>📊 Today's Performance:</b>
┌─────────────────────────────────────┐
│  Trades:    {DAILY_STATS['trades']:>6}                  │
│  Wins:      {DAILY_STATS['wins']:>6}  ✅               │
│  Losses:    {DAILY_STATS['losses']:>6}  ❌               │
│  Win Rate:  {win_rate:>6.1f}%               │
│  P&L:       {pnl_emoji} ${DAILY_STATS['pnl']:>10,.2f}       │
└─────────────────────────────────────┘

<b>🔧 Mode:</b> {BOT_SETTINGS.get('mode', 'Unknown')}
<b>📡 Alerts:</b> {'🟢 ON' if BOT_SETTINGS.get('notifications_enabled', True) else '🔴 OFF'}

━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        await message_obj.reply_text(message, parse_mode='HTML')
    
    async def _positions_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /positions command - detailed position info"""
        message_obj = update.message or update.callback_query.message
        
        if not CURRENT_POSITIONS:
            await message_obj.reply_text("📭 No open positions currently.")
            return
        
        lines = []
        total_unrealized = 0
        
        for symbol, pos in CURRENT_POSITIONS.items():
            direction = "🟢 LONG" if pos.get('direction', 0) == 1 else "🔴 SHORT"
            entry = pos.get('entry', 0)
            stop = pos.get('stop', 0)
            target = pos.get('target', 0)
            qty = pos.get('qty', 0)
            confluence = pos.get('confluence', 0)
            
            current = LAST_MARKET_DATA.get(symbol, {}).get('price', entry)
            
            if pos.get('direction', 0) == 1:
                unrealized = (current - entry) * qty
            else:
                unrealized = (entry - current) * qty
            
            total_unrealized += unrealized
            pnl_emoji = "📈" if unrealized >= 0 else "📉"
            
            lines.append(f"""
<b>{symbol}</b> {direction}
┌─────────────────────────────────────┐
│  Entry:     ${entry:>12,.2f}          │
│  Current:   ${current:>12,.2f}          │
│  Stop:      ${stop:>12,.2f}          │
│  Target:    ${target:>12,.2f}          │
│  Qty:       {qty:>12}              │
│  Conf:      {confluence:>12}/100          │
│  {pnl_emoji} P&L:     ${unrealized:>12,.2f}          │
└─────────────────────────────────────┘""")
        
        total_emoji = "🟢" if total_unrealized >= 0 else "🔴"
        
        message = f"""
╔══════════════════════════════════════╗
║       📈 <b>OPEN POSITIONS</b>             ║
╚══════════════════════════════════════╝
{"".join(lines)}

━━━━━━━━━━━━━━━━━━━━
<b>Total Unrealized:</b> {total_emoji} ${total_unrealized:,.2f}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        await message_obj.reply_text(message, parse_mode='HTML')
    
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
        """Handle /help command"""
        message_obj = update.message or update.callback_query.message
        
        message = """
╔══════════════════════════════════════╗
║         ❓ <b>HELP & COMMANDS</b>          ║
╚══════════════════════════════════════╝

<b>📊 Status Commands:</b>
/start - Main menu with buttons
/status - Current positions & daily stats
/positions - Detailed position information
/trades - Recent trade history
/pnl - P&L breakdown by symbol

<b>📈 Analysis Commands:</b>
/price - Live prices from IBKR
/bias - Market bias with live prices
/confluence - Signal strength levels
/chart SYMBOL - Detailed symbol info
/performance - Detailed performance stats
/compare - Compare symbol performance

<b>💰 Trading Commands:</b>
/pnl - P&L breakdown by symbol
/risk - Risk exposure dashboard
/summary - Detailed daily summary
/export - Export trades to CSV
/alert SYMBOL PRICE - Set price alerts

<b>⚙️ Settings:</b>
/settings - View bot configuration
/alerts - Toggle notification settings

<b>📖 Legend:</b>
🟢 = Bullish/Profit/High Confluence
🔴 = Bearish/Loss/Low Confluence
🟡 = Neutral/Medium Confluence
🌙 = In Kill Zone (London/NYC session)
☀️ = Outside Kill Zone

<b>💡 Tips:</b>
• Use buttons for quick access
• Confluence 60+ = Strong signal
• Kill Zone = Higher probability setups

━━━━━━━━━━━━━━━━━━━━
V5 ICT Trading Bot
"""
        await message_obj.reply_text(message, parse_mode='HTML')
    
    async def _button_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle button callbacks"""
        query = update.callback_query
        await query.answer()
        
        callback_data = query.data
        
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
            "help": self._help_command
        }
        
        handler = command_map.get(callback_data)
        if handler:
            await handler(update, context)


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


def update_market_data(symbol: str, data: Dict):
    """Update market data for a symbol"""
    global LAST_MARKET_DATA
    LAST_MARKET_DATA[symbol] = {
        **data,
        'updated_at': datetime.now().isoformat()
    }


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
    """Add a completed trade to history"""
    global TRADE_HISTORY, DAILY_STATS
    
    trade_data['timestamp'] = datetime.now().isoformat()
    TRADE_HISTORY.append(trade_data)
    
    DAILY_STATS['trades'] += 1
    DAILY_STATS['pnl'] += trade_data.get('pnl', 0)
    
    if trade_data.get('pnl', 0) > 0:
        DAILY_STATS['wins'] += 1
    else:
        DAILY_STATS['losses'] += 1


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


# === Notification Functions ===

def send_signal(symbol, direction, confluence, price, tp, sl, htf, ltf, kz, pp):
    """Send signal notification"""
    if not BOT_SETTINGS.get('signal_alerts', True):
        return
    
    direction_emoji = "🟢" if direction == 1 else "🔴"
    direction_text = "LONG" if direction == 1 else "SHORT"
    
    htf_emoji = "⬆️" if htf == 1 else "⬇️" if htf == -1 else "➡️"
    htf_text = "BULLISH" if htf == 1 else "BEARISH" if htf == -1 else "NEUTRAL"
    ltf_emoji = "⬆️" if ltf >= 0 else "⬇️"
    ltf_text = "BULLISH" if ltf >= 0 else "BEARISH"
    
    kz_emoji = "🌙" if kz else "☀️"
    
    if direction == 1:
        risk = price - sl
        reward = tp - price
    else:
        risk = sl - price
        reward = price - tp
    rr = reward / risk if risk > 0 else 0
    
    message = f"""
╔══════════════════════════════════════╗
║       📊 <b>SIGNAL DETECTED</b>            ║
╚══════════════════════════════════════╝

{direction_emoji} <b>{direction_text}</b> on <b>{symbol}</b>

┌─────────────────────────────────────┐
│  📈 Entry:    ${price:>12,.4f}       │
│  🎯 Target:   ${tp:>12,.4f}       │
│  🛡️ Stop:     ${sl:>12,.4f}       │
│  📊 R:R:      1:{rr:>11.1f}       │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  ⚡ Confluence:  {confluence:>10}/100       │
│  {htf_emoji} HTF:          {htf_text:>10}       │
│  {ltf_emoji} LTF:          {ltf_text:>10}       │
│  {kz_emoji} Kill Zone:    {'Yes' if kz else 'No':>10}       │
│  📍 Price Pos:   {pp:>10.0%}       │
└─────────────────────────────────────┘

⏰ {datetime.now().strftime('%H:%M:%S')} | {datetime.now().strftime('%Y-%m-%d')}
"""
    get_notifier().send_message_async(message)


def send_trade_entry(symbol, direction, qty, entry_price, confluence, tp, sl):
    """Send trade entry notification"""
    if not BOT_SETTINGS.get('trade_alerts', True):
        return
    
    direction_emoji = "🟢" if direction == 1 else "🔴"
    direction_text = "LONG" if direction == 1 else "SHORT"
    
    # Update position tracking
    update_position(symbol, {
        'direction': direction,
        'entry': entry_price,
        'qty': qty,
        'stop': sl,
        'target': tp,
        'confluence': confluence
    })
    
    message = f"""
╔══════════════════════════════════════╗
║    ✅ <b>TRADE ENTERED</b>                 ║
╚══════════════════════════════════════╝

{direction_emoji} <b>{direction_text}</b> <b>{symbol}</b>

┌─────────────────────────────────────┐
│  📦 Quantity:     {qty:>12}       │
│  ⚡ Confluence:   {confluence:>12}/100   │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  💵 Entry:       ${entry_price:>12,.4f}   │
│  🎯 Target:      ${tp:>12,.4f}   │
│  🛡️ Stop:        ${sl:>12,.4f}   │
└─────────────────────────────────────┘

<b>⚡ Trade Active</b>

⏰ {datetime.now().strftime('%H:%M:%S')} | {datetime.now().strftime('%Y-%m-%d')}
"""
    get_notifier().send_message_async(message)


def send_trade_exit(symbol, direction, pnl, exit_reason, entry_price, exit_price, bars_held):
    """Send trade exit notification"""
    if not BOT_SETTINGS.get('trade_alerts', True):
        return
    
    is_win = pnl > 0
    emoji = "✅" if is_win else "❌"
    win_loss = "WIN" if is_win else "LOSS"
    pnl_emoji = "💰" if is_win else "💸"
    
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
    
    message = f"""
╔══════════════════════════════════════╗
║    {emoji} <b>TRADE CLOSED - {win_loss}</b>          ║
╚══════════════════════════════════════╝

<b>{symbol}</b>

┌─────────────────────────────────────┐
│  💵 Entry:       ${entry_price:>12,.4f}   │
│  🚪 Exit:        ${exit_price:>12,.4f}   │
│  ⏳ Bars Held:   {bars_held:>12}       │
└─────────────────────────────────────┘

{pnl_emoji} <b>P&L: ${pnl:,.2f}</b>

<b>Exit Reason:</b> {exit_reason.upper().replace('_', ' ')}

<b>📊 Daily Stats:</b> W:{DAILY_STATS['wins']} L:{DAILY_STATS['losses']} | ${DAILY_STATS['pnl']:,.2f}

⏰ {datetime.now().strftime('%H:%M:%S')} | {datetime.now().strftime('%Y-%m-%d')}
"""
    get_notifier().send_message_async(message)


def send_startup(symbols, risk_pct, interval, mode):
    """Send startup notification"""
    update_settings({
        'symbols': symbols,
        'risk_per_trade': risk_pct,
        'mode': mode
    })
    
    symbols_str = ", ".join(symbols)
    
    message = f"""
╔══════════════════════════════════════╗
║    🚀 <b>V5 TRADING BOT STARTED</b>        ║
╚══════════════════════════════════════╝

<b>Mode:</b> {mode}

<b>Symbols ({len(symbols)}):</b>
{symbols_str}

<b>Settings:</b>
┌─────────────────────────────────────┐
│  Risk per Trade:  {risk_pct*100:>10.1f}%      │
│  Check Interval:  {interval:>10}s      │
└─────────────────────────────────────┘

<b>Commands:</b> /start for menu

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    get_notifier().send_message(message)


def send_daily_summary():
    """Send daily trading summary"""
    if not BOT_SETTINGS.get('daily_summary', True):
        return
    
    if DAILY_STATS['trades'] == 0:
        return
    
    win_rate = (DAILY_STATS['wins'] / max(DAILY_STATS['trades'], 1)) * 100
    pnl_emoji = "🟢" if DAILY_STATS['pnl'] >= 0 else "🔴"
    
    # Calculate profit factor
    gross_profit = sum(t['pnl'] for t in TRADE_HISTORY if t.get('pnl', 0) > 0)
    gross_loss = abs(sum(t['pnl'] for t in TRADE_HISTORY if t.get('pnl', 0) < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    message = f"""
╔══════════════════════════════════════╗
║    📊 <b>DAILY TRADING SUMMARY</b>         ║
╚══════════════════════════════════════╝

<b>Performance:</b>
┌─────────────────────────────────────┐
│  Total Trades:    {DAILY_STATS['trades']:>10}       │
│  Wins:            {DAILY_STATS['wins']:>10} ✅     │
│  Losses:          {DAILY_STATS['losses']:>10} ❌     │
│  Win Rate:        {win_rate:>10.1f}%      │
│  Profit Factor:   {profit_factor:>10.2f}       │
└─────────────────────────────────────┘

{pnl_emoji} <b>Total P&L: ${DAILY_STATS['pnl']:,.2f}</b>

See you tomorrow! 🌙

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    get_notifier().send_message(message)


def test_connection():
    """Test Telegram connection"""
    notifier = get_notifier()
    if not notifier.init():
        print("Failed to initialize bot")
        return False
    
    keyboard = [
        [
            InlineKeyboardButton("📊 Status", callback_data="status"),
            InlineKeyboardButton("📈 Positions", callback_data="positions")
        ],
        [
            InlineKeyboardButton("📜 Trades", callback_data="trades"),
            InlineKeyboardButton("💰 P&L", callback_data="pnl")
        ],
        [
            InlineKeyboardButton("🔮 Bias", callback_data="bias"),
            InlineKeyboardButton("⚡ Confluence", callback_data="confluence")
        ]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    message = f"""
╔══════════════════════════════════════╗
║  ✅ <b>V5 BOT CONNECTED</b>                ║
╚══════════════════════════════════════╝

<b>Telegram bot is now active!</b>

Use /start for the full menu or try:
• /status - Current positions & stats
• /positions - Detailed positions
• /trades - Recent trades  
• /pnl - P&L breakdown
• /bias - Market bias
• /confluence - Signal levels
• /chart - Price info
• /settings - Bot settings
• /alerts - Toggle notifications

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    success = notifier.send_message(message, reply_markup)
    if success:
        print("✅ Test message sent successfully!")
    else:
        print("❌ Failed to send test message")
    return success


def run_polling():
    """Run the bot in polling mode (blocking)"""
    notifier = get_notifier()
    if not notifier._initialized:
        notifier.init()
    
    if notifier.app:
        print("Starting Telegram bot polling...")
        notifier.app.run_polling()


if __name__ == "__main__":
    test_connection()
