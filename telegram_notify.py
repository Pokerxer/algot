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
                InlineKeyboardButton("📈 Positions", callback_data="positions")
            ],
            [
                InlineKeyboardButton("📜 Trades", callback_data="trades"),
                InlineKeyboardButton("💰 P&L", callback_data="pnl")
            ],
            [
                InlineKeyboardButton("🔮 Bias", callback_data="bias"),
                InlineKeyboardButton("⚡ Confluence", callback_data="confluence")
            ],
            [
                InlineKeyboardButton("⚙️ Settings", callback_data="settings"),
                InlineKeyboardButton("🔔 Alerts", callback_data="alerts")
            ],
            [
                InlineKeyboardButton("📉 Charts", callback_data="chart"),
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
        """Handle /bias command"""
        message_obj = update.message or update.callback_query.message
        
        if not LAST_MARKET_DATA:
            await message_obj.reply_text("📭 No market data available. Start the trading bot first!")
            return
        
        bullish = []
        bearish = []
        neutral = []
        
        for symbol, data in LAST_MARKET_DATA.items():
            htf = data.get('htf_trend', 0)
            ltf = data.get('ltf_trend', 0)
            price = data.get('price', 0)
            kz = data.get('kill_zone', False)
            
            kz_icon = "🌙" if kz else "☀️"
            
            if htf == 1 and ltf >= 0:
                bullish.append(f"  {kz_icon} <b>{symbol}</b>: ${price:,.2f}")
            elif htf == -1 and ltf <= 0:
                bearish.append(f"  {kz_icon} <b>{symbol}</b>: ${price:,.2f}")
            else:
                neutral.append(f"  {kz_icon} <b>{symbol}</b>: ${price:,.2f}")
        
        message = f"""
╔══════════════════════════════════════╗
║       🔮 <b>MARKET BIAS</b>                ║
╚══════════════════════════════════════╝

🟢 <b>BULLISH ({len(bullish)}):</b>
{chr(10).join(bullish) if bullish else '  None'}

🔴 <b>BEARISH ({len(bearish)}):</b>
{chr(10).join(bearish) if bearish else '  None'}

⚪ <b>NEUTRAL ({len(neutral)}):</b>
{chr(10).join(neutral) if neutral else '  None'}

━━━━━━━━━━━━━━━━━━━━
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
        """Handle /chart command"""
        message_obj = update.message or update.callback_query.message
        
        # Check if symbol argument provided
        args = context.args if context.args else []
        
        if args:
            symbol = args[0].upper()
            if symbol in LAST_MARKET_DATA:
                data = LAST_MARKET_DATA[symbol]
                price = data.get('price', 0)
                htf = data.get('htf_trend', 0)
                ltf = data.get('ltf_trend', 0)
                kz = data.get('kill_zone', False)
                pp = data.get('price_position', 0.5)
                conf = data.get('confluence', 0)
                
                htf_text = "BULLISH ⬆️" if htf == 1 else "BEARISH ⬇️" if htf == -1 else "NEUTRAL ➡️"
                ltf_text = "BULLISH ⬆️" if ltf >= 0 else "BEARISH ⬇️"
                
                message = f"""
╔══════════════════════════════════════╗
║       📉 <b>{symbol}</b>                    ║
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
                available = ", ".join(LAST_MARKET_DATA.keys()) or "None"
                await message_obj.reply_text(
                    f"❌ Symbol '{symbol}' not found.\n\n"
                    f"<b>Available symbols:</b>\n{available}",
                    parse_mode='HTML'
                )
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

<b>🔮 Analysis Commands:</b>
/bias - Market bias for all symbols
/confluence - Signal strength levels
/chart - Price overview (or /chart SYMBOL)

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
