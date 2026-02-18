"""
Telegram Notification Module for V5 Trading Bot
=============================================
Beautifully designed notifications with interactive commands.
"""

import os
import asyncio
import json
from datetime import datetime
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '8373721073:AAEBSdP3rmREEccpRiKznTFJtwNKsmXJEts')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '809192622')

# Global state
TRADE_HISTORY = []
CURRENT_POSITIONS = {}
DAILY_STATS = {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0}
LAST_MARKET_DATA = {}

app = None

def init_bot():
    """Initialize the Telegram bot with commands"""
    global app
    
    try:
        app = Application.builder().token(TELEGRAM_TOKEN).build()
        
        # Add command handlers
        app.add_handler(CommandHandler("start", start_command))
        app.add_handler(CommandHandler("status", status_command))
        app.add_handler(CommandHandler("trades", trades_command))
        app.add_handler(CommandHandler("pnl", pnl_command))
        app.add_handler(CommandHandler("bias", bias_command))
        app.add_handler(CommandHandler("confluence", confluence_command))
        app.add_handler(CommandHandler("help", help_command))
        app.add_handler(CallbackQueryHandler(button_click))
        
        return app
    except Exception as e:
        print(f"Error initializing bot: {e}")
        return None


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command"""
    keyboard = [
        [InlineKeyboardButton("📊 Status", callback_data="status")],
        [InlineKeyboardButton("📜 Past Trades", callback_data="trades")],
        [InlineKeyboardButton("💰 P&L", callback_data="pnl")],
        [InlineKeyboardButton("📈 Market Bias", callback_data="bias")],
        [InlineKeyboardButton("⚡ Confluence", callback_data="confluence")],
        [InlineKeyboardButton("❓ Help", callback_data="help")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "╔══════════════════════════════════════╗\n"
        "║    🚀 WELCOME TO V5 TRADING BOT    ║\n"
        "╚══════════════════════════════════════╝\n\n"
        "I'm your trading assistant! Use the buttons below or commands to get info:",
        reply_markup=reply_markup
    )


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /status command - show current positions and daily stats"""
    global DAILY_STATS, CURRENT_POSITIONS
    
    if not CURRENT_POSITIONS:
        positions_text = "No open positions"
    else:
        lines = []
        for symbol, pos in CURRENT_POSITIONS.items():
            direction = "🟢 LONG" if pos['direction'] == 1 else "🔴 SHORT"
            lines.append(f"{symbol}: {direction} @ ${pos['entry']:,.2f}")
        positions_text = "\n".join(lines)
    
    message = f"""
╔══════════════════════════════════════╗
║         📊 CURRENT STATUS          ║
╚══════════════════════════════════════╝

📈 <b>Open Positions:</b>
{positions_text}

📊 <b>Today's Stats:</b>
• Trades: {DAILY_STATS['trades']}
• Wins: {DAILY_STATS['wins']} | Losses: {DAILY_STATS['losses']}
• P&L: ${DAILY_STATS['pnl']:,.2f}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    await update.message.reply_text(message, parse_mode='HTML')


async def trades_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /trades command - show past trades"""
    global TRADE_HISTORY
    
    if not TRADE_HISTORY:
        await update.message.reply_text("No trades yet today!")
        return
    
    # Show last 10 trades
    recent = TRADE_HISTORY[-10:]
    lines = []
    for i, trade in enumerate(recent, 1):
        emoji = "✅" if trade['pnl'] > 0 else "❌"
        lines.append(
            f"{i}. {trade['symbol']} {trade['direction']} | "
            f"Entry: ${trade['entry']:,.2f} | Exit: ${trade['exit']:,.2f} | "
            f"{emoji} ${trade['pnl']:,.2f}"
        )
    
    message = f"""
╔══════════════════════════════════════╗
║       📜 RECENT TRADES             ║
╚══════════════════════════════════════╝

{chr(10).join(lines)}

Total: {len(TRADE_HISTORY)} trades today
"""
    await update.message.reply_text(message)


async def pnl_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /pnl command - show P&L breakdown"""
    global TRADE_HISTORY, DAILY_STATS
    
    # Calculate by symbol
    symbol_pnl = {}
    for trade in TRADE_HISTORY:
        sym = trade['symbol']
        if sym not in symbol_pnl:
            symbol_pnl[sym] = {'pnl': 0, 'wins': 0, 'losses': 0}
        symbol_pnl[sym]['pnl'] += trade['pnl']
        if trade['pnl'] > 0:
            symbol_pnl[sym]['wins'] += 1
        else:
            symbol_pnl[sym]['losses'] += 1
    
    lines = []
    for sym, data in sorted(symbol_pnl.items(), key=lambda x: x[1]['pnl'], reverse=True):
        emoji = "🟢" if data['pnl'] > 0 else "🔴"
        lines.append(f"{sym}: {emoji} ${data['pnl']:,.2f} (W:{data['wins']} L:{data['losses']})")
    
    total_pnl = sum(t['pnl'] for t in TRADE_HISTORY)
    emoji = "🟢" if total_pnl > 0 else "🔴"
    
    message = f"""
╔══════════════════════════════════════╗
║         💰 P&L BREAKDOWN            ║
╚══════════════════════════════════════╝

{chr(10).join(lines) if lines else 'No trades yet'}

{'─'*40}
Total: {emoji} <b>${total_pnl:,.2f}</b>

Today's: Wins: {DAILY_STATS['wins']} | Losses: {DAILY_STATS['losses']}
Win Rate: {DAILY_STATS['wins']/max(DAILY_STATS['trades'],1)*100:.1f}%
"""
    await update.message.reply_text(message, parse_mode='HTML')


async def bias_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /bias command - show market bias for all symbols"""
    global LAST_MARKET_DATA
    
    if not LAST_MARKET_DATA:
        await update.message.reply_text("No market data available yet. Run the bot first!")
        return
    
    lines = []
    for symbol, data in LAST_MARKET_DATA.items():
        htf = data.get('htf_trend', 0)
        ltf = data.get('ltf_trend', 0)
        
        # Determine bias
        if htf == 1 and ltf >= 0:
            bias = "🟢 BULLISH"
            bias_emoji = "⬆️"
        elif htf == -1 and ltf <= 0:
            bias = "🔴 BEARISH"
            bias_emoji = "⬇️"
        elif htf == 1:
            bias = "🟡 BULLISH (neutral LTF)"
            bias_emoji = "↗️"
        elif htf == -1:
            bias = "🟠 BEARISH (neutral LTF)"
            bias_emoji = "↘️"
        else:
            bias = "⚪ NEUTRAL"
            bias_emoji = "➡️"
        
        price = data.get('price', 0)
        kz = "✅" if data.get('kill_zone', False) else "❌"
        
        lines.append(f"{symbol}: {bias_emoji} {bias} | ${price:,.2f} | KZ: {kz}")
    
    message = f"""
╔══════════════════════════════════════╗
║       📈 MARKET BIAS              ║
╚══════════════════════════════════════╝

{chr(10).join(lines)}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    await update.message.reply_text(message)


async def confluence_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /confluence command - show confluence for all symbols"""
    global LAST_MARKET_DATA
    
    if not LAST_MARKET_DATA:
        await update.message.reply_text("No market data available yet. Run the bot first!")
        return
    
    lines = []
    for symbol, data in LAST_MARKET_DATA.items():
        conf = data.get('confluence', 0)
        htf = data.get('htf_trend', 0)
        ltf = data.get('ltf_trend', 0)
        kz = data.get('kill_zone', False)
        pp = data.get('price_position', 0.5)
        
        # Color based on confluence
        if conf >= 60:
            emoji = "🟢"
        elif conf >= 40:
            emoji = "🟡"
        else:
            emoji = "🔴"
        
        kz_icon = "🌙" if kz else "☀️"
        
        lines.append(
            f"{emoji} {symbol}: {conf}/100 | "
            f"HTF:{htf} LTF:{ltf} | {kz_icon} KZ | PP:{pp:.0%}"
        )
    
    message = f"""
╔══════════════════════════════════════╗
║      ⚡ CONF LUENCE STATUS         ║
╚══════════════════════════════════════╝

{chr(10).join(lines)}

{'─'*40}
🟢 60+ = Signal Zone
🟡 40-59 = Watching
🔴 <40 = No Signal

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    await update.message.reply_text(message)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /help command"""
    message = """
╔══════════════════════════════════════╗
║         ❓ AVAILABLE COMMANDS      ║
╚══════════════════════════════════════╝

/start - Show menu
/status - Current positions & stats
/trades - Past trades
/pnl - P&L breakdown by symbol
/bias - Market bias for all symbols
/confluence - Confluence levels
/help - Show this help message

╔══════════════════════════════════════╗
║         📊 STATUS CODES            ║
╚══════════════════════════════════════╝

🟢 BULLISH - HTF & LTF aligned up
🔴 BEARISH - HTF & LTF aligned down
🟡/🟠 - Mixed signals
⚪ NEUTRAL - No clear trend
"""
    await update.message.reply_text(message)


async def button_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle button clicks"""
    query = update.callback_query
    await query.answer()
    
    if query.data == "status":
        await status_command(query, context)
    elif query.data == "trades":
        await trades_command(query, context)
    elif query.data == "pnl":
        await pnl_command(query, context)
    elif query.data == "bias":
        await bias_command(query, context)
    elif query.data == "confluence":
        await confluence_command(query, context)
    elif query.data == "help":
        await help_command(query, context)


# === Notification Functions ===

def send_message(message):
    """Send message to Telegram (non-async)"""
    if app is None:
        init_bot()
    if app is None:
        return
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(
            app.bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='HTML')
        )
    except Exception as e:
        print(f"Failed to send message: {e}")


def update_market_data(symbol, data):
    """Update market data for all symbols"""
    global LAST_MARKET_DATA
    LAST_MARKET_DATA[symbol] = data


def update_position(symbol, position_data):
    """Update current position"""
    global CURRENT_POSITIONS
    if position_data is None:
        CURRENT_POSITIONS.pop(symbol, None)
    else:
        CURRENT_POSITIONS[symbol] = position_data


def add_trade(trade_data):
    """Add trade to history"""
    global TRADE_HISTORY, DAILY_STATS
    
    TRADE_HISTORY.append(trade_data)
    DAILY_STATS['trades'] += 1
    DAILY_STATS['pnl'] += trade_data['pnl']
    
    if trade_data['pnl'] > 0:
        DAILY_STATS['wins'] += 1
    else:
        DAILY_STATS['losses'] += 1


def reset_daily_stats():
    """Reset daily statistics"""
    global TRADE_HISTORY, DAILY_STATS
    TRADE_HISTORY = []
    DAILY_STATS = {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0}


# === Original notification functions (updated) ===

def send_signal(symbol, direction, confluence, price, tp, sl, htf, ltf, kz, pp):
    """Send signal notification"""
    emoji = "🔵" if direction == 1 else "🔴"
    direction_text = "LONG" if direction == 1 else "SHORT"
    direction_emoji = "🟢" if direction == 1 else "🔴"
    
    htf_emoji = "⬆️" if htf == 1 else "⬇️" if htf == -1 else "➡️"
    htf_text = "BULLISH" if htf == 1 else "BEARISH" if htf == -1 else "NEUTRAL"
    ltf_emoji = "🟢" if ltf >= 0 else "🔴"
    
    kz_emoji = "✅" if kz else "❌"
    
    if direction == 1:
        risk = price - sl
        reward = tp - price
    else:
        risk = sl - price
        reward = price - tp
    rr = reward / risk if risk > 0 else 0
    
    message = f"""
╔══════════════════════════════════════╗
║       📊 SIGNAL DETECTED            ║
╚══════════════════════════════════════╝

{direction_emoji} <b>{direction_text}</b> on <b>{symbol}</b>

┌─────────────────────────────────────┐
│  📈 Entry:    ${price:,.2f}         │
│  🎯 TP:       ${tp:,.2f}         │
│  🛡️  SL:       ${sl:,.2f}         │
│  📊 R/R:      1:{rr:.1f}            │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  🔹 Confluence:     <b>{confluence}</b>             │
│  ⏱️  HTF Trend:     {htf_emoji} {htf_text}       │
│  ⏱️  LTF Trend:     {ltf_emoji} {"BULLISH" if ltf >= 0 else "BEARISH"}         │
│  🌙 Kill Zone:      {kz_emoji} {"Yes" if kz else "No"}            │
│  📍 Price Pos:      {pp:.0%}            │
└─────────────────────────────────────┘

⏰ {datetime.now().strftime('%H:%M:%S')} | {datetime.now().strftime('%Y-%m-%d')}
"""
    send_message(message)


def send_trade_entry(symbol, direction, qty, entry_price, confluence, tp, sl):
    """Send trade entry notification"""
    direction_emoji = "🟢" if direction == 1 else "🔴"
    direction_text = "LONG" if direction == 1 else "SHORT"
    
    message = f"""
╔══════════════════════════════════════╗
║    ✅ TRADE ENTERED                 ║
╚══════════════════════════════════════╝

{direction_emoji} <b>{direction_text}</b> <b>{symbol}</b>

┌─────────────────────────────────────┐
│  📦 Quantity:      <b>{qty}</b>               │
│  🎯 Confluence:   <b>{confluence}</b>              │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│  💵 Entry:       ${entry_price:,.2f}         │
│  🎯 TP:          ${tp:,.2f}         │
│  🛡️  SL:          ${sl:,.2f}         │
└─────────────────────────────────────┘

<b>⚡ Trade Active</b>

⏰ {datetime.now().strftime('%H:%M:%S')} | {datetime.now().strftime('%Y-%m-%d')}
"""
    send_message(message)


def send_trade_exit(symbol, direction, pnl, exit_reason, entry_price, exit_price, bars_held):
    """Send trade exit notification"""
    is_win = pnl > 0
    emoji = "✅" if is_win else "❌"
    win_loss = "WIN" if is_win else "LOSS"
    win_emoji = "💰" if is_win else "💸"
    
    # Add to trade history
    trade_data = {
        'symbol': symbol,
        'direction': 'LONG' if direction == 1 else 'SHORT',
        'entry': entry_price,
        'exit': exit_price,
        'pnl': pnl,
        'exit_reason': exit_reason,
        'time': datetime.now().isoformat()
    }
    add_trade(trade_data)
    
    message = f"""
╔══════════════════════════════════════╗
║    {emoji} TRADE CLOSED - {win_loss}            ║
╚══════════════════════════════════════╝

<b>{symbol}</b>

┌─────────────────────────────────────┐
│  💵 Entry:       ${entry_price:,.2f}         │
│  🚪 Exit:        ${exit_price:,.2f}         │
│  ⏳ Bars Held:   <b>{bars_held}</b>               │
└─────────────────────────────────────┘

{win_emoji} <b>P&L: ${pnl:,.2f}</b>

<b>Exit Reason:</b> {exit_reason.upper()}

⏰ {datetime.now().strftime('%H:%M:%S')} | {datetime.now().strftime('%Y-%m-%d')}
"""
    send_message(message)


def send_startup(symbols, risk_pct, interval, mode):
    """Send startup notification"""
    symbols_str = ", ".join(symbols)
    
    message = f"""
╔══════════════════════════════════════╗
║    🚀 V5 TRADING BOT STARTED        ║
╚══════════════════════════════════════╝

<b>Mode:</b> {mode}

<b>Symbols ({len(symbols)}):</b>
{symbols_str}

<b>Risk:</b> {risk_pct*100}%
<b>Interval:</b> {interval}s

⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    send_message(message)


def test_connection():
    """Test Telegram connection"""
    init_bot()
    
    keyboard = [
        [InlineKeyboardButton("📊 Status", callback_data="status")],
        [InlineKeyboardButton("📜 Trades", callback_data="trades")],
        [InlineKeyboardButton("💰 P&L", callback_data="pnl")],
        [InlineKeyboardButton("📈 Bias", callback_data="bias")],
        [InlineKeyboardButton("⚡ Confluence", callback_data="confluence")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    message = """
╔══════════════════════════════════════╗
║  ✅ V5 BOT CONNECTED                ║
╚══════════════════════════════════════╝

<b>Telegram commands are now active!</b>

Use /start for menu or these commands:
• /status - Positions & stats
• /trades - Recent trades  
• /pnl - P&L breakdown
• /bias - Market bias
• /confluence - Signal levels

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    if app:
        try:
            loop.run_until_complete(
                app.bot.send_message(
                    chat_id=TELEGRAM_CHAT_ID, 
                    text=message, 
                    parse_mode='HTML',
                    reply_markup=reply_markup
                )
            )
            print("Test message sent!")
        except Exception as e:
            print(f"Error: {e}")


def run_polling():
    """Run the bot in polling mode"""
    if app:
        print("Starting Telegram bot polling...")
        app.run_polling()


if __name__ == "__main__":
    test_connection()
