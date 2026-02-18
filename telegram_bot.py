"""
Telegram Bot Runner
Run this separately to handle interactive commands and buttons.
"""

import asyncio
import os
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, ContextTypes

TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '8373721073:AAEBSdP3rmREEccpRiKznTFJtwNKsmXJEts')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '809192622')

# Global state - shared with V5
TRADE_HISTORY = []
CURRENT_POSITIONS = {}
DAILY_STATS = {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0}
LAST_MARKET_DATA = {}


def update_trade_history(history):
    global TRADE_HISTORY
    TRADE_HISTORY = history


def update_positions(positions):
    global CURRENT_POSITIONS
    CURRENT_POSITIONS = positions


def update_daily_stats(stats):
    global DAILY_STATS
    DAILY_STATS = stats


def update_market_data(data):
    global LAST_MARKET_DATA
    LAST_MARKET_DATA = data


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    keyboard = [
        [InlineKeyboardButton("📊 Status", callback_data="status")],
        [InlineKeyboardButton("📜 Trades", callback_data="trades")],
        [InlineKeyboardButton("💰 P&L", callback_data="pnl")],
        [InlineKeyboardButton("📈 Bias", callback_data="bias")],
        [InlineKeyboardButton("⚡ Confluence", callback_data="confluence")]
    ]
    reply_markup = InlineKeyboardMarkup(keyboard)
    
    await update.message.reply_text(
        "╔══════════════════════════════════════╗\n"
        "║    🚀 V5 TRADING BOT MENU       ║\n"
        "╚══════════════════════════════════════╝\n\n"
        "Select an option:",
        reply_markup=reply_markup
    )


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
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
"""
    await update.message.reply_text(message, parse_mode='HTML')


async def trades_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not TRADE_HISTORY:
        await update.message.reply_text("No trades yet today!")
        return
    
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
    if not LAST_MARKET_DATA:
        await update.message.reply_text("No market data available!")
        return
    
    lines = []
    for symbol, data in LAST_MARKET_DATA.items():
        htf = data.get('htf_trend', 0)
        ltf = data.get('ltf_trend', 0)
        
        if htf == 1 and ltf >= 0:
            bias = "🟢 BULLISH"
        elif htf == -1 and ltf <= 0:
            bias = "🔴 BEARISH"
        elif htf == 1:
            bias = "🟡 BULLISH"
        elif htf == -1:
            bias = "🟠 BEARISH"
        else:
            bias = "⚪ NEUTRAL"
        
        price = data.get('price', 0)
        kz = "✅" if data.get('kill_zone', False) else "❌"
        
        lines.append(f"{symbol}: {bias} | ${price:,.2f} | KZ: {kz}")
    
    message = f"""
╔══════════════════════════════════════╗
║       📈 MARKET BIAS              ║
╚══════════════════════════════════════╝

{chr(10).join(lines)}
"""
    await update.message.reply_text(message)


async def confluence_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not LAST_MARKET_DATA:
        await update.message.reply_text("No market data available!")
        return
    
    lines = []
    for symbol, data in LAST_MARKET_DATA.items():
        conf = data.get('confluence', 0)
        htf = data.get('htf_trend', 0)
        ltf = data.get('ltf_trend', 0)
        kz = data.get('kill_zone', False)
        pp = data.get('price_position', 0.5)
        
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
"""
    await update.message.reply_text(message)


async def button_click(update: Update, context: ContextTypes.DEFAULT_TYPE):
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


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    message = """
╔══════════════════════════════════════╗
║         ❓ AVAILABLE COMMANDS       ║
╚══════════════════════════════════════╝

/start - Show menu
/status - Positions & stats
/trades - Recent trades
/pnl - P&L breakdown
/bias - Market bias
/confluence - Confluence levels
"""
    await update.message.reply_text(message)


def run_bot():
    """Run the Telegram bot in polling mode"""
    app = Application.builder().token(TELEGRAM_TOKEN).build()
    
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("status", status_command))
    app.add_handler(CommandHandler("trades", trades_command))
    app.add_handler(CommandHandler("pnl", pnl_command))
    app.add_handler(CommandHandler("bias", bias_command))
    app.add_handler(CommandHandler("confluence", confluence_command))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CallbackQueryHandler(button_click))
    
    print("🤖 Telegram bot starting...")
    print("Use /start in Telegram to see the menu")
    app.run_polling()


if __name__ == "__main__":
    run_bot()
