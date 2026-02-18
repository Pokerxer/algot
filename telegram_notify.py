"""
Telegram Notification Module for V5 Trading Bot
=============================================
Beautifully designed notifications for trading signals and trades.
"""

import os
import asyncio
import telegram
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from datetime import datetime

TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '8373721073:AAEBSdP3rmREEccpRiKznTFJtwNKsmXJEts')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '809192622')

bot = None

def init_bot():
    """Initialize the bot"""
    global bot
    try:
        bot = telegram.Bot(token=TELEGRAM_TOKEN)
    except Exception as e:
        print(f"Warning: Telegram bot initialization failed: {e}")
        bot = None
    return bot


def send_message(message, keyboard=None):
    """Send message to Telegram"""
    if bot is None:
        init_bot()
    if bot is None:
        return
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    try:
        if keyboard:
            loop.run_until_complete(bot.send_message(
                chat_id=TELEGRAM_CHAT_ID, 
                text=message, 
                parse_mode='HTML',
                reply_markup=keyboard
            ))
        else:
            loop.run_until_complete(bot.send_message(
                chat_id=TELEGRAM_CHAT_ID, 
                text=message, 
                parse_mode='HTML'
            ))
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")


def send_signal(symbol, direction, confluence, price, tp, sl, htf, ltf, kz, pp):
    """Send signal notification to Telegram - beautifully designed"""
    
    # Direction
    emoji = "🔵" if direction == 1 else "🔴"
    direction_text = "LONG" if direction == 1 else "SHORT"
    direction_emoji = "🟢" if direction == 1 else "🔴"
    
    # Trends
    htf_emoji = "⬆️" if htf == 1 else "⬇️" if htf == -1 else "➡️"
    htf_text = "BULLISH" if htf == 1 else "BEARISH" if htf == -1 else "NEUTRAL"
    ltf_emoji = "🟢" if ltf >= 0 else "🔴"
    
    # Kill zone
    kz_emoji = "✅" if kz else "❌"
    
    # Calculate risk/reward
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
    """Send trade entry notification - beautifully designed"""
    
    emoji = "🟢" if direction == 1 else "🔴"
    direction_text = "LONG" if direction == 1 else "SHORT"
    direction_emoji = "🟢" if direction == 1 else "🔴"
    
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
    """Send trade exit notification - beautifully designed"""
    
    is_win = pnl > 0
    emoji = "✅" if is_win else "❌"
    win_loss = "WIN" if is_win else "LOSS"
    win_emoji = "💰" if is_win else "💸"
    
    pnl_emoji = "📈" if is_win else "📉"
    color = "green" if is_win else "red"
    
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


def send_daily_summary(total_trades, wins, losses, total_pnl, symbols_traded):
    """Send daily summary notification - beautifully designed"""
    
    is_profit = total_pnl > 0
    emoji = "📈" if is_profit else "📉"
    win_emoji = "🟢" if is_profit else "🔴"
    
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    
    symbols_str = ", ".join(symbols_traded) if symbols_traded else "None"
    
    message = f"""
╔══════════════════════════════════════╗
║      📊 DAILY SUMMARY               ║
╚══════════════════════════════════════╝

┌─────────────────────────────────────┐
│  📊 Total Trades:   <b>{total_trades}</b>           │
│  🟢 Wins:           <b>{wins}</b>               │
│  🔴 Losses:         <b>{losses}</b>               │
│  📈 Win Rate:       <b>{win_rate:.1f}%</b>             │
└─────────────────────────────────────┘

{win_emoji} <b>Total P&L: ${total_pnl:,.2f}</b>

<b>Symbols Traded:</b>
{symbols_str}

📅 {datetime.now().strftime('%Y-%m-%d')}
"""
    send_message(message)


def send_error(error_type, message_text):
    """Send error notification"""
    
    message = f"""
╔══════════════════════════════════════╗
║      ⚠️ ERROR ALERT                  ║
╚══════════════════════════════════════╝

<b>Type:</b> {error_type}
<b>Message:</b> {message_text}

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
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
    """Test Telegram connection with a nice welcome message"""
    
    message = f"""
╔══════════════════════════════════════╗
║  ✅ V5 BOT CONNECTED                ║
╚══════════════════════════════════════╝

<b>Telegram notifications are now active!</b>

You will receive notifications for:
• 📊 Trading Signals
• ✅ Trade Entries  
• ❌ Trade Exits
• 📊 Daily Summaries

⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    send_message(message)


if __name__ == "__main__":
    test_connection()
