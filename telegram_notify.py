"""
Telegram Notification Module for V5 Trading Bot
"""

import os
import telegram
from datetime import datetime

TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN', '8373721073:AAEBSdP3rmREEccpRiKznTFJtwNKsmXJEts')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '809192622')

try:
    bot = telegram.Bot(token=TELEGRAM_TOKEN)
except Exception as e:
    print(f"Warning: Telegram bot initialization failed: {e}")
    bot = None


def send_message(message):
    """Send message to Telegram"""
    if bot is None:
        return
    try:
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message, parse_mode='HTML')
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")


def send_signal(symbol, direction, confluence, price, tp, sl, htf, ltf, kz, pp):
    """Send signal notification to Telegram"""
    emoji = "🟢" if direction == 1 else "🔴"
    direction_text = "LONG" if direction == 1 else "SHORT"
    htf_text = "BULLISH" if htf == 1 else "BEARISH" if htf == -1 else "NEUTRAL"
    ltf_text = "BULLISH" if ltf >= 0 else "BEARISH"
    kz_text = "✅ Yes" if kz else "❌ No"
    
    message = f"""
{emoji} <b>SIGNAL DETECTED</b>
━━━━━━━━━━━━━━━━━━━━
<b>Symbol:</b> {symbol}
<b>Direction:</b> {direction_text}
<b>Confluence:</b> {confluence}
━━━━━━━━━━━━━━━━━━━━
<b>HTF Trend:</b> {htf_text}
<b>LTF Trend:</b> {ltf_text}
<b>Kill Zone:</b> {kz_text}
<b>Price Position:</b> {pp:.2f}
━━━━━━━━━━━━━━━━━━━━
<b>Entry:</b> ${price:,.2f}
<b>Take Profit:</b> ${tp:,.2f}
<b>Stop Loss:</b> ${sl:,.2f}
<b>Risk/Reward:</b> 1:2
━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    send_message(message)


def send_trade_entry(symbol, direction, qty, entry_price, confluence, tp, sl):
    """Send trade entry notification"""
    emoji = "🟢" if direction == 1 else "🔴"
    direction_text = "LONG" if direction == 1 else "SHORT"
    
    message = f"""
✅ <b>TRADE ENTERED</b>
━━━━━━━━━━━━━━━━━━━━
<b>Symbol:</b> {symbol}
<b>Direction:</b> {direction_text}
<b>Quantity:</b> {qty}
<b>Confluence:</b> {confluence}
━━━━━━━━━━━━━━━━━━━━
<b>Entry:</b> ${entry_price:,.2f}
<b>Take Profit:</b> ${tp:,.2f}
<b>Stop Loss:</b> ${sl:,.2f}
━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    send_message(message)


def send_trade_exit(symbol, direction, pnl, exit_reason, entry_price, exit_price, bars_held):
    """Send trade exit notification"""
    emoji = "✅" if pnl > 0 else "❌"
    win = "WIN" if pnl > 0 else "LOSS"
    
    message = f"""
{emoji} <b>TRADE CLOSED - {win}</b>
━━━━━━━━━━━━━━━━━━━━
<b>Symbol:</b> {symbol}
<b>Direction:</b> {"LONG" if direction == 1 else "SHORT"}
<b>Exit Reason:</b> {exit_reason.upper()}
━━━━━━━━━━━━━━━━━━━━
<b>Entry:</b> ${entry_price:,.2f}
<b>Exit:</b> ${exit_price:,.2f}
<b>PnL:</b> <b>${pnl:,.2f}</b>
<b>Bars Held:</b> {bars_held}
━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    send_message(message)


def send_daily_summary(total_trades, wins, losses, total_pnl, symbols_traded):
    """Send daily summary notification"""
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
    emoji = "📈" if total_pnl > 0 else "📉"
    
    message = f"""
📊 <b>DAILY SUMMARY</b>
━━━━━━━━━━━━━━━━━━━━
<b>Total Trades:</b> {total_trades}
<b>Wins:</b> {wins} | <b>Losses:</b> {losses}
<b>Win Rate:</b> {win_rate:.1f}%
<b>Total PnL:</b> <b>${total_pnl:,.2f}</b>
━━━━━━━━━━━━━━━━━━━━
<b>Symbols Traded:</b>
{', '.join(symbols_traded)}
━━━━━━━━━━━━━━━━━━━━
📅 {datetime.now().strftime('%Y-%m-%d')}
"""
    send_message(message)


def send_error(error_type, message_text):
    """Send error notification"""
    message = f"""
⚠️ <b>ERROR ALERT</b>
━━━━━━━━━━━━━━━━━━━━
<b>Type:</b> {error_type}
<b>Message:</b> {message_text}
━━━━━━━━━━━━━━━━━━━━
⏰ {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
    send_message(message)


def test_connection():
    """Test Telegram connection"""
    if bot is None:
        print("Telegram bot not initialized")
        return False
    try:
        bot.send_message(chat_id=TELEGRAM_CHAT_ID, text="✅ <b>V5 Bot Connected!</b>\n\nTelegram notifications are now active.", parse_mode='HTML')
        print("Telegram test message sent successfully!")
        return True
    except Exception as e:
        print(f"Failed to send Telegram message: {e}")
        return False


if __name__ == "__main__":
    test_connection()
