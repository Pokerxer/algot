"""
ICT V2 Live Trading with Alpaca Broker
======================================

Usage:
    # Test connection (paper)
    python3 ict_v2_alpaca_live.py --test
    
    # Run live trading (paper)
    python3 ict_v2_alpaca_live.py --mode paper --symbols "NQ=F,YM=F,EURUSD=X"
    
    # Run live trading (real money)
    python3 ict_v2_alpaca_live.py --mode live --symbols "NQ=F"
    
    # Signal only mode
    python3 ict_v2_alpaca_live.py --mode signal --symbols "NQ=F"

Requirements:
    - Alpaca API keys (paper or live)
    - Set environment variables:
        export ALPACA_API_KEY=your_api_key
        export ALPACA_SECRET_KEY=your_secret_key
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
import argparse
import logging
from dataclasses import dataclass
from typing import Optional, Dict, List
import os

from alpaca_broker import AlpacaBroker

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('v2_alpaca_trades.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class TradeSignal:
    timestamp: str
    symbol: str
    direction: str
    entry_price: float
    stop_loss: float
    take_profit: float
    confluence: int
    grade: str
    risk_pct: float = 0.01
    position_size: float = 0.0


class ICTV2AlpacaTrader:
    def __init__(self, symbols: List[str], capital: float = 10000.0, 
                 mode: str = 'paper', risk_per_trade: float = 0.01):
        self.symbols = symbols
        self.capital = capital
        self.mode = mode
        self.risk_per_trade = risk_per_trade
        
        # Initialize broker
        self.broker = AlpacaBroker(paper=(mode == 'paper'))
        
        # Trade history
        self.trade_history: List[Dict] = []
        self.open_trades: Dict[str, Dict] = {}
        
    def fetch_data(self, symbol: str, days: int = 5) -> Optional[Dict]:
        """Fetch data for symbol"""
        try:
            df = yf.Ticker(symbol).history(period=f"{days}d", interval="1h")
            df_daily = yf.Ticker(symbol).history(period=f"{days}d", interval="1d")
            if df.empty:
                return None
            return {'df': df, 'df_daily': df_daily}
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None
    
    def calculate_indicators(self, df, df_daily) -> Dict:
        """Calculate V2 indicators"""
        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        opens = df['Open'].values
        
        # FVGs
        bullish_fvgs = [{'idx': i, 'low': highs[i-2], 'high': lows[i], 'mid': (highs[i-2]+lows[i])/2} 
                       for i in range(3, len(df)) if lows[i] > highs[i-2]]
        bearish_fvgs = [{'idx': i, 'low': highs[i], 'high': lows[i-2], 'mid': (highs[i]+lows[i-2])/2} 
                        for i in range(3, len(df)) if highs[i] < lows[i-2]]
        
        # Order Blocks
        bullish_obs = [{'idx': i, 'high': highs[i-1], 'low': lows[i-1]} 
                      for i in range(5, len(df)) if closes[i-1] < opens[i-1] and closes[i] > opens[i] and lows[i] < lows[i-1]]
        bearish_obs = [{'idx': i, 'high': highs[i-1], 'low': lows[i-1]} 
                      for i in range(5, len(df)) if closes[i-1] > opens[i-1] and closes[i] < opens[i] and highs[i] > highs[i-1]]
        
        # HTF Trend
        daily_highs = df_daily['High'].values
        daily_lows = df_daily['Low'].values
        df_daily_index = pd.DatetimeIndex(df_daily.index).tz_localize(None)
        df_index = pd.DatetimeIndex(df.index).tz_localize(None)
        
        htf_trend = []
        for i in range(1, len(df_daily)):
            if len(df_daily) < 5:
                htf_trend.append(0)
            elif daily_highs[i] > daily_highs[max(0,i-5):i].max() and daily_lows[i] > daily_lows[max(0,i-5):i].min():
                htf_trend.append(1)
            elif daily_highs[i] < daily_highs[max(0,i-5):i].max() and daily_lows[i] < daily_lows[max(0,i-5):i].min():
                htf_trend.append(-1)
            else:
                htf_trend.append(0)
        
        htf_trend_hourly = np.zeros(len(df))
        for i in range(len(df)):
            bar_time = df_index[i]
            for j in range(len(df_daily)-1, -1, -1):
                if df_daily_index[j] <= bar_time:
                    htf_trend_hourly[i] = htf_trend[j] if j < len(htf_trend) else 0
                    break
        
        # LTF Trend
        trend = np.zeros(len(df))
        for i in range(20, len(df)):
            rh = highs[max(0,i-20):i].max()
            rl = lows[max(0,i-20):i].min()
            if rh > highs[i-5] and rl > lows[i-5]:
                trend[i] = 1
            elif rh < highs[i-5] and rl < lows[i-5]:
                trend[i] = -1
        
        # Price Position
        price_position = np.zeros(len(df))
        for i in range(20, len(df)):
            ph = highs[i-20:i].max()
            pl = lows[i-20:i].min()
            range_size = ph - pl
            if range_size < 0.001:
                range_size = 0.001
            price_position[i] = (closes[i] - pl) / range_size
        
        # Kill Zones
        hours = pd.to_datetime(df.index).hour.values
        kill_zone = np.array([(1 <= h < 5) or (7 <= h < 12) or (13.5 <= h < 16) for h in hours])
        
        return {
            'highs': highs, 'lows': lows, 'closes': closes,
            'bullish_fvgs': bullish_fvgs, 'bearish_fvgs': bearish_fvgs,
            'bullish_obs': bullish_obs, 'bearish_obs': bearish_obs,
            'htf_trend_hourly': htf_trend_hourly, 'trend': trend,
            'price_position': price_position, 'kill_zone': kill_zone
        }
    
    def check_signal(self, symbol: str) -> Optional[TradeSignal]:
        """Check for trade signal"""
        data = self.fetch_data(symbol)
        if not data:
            return None
        
        ind = self.calculate_indicators(data['df'], data['df_daily'])
        idx = len(ind['closes']) - 1
        current_price = ind['closes'][idx]
        
        current_trend = ind['trend'][idx]
        htf_bias = ind['htf_trend_hourly'][idx]
        kz = ind['kill_zone'][idx]
        pp = ind['price_position'][idx]
        
        # Nearest levels
        nearest_bull = next((ob for ob in reversed(ind['bullish_obs']) if ob['idx'] < idx), None)
        nearest_bear = next((ob for ob in reversed(ind['bearish_obs']) if ob['idx'] < idx), None)
        near_bull_fvg = next((f for f in reversed(ind['bullish_fvgs']) if f['idx'] < idx and f['mid'] < current_price < f['high']), None)
        near_bear_fvg = next((f for f in reversed(ind['bearish_fvgs']) if f['idx'] < idx and f['low'] < current_price < f['mid']), None)
        
        # Confluence
        confluence = 0
        if kz:
            confluence += 15
        if htf_bias == 1 and current_trend >= 0:
            confluence += 25
        elif htf_bias == -1 and current_trend <= 0:
            confluence += 25
        elif htf_bias != 0:
            confluence += 10
        if pp < 0.25:
            confluence += 20
        elif pp < 0.35:
            confluence += 15
        elif pp > 0.75:
            confluence += 20
        elif pp > 0.65:
            confluence += 15
        if near_bull_fvg and current_trend >= 0:
            confluence += 15
        if near_bear_fvg and current_trend <= 0:
            confluence += 15
        if nearest_bull and current_price > nearest_bull['high']:
            confluence += 10
        if nearest_bear and current_price < nearest_bear['low']:
            confluence += 10
        
        grade = 'F'
        if confluence >= 75:
            grade = 'A+'
        elif confluence >= 70:
            grade = 'A'
        
        # AI Filter
        if confluence < 70 or grade not in ['A+', 'A']:
            return None
        if htf_bias == 0 or not kz:
            return None
        if not (near_bull_fvg or nearest_bull or near_bear_fvg or nearest_bear):
            return None
        
        atr = (ind['highs'][idx-14:idx] - ind['lows'][idx-14:idx]).mean() if idx > 14 else 50
        if atr <= 0:
            atr = 50
        
        # Long
        if pp < 0.40 and htf_bias == 1:
            if near_bull_fvg and current_price > near_bull_fvg['mid']:
                sl = near_bull_fvg['low'] - atr * 0.5
                tp = current_price + atr * 2.5
                return self._create_signal(symbol, idx, 'long', current_price, sl, tp, confluence, grade)
            elif nearest_bull and current_price > nearest_bull['high']:
                sl = nearest_bull['low'] - atr * 0.5
                tp = current_price + atr * 2.5
                return self._create_signal(symbol, idx, 'long', current_price, sl, tp, confluence, grade)
        
        # Short
        elif pp > 0.60 and htf_bias == -1:
            if near_bear_fvg and current_price < near_bear_fvg['mid']:
                sl = near_bear_fvg['high'] + atr * 0.5
                tp = current_price - atr * 2.5
                return self._create_signal(symbol, idx, 'short', current_price, sl, tp, confluence, grade)
            elif nearest_bear and current_price < nearest_bear['low']:
                sl = nearest_bear['high'] + atr * 0.5
                tp = current_price - atr * 2.5
                return self._create_signal(symbol, idx, 'short', current_price, sl, tp, confluence, grade)
        
        return None
    
    def _create_signal(self, symbol: str, idx: int, direction: str, 
                       entry: float, sl: float, tp: float, confluence: int, grade: str) -> TradeSignal:
        """Create trade signal with position sizing"""
        risk_amt = self.capital * self.risk_per_trade
        risk = abs(entry - sl)
        if risk <= 0:
            return None  # Skip if invalid risk
        size = risk_amt / risk if risk > 0 else 1.0
        
        return TradeSignal(
            timestamp=str(self.fetch_data(symbol)['df'].index[idx]),
            symbol=symbol,
            direction=direction,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            confluence=confluence,
            grade=grade,
            risk_pct=self.risk_per_trade,
            position_size=size
        )
    
    def execute_trade(self, signal: TradeSignal) -> bool:
        """Execute trade via Alpaca"""
        try:
            logger.info(f"{'='*60}")
            logger.info(f"EXECUTING TRADE: {signal.direction.upper()}")
            logger.info(f"{'='*60}")
            logger.info(f"Symbol: {signal.symbol}")
            logger.info(f"Entry: {signal.entry_price:.2f}")
            logger.info(f"SL: {signal.stop_loss:.2f}")
            logger.info(f"TP: {signal.take_profit:.2f}")
            logger.info(f"Confluence: {signal.confluence} ({signal.grade})")
            logger.info(f"Size: {signal.position_size:.2f}")
            
            # Submit order with bracket (TP/SL)
            order = self.broker.submit_order(
                symbol=signal.symbol,
                qty=int(signal.position_size),
                side='buy' if signal.direction == 'long' else 'sell',
                type='limit',
                time_in_force='gtc',
                limit_price=signal.entry_price,
                take_profit=signal.take_profit,
                stop_loss=signal.stop_loss
            )
            
            if order:
                self.open_trades[signal.symbol] = {
                    'signal': signal,
                    'order_id': order.id,
                    'entry_price': signal.entry_price,
                    'sl': signal.stop_loss,
                    'tp': signal.take_profit,
                    'direction': signal.direction
                }
                logger.info(f"Order submitted: {order.id}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error executing trade: {e}")
            return False
    
    def check_positions(self):
        """Check open positions"""
        try:
            positions = self.broker.get_all_positions()
            for pos in positions:
                symbol = pos['symbol']
                if symbol in self.open_trades:
                    trade = self.open_trades[symbol]
                    pnl_pct = pos['unrealized_plpc'] * 100
                    logger.info(f"{symbol}: {pnl_pct:+.2f}% | MV: ${pos['market_value']:,.0f}")
                    
                    # Check if position is closed
                    if float(pos['qty']) == 0:
                        self._close_trade(symbol, pos)
                        
        except Exception as e:
            logger.error(f"Error checking positions: {e}")
    
    def _close_trade(self, symbol: str, position: Dict):
        """Record closed trade"""
        if symbol in self.open_trades:
            trade = self.open_trades.pop(symbol)
            self.trade_history.append({
                'symbol': symbol,
                'direction': trade['direction'],
                'entry_price': trade['entry_price'],
                'exit_price': position.get('market_value', 0),
                'pnl': position.get('unrealized_pl', 0),
                'confluence': trade['signal'].confluence,
                'grade': trade['signal'].grade
            })
            logger.info(f"Trade closed: {symbol} | PnL: ${position.get('unrealized_pl', 0):+,.2f}")
    
    def run(self, check_interval: int = 60):
        """Main trading loop"""
        logger.info(f"{'='*60}")
        logger.info(f"ICT V2 ALPACA TRADING - {self.mode.upper()} MODE")
        logger.info(f"{'='*60}")
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info(f"Capital: ${self.capital:,.2f}")
        logger.info(f"Risk/Trade: {self.risk_per_trade*100}%")
        logger.info(f"Check Interval: {check_interval}s")
        
        # Show account info
        account = self.broker.get_account()
        if account:
            logger.info(f"Alpaca Account: ${account['portfolio_value']:,.2f}")
        
        while True:
            try:
                for symbol in self.symbols:
                    try:
                        # Skip if already have position
                        if symbol in self.open_trades:
                            continue
                        
                        signal = self.check_signal(symbol)
                        if signal:
                            if self.mode in ['paper', 'live']:
                                self.execute_trade(signal)
                    except Exception as e:
                        logger.error(f"Error checking {symbol}: {e}")
                        import traceback
                        traceback.print_exc()
                
                # Check existing positions
                self.check_positions()
                
                # Log status
                self._log_status()
                
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                logger.info("Stopping trading...")
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                time.sleep(check_interval)
        
        # Save trade history
        with open('v2_alpaca_trades.json', 'w') as f:
            json.dump(self.trade_history, f, indent=2)
    
    def _log_status(self):
        """Log current status"""
        wins = len([t for t in self.trade_history if t['pnl'] > 0])
        total = len(self.trade_history)
        total_pnl = sum(t['pnl'] for t in self.trade_history)
        win_rate = (wins / total * 100) if total > 0 else 0
        
        logger.info(f"\n{'='*50}")
        logger.info(f"STATUS | Trades: {total} | Win: {wins}/{total} ({win_rate:.0f}% | PnL: ${total_pnl:+,.0f} | Open: {len(self.open_trades)}")
        logger.info(f"{'='*50}\n")
    
    def run_signal_check(self):
        """Check signals once and exit"""
        logger.info(f"{'='*60}")
        logger.info(f"ICT V2 SIGNAL CHECK - {self.mode.upper()} MODE")
        logger.info(f"{'='*60}")
        
        for symbol in self.symbols:
            signal = self.check_signal(symbol)
            if signal:
                logger.info(f"\n{'!'*60}")
                logger.info(f"SIGNAL: {symbol}")
                logger.info(f"{'!'*60}")
                logger.info(f"Direction: {signal.direction.upper()}")
                logger.info(f"Entry: {signal.entry_price:.2f}")
                logger.info(f"SL: {signal.stop_loss:.2f}")
                logger.info(f"TP: {signal.take_profit:.2f}")
                logger.info(f"Confluence: {signal.confluence} ({signal.grade})")
                logger.info(f"Size: {signal.position_size:.2f}")
                logger.info(f"{'!'*60}\n")
            else:
                # Get confluence info
                data = self.fetch_data(symbol)
                if data:
                    ind = self.calculate_indicators(data['df'], data['df_daily'])
                    idx = len(ind['closes']) - 1
                    conf = 0
                    if ind['kill_zone'][idx]: conf += 15
                    if ind['htf_trend_hourly'][idx] != 0: conf += 10
                    conf += 15 if ind['price_position'][idx] < 0.35 else 0
                    logger.info(f"{symbol}: No signal (Confluence ~{conf})")


def main():
    parser = argparse.ArgumentParser(description='ICT V2 Alpaca Live Trading')
    parser.add_argument('--mode', type=str, default='signal',
                       choices=['signal', 'paper', 'live'], help='Trading mode')
    parser.add_argument('--symbols', type=str, default='NQ=F',
                       help='Symbols to trade (comma-separated)')
    parser.add_argument('--capital', type=float, default=10000,
                       help='Capital for position sizing')
    parser.add_argument('--risk', type=float, default=0.01,
                       help='Risk per trade')
    parser.add_argument('--interval', type=int, default=60,
                       help='Check interval in seconds')
    parser.add_argument('--test', action='store_true',
                       help='Test Alpaca connection')
    
    args = parser.parse_args()
    
    symbols = [s.strip() for s in args.symbols.split(',')]
    
    if args.test:
        from alpaca_broker import test_connection
        test_connection()
        return
    
    trader = ICTV2AlpacaTrader(
        symbols=symbols,
        capital=args.capital,
        mode=args.mode,
        risk_per_trade=args.risk
    )
    
    if args.mode == 'signal':
        trader.run_signal_check()
    else:
        trader.run(check_interval=args.interval)


if __name__ == "__main__":
    main()
