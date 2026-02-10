"""
ICT V2 Live Trading System
Based on backtest_unified_v2.py logic

Usage:
    python ict_v2_live.py --mode paper    # Paper trading
    python ict_v2_live.py --mode live      # Live trading (requires broker)
    python ict_v2_live.py --mode signal    # Signal only mode
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import argparse
import time
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, List
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TradeDirection(Enum):
    LONG = "long"
    SHORT = "short"
    NONE = "none"


@dataclass
class TradeSignal:
    timestamp: str
    symbol: str
    direction: TradeDirection
    entry_price: float
    stop_loss: float
    take_profit: float
    confluence: int
    grade: str
    risk_pct: float = 0.01
    position_size: float = 0.0


class ICTV2LiveTrader:
    def __init__(self, symbol: str = "NQ=F", mode: str = "paper"):
        self.symbol = symbol
        self.mode = mode
        self.capital = 10000.0
        self.risk_per_trade = 0.01  # 1%
        
        self.df = None
        self.df_daily = None
        self.bullish_fvgs = []
        self.bearish_fvgs = []
        self.bullish_obs = []
        self.bearish_obs = []
        self.htf_trend_hourly = np.array([])
        self.trend = np.array([])
        self.price_position = np.array([])
        self.kill_zone = np.array([])
        
        self.current_signal: Optional[TradeSignal] = None
        self.open_position = None
        
    def fetch_data(self, days: int = 60):
        """Fetch hourly and daily data"""
        logger.info(f"Fetching {self.symbol} data for {days} days...")
        self.df = yf.Ticker(self.symbol).history(period=f"{days}d", interval="1h")
        self.df = self.df.dropna()
        self.df = self.df[~self.df.index.duplicated(keep='first')]
        
        self.df_daily = yf.Ticker(self.symbol).history(period=f"{days}d", interval="1d")
        logger.info(f"Data: {len(self.df)} hourly bars, {len(self.df_daily)} daily bars")
        
    def calculate_indicators(self):
        """Pre-calculate all V2 indicators"""
        logger.info("Calculating indicators...")
        
        highs = self.df['High'].values
        lows = self.df['Low'].values
        closes = self.df['Close'].values
        opens = self.df['Open'].values
        
        # FVGs
        self.bullish_fvgs = []
        self.bearish_fvgs = []
        for i in range(3, len(self.df)):
            if lows[i] > highs[i-2]:
                self.bullish_fvgs.append({'idx': i, 'low': highs[i-2], 'high': lows[i], 'mid': (highs[i-2]+lows[i])/2})
            if highs[i] < lows[i-2]:
                self.bearish_fvgs.append({'idx': i, 'low': highs[i], 'high': lows[i-2], 'mid': (highs[i]+lows[i-2])/2})
        
        # Order Blocks
        self.bullish_obs = []
        self.bearish_obs = []
        for i in range(5, len(self.df)):
            if closes[i-1] < opens[i-1] and closes[i] > opens[i] and lows[i] < lows[i-1]:
                self.bullish_obs.append({'idx': i, 'high': highs[i-1], 'low': lows[i-1]})
            if closes[i-1] > opens[i-1] and closes[i] < opens[i] and highs[i] > highs[i-1]:
                self.bearish_obs.append({'idx': i, 'high': highs[i-1], 'low': lows[i-1]})
        
        # HTF Trend
        daily_highs = self.df_daily['High'].values
        daily_lows = self.df_daily['Low'].values
        df_daily_index = pd.DatetimeIndex(self.df_daily.index).tz_localize(None)
        df_index = pd.DatetimeIndex(self.df.index).tz_localize(None)
        
        htf_trend = []
        for i in range(1, len(self.df_daily)):
            if len(self.df_daily) < 5:
                htf_trend.append(0)
            elif daily_highs[i] > daily_highs[max(0,i-5):i].max() and daily_lows[i] > daily_lows[max(0,i-5):i].min():
                htf_trend.append(1)
            elif daily_highs[i] < daily_highs[max(0,i-5):i].max() and daily_lows[i] < daily_lows[max(0,i-5):i].min():
                htf_trend.append(-1)
            else:
                htf_trend.append(0)
        
        self.htf_trend_hourly = np.zeros(len(self.df))
        for i in range(len(self.df)):
            bar_time = df_index[i]
            for j in range(len(self.df_daily)-1, -1, -1):
                if df_daily_index[j] <= bar_time:
                    self.htf_trend_hourly[i] = htf_trend[j] if j < len(htf_trend) else 0
                    break
        
        # LTF Trend
        self.trend = np.zeros(len(self.df))
        for i in range(20, len(self.df)):
            rh = highs[max(0,i-20):i].max()
            rl = lows[max(0,i-20):i].min()
            if rh > highs[i-5] and rl > lows[i-5]:
                self.trend[i] = 1
            elif rh < highs[i-5] and rl < lows[i-5]:
                self.trend[i] = -1
        
        # Price Position
        self.price_position = np.zeros(len(self.df))
        for i in range(20, len(self.df)):
            ph = highs[i-20:i].max()
            pl = lows[i-20:i].min()
            self.price_position[i] = (closes[i] - pl) / (ph - pl + 0.001)
        
        # Kill Zones
        hours = pd.to_datetime(self.df.index).hour.values
        self.kill_zone = np.zeros(len(self.df), dtype=bool)
        for i in range(len(hours)):
            h = hours[i]
            self.kill_zone[i] = (1 <= h < 5) or (7 <= h < 12) or (13.5 <= h < 16)
        
        logger.info("Indicators calculated")
        
    def get_current_confluence(self, idx: int) -> tuple:
        """Calculate confluence score for current bar"""
        highs = self.df['High'].values
        lows = self.df['Low'].values
        closes = self.df['Close'].values
        
        current_price = closes[idx]
        current_trend = self.trend[idx]
        htf_bias = self.htf_trend_hourly[idx]
        kz = self.kill_zone[idx]
        pp = self.price_position[idx]
        
        nearest_bull = next((ob for ob in reversed(self.bullish_obs) if ob['idx'] < idx), None)
        nearest_bear = next((ob for ob in reversed(self.bearish_obs) if ob['idx'] < idx), None)
        near_bull_fvg = next((f for f in reversed(self.bullish_fvgs) if f['idx'] < idx and f['mid'] < current_price < f['high']), None)
        near_bear_fvg = next((f for f in reversed(self.bearish_fvgs) if f['idx'] < idx and f['low'] < current_price < f['mid']), None)
        
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
        elif confluence >= 60:
            grade = 'B'
        elif confluence >= 50:
            grade = 'C'
        
        return confluence, grade, {
            'nearest_bull': nearest_bull,
            'nearest_bear': nearest_bear,
            'near_bull_fvg': near_bull_fvg,
            'near_bear_fvg': near_bear_fvg
        }
    
    def check_entry_signal(self, idx: int) -> Optional[TradeSignal]:
        """Check for entry signal at current bar"""
        highs = self.df['High'].values
        lows = self.df['Low'].values
        closes = self.df['Close'].values
        
        current_price = closes[idx]
        confluence, grade, ctx = self.get_current_confluence(idx)
        
        # AI Filter (V2 rules)
        if confluence < 70 or grade not in ['A+', 'A']:
            return None
        
        htf_bias = self.htf_trend_hourly[idx]
        kz = self.kill_zone[idx]
        pp = self.price_position[idx]
        
        if htf_bias == 0 or not kz:
            return None
        
        if not (ctx['near_bull_fvg'] or ctx['nearest_bull'] or ctx['near_bear_fvg'] or ctx['nearest_bear']):
            return None
        
        atr = (highs[idx-14:idx] - lows[idx-14:idx]).mean() if idx > 14 else 50
        
        # Long entry
        if pp < 0.40 and htf_bias == 1:
            if ctx['near_bull_fvg'] and current_price > ctx['near_bull_fvg']['mid']:
                sl = ctx['near_bull_fvg']['low'] - atr * 0.5
                tp = current_price + atr * 2.5
                return self._create_signal(idx, TradeDirection.LONG, current_price, sl, tp, confluence, grade)
            elif ctx['nearest_bull'] and current_price > ctx['nearest_bull']['high']:
                sl = ctx['nearest_bull']['low'] - atr * 0.5
                tp = current_price + atr * 2.5
                return self._create_signal(idx, TradeDirection.LONG, current_price, sl, tp, confluence, grade)
        
        # Short entry
        elif pp > 0.60 and htf_bias == -1:
            if ctx['near_bear_fvg'] and current_price < ctx['near_bear_fvg']['mid']:
                sl = ctx['near_bear_fvg']['high'] + atr * 0.5
                tp = current_price - atr * 2.5
                return self._create_signal(idx, TradeDirection.SHORT, current_price, sl, tp, confluence, grade)
            elif ctx['nearest_bear'] and current_price < ctx['nearest_bear']['low']:
                sl = ctx['nearest_bear']['high'] + atr * 0.5
                tp = current_price - atr * 2.5
                return self._create_signal(idx, TradeDirection.SHORT, current_price, sl, tp, confluence, grade)
        
        return None
    
    def _create_signal(self, idx: int, direction: TradeDirection, entry: float, 
                       sl: float, tp: float, confluence: int, grade: str) -> TradeSignal:
        """Create a trade signal with position sizing"""
        risk_amt = self.capital * self.risk_per_trade
        risk = abs(entry - sl)
        size = risk_amt / risk if risk > 0 else 1.0
        
        return TradeSignal(
            timestamp=str(self.df.index[idx]),
            symbol=self.symbol,
            direction=direction,
            entry_price=entry,
            stop_loss=sl,
            take_profit=tp,
            confluence=confluence,
            grade=grade,
            risk_pct=self.risk_per_trade,
            position_size=size
        )
    
    def run_analysis(self, lookback_bars: int = 100):
        """Run full analysis on recent data"""
        self.fetch_data()
        self.calculate_indicators()
        
        closes = self.df['Close'].values
        logger.info(f"\n{'='*60}")
        logger.info(f"ICT V2 LIVE ANALYSIS - {self.symbol}")
        logger.info(f"{'='*60}")
        logger.info(f"Current Price: {closes[-1]:.2f}")
        logger.info(f"HTF Trend: {self.htf_trend_hourly[-1]}")
        logger.info(f"LTF Trend: {self.trend[-1]}")
        logger.info(f"Price Position: {self.price_position[-1]:.2f}")
        logger.info(f"In Kill Zone: {self.kill_zone[-1]}")
        
        # Check last 10 bars for signals
        logger.info(f"\nChecking recent bars for signals...")
        signals_found = 0
        for idx in range(-20, 0):
            signal = self.check_entry_signal(idx)
            if signal:
                signals_found += 1
                logger.info(f"  SIGNAL #{signals_found}: {signal.direction.value.upper()} at {signal.entry_price:.2f}")
                logger.info(f"    Confluence: {signal.confluence} ({signal.grade})")
                logger.info(f"    SL: {signal.stop_loss:.2f}, TP: {signal.take_profit:.2f}")
        
        if signals_found == 0:
            logger.info("  No signals found in recent bars")
        
        # Current bar analysis
        current_conf, current_grade, _ = self.get_current_confluence(len(self.df) - 1)
        logger.info(f"\nCurrent Bar Analysis:")
        logger.info(f"  Confluence Score: {current_conf} ({current_grade})")
        logger.info(f"  Signal Ready: {'YES' if current_conf >= 70 else 'NO'}")
        
        return signals_found > 0
    
    def run_live_monitoring(self, check_interval: int = 60):
        """Run continuous live monitoring"""
        logger.info(f"Starting live monitoring in {self.mode} mode...")
        logger.info(f"Check interval: {check_interval} seconds")
        
        while True:
            try:
                # Refresh data
                self.fetch_data(days=3)  # Only need recent data
                self.calculate_indicators()
                
                # Check for signals
                idx = len(self.df) - 1
                signal = self.check_entry_signal(idx)
                
                if signal:
                    logger.info(f"\n{'!'*60}")
                    logger.info(f"NEW SIGNAL: {signal.direction.value.upper()}")
                    logger.info(f"Symbol: {signal.symbol}")
                    logger.info(f"Entry: {signal.entry_price:.2f}")
                    logger.info(f"Stop Loss: {signal.stop_loss:.2f}")
                    logger.info(f"Take Profit: {signal.take_profit:.2f}")
                    logger.info(f"Confluence: {signal.confluence} ({signal.grade})")
                    logger.info(f"Position Size: {signal.position_size:.2f} contracts")
                    logger.info(f"{'!'*60}\n")
                    
                    if self.mode == "live":
                        self.execute_trade(signal)
                    elif self.mode == "paper":
                        self.execute_paper_trade(signal)
                
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                logger.info("Stopping live monitoring...")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                time.sleep(check_interval)
    
    def execute_trade(self, signal: TradeSignal):
        """Execute live trade (placeholder for broker integration)"""
        logger.info(f"EXECUTING LIVE TRADE: {signal.direction.value.upper()} {signal.symbol}")
        logger.info(f"  Entry: {signal.entry_price}")
        logger.info(f"  Size: {signal.position_size}")
        # Add broker API calls here
    
    def execute_paper_trade(self, signal: TradeSignal):
        """Execute paper trade"""
        logger.info(f"PAPER TRADE: {signal.direction.value.upper()} {signal.symbol}")
        logger.info(f"  Entry: {signal.entry_price}")
        logger.info(f"  Size: {signal.position_size}")
        logger.info(f"  SL: {signal.stop_loss}")
        logger.info(f"  TP: {signal.take_profit}")
        
        self.open_position = {
            'signal': signal,
            'entry_time': datetime.now(),
            'entry_price': signal.entry_price
        }
        logger.info(f"  Position opened. Monitoring...")


def main():
    parser = argparse.ArgumentParser(description='ICT V2 Live Trading System')
    parser.add_argument('--symbol', type=str, default='NQ=F', help='Symbol to trade')
    parser.add_argument('--mode', type=str, default='signal', 
                       choices=['signal', 'paper', 'live'], help='Trading mode')
    parser.add_argument('--interval', type=int, default=60, help='Check interval in seconds')
    args = parser.parse_args()
    
    trader = ICTV2LiveTrader(symbol=args.symbol, mode=args.mode)
    
    if args.mode == 'signal':
        trader.run_analysis()
    else:
        trader.run_live_monitoring(check_interval=args.interval)


if __name__ == "__main__":
    main()
