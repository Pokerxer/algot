"""
ICT V2 Multi-Symbol Paper Trader
Monitors NQ, YM, EURUSD for signals in real-time
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import time
import argparse
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Dict, List
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.FileHandler('v2_paper_trades.log'),
        logging.StreamHandler()
    ]
)
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


@dataclass
class OpenPosition:
    signal: TradeSignal
    entry_time: datetime
    entry_price: float
    current_pnl: float = 0.0
    status: str = "open"


class ICTV2Trader:
    def __init__(self, symbols: List[str] = None, capital: float = 10000.0):
        self.symbols = symbols or ["NQ=F", "YM=F", "EURUSD=X"]
        self.capital = capital
        self.risk_per_trade = 0.01
        self.positions: Dict[str, OpenPosition] = {}
        self.trade_history: List[Dict] = []
        
        self.data_cache: Dict[str, Dict] = {}
        
    def fetch_symbol_data(self, symbol: str, days: int = 5) -> Dict:
        """Fetch and cache recent data"""
        if symbol in self.data_cache:
            age = datetime.now() - self.data_cache[symbol]['fetch_time']
            if age.total_seconds() < 300:  # 5 min cache
                return self.data_cache[symbol]
        
        try:
            df = yf.Ticker(symbol).history(period=f"{days}d", interval="1h")
            df_daily = yf.Ticker(symbol).history(period=f"{days}d", interval="1d")
            
            result = {
                'df': df,
                'df_daily': df_daily,
                'fetch_time': datetime.now()
            }
            self.data_cache[symbol] = result
            return result
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None
    
    def calculate_indicators(self, df, df_daily):
        """Calculate all V2 indicators"""
        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        opens = df['Open'].values
        
        # FVGs
        bullish_fvgs = []
        bearish_fvgs = []
        for i in range(3, len(df)):
            if lows[i] > highs[i-2]:
                bullish_fvgs.append({'idx': i, 'low': highs[i-2], 'high': lows[i], 'mid': (highs[i-2]+lows[i])/2})
            if highs[i] < lows[i-2]:
                bearish_fvgs.append({'idx': i, 'low': highs[i], 'high': lows[i-2], 'mid': (highs[i]+lows[i-2])/2})
        
        # Order Blocks
        bullish_obs = []
        bearish_obs = []
        for i in range(5, len(df)):
            if closes[i-1] < opens[i-1] and closes[i] > opens[i] and lows[i] < lows[i-1]:
                bullish_obs.append({'idx': i, 'high': highs[i-1], 'low': lows[i-1]})
            if closes[i-1] > opens[i-1] and closes[i] < opens[i] and highs[i] > highs[i-1]:
                bearish_obs.append({'idx': i, 'high': highs[i-1], 'low': lows[i-1]})
        
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
            price_position[i] = (closes[i] - pl) / (ph - pl + 0.001)
        
        # Kill Zones
        hours = pd.to_datetime(df.index).hour.values
        kill_zone = np.zeros(len(df), dtype=bool)
        for i in range(len(hours)):
            h = hours[i]
            kill_zone[i] = (1 <= h < 5) or (7 <= h < 12) or (13.5 <= h < 16)
        
        return {
            'highs': highs,
            'lows': lows,
            'closes': closes,
            'bullish_fvgs': bullish_fvgs,
            'bearish_fvgs': bearish_fvgs,
            'bullish_obs': bullish_obs,
            'bearish_obs': bearish_obs,
            'htf_trend_hourly': htf_trend_hourly,
            'trend': trend,
            'price_position': price_position,
            'kill_zone': kill_zone
        }
    
    def check_signal(self, symbol: str) -> Optional[TradeSignal]:
        """Check for trade signal on symbol"""
        data = self.fetch_symbol_data(symbol)
        if not data:
            return None
        
        ind = self.calculate_indicators(data['df'], data['df_daily'])
        
        idx = len(ind['closes']) - 1
        current_price = ind['closes'][idx]
        current_trend = ind['trend'][idx]
        htf_bias = ind['htf_trend_hourly'][idx]
        kz = ind['kill_zone'][idx]
        pp = ind['price_position'][idx]
        
        # Nearest OB/FVG
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
        
        # Grade
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
        
        # Long
        if pp < 0.40 and htf_bias == 1:
            if near_bull_fvg and current_price > near_bull_fvg['mid']:
                sl = near_bull_fvg['low'] - atr * 0.5
                tp = current_price + atr * 2.5
                return self._create_signal(symbol, idx, TradeDirection.LONG, current_price, sl, tp, confluence, grade)
            elif nearest_bull and current_price > nearest_bull['high']:
                sl = nearest_bull['low'] - atr * 0.5
                tp = current_price + atr * 2.5
                return self._create_signal(symbol, idx, TradeDirection.LONG, current_price, sl, tp, confluence, grade)
        
        # Short
        elif pp > 0.60 and htf_bias == -1:
            if near_bear_fvg and current_price < near_bear_fvg['mid']:
                sl = near_bear_fvg['high'] + atr * 0.5
                tp = current_price - atr * 2.5
                return self._create_signal(symbol, idx, TradeDirection.SHORT, current_price, sl, tp, confluence, grade)
            elif nearest_bear and current_price < nearest_bear['low']:
                sl = nearest_bear['high'] + atr * 0.5
                tp = current_price - atr * 2.5
                return self._create_signal(symbol, idx, TradeDirection.SHORT, current_price, sl, tp, confluence, grade)
        
        return None
    
    def _create_signal(self, symbol: str, idx: int, direction: TradeDirection, 
                       entry: float, sl: float, tp: float, confluence: int, grade: str) -> TradeSignal:
        risk_amt = self.capital * self.risk_per_trade
        risk = abs(entry - sl)
        size = risk_amt / risk if risk > 0 else 1.0
        
        return TradeSignal(
            timestamp=str(self.data_cache[symbol]['df'].index[idx]),
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
    
    def open_paper_trade(self, signal: TradeSignal):
        """Open a paper trade"""
        if signal.symbol in self.positions:
            logger.info(f"{signal.symbol} - Already have open position, skipping")
            return
        
        self.positions[signal.symbol] = OpenPosition(
            signal=signal,
            entry_time=datetime.now(),
            entry_price=signal.entry_price,
            status="open"
        )
        
        logger.info(f"=" * 60)
        logger.info(f"PAPER TRADE OPENED")
        logger.info(f"Symbol: {signal.symbol}")
        logger.info(f"Direction: {signal.direction.value.upper()}")
        logger.info(f"Entry: {signal.entry_price:.2f}")
        logger.info(f"SL: {signal.stop_loss:.2f}")
        logger.info(f"TP: {signal.take_profit:.2f}")
        logger.info(f"Confluence: {signal.confluence} ({signal.grade})")
        logger.info(f"Size: {signal.position_size:.2f}")
        logger.info(f"=" * 60)
    
    def check_positions(self):
        """Check open positions for SL/TP hits"""
        for symbol, pos in list(self.positions.items()):
            data = self.fetch_symbol_data(symbol, days=1)
            if not data:
                continue
            
            current_price = data['df']['Close'].values[-1]
            signal = pos.signal
            
            pnl = (current_price - signal.entry_price) * signal.position_size * 20 if signal.direction == TradeDirection.LONG else (signal.entry_price - current_price) * signal.position_size * 20
            
            if signal.direction == TradeDirection.LONG:
                if current_price <= signal.stop_loss:
                    self._close_position(symbol, current_price, "STOP_HIT", pnl)
                elif current_price >= signal.take_profit:
                    self._close_position(symbol, current_price, "TP_HIT", pnl)
            else:
                if current_price >= signal.stop_loss:
                    self._close_position(symbol, current_price, "STOP_HIT", pnl)
                elif current_price <= signal.take_profit:
                    self._close_position(symbol, current_price, "TP_HIT", pnl)
            
            pos.current_pnl = pnl
    
    def _close_position(self, symbol: str, exit_price: float, reason: str, pnl: float):
        """Close a position"""
        pos = self.positions.pop(symbol)
        trade_record = {
            'symbol': symbol,
            'direction': pos.signal.direction.value,
            'entry_price': pos.signal.entry_price,
            'exit_price': exit_price,
            'exit_reason': reason,
            'pnl': pnl,
            'confluence': pos.signal.confluence,
            'grade': pos.signal.grade,
            'entry_time': pos.entry_time.isoformat(),
            'exit_time': datetime.now().isoformat()
        }
        self.trade_history.append(trade_record)
        
        logger.info(f"POSITION CLOSED: {symbol}")
        logger.info(f"  Reason: {reason}")
        logger.info(f"  Exit: {exit_price:.2f}")
        logger.info(f"  PnL: ${pnl:+,.2f}")
        
        self.capital += pnl
        logger.info(f"  Account Equity: ${self.capital:,.2f}")
    
    def print_status(self):
        """Print current status"""
        logger.info(f"\n{'='*60}")
        logger.info(f"V2 PAPER TRADING STATUS")
        logger.info(f"{'='*60}")
        logger.info(f"Account: ${self.capital:,.2f}")
        logger.info(f"Open Positions: {len(self.positions)}")
        logger.info(f"Completed Trades: {len(self.trade_history)}")
        
        if self.positions:
            logger.info(f"\nOpen Positions:")
            for symbol, pos in self.positions.items():
                logger.info(f"  {symbol}: {pos.signal.direction.value.upper()} @ {pos.signal.entry_price:.2f} | PnL: ${pos.current_pnl:+,.0f}")
        
        wins = len([t for t in self.trade_history if t['pnl'] > 0])
        if self.trade_history:
            logger.info(f"\nWin Rate: {wins}/{len(self.trade_history)} ({wins/len(self.trade_history)*100:.1f}%)")
        logger.info(f"{'='*60}\n")
    
    def run(self, check_interval: int = 60):
        """Run paper trading loop"""
        logger.info(f"Starting V2 Paper Trading")
        logger.info(f"Symbols: {', '.join(self.symbols)}")
        logger.info(f"Capital: ${self.capital:,.2f}")
        logger.info(f"Risk per Trade: {self.risk_per_trade*100}%")
        logger.info(f"Check Interval: {check_interval}s")
        
        while True:
            try:
                # Check for new signals
                for symbol in self.symbols:
                    signal = self.check_signal(symbol)
                    if signal:
                        self.open_paper_trade(signal)
                
                # Check existing positions
                self.check_positions()
                
                # Print status every 5 checks
                self.print_status()
                
                time.sleep(check_interval)
                
            except KeyboardInterrupt:
                logger.info("Stopping paper trading...")
                break
            except Exception as e:
                logger.error(f"Error: {e}")
                time.sleep(check_interval)
        
        # Save trade history
        with open('v2_paper_trades.json', 'w') as f:
            json.dump(self.trade_history, f, indent=2)
        logger.info(f"Trade history saved to v2_paper_trades.json")


def main():
    parser = argparse.ArgumentParser(description='ICT V2 Paper Trading')
    parser.add_argument('--symbols', type=str, default="NQ=F,YM=F,EURUSD=X", help='Symbols to monitor')
    parser.add_argument('--capital', type=float, default=10000, help='Starting capital')
    parser.add_argument('--interval', type=int, default=60, help='Check interval in seconds')
    args = parser.parse_args()
    
    symbols = [s.strip() for s in args.symbols.split(',')]
    trader = ICTV2Trader(symbols=symbols, capital=args.capital)
    trader.run(check_interval=args.interval)


if __name__ == "__main__":
    main()
