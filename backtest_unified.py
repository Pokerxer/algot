"""
ICT Unified Handler - 6 Month Backtest
=======================================

Backtest the complete Phase 1 integrated handler on NQ with 6 months of live data.
Starting Capital: $10,000

Uses all Phase 1 components via ICTUnifiedHandler:
- TimeframeHandler
- MarketStructureHandler
- OrderBlockHandler
- FVGHandler
- LiquidityHandler
- PDArrayHandler
- TradingModelHandler
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Optional
from enum import Enum
import json
import logging
import pytz

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class TradeDirection(Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


@dataclass
class Trade:
    entry_idx: int
    entry_time: str
    entry_price: float
    direction: TradeDirection
    size: float
    stop_loss: float
    take_profit: float
    exit_idx: int = 0
    exit_time: str = ""
    exit_price: float = 0.0
    pnl: float = 0.0
    status: str = "OPEN"
    setup_type: str = ""
    confidence: float = 0.0
    grade: str = "C"
    confluence_score: int = 0


class Backtester:
    """Backtester using ICT Unified Handler"""
    
    def __init__(self, initial_capital: float = 10000):
        self.capital = initial_capital
        self.initial_capital = initial_capital
        self.trades: List[Trade] = []
        self.equity_curve = []
        
        # Import unified handler
        from ict_unified_handler import ICTUnifiedHandler
        self.handler = ICTUnifiedHandler(symbol="NQ")
        
        logger.info(f"Backtester initialized | Capital: ${initial_capital:,.0f}")
    
    def run(self, df: pd.DataFrame) -> Dict:
        """Run backtest"""
        logger.info(f"Starting backtest: {len(df)} bars")
        logger.info(f"Period: {df.index[0]} to {df.index[-1]}")
        
        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        timestamps = df.index.values
        
        open_trade = None
        
        for idx in range(50, len(df)):
            # Get analysis using unified handler
            current_df = df.iloc[:idx+1]
            analysis = self.handler.analyze(current_df)
            
            current_price = closes[idx]
            current_time = str(df.index[idx])[:19]
            
            # Update equity
            self.equity_curve.append({
                'date': str(df.index[idx])[:10],
                'equity': self.capital
            })
            
            # Check exits for open trade
            if open_trade:
                if open_trade.direction == TradeDirection.LONG:
                    if current_price <= open_trade.stop_loss:
                        open_trade.exit_idx = idx
                        open_trade.exit_time = current_time
                        open_trade.exit_price = open_trade.stop_loss
                        open_trade.pnl = (open_trade.exit_price - open_trade.entry_price) * open_trade.size * 20
                        open_trade.status = "STOP_HIT"
                        self.capital += open_trade.pnl
                        self.trades.append(open_trade)
                        open_trade = None
                    
                    elif current_price >= open_trade.take_profit:
                        open_trade.exit_idx = idx
                        open_trade.exit_time = current_time
                        open_trade.exit_price = open_trade.take_profit
                        open_trade.pnl = (open_trade.exit_price - open_trade.entry_price) * open_trade.size * 20
                        open_trade.status = "TP_HIT"
                        self.capital += open_trade.pnl
                        self.trades.append(open_trade)
                        open_trade = None
                
                else:  # SHORT
                    if current_price >= open_trade.stop_loss:
                        open_trade.exit_idx = idx
                        open_trade.exit_time = current_time
                        open_trade.exit_price = open_trade.stop_loss
                        open_trade.pnl = (open_trade.entry_price - open_trade.exit_price) * open_trade.size * 20
                        open_trade.status = "STOP_HIT"
                        self.capital += open_trade.pnl
                        self.trades.append(open_trade)
                        open_trade = None
                    
                    elif current_price <= open_trade.take_profit:
                        open_trade.exit_idx = idx
                        open_trade.exit_time = current_time
                        open_trade.exit_price = open_trade.take_profit
                        open_trade.pnl = (open_trade.entry_price - open_trade.exit_price) * open_trade.size * 20
                        open_trade.status = "TP_HIT"
                        self.capital += open_trade.pnl
                        self.trades.append(open_trade)
                        open_trade = None
                
                # Time exit after 20 bars
                if open_trade and (idx - open_trade.entry_idx) > 20:
                    open_trade.exit_idx = idx
                    open_trade.exit_time = current_time
                    open_trade.exit_price = current_price
                    open_trade.pnl = (open_trade.entry_price - current_price) * open_trade.size * 20 if open_trade.direction == TradeDirection.SHORT else (current_price - open_trade.entry_price) * open_trade.size * 20
                    open_trade.status = "TIME_EXIT"
                    self.capital += open_trade.pnl
                    self.trades.append(open_trade)
                    open_trade = None
            
            # Check for new signals (only if no open trade)
            if open_trade is None and analysis.confidence > 0.5:
                risk_pct = self.capital * 0.02
                atr = (highs[max(0,idx-14):idx] - lows[max(0,idx-14):idx]).mean() if idx > 14 else 50
                
                if analysis.long_signal:
                    direction = TradeDirection.LONG
                    sl = current_price - atr * 1.5
                    tp = current_price + atr * 3
                    risk = current_price - sl
                    size = risk_pct / risk if risk > 0 else 1
                    
                    open_trade = Trade(
                        entry_idx=idx,
                        entry_time=current_time,
                        entry_price=current_price,
                        direction=direction,
                        size=size,
                        stop_loss=sl,
                        take_profit=tp,
                        setup_type="Unified Long",
                        confidence=analysis.confidence,
                        grade=analysis.grade,
                        confluence_score=analysis.confluence_score
                    )
                
                elif analysis.short_signal:
                    direction = TradeDirection.SHORT
                    sl = current_price + atr * 1.5
                    tp = current_price - atr * 3
                    risk = sl - current_price
                    size = risk_pct / risk if risk > 0 else 1
                    
                    open_trade = Trade(
                        entry_idx=idx,
                        entry_time=current_time,
                        entry_price=current_price,
                        direction=direction,
                        size=size,
                        stop_loss=sl,
                        take_profit=tp,
                        setup_type="Unified Short",
                        confidence=analysis.confidence,
                        grade=analysis.grade,
                        confluence_score=analysis.confluence_score
                    )
            
            if idx % 500 == 0:
                logger.info(f"Progress: {idx}/{len(df)} | Equity: ${self.capital:,.0f}")
        
        # Close open trade at end
        if open_trade:
            open_trade.exit_idx = len(df) - 1
            open_trade.exit_time = str(df.index[-1])[:19]
            open_trade.exit_price = closes[-1]
            open_trade.pnl = (open_trade.entry_price - closes[-1]) * open_trade.size * 20 if open_trade.direction == TradeDirection.SHORT else (closes[-1] - open_trade.entry_price) * open_trade.size * 20
            open_trade.status = "EOD"
            self.trades.append(open_trade)
        
        # Calculate statistics
        return self._calculate_stats()
    
    def _calculate_stats(self) -> Dict:
        closed = [t for t in self.trades if t.status != "OPEN"]
        winners = [t for t in closed if t.pnl > 0]
        losers = [t for t in closed if t.pnl <= 0]
        
        total_return = (self.capital - self.initial_capital) / self.initial_capital * 100
        win_rate = len(winners) / len(closed) * 100 if closed else 0
        
        max_eq = max(e['equity'] for e in self.equity_curve) if self.equity_curve else self.initial_capital
        min_eq = min(e['equity'] for e in self.equity_curve) if self.equity_curve else self.initial_capital
        max_dd = (max_eq - min_eq) / max_eq * 100 if max_eq > 0 else 0
        
        profit = sum(t.pnl for t in winners)
        loss = abs(sum(t.pnl for t in losers))
        pf = profit / loss if loss > 0 else float('inf')
        
        # By setup type
        setups = {}
        for t in closed:
            if t.setup_type:
                key = t.setup_type.split()[0]
                if key not in setups:
                    setups[key] = {'count': 0, 'wins': 0, 'pnl': 0}
                setups[key]['count'] += 1
                if t.pnl > 0:
                    setups[key]['wins'] += 1
                setups[key]['pnl'] += t.pnl
        
        # By grade
        grades = {}
        for t in closed:
            if t.grade not in grades:
                grades[t.grade] = {'count': 0, 'wins': 0, 'pnl': 0}
            grades[t.grade]['count'] += 1
            if t.pnl > 0:
                grades[t.grade]['wins'] += 1
            grades[t.grade]['pnl'] += t.pnl
        
        # By confluence score bucket
        confluence_buckets = {'High (70+)': {'count': 0, 'wins': 0, 'pnl': 0}, 
                            'Medium (40-69)': {'count': 0, 'wins': 0, 'pnl': 0}, 
                            'Low (<40)': {'count': 0, 'wins': 0, 'pnl': 0}}
        for t in closed:
            if t.confluence_score >= 70:
                confluence_buckets['High (70+)']['count'] += 1
                if t.pnl > 0:
                    confluence_buckets['High (70+)']['wins'] += 1
                confluence_buckets['High (70+)']['pnl'] += t.pnl
            elif t.confluence_score >= 40:
                confluence_buckets['Medium (40-69)']['count'] += 1
                if t.pnl > 0:
                    confluence_buckets['Medium (40-69)']['wins'] += 1
                confluence_buckets['Medium (40-69)']['pnl'] += t.pnl
            else:
                confluence_buckets['Low (<40)']['count'] += 1
                if t.pnl > 0:
                    confluence_buckets['Low (<40)']['wins'] += 1
                confluence_buckets['Low (<40)']['pnl'] += t.pnl
        
        return {
            'period': {
                'start': str(self.equity_curve[0]['date']) if self.equity_curve else 'N/A',
                'end': str(self.equity_curve[-1]['date']) if self.equity_curve else 'N/A',
                'bars': len(self.equity_curve)
            },
            'capital': {
                'initial': self.initial_capital,
                'final': self.capital,
                'return_pct': total_return
            },
            'trades': {
                'total': len(closed),
                'winners': len(winners),
                'losers': len(losers),
                'win_rate': win_rate
            },
            'pnl': {
                'gross_profit': profit,
                'gross_loss': loss,
                'net_pnl': self.capital - self.initial_capital,
                'profit_factor': pf
            },
            'risk': {
                'max_drawdown_pct': max_dd
            },
            'setup_breakdown': setups,
            'grade_breakdown': grades,
            'confluence_breakdown': confluence_buckets
        }
    
    def save_results(self, filepath: str = "unified_backtest_results.json"):
        """Save results to JSON"""
        results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'version': '1.0.0',
                'handler': 'ICTUnifiedHandler (Phase 1 Complete)'
            },
            'statistics': self._calculate_stats(),
            'equity_curve': self.equity_curve,
            'trades': [
                {
                    'entry_time': t.entry_time,
                    'entry_price': t.entry_price,
                    'direction': t.direction.value,
                    'size': t.size,
                    'stop_loss': t.stop_loss,
                    'take_profit': t.take_profit,
                    'exit_time': t.exit_time,
                    'exit_price': t.exit_price,
                    'pnl': t.pnl,
                    'status': t.status,
                    'setup_type': t.setup_type,
                    'grade': t.grade,
                    'confidence': t.confidence,
                    'confluence_score': t.confluence_score
                }
                for t in self.trades
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {filepath}")
        return filepath


def main():
    print("=" * 70)
    print("ICT UNIFIED HANDLER - 6 MONTH BACKTEST")
    print("=" * 70)
    print(f"Starting Capital: $10,000")
    print()
    
    # Fetch data
    logger.info("Fetching NQ data...")
    df = yf.Ticker("NQ=F").history(period="6mo", interval="1h")
    df = df.dropna()
    df = df[~df.index.duplicated(keep='first')]
    
    logger.info(f"Data: {len(df)} bars | {df.index[0]} to {df.index[-1]}")
    print()
    
    # Run backtest
    backtester = Backtester(initial_capital=10000)
    stats = backtester.run(df)
    
    # Save results
    output_file = backtester.save_results("unified_backtest_results.json")
    
    # Print summary
    print()
    print("=" * 70)
    print("BACKTEST RESULTS - ICT UNIFIED HANDLER (PHASE 1)")
    print("=" * 70)
    print(f"Period: {stats['period']['start']} to {stats['period']['end']} ({stats['period']['bars']} bars)")
    print()
    print(f"CAPITAL:")
    print(f"  Initial:    ${stats['capital']['initial']:>10,.0f}")
    print(f"  Final:      ${stats['capital']['final']:>10,.0f}")
    print(f"  Return:     {stats['capital']['return_pct']:>10.2f}%")
    print()
    print(f"TRADES: {stats['trades']['total']} total | Win Rate: {stats['trades']['win_rate']:.1f}%")
    print(f"  Winners: {stats['trades']['winners']} | Losers: {stats['trades']['losers']}")
    print()
    print(f"P&L:")
    print(f"  Gross Profit:  ${stats['pnl']['gross_profit']:>10,.0f}")
    print(f"  Gross Loss:    ${stats['pnl']['gross_loss']:>10,.0f}")
    print(f"  Net PnL:       ${stats['pnl']['net_pnl']:>10,.0f}")
    print(f"  Profit Factor: {stats['pnl']['profit_factor']:>10.2f}")
    print()
    print(f"RISK:")
    print(f"  Max Drawdown:  {stats['risk']['max_drawdown_pct']:.2f}%")
    print()
    print("SETUP BREAKDOWN:")
    for setup, data in stats['setup_breakdown'].items():
        wr = data['wins'] / data['count'] * 100 if data['count'] > 0 else 0
        print(f"  {setup:12} | {data['count']:3} trades | {wr:5.1f}% win | ${data['pnl']:+,.0f}")
    print()
    print("GRADE BREAKDOWN:")
    for grade, data in sorted(stats['grade_breakdown'].items()):
        wr = data['wins'] / data['count'] * 100 if data['count'] > 0 else 0
        print(f"  Grade {grade}: {data['count']:2} trades | {wr:5.1f}% win | ${data['pnl']:+,.0f}")
    print()
    print("CONFLUENCE SCORE BREAKDOWN:")
    for bucket, data in stats['confluence_breakdown'].items():
        wr = data['wins'] / data['count'] * 100 if data['count'] > 0 else 0
        print(f"  {bucket}: {data['count']:3} trades | {wr:5.1f}% win | ${data['pnl']:+,.0f}")
    print()
    print(f"Results saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
