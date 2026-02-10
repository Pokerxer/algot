"""
ICT Backtester - NQ 6-Month Test (Optimized)
=============================================
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List, Tuple
from enum import Enum
import json
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class TradeDirection(Enum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


@dataclass
class Trade:
    entry_idx: int
    entry_price: float
    direction: TradeDirection
    size: float
    stop_loss: float
    take_profit: float
    exit_idx: int = 0
    exit_price: float = 0.0
    pnl: float = 0.0
    status: str = "OPEN"
    setup_type: str = ""


def identify_fvg_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized FVG identification"""
    highs = df['High'].values
    lows = df['Low'].values
    
    fvg_data = []
    for i in range(3, len(df)):
        # Bullish FVG
        if lows[i] > highs[i-2]:
            fvg_data.append({
                'idx': i, 'type': 'BULLISH', 
                'low': highs[i-2], 'high': lows[i],
                'size': lows[i] - highs[i-2], 'filled': False
            })
        # Bearish FVG
        if highs[i] < lows[i-2]:
            fvg_data.append({
                'idx': i, 'type': 'BEARISH',
                'low': highs[i], 'high': lows[i-2],
                'size': lows[i-2] - highs[i], 'filled': False
            })
    
    return pd.DataFrame(fvg_data)


def identify_ob_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized Order Block identification"""
    opens = df['Open'].values
    closes = df['Close'].values
    highs = df['High'].values
    lows = df['Low'].values
    
    ob_data = []
    for i in range(5, len(df)):
        # Bullish OB (bearish candle followed by break of low)
        if closes[i-1] < opens[i-1] and closes[i] > opens[i] and lows[i] < lows[i-1]:
            ob_data.append({
                'idx': i, 'type': 'BULLISH',
                'low': lows[i-1], 'high': highs[i-1],
                'strength': 0.5
            })
        # Bearish OB
        if closes[i-1] > opens[i-1] and closes[i] < opens[i] and highs[i] > highs[i-1]:
            ob_data.append({
                'idx': i, 'type': 'BEARISH',
                'low': lows[i-1], 'high': highs[i-1],
                'strength': 0.5
            })
    
    return pd.DataFrame(ob_data)


def analyze_structure_vectorized(df: pd.DataFrame) -> np.ndarray:
    """Vectorized structure analysis"""
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    
    trend = np.zeros(len(df))  # 1=bullish, -1=bearish, 0=neutral
    
    for i in range(20, len(df)):
        recent_highs = highs[max(0,i-20):i+1]
        recent_lows = lows[max(0,i-20):i+1]
        
        if len(recent_highs) >= 2 and len(recent_lows) >= 2:
            higher_highs = recent_highs[-1] > recent_highs[-2]
            higher_lows = recent_lows[-1] > recent_lows[-2]
            lower_highs = recent_highs[-1] < recent_highs[-2]
            lower_lows = recent_lows[-1] < recent_lows[-2]
            
            if higher_highs and higher_lows:
                trend[i] = 1
            elif lower_highs and lower_lows:
                trend[i] = -1
    
    return trend


def run_backtest(df: pd.DataFrame, initial_capital: float = 10000) -> Dict:
    """Fast backtest execution"""
    logger.info(f"Running backtest: {len(df)} bars | Capital: ${initial_capital:,.0f}")
    
    # Pre-compute all indicators
    logger.info("Computing indicators...")
    fvgs = identify_fvg_vectorized(df)
    obs = identify_ob_vectorized(df)
    trend = analyze_structure_vectorized(df)
    
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    timestamps = df.index.values
    
    # Trading parameters
    capital = initial_capital
    trades = []
    open_trade = None
    equity_curve = [capital]
    
    # Session detection (vectorized hour extraction)
    hours = pd.to_datetime(timestamps).hour.values
    
    def get_session(hour: int) -> str:
        if 0 <= hour < 5: return 'asian'
        elif 5 <= hour < 12: return 'london'
        elif 9.5 <= hour < 12: return 'ny_am'
        elif 12 <= hour < 13.5: return 'ny_lunch'
        elif 13.5 <= hour < 16: return 'ny_pm'
        else: return 'overnight'
    
    logger.info("Simulating trades...")
    
    for idx in range(50, len(df)):
        current_price = closes[idx]
        session = get_session(hours[idx])
        session_conf = 0.7 if session in ['london', 'ny_am'] else 0.3
        
        current_trend = trend[idx]
        atr = (highs[idx-14:idx] - lows[idx-14:idx]).mean() if idx > 14 else 50
        
        # Check exits
        if open_trade:
            if open_trade.direction == TradeDirection.LONG:
                if current_price <= open_trade.stop_loss:
                    open_trade.exit_idx = idx
                    open_trade.exit_price = open_trade.stop_loss
                    open_trade.pnl = (open_trade.exit_price - open_trade.entry_price) * open_trade.size * 20
                    open_trade.status = "STOP_HIT"
                    capital += open_trade.pnl
                    trades.append(open_trade)
                    open_trade = None
                elif current_price >= open_trade.take_profit:
                    open_trade.exit_idx = idx
                    open_trade.exit_price = open_trade.take_profit
                    open_trade.pnl = (open_trade.exit_price - open_trade.entry_price) * open_trade.size * 20
                    open_trade.status = "TP_HIT"
                    capital += open_trade.pnl
                    trades.append(open_trade)
                    open_trade = None
            else:
                if current_price >= open_trade.stop_loss:
                    open_trade.exit_idx = idx
                    open_trade.exit_price = open_trade.stop_loss
                    open_trade.pnl = (open_trade.entry_price - open_trade.exit_price) * open_trade.size * 20
                    open_trade.status = "STOP_HIT"
                    capital += open_trade.pnl
                    trades.append(open_trade)
                    open_trade = None
                elif current_price <= open_trade.take_profit:
                    open_trade.exit_idx = idx
                    open_trade.exit_price = open_trade.take_profit
                    open_trade.pnl = (open_trade.entry_price - open_trade.exit_price) * open_trade.size * 20
                    open_trade.status = "TP_HIT"
                    capital += open_trade.pnl
                    trades.append(open_trade)
                    open_trade = None
            
            # Time exit
            if open_trade and (idx - open_trade.entry_idx) > 20:
                open_trade.exit_idx = idx
                open_trade.exit_price = current_price
                open_trade.pnl = (open_trade.entry_price - current_price) * open_trade.size * 20 if open_trade.direction == TradeDirection.SHORT else (current_price - open_trade.entry_price) * open_trade.size * 20
                open_trade.status = "TIME_EXIT"
                capital += open_trade.pnl
                trades.append(open_trade)
                open_trade = None
        
        # Check entries
        if open_trade is None and session_conf > 0.4:
            # Get recent FVGs
            recent_fvgs = fvgs[(fvgs['idx'] < idx) & (~fvgs['filled'])].tail(10)
            nearest_obs = obs[obs['idx'] < idx].tail(5)
            
            # Price position
            period_high = highs[idx-20:idx].max()
            period_low = lows[idx-20:idx].min()
            price_pos = (current_price - period_low) / (period_high - period_low + 0.001)
            
            signal = None
            
            # Long signals (discount zone, bullish trend/FVG)
            if price_pos < 0.4 and current_trend >= 0:
                for _, fvg in recent_fvgs.iterrows():
                    if fvg['type'] == 'BULLISH' and fvg['low'] < current_price < fvg['high']:
                        signal = ('FVG Long', TradeDirection.LONG, 0.65)
                        break
                
                if not signal:
                    for _, ob in nearest_obs.iterrows():
                        if ob['type'] == 'BULLISH' and current_price > ob['high']:
                            signal = ('OB Long', TradeDirection.LONG, 0.70)
                            break
            
            # Short signals (premium zone, bearish trend/FVG)
            elif price_pos > 0.6 and current_trend <= 0:
                for _, fvg in recent_fvgs.iterrows():
                    if fvg['type'] == 'BEARISH' and fvg['low'] < current_price < fvg['high']:
                        signal = ('FVG Short', TradeDirection.SHORT, 0.65)
                        break
                
                if not signal:
                    for _, ob in nearest_obs.iterrows():
                        if ob['type'] == 'BEARISH' and current_price < ob['low']:
                            signal = ('OB Short', TradeDirection.SHORT, 0.70)
                            break
            
            if signal:
                setup_type, direction, confidence = signal
                risk_pct = capital * 0.02
                
                if direction == TradeDirection.LONG:
                    sl = current_price - atr * 1.5
                    tp = current_price + atr * 3
                else:
                    sl = current_price + atr * 1.5
                    tp = current_price - atr * 3
                
                risk = abs(current_price - sl)
                size = risk_pct / risk if risk > 0 else 1
                
                open_trade = Trade(
                    entry_idx=idx,
                    entry_price=current_price,
                    direction=direction,
                    size=size,
                    stop_loss=sl,
                    take_profit=tp,
                    setup_type=setup_type
                )
        
        equity_curve.append(capital)
        
        if idx % 500 == 0:
            logger.info(f"Progress: {idx}/{len(df)} | Equity: ${capital:,.0f}")
    
    # Close open trade
    if open_trade:
        open_trade.exit_idx = len(df) - 1
        open_trade.exit_price = closes[-1]
        open_trade.pnl = (open_trade.entry_price - closes[-1]) * open_trade.size * 20 if open_trade.direction == TradeDirection.SHORT else (closes[-1] - open_trade.entry_price) * open_trade.size * 20
        open_trade.status = "EOD"
        trades.append(open_trade)
    
    # Calculate statistics
    closed = [t for t in trades if t.status != "OPEN"]
    winners = [t for t in closed if t.pnl > 0]
    losers = [t for t in closed if t.pnl <= 0]
    
    total_return = (capital - initial_capital) / initial_capital * 100
    win_rate = len(winners) / len(closed) * 100 if closed else 0
    
    max_eq = max(equity_curve) if equity_curve else initial_capital
    min_eq = min(equity_curve) if equity_curve else initial_capital
    max_dd = (max_eq - min_eq) / max_eq * 100 if max_eq > 0 else 0
    
    profit = sum(t.pnl for t in winners)
    loss = abs(sum(t.pnl for t in losers))
    pf = profit / loss if loss > 0 else float('inf')
    
    # Setup breakdown
    setups = {}
    for t in closed:
        if t.setup_type:
            if t.setup_type not in setups:
                setups[t.setup_type] = {'count': 0, 'wins': 0, 'pnl': 0}
            setups[t.setup_type]['count'] += 1
            if t.pnl > 0:
                setups[t.setup_type]['wins'] += 1
            setups[t.setup_type]['pnl'] += t.pnl
    
    stats = {
        'period': {
            'start': str(df.index[0])[:10],
            'end': str(df.index[-1])[:10],
            'bars': len(df)
        },
        'capital': {
            'initial': initial_capital,
            'final': capital,
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
            'net_pnl': capital - initial_capital,
            'profit_factor': pf
        },
        'risk': {
            'max_drawdown_pct': max_dd
        },
        'setup_breakdown': setups
    }
    
    return stats, trades, equity_curve


def main():
    print("=" * 70)
    print("ICT BACKTESTER - NQ 6-MONTH TEST ($10,000)")
    print("=" * 70)
    
    # Fetch data
    logger.info("Fetching NQ data...")
    df = yf.Ticker("NQ=F").history(period="6mo", interval="1h")
    df = df.dropna()
    df = df[~df.index.duplicated(keep='first')]
    
    logger.info(f"Data: {len(df)} bars | {df.index[0]} to {df.index[-1]}")
    print()
    
    # Run backtest
    stats, trades, equity = run_backtest(df, initial_capital=10000)
    
    # Save results
    results = {
        'metadata': {'timestamp': datetime.now().isoformat(), 'symbol': 'NQ', 'timeframe': '1h'},
        'statistics': stats,
        'trades': [
            {
                'entry_time': str(df.index[t.entry_idx]),
                'entry_price': t.entry_price,
                'direction': t.direction.value,
                'size': t.size,
                'stop_loss': t.stop_loss,
                'take_profit': t.take_profit,
                'exit_time': str(df.index[t.exit_idx]) if t.exit_idx > 0 else None,
                'exit_price': t.exit_price,
                'pnl': t.pnl,
                'status': t.status,
                'setup_type': t.setup_type
            }
            for t in trades
        ],
        'equity_curve': [{'date': str(df.index[i])[:10], 'equity': e} for i, e in enumerate(equity)]
    }
    
    with open('backtest_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print()
    print("=" * 70)
    print("BACKTEST RESULTS")
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
    for name, data in stats['setup_breakdown'].items():
        wr = data['wins'] / data['count'] * 100 if data['count'] > 0 else 0
        print(f"  {name:12} | {data['count']:3} trades | {wr:5.1f}% win | ${data['pnl']:+,.0f}")
    
    print()
    print(f"Results saved to: backtest_results.json")
    print("=" * 70)


if __name__ == "__main__":
    main()
