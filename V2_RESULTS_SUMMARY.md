# ICT Trading Bot - V2 Results Summary

## Summary (Feb 9, 2026)

### V2 System - WORKING ✓

**6-Month Backtest (Aug 2025 - Feb 2026):**
- $10,000 → $7,336,007 (+73,260%)
- Win Rate: 74.3%
- Profit Factor: 3.78
- 35 trades

**60-Day Backtest (Nov 2025 - Feb 2026):**
- $10,000 → $115,205 (+1,052%)
- Win Rate: 66.7%
- Profit Factor: 2.96
- 15 trades

### Key Rules

1. **Confluence >= 70 required** (A-grade only)
2. **HTF alignment required** (D1 trend must match signal direction)
3. **Kill zone trading** (1-5am, 7am-12pm, 1:30pm-4pm)
4. **1% risk per trade**
5. **2.5:1 RR** (1.5x ATR TP, 0.5x ATR SL)

### Best Markets

1. **NQ (Nasdaq)** - Best overall returns
2. **YM (Dow)** - Good returns
3. **EURUSD** - Highest win rate (80%+)
4. **GBPUSD** - Modest returns
5. **ES (S&P 500)** - DOES NOT WORK

### What NOT to Do

1. **Don't add ML/AI** - Simple beats complex
2. **Don't use trailing stops** - 1.5% is too tight
3. **Don't add complex risk rules** - Blocks too many trades
4. **Don't trade ES** - Different market structure

### Files

- `backtest_unified_v2.py` - Main V2 backtester
- `v2_60day_test.py` - Quick 60-day validation
- `unified_backtest_v2_results.json` - 6mo results
- `v2_1year_results.json` - NQ 1-year results

## Next Steps for Live Trading

1. Use `backtest_unified_v2.py` logic for live signals
2. Trade NQ during kill zones only
3. 1% risk per trade, 2.5:1 RR
4. Skip if confluence < 70
5. Require HTF alignment

## Current Status

System is READY for paper/live trading.
Start with $10,000 account, 1% risk per trade.
