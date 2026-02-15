# ICT V2 - Moving Forward Plan

## Current Status (Feb 9, 2026)

### Backtest Results ✓
| Period | Return | Win Rate | PF | Trades |
|--------|--------|----------|-----|--------|
| 6 Months (Aug 2025 - Feb 2026) | +73,260% | 74.3% | 3.78 | 35 |
| 60 Days (Nov 2025 - Feb 2026) | +1,052% | 66.7% | 2.96 | 15 |

### Files Created
- `backtest_unified_v2.py` - Main backtester
- `ict_v2_live.py` - Single symbol live monitor
- `ict_v2_paper_trader.py` - Multi-symbol paper trader

---

## Phase 1: Paper Trading (Now)

### Run Paper Trader
```bash
# Monitor NQ, YM, EURUSD
python3 ict_v2_paper_trader.py --symbols "NQ=F,YM=F,EURUSD=X" --capital 10000 --interval 60

# Single symbol
python3 ict_v2_paper_trader.py --symbols "NQ=F" --capital 10000 --interval 30
```

### Paper Trading Goals
1. Validate signals in real-time
2. Track win rate and PF
3. Build confidence before live trading
4. Expected: ~60-70% win rate, 2.5+ PF

---

## Phase 2: Live Trading Preparation

### Broker Integration Options
1. **IBKR** (Interactive Brokers) - Most flexible
2. **Alpaca** - Python-friendly, free API
3. **MetaTrader 5** - Popular for forex

### Example: Alpaca Integration
```python
import alpaca_trade_api as tradeapi

api = tradeapi.REST('API_KEY', 'SECRET_KEY', base_url='https://paper-api.alpaca.markets')

# Place order
api.submit_order(
    symbol='NQH25',  # Futures contract
    qty=position_size,
    side='buy' if direction == 'long' else 'sell',
    type='market',
    time_in_force='day',
    stop_loss=sl_price,
    take_profit=tp_price
)
```

---

## Phase 3: Live Trading

### Launch Live Trader
```bash
# Start live monitoring
python3 ict_v2_live.py --symbol NQ=F --mode live --interval 30
```

### Risk Management Rules
| Rule | Value | Purpose |
|------|-------|---------|
| Max Risk/Trade | 1% | Limit single trade loss |
| Max Daily Loss | 3% | Stop trading if reached |
| Max Positions | 2 | Avoid overtrading |
| Kill Zone Only | Yes | Trade during optimal hours |

---

## V2 Rules Summary

### Entry Conditions (ALL must be met)
1. **Confluence >= 70** (A-grade only)
2. **HTF Alignment** (D1 trend matches direction)
3. **Kill Zone** (1-5am, 7am-12pm, 1:30pm-4pm ET)
4. **Price Position** (<0.40 for longs, >0.60 for shorts)
5. **OB or FVG** present near entry

### Risk/Reward
- Stop Loss: 0.5x ATR below/above OB/FVG
- Take Profit: 2.5x ATR from entry
- RR: 2.5:1 minimum

### Position Sizing
```
Risk Amount = Account × 1%
Size = Risk Amount ÷ (Entry - SL)
```

---

## Expected Performance

### Per Trade
| Outcome | Probability | PnL ($1K account) |
|---------|-------------|-------------------|
| Win (2.5:1) | ~67% | +$25 |
| Loss (1R) | ~33% | -$10 |
| Net Expected | - | +$12.50/trade |

### Monthly (20 trades)
- Expected Return: +$250 (2.5%)
- With PF 3.0: +$400 (4%)

---

## Next Steps Checklist

- [ ] Run paper trader for 2-4 weeks
- [ ] Validate real-time win rate matches backtest
- [ ] Choose broker (recommend Alpaca for start)
- [ ] Set up paper account with broker
- [ ] Test broker API connection
- [ ] Paper trade for 2 more weeks
- [ ] Go live with $5,000
- [ ] Scale to $10,000 after 1 month

---

## Important Notes

1. **No ML/AI** - V2 works because it's simple
2. **No Trailing Stops** - They reduce returns
3. **No ES** - Doesn't work with ICT concepts
4. **NQ is best** - Start here
5. **Patience** - Only ~1-2 signals per week
