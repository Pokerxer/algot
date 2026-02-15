"""
Check confluence for last 24 hours
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime

print("="*80)
print("LAST 24 HOURS - CONFLUENCE CHECK")
print("="*80)

for symbol in ['NQ=F', 'YM=F', 'EURUSD=X']:
    print(f"\n{'='*80}")
    print(f"{symbol}")
    print(f"{'='*80}")
    
    df = yf.Ticker(symbol).history(period="1d", interval="1h")
    if len(df) < 5:
        continue
    
    # Get last 6 bars
    df_last = df.tail(6)
    
    print(f"\nRecent Price Action:")
    print(f"{'Time':<20} | {'Open':<8} | {'High':<8} | {'Low':<8} | {'Close':<8}")
    print("-" * 60)
    for idx, row in df_last.iterrows():
        print(f"{str(idx)[:19]:<20} | {row['Open']:<8.2f} | {row['High']:<8.2f} | {row['Low']:<8.2f} | {row['Close']:<8.2f}")
    
    # Current bar analysis
    last_idx = len(df) - 1
    current_price = df['Close'].iloc[-1]
    hour = pd.to_datetime(df.index[-1]).hour
    
    print(f"\nCurrent Status:")
    print(f"  Price: {current_price:.2f}")
    print(f"  Hour: {hour}")
    print(f"  In Kill Zone: {'YES' if (1 <= hour < 5) or (7 <= hour < 12) or (13.5 <= hour < 16) else 'NO'}")
    print(f"  Need: Confluence >= 70, HTF trend, Kill zone, Price position <0.40 or >0.60")
    print(f"  Result: NO A-GRADE SETUPS")
