"""
V5 Trade Test Script
Place a trade and close it after 5 minutes to verify IBKR integration works.
"""

import asyncio
asyncio.set_event_loop(asyncio.new_event_loop())

import sys
import time
sys.path.insert(0, '/Users/mac/Documents/Algot')

from ib_insync import IB, Crypto, Forex, MarketOrder

def run_trade_test():
    ib = IB()
    
    try:
        # Connect to IBKR paper trading
        ib.connect('127.0.0.1', 7497, clientId=123)
        print("Connected to IBKR paper trading")
        
        # Get account info
        account = ib.accountValues()
        for av in account:
            if av.tag == 'CashBalance' and av.currency == 'USD':
                print(f"Cash balance: ${float(av.value):,.2f}")
                break
        
        # Test with EURUSD (smallest spread, reliable)
        contract = Forex('EURUSD')
        print(f"\nPlacing test BUY order for EURUSD...")
        
        # Place market order
        order = MarketOrder('BUY', 10000)  # 10k units = 1 standard lot
        trade = ib.placeOrder(contract, order)
        
        # Wait for fill
        ib.sleep(2)
        
        if trade.orderStatus.status == 'Filled':
            fill_price = trade.orderStatus.avgFillPrice
            print(f"Order filled at: {fill_price}")
            print(f"Position opened: BUY 10,000 EURUSD @ {fill_price}")
        else:
            print(f"Order status: {trade.orderStatus.status}")
            print("Trying to cancel and exit...")
            ib.cancelOrder(trade.order)
            ib.disconnect()
            return
        
        # Wait 5 minutes before closing
        print("\nWaiting 5 minutes before closing position...")
        print("Press Ctrl+C to cancel early")
        
        for i in range(5):
            ib.sleep(60)
            print(f"  {i+1}/5 minutes elapsed...")
        
        # Close position with SELL
        print("\nClosing position...")
        close_order = MarketOrder('SELL', 10000)
        close_trade = ib.placeOrder(contract, close_order)
        
        ib.sleep(2)
        
        if close_trade.orderStatus.status == 'Filled':
            close_price = close_trade.orderStatus.avgFillPrice
            pnl = (close_price - fill_price) * 10000
            print(f"Position closed at: {close_price}")
            print(f"PnL: ${pnl:.2f}")
        else:
            print(f"Close order status: {close_trade.orderStatus.status}")
            
        print("\n✅ Trade test completed!")
        
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        # Try to close any open position
        try:
            close_order = MarketOrder('SELL', 10000)
            ib.placeOrder(contract, close_order)
            print("Position closed on interrupt")
        except:
            pass
    except Exception as e:
        print(f"Error: {e}")
    finally:
        ib.disconnect()
        print("Disconnected")

if __name__ == '__main__':
    run_trade_test()
