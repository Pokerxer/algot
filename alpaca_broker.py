"""
ICT V2 - Alpaca Broker Integration
"""

import alpaca_trade_api as tradeapi
from typing import Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class AlpacaBroker:
    def __init__(self, api_key: str = None, secret_key: str = None, 
                 paper: bool = True, base_url: str = None):
        """
        Initialize Alpaca broker connection.
        
        Args:
            api_key: Alpaca API key (or set ALPACA_API_KEY env var)
            secret_key: Alpaca secret key (or set ALPACA_SECRET_KEY env var)
            paper: Use paper trading if True (default)
            base_url: Custom base URL (overrides paper/live)
        """
        import os
        
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')
        
        if not self.api_key or not self.secret_key:
            raise ValueError("API keys required. Set ALPACA_API_KEY and ALPACA_SECRET_KEY env vars.")
        
        if base_url:
            self.base_url = base_url
        elif paper:
            self.base_url = 'https://paper-api.alpaca.markets'
        else:
            self.base_url = 'https://api.alpaca.markets'
        
        self.api = tradeapi.REST(self.api_key, self.secret_key, self.base_url, api_version='v2')
        self.account = None
        
    def get_account(self) -> Dict:
        """Get account information"""
        try:
            self.account = self.api.get_account()
            return {
                'id': self.account.id,
                'cash': float(self.account.cash),
                'portfolio_value': float(self.account.portfolio_value),
                'buying_power': float(self.account.buying_power),
                'equity': float(self.account.equity),
                'last_equity': float(self.account.last_equity),
                'pattern_day_trader': self.account.pattern_day_trader,
                'trading_blocked': self.account.trading_blocked,
                'account_blocked': self.account.account_blocked,
            }
        except Exception as e:
            logger.error(f"Error getting account: {e}")
            return None
    
    def get_position(self, symbol: str) -> Optional[Dict]:
        """Get open position for symbol"""
        try:
            position = self.api.get_position(symbol)
            return {
                'symbol': position.symbol,
                'qty': float(position.qty),
                'avg_entry_price': float(position.avg_entry_price),
                'market_value': float(position.market_value),
                'unrealized_pl': float(position.unrealized_pl),
                'unrealized_plpc': float(position.unrealized_plpc),
            }
        except Exception as e:
            return None
    
    def get_all_positions(self) -> List[Dict]:
        """Get all open positions"""
        try:
            positions = self.api.list_positions()
            return [{
                'symbol': p.symbol,
                'qty': float(p.qty),
                'avg_entry_price': float(p.avg_entry_price),
                'market_value': float(p.market_value),
                'unrealized_pl': float(p.unrealized_pl),
            } for p in positions]
        except Exception as e:
            logger.error(f"Error getting positions: {e}")
            return []
    
    def get_barset(self, symbols: List[str], timeframe: str = '1H', 
                   limit: int = 100) -> Dict:
        """Get latest bar data for symbols"""
        try:
            barset = self.api.get_barset(symbols, timeframe, limit=limit)
            return {symbol: [{
                't': str(bar.t),
                'o': bar.o,
                'h': bar.h,
                'l': bar.l,
                'c': bar.c,
                'v': bar.v,
            } for bar in barset[symbol]] for symbol in symbols if symbol in barset}
        except Exception as e:
            logger.error(f"Error getting barset: {e}")
            return {}
    
    def submit_order(self, symbol: str, qty: int, side: str, 
                    order_type: str = 'market', 
                    time_in_force: str = 'day',
                    limit_price: float = None,
                    stop_price: float = None,
                    take_profit: float = None,
                    stop_loss: float = None) -> Optional[Order]:
        """
        Submit a trade order.
        
        Args:
            symbol: Symbol to trade
            qty: Quantity (positive for buy, negative for sell)
            side: 'buy' or 'sell'
            order_type: 'market', 'limit', 'stop', 'stop_limit'
            time_in_force: 'day', 'gtc', 'ioc', 'fok'
            limit_price: Limit price for limit orders
            stop_price: Stop price for stop orders
            take_profit: Take profit price (bracket order)
            stop_loss: Stop loss price (bracket order)
        """
        try:
            # Build order arguments
            order_args = {
                'symbol': symbol,
                'qty': abs(qty),
                'side': side,
                'type': order_type,
                'time_in_force': time_in_force,
            }
            
            if limit_price:
                order_args['limit_price'] = limit_price
            
            if stop_price:
                order_args['stop_price'] = stop_price
            
            # Submit order
            order = self.api.submit_order(**order_args)
            logger.info(f"Order submitted: {order.id} - {symbol} {side} {qty}")
            
            # Add bracket orders (TP/SL) if specified
            if take_profit and stop_loss:
                self._add_bracket_orders(order.id, take_profit, stop_loss, side)
            
            return order
            
        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            return None
    
    def _add_bracket_orders(self, parent_order_id: str, take_profit: float, 
                           stop_loss: float, side: str):
        """Add bracket orders (TP/SL) to parent order"""
        try:
            # Get the filled order to get actual fill price
            order = self.api.get_order(parent_order_id)
            
            if order.status != 'filled':
                logger.warning(f"Parent order not filled yet: {order.status}")
                return
            
            # Determine bracket order sides
            if side == 'buy':
                tp_side = 'sell'
                sl_side = 'sell'
                tp_price = take_profit
                sl_price = stop_loss
            else:
                tp_side = 'buy'
                sl_side = 'buy'
                tp_price = take_profit
                sl_price = stop_loss
            
            # Take profit order
            self.api.submit_order(
                symbol=order.symbol,
                qty=order.qty,
                side=tp_side,
                type='limit',
                time_in_force='gtc',
                limit_price=tp_price,
                order_class='oto',
                sibling_order_id=parent_order_id
            )
            
            # Stop loss order
            self.api.submit_order(
                symbol=order.symbol,
                qty=order.qty,
                side=sl_side,
                type='stop',
                time_in_force='gtc',
                stop_price=sl_price,
                order_class='oto',
                sibling_order_id=parent_order_id
            )
            
            logger.info(f"Bracket orders added: TP={tp_price}, SL={sl_price}")
            
        except Exception as e:
            logger.error(f"Error adding bracket orders: {e}")
    
    def close_position(self, symbol: str, qty: int = None, 
                      side: str = None) -> Optional[Order]:
        """Close position (market order)"""
        try:
            if qty:
                return self.api.submit_order(
                    symbol=symbol,
                    qty=qty,
                    side=side or 'sell',
                    type='market',
                    time_in_force='day'
                )
            else:
                return self.api.close_position(symbol)
        except Exception as e:
            logger.error(f"Error closing position: {e}")
            return None
    
    def get_orders(self, status: str = 'all', limit: int = 50) -> List[Dict]:
        """Get orders"""
        try:
            orders = self.api.list_orders(status=status, limit=limit)
            return [{
                'id': o.id,
                'symbol': o.symbol,
                'qty': float(o.qty),
                'side': o.side,
                'type': o.type,
                'status': o.status,
                'filled_qty': float(o.filled_qty),
                'filled_avg_price': float(o.filled_avg_price) if o.filled_avg_price else None,
                'limit_price': float(o.limit_price) if o.limit_price else None,
                'stop_price': float(o.stop_price) if o.stop_price else None,
                'created_at': str(o.created_at),
            } for o in orders]
        except Exception as e:
            logger.error(f"Error getting orders: {e}")
            return []
    
    def cancel_all_orders(self) -> bool:
        """Cancel all open orders"""
        try:
            self.api.cancel_all_orders()
            logger.info("All orders cancelled")
            return True
        except Exception as e:
            logger.error(f"Error cancelling orders: {e}")
            return False
    
    def liquidate(self) -> bool:
        """Liquidate all positions"""
        try:
            self.api.close_all_positions()
            logger.info("All positions liquidated")
            return True
        except Exception as e:
            logger.error(f"Error liquidating: {e}")
            return False


def test_connection():
    """Test Alpaca connection"""
    try:
        broker = AlpacaBroker(paper=True)
        account = broker.get_account()
        if account:
            print("=" * 50)
            print("ALPACA CONNECTION SUCCESSFUL")
            print("=" * 50)
            print(f"Account ID: {account['id']}")
            print(f"Cash: ${account['cash']:,.2f}")
            print(f"Portfolio Value: ${account['portfolio_value']:,.2f}")
            print(f"Buying Power: ${account['buying_power']:,.2f}")
            print("=" * 50)
            return True
    except ValueError as e:
        print(f"Connection failed: {e}")
        print("\nTo set up Alpaca:")
        print("1. Sign up at https://alpaca.markets")
        print("2. Get your API keys from dashboard")
        print("3. Set environment variables:")
        print("   export ALPACA_API_KEY=your_api_key")
        print("   export ALPACA_SECRET_KEY=your_secret_key")
        return False
    except Exception as e:
        print(f"Connection error: {e}")
        return False


if __name__ == "__main__":
    test_connection()
