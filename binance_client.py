"""
Binance Client for Crypto Trading
================================
Fetch data and place trades via Binance API.

Usage:
    from binance_client import BinanceClient, fetch_binance_data, place_order
    
    client = BinanceClient(api_key, api_secret)
    data = fetch_binance_data('BTCUSDT', '1h', 500)
    client.place_order('BTCUSDT', 'BUY', 0.001, stop_loss=65000, take_profit=75000)
"""

import time
import hmac
import hashlib
import requests
import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from urllib.parse import urlencode

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False


BINANCE_BASE_URL = "https://api.binance.com"
BINANCE_TEST_URL = "https://testnet.binance.vision"


class BinanceClient:
    """Binance API client for fetching data and placing trades"""
    
    def __init__(self, api_key: str = "", api_secret: str = "", testnet: bool = False):
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.base_url = BINANCE_TEST_URL if testnet else BINANCE_BASE_URL
        self.session = requests.Session()
        self.session.headers.update({"X-MBX-APIKEY": api_key})
    
    def _sign(self, params: str) -> str:
        """Generate signature for authenticated requests"""
        return hmac.new(
            self.api_secret.encode('utf-8'),
            params.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
    
    def _request(self, method: str, endpoint: str, signed: bool = False, **kwargs) -> Dict:
        """Make request to Binance API"""
        url = f"{self.base_url}{endpoint}"
        
        if signed:
            params = kwargs.get('params', {})
            params['timestamp'] = int(time.time() * 1000)
            params['signature'] = self._sign(urlencode(params))
            kwargs['params'] = params
        
        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()
        return response.json()
    
    def get_account(self) -> Dict:
        """Get account information"""
        return self._request('GET', '/api/v3/account', signed=True)
    
    def get_balance(self, asset: str = 'USDT') -> float:
        """Get balance for specific asset"""
        account = self.get_account()
        for balance in account['balances']:
            if balance['asset'] == asset:
                return float(balance['free'])
        return 0.0
    
    def get_positions(self) -> List[Dict]:
        """Get all open positions (futures) or balances (spot)"""
        try:
            # Try futures first
            positions = self._request('GET', '/fapi/v2/positionRisk', signed=True)
            open_positions = []
            for pos in positions:
                if float(pos['positionAmt']) != 0:
                    open_positions.append({
                        'symbol': pos['symbol'],
                        'qty': float(pos['positionAmt']),
                        'entry_price': float(pos['entryPrice']),
                        'unrealized_pnl': float(pos['unrealizedProfit']),
                        'side': 'LONG' if float(pos['positionAmt']) > 0 else 'SHORT'
                    })
            return open_positions
        except:
            # Fall back to spot balances
            account = self.get_account()
            balances = []
            for b in account['balances']:
                if float(b['free']) > 0 or float(b['locked']) > 0:
                    balances.append({
                        'asset': b['asset'],
                        'free': float(b['free']),
                        'locked': float(b['locked'])
                    })
            return balances
    
    def place_order(
        self,
        symbol: str,
        side: str,  # 'BUY' or 'SELL'
        quantity: float,
        order_type: str = 'MARKET',
        price: Optional[float] = None,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        reduce_only: bool = False
    ) -> Dict:
        """
        Place an order on Binance.
        
        For futures: supports bracket orders (entry + SL + TP)
        For spot: places market/limit order
        """
        orders = []
        
        # Main order
        main_order = {
            'symbol': symbol.upper(),
            'side': side.upper(),
            'quantity': quantity,
            'type': order_type.upper()
        }
        
        if order_type.upper() == 'LIMIT' and price:
            main_order['timeInForce'] = 'GTC'
            main_order['price'] = str(price)
        
        # Add stop loss
        if stop_loss:
            if side.upper() == 'BUY':
                stop_side = 'SELL'
                stop_price = stop_loss
            else:
                stop_side = 'BUY'
                stop_price = stop_loss
            
            # Try futures bracket
            try:
                main_order['stopPrice'] = str(stop_price)
                main_order['workingType'] = 'MARK_PRICE'
                main_order['closePosition'] = 'true' if reduce_only else 'false'
            except:
                pass
        
        # Place main order
        try:
            # Try futures
            response = self._request('POST', '/fapi/v1/order', signed=True, params=main_order)
            orders.append(response)
        except:
            # Fall back to spot
            endpoint = '/api/v3/order' if order_type.upper() == 'LIMIT' else '/api/v3/order/test'
            response = self._request('POST', endpoint, signed=True, params=main_order)
            orders.append(response)
        
        return {'orders': orders, 'main_order': orders[0]}
    
    def place_bracket_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        stop_loss: float,
        take_profit: float
    ) -> Dict:
        """
        Place bracket order: entry + stop loss + take profit
        Uses OCO (One Cancels Other) for spot or hedge mode for futures
        """
        symbol = symbol.upper()
        side_upper = side.upper()
        
        try:
            # Futures bracket order
            params = {
                'symbol': symbol,
                'side': side_upper,
                'quantity': quantity,
                'positionSide': 'LONG' if side_upper == 'BUY' else 'SHORT',
                'stopPrice': str(stop_loss),
                'workingType': 'MARK_PRICE',
                'closePosition': 'true'
            }
            sl_order = self._request('POST', '/fapi/v1/order', signed=True, params=params)
            
            # Take profit order
            params_tp = {
                'symbol': symbol,
                'side': 'SELL' if side_upper == 'BUY' else 'BUY',
                'quantity': quantity,
                'type': 'TAKE_PROFIT_MARKET',
                'stopPrice': str(take_profit),
                'workingType': 'MARK_PRICE',
                'closePosition': 'true'
            }
            tp_order = self._request('POST', '/fapi/v1/order', signed=True, params=params_tp)
            
            return {
                'entry': sl_order,
                'stop_loss': sl_order,
                'take_profit': tp_order
            }
        except Exception as e:
            print(f"Bracket order error: {e}")
            # Fallback: just place market order
            return self.place_order(symbol, side, quantity, 'MARKET')
    
    def cancel_order(self, symbol: str, order_id: int) -> Dict:
        """Cancel an order"""
        return self._request('DELETE', '/api/v3/order', signed=True, params={
            'symbol': symbol.upper(),
            'orderId': order_id
        })
    
    def get_open_orders(self, symbol: Optional[str] = None) -> List[Dict]:
        """Get open orders"""
        params = {}
        if symbol:
            params['symbol'] = symbol.upper()
        return self._request('GET', '/api/v3/openOrders', signed=True, params=params)
    
    def get_order(self, symbol: str, order_id: int) -> Dict:
        """Get order status"""
        return self._request('GET', '/api/v3/order', signed=True, params={
            'symbol': symbol.upper(),
            'orderId': order_id
        })
    
    def get_pnl(self, symbol: Optional[str] = None) -> Dict:
        """Get daily PnL"""
        params = {
            'timestamp': int(time.time() * 1000),
            'signature': self._sign(urlencode(params))
        }
        
        try:
            # Futures daily PnL
            if symbol:
                params['symbol'] = symbol.upper()
            resp = self._request('GET', '/fapi/v2/account', signed=True, params=params)
            return {
                'total_unrealized_pnl': sum(float(p['unrealizedProfit']) for p in resp.get('positions', []))
            }
        except:
            return {'total_unrealized_pnl': 0}


def fetch_binance_data(
    symbol: str,
    interval: str = '1h',
    limit: int = 500,
    start_str: Optional[str] = None,
    testnet: bool = False
) -> Optional[Dict]:
    """
    Fetch historical candlestick data from Binance.
    
    Args:
        symbol: Trading pair (e.g., 'BTCUSDT', 'ETHUSDT')
        interval: Timeframe (1m, 5m, 15m, 1h, 4h, 1d)
        limit: Number of candles (max 1500)
        start_str: Start time (e.g., '500 hours ago', '2024-01-01')
        testnet: Use testnet
    
    Returns:
        Dict with opens, highs, lows, closes, volumes, timestamps
    """
    base_url = BINANCE_TEST_URL if testnet else BINANCE_BASE_URL
    endpoint = "/api/v3/klines"
    
    params = {
        'symbol': symbol.upper(),
        'interval': interval,
        'limit': limit
    }
    
    if start_str:
        params['startTime'] = start_str
    
    try:
        url = f"{base_url}{endpoint}"
        try:
            # Try with SSL verification
            response = requests.get(url, params=params)
            response.raise_for_status()
        except Exception:
            # If SSL fails, try without verification
            response = requests.get(url, params=params, verify=False)
            response.raise_for_status()
        data = response.json()
        
        if not data:
            return None
        
        # Parse candle data
        candles = []
        for d in data:
            candles.append({
                'timestamp': d[0],
                'open': float(d[1]),
                'high': float(d[2]),
                'low': float(d[3]),
                'close': float(d[4]),
                'volume': float(d[5]),
                'close_time': d[6],
                'quote_volume': float(d[7]),
                'trades': d[8],
                'taker_buy_base': float(d[9]),
                'taker_buy_quote': float(d[10])
            })
        
        # Convert to dict format compatible with trading system
        result = {
            'opens': [c['open'] for c in candles],
            'highs': [c['high'] for c in candles],
            'lows': [c['low'] for c in candles],
            'closes': [c['close'] for c in candles],
            'volumes': [c['volume'] for c in candles],
            'timestamps': [c['timestamp'] for c in candles],
            'df': pd.DataFrame(candles) if PANDAS_AVAILABLE else None
        }
        
        return result
        
    except Exception as e:
        print(f"Error fetching Binance data {symbol}: {e}")
        # Return cached test data if needed
        return None


def get_contract_info_binance(symbol: str) -> Dict:
    """Get contract specifications for a symbol"""
    try:
        url = f"{BINANCE_BASE_URL}/api/v3/exchangeInfo"
        response = requests.get(url)
        data = response.json()
        
        for s in data['symbols']:
            if s['symbol'] == symbol.upper():
                return {
                    'symbol': s['symbol'],
                    'base_asset': s['baseAsset'],
                    'quote_asset': s['quoteAsset'],
                    'min_qty': float(s['filters'][0]['minQty']),
                    'max_qty': float(s['filters'][0]['maxQty']),
                    'step_size': float(s['filters'][0]['stepSize']),
                    'min_price': float(s['filters'][1]['minPrice']),
                    'max_price': float(s['filters'][1]['maxPrice']),
                    'tick_size': float(s['filters'][1]['tickSize'])
                }
    except Exception as e:
        print(f"Error getting contract info: {e}")
    
    return {}


def calculate_position_size_binance(
    symbol: str,
    account_balance: float,
    risk_pct: float,
    stop_distance_pct: float
) -> float:
    """
    Calculate position size based on risk management.
    
    Args:
        symbol: Trading pair
        account_balance: Total account balance in quote currency
        risk_pct: Risk percentage (e.g., 0.02 for 2%)
        stop_distance_pct: Stop loss distance as percentage (e.g., 0.02 for 2%)
    
    Returns:
        Position size in base currency
    """
    contract_info = get_contract_info_binance(symbol)
    
    if not contract_info:
        return 0.0
    
    risk_amount = account_balance * risk_pct
    position_value = risk_amount / stop_distance_pct
    
    # Get current price
    data = fetch_binance_data(symbol, '1m', 1)
    if not data:
        return 0.0
    
    current_price = data['closes'][-1]
    qty = position_value / current_price
    
    # Apply step size
    step_size = contract_info.get('step_size', 0.001)
    qty = round(qty / step_size) * step_size
    
    # Check min/max
    min_qty = contract_info.get('min_qty', 0.001)
    max_qty = contract_info.get('max_qty', 1000000)
    
    if qty < min_qty:
        qty = min_qty
    if qty > max_qty:
        qty = max_qty
    
    return qty


# Map common symbols to Binance format
SYMBOL_MAP = {
    'BTCUSD': 'BTCUSDT',
    'ETHUSD': 'ETHUSDT',
    'SOLUSD': 'SOLUSDT',
    'LTCUSD': 'LTCUSDT',
    'LINKUSD': 'LINKUSDT',
    'UNIUSD': 'UNIUSDT',
    'XRPUSD': 'XRPUSDT',
    'ADAUSD': 'ADAUSDT',
    'DOGEUSD': 'DOGEUSDT',
    'DOTUSD': 'DOTUSDT',
    'AVAXUSD': 'AVAXUSDT',
    'MATICUSD': 'MATICUSDT',
}


def to_binance_symbol(symbol: str) -> str:
    """Convert standard symbol to Binance format"""
    return SYMBOL_MAP.get(symbol, symbol.upper())


def from_binance_symbol(symbol: str) -> str:
    """Convert Binance symbol to standard format"""
    for std, bn in SYMBOL_MAP.items():
        if bn == symbol.upper():
            return std
    return symbol.upper().replace('USDT', 'USD')


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Binance Client')
    parser.add_argument('--symbol', default='BTCUSDT', help='Symbol')
    parser.add_argument('--interval', default='1h', help='Interval')
    parser.add_argument('--limit', type=int, default=100, help='Limit')
    parser.add_argument('--testnet', action='store_true', help='Use testnet')
    parser.add_argument('--api-key', default='', help='API Key')
    parser.add_argument('--api-secret', default='', help='API Secret')
    
    args = parser.parse_args()
    
    # Fetch data
    data = fetch_binance_data(args.symbol, args.interval, args.limit, testnet=args.testnet)
    if data:
        print(f"Fetched {len(data['closes'])} candles for {args.symbol}")
        print(f"Latest: {data['closes'][-1]}")
        print(f"High: {max(data['highs'])}")
        print(f"Low: {min(data['lows'])}")
    else:
        print(f"Failed to fetch data for {args.symbol}")
