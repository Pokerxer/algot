"""
ICT Core Engine Test with Live Data
===================================

Test the ICT Core Engine with live data fetched from Yahoo Finance.
Uses EURUSD as the primary pair for testing.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Optional
import logging
import pytz

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EST = pytz.timezone('US/Eastern')


def fetch_live_forex_data(symbol: str = "EURUSD=X", timeframe: str = "1h", periods: int = 100) -> pd.DataFrame:
    """
    Fetch live forex data from Yahoo Finance.
    
    Args:
        symbol: Forex pair symbol (e.g., "EURUSD=X", "GBPUSD=X", "USDJPY=X")
        timeframe: Data timeframe ("1m", "5m", "15m", "30m", "1h", "4h", "1d")
        periods: Number of periods to fetch
    
    Returns:
        DataFrame with OHLCV data
    """
    logger.info(f"Fetching live data for {symbol}...")
    
    try:
        ticker = yf.Ticker(symbol)
        
        # Fetch data
        df = ticker.history(period=f"{periods}{timeframe}", interval=timeframe)
        
        if df.empty:
            logger.warning(f"No data returned for {symbol}, trying different approach")
            # Try with explicit start/end dates
            end = datetime.now()
            start = end - timedelta(days=10)
            df = ticker.history(start=start, end=end, interval=timeframe)
        
        # Clean data
        df = df.dropna()
        df = df[~df.index.duplicated(keep='first')]
        
        logger.info(f"Fetched {len(df)} bars for {symbol}")
        logger.info(f"Latest: {df.index[-1]} | O: {df['Open'].iloc[-1]:.5f} H: {df['High'].iloc[-1]:.5f} L: {df['Low'].iloc[-1]:.5f} C: {df['Close'].iloc[-1]:.5f}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        return pd.DataFrame()


def prepare_data_for_ict(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare data for ICT analysis by adding required columns.
    """
    df = df.copy()
    
    # Ensure we have all required columns
    df['volume'] = df.get('Volume', 0)
    
    # Calculate typical price if not present
    if 'Typical' not in df.columns:
        df['Typical'] = (df['High'] + df['Low'] + df['Close']) / 3
    
    # Add datetime columns for ICT analysis
    df['timestamp'] = df.index
    df['hour'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    
    return df


def create_market_data_dict(df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
    """
    Create market data dictionary for ICT Core Engine.
    """
    return {
        'symbol': symbol,
        'data': df,
        'current_price': df['Close'].iloc[-1],
        'current_time': datetime.now(EST),
        'bid': df['Close'].iloc[-1] - 0.0001,
        'ask': df['Close'].iloc[-1] + 0.0001,
    }


def run_ict_analysis():
    """
    Main function to run ICT analysis with live data.
    """
    logger.info("=" * 60)
    logger.info("ICT CORE ENGINE - LIVE DATA TEST")
    logger.info("=" * 60)
    
    # Fetch live data
    symbols = ["EURUSD=X", "GBPUSD=X", "USDJPY=X"]
    live_data = {}
    
    for symbol in symbols:
        df = fetch_live_forex_data(symbol, timeframe="1h", periods=100)
        if not df.empty:
            live_data[symbol] = {
                'df': prepare_data_for_ict(df),
                'ohlcv': df
            }
    
    if not live_data:
        logger.error("No data fetched. Exiting.")
        return
    
    # Test with primary pair
    primary_symbol = "EURUSD=X"
    if primary_symbol in live_data:
        df = live_data[primary_symbol]['df']
        
        logger.info("\n" + "=" * 60)
        logger.info(f"TESTING ICT CORE ENGINE WITH {primary_symbol}")
        logger.info("=" * 60)
        
        # Import and test ICT Core Engine
        try:
            from ict_core_engine import (
                ICTCoreEngine, TradeDirection
            )
            
            # Initialize engine
            engine = ICTCoreEngine()
            
            logger.info("\nICT Core Engine initialized")
            
            # Analyze current market conditions using available methods
            current_time = datetime.now(EST)
            current_hour = current_time.hour
            
            # Determine session
            session = engine._get_session(current_time.time())
            kill_zone = engine._get_kill_zone(current_time.time())
            is_macro, macro_window = engine._check_macro_time(current_time.time())
            recommendation = engine._get_time_recommendation(current_time.time())
            
            logger.info(f"\nCurrent Time Context:")
            logger.info(f"  Local Time: {current_time}")
            logger.info(f"  Session: {session}")
            logger.info(f"  Kill Zone: {kill_zone}")
            logger.info(f"  Is Macro Time: {is_macro} ({macro_window})")
            logger.info(f"  Recommendation: {recommendation}")
            
            # Create data dictionary for multi-timeframe analysis
            data_dict = {}
            for symbol, data in live_data.items():
                clean_symbol = symbol.replace("=X", "").replace("=", "")
                data_dict[clean_symbol] = data['ohlcv']
            
            # Run multi-timeframe analysis
            mtf_context = engine._analyze_multi_timeframe(data_dict)
            logger.info(f"\nMulti-Timeframe Analysis:")
            logger.info(f"  HTF Bias: {mtf_context.htf_bias.value}")
            logger.info(f"  ITF Bias: {mtf_context.itf_bias.value}")
            logger.info(f"  LTF Bias: {mtf_context.ltf_bias.value}")
            logger.info(f"  Bias Alignment: {mtf_context.bias_alignment}")
            logger.info(f"  Alignment Score: {mtf_context.alignment_score:.1f}")
            
            # Get overall bias
            overall_bias = mtf_context.get_overall_bias()
            logger.info(f"\nOverall Bias: {overall_bias.value}")
            
            # Analyze trading models
            pd_arrays = engine._identify_pd_arrays(data_dict, overall_bias)
            liquidity = engine._analyze_liquidity(data_dict)
            models = engine._analyze_trading_models(data_dict, mtf_context, pd_arrays)
            
            logger.info(f"\nTrading Models Status:")
            for model_name, model_data in models.items():
                logger.info(f"  {model_name}: {model_data}")
            
            # Generate trade setups
            time_context_full = {
                'session': session,
                'kill_zone': kill_zone,
                'is_macro_time': is_macro,
                'macro_window': macro_window,
                'recommendation': recommendation,
            }
            trade_setups = engine._generate_trade_setups(
                primary_symbol.replace("=X", ""),
                data_dict,
                mtf_context,
                pd_arrays,
                liquidity,
                models,
                time_context_full
            )
            
            logger.info(f"\nTrade Setups Found: {len(trade_setups)}")
            for i, setup in enumerate(trade_setups):
                logger.info(f"  Setup {i+1}: {setup.direction.value} | Grade: {setup.grade.value} | Confidence: {setup.confidence:.1%}")
            
            logger.info("\n✅ ICT Core Engine test completed successfully!")
            
        except ImportError as e:
            logger.error(f"Import error: {e}")
        except Exception as e:
            logger.error(f"Error during analysis: {e}")
            import traceback
            traceback.print_exc()
    else:
        logger.error(f"Primary symbol {primary_symbol} not available")


def run_quick_test():
    """
    Quick test with just EURUSD data.
    """
    logger.info("\n" + "=" * 60)
    logger.info("QUICK TEST - FETCHING EURUSD DATA")
    logger.info("=" * 60)
    
    # Fetch 1 hour data
    df = fetch_live_forex_data("EURUSD=X", timeframe="1h", periods=48)
    
    if df.empty:
        logger.error("Failed to fetch data")
        return
    
    # Display data summary
    logger.info(f"\nData Summary:")
    logger.info(f"  Period: {df.index[0]} to {df.index[-1]}")
    logger.info(f"  Bars: {len(df)}")
    logger.info(f"  Open: {df['Open'].iloc[0]:.5f}")
    logger.info(f"  High: {df['High'].max():.5f}")
    logger.info(f"  Low: {df['Low'].min():.5f}")
    logger.info(f"  Close: {df['Close'].iloc[-1]:.5f}")
    logger.info(f"  Volume: {df['Volume'].sum():,}")
    
    # Calculate basic ICT metrics
    current_price = df['Close'].iloc[-1]
    daily_open = df['Open'].iloc[0]
    daily_range = df['High'].max() - df['Low'].min()
    
    # Premium/Discount calculation
    if daily_range > 0:
        price_position = (current_price - df['Low'].min()) / daily_range * 100
        zone = "PREMIUM" if price_position > 50 else "DISCOUNT"
        logger.info(f"\nICT Analysis:")
        logger.info(f"  Current Price: {current_price:.5f}")
        logger.info(f"  Daily Range: {daily_range:.5f}")
        logger.info(f"  Price Position: {price_position:.1f}% (from low)")
        logger.info(f"  Zone: {zone}")
    
    return df


if __name__ == "__main__":
    import pytz
    
    # Run quick test first
    run_quick_test()
    
    # Run full ICT analysis
    run_ict_analysis()
