"""
Data Fetcher module for retrieving market data from various sources.

Supports:
- yfinance: Stocks, ETFs, indices
- ccxt: Cryptocurrency exchanges (Binance, Coinbase, etc.)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional, List, Literal
import logging

logger = logging.getLogger(__name__)


class DataFetcher:
    """
    Unified data fetcher supporting multiple data sources.
    
    Supports:
    - yfinance for stocks/ETFs
    - ccxt for cryptocurrency data
    """
    
    def __init__(self, source: Literal["yfinance", "ccxt"] = "yfinance"):
        """
        Initialize the data fetcher.
        
        Args:
            source: Data source to use ("yfinance" or "ccxt")
        """
        self.source = source
        self._validate_source()
        
    def _validate_source(self):
        """Validate that the required library is available."""
        if self.source == "yfinance":
            try:
                import yfinance
                self.yf = yfinance
            except ImportError:
                raise ImportError("yfinance is required. Install with: pip install yfinance")
        elif self.source == "ccxt":
            try:
                import ccxt
                self.ccxt = ccxt
            except ImportError:
                raise ImportError("ccxt is required. Install with: pip install ccxt")
    
    def fetch_stock_data(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch stock/ETF data using yfinance.
        
        Args:
            symbol: Stock ticker symbol (e.g., "AAPL", "SPY")
            start_date: Start date in format "YYYY-MM-DD"
            end_date: End date in format "YYYY-MM-DD"
            interval: Data interval ("1m", "5m", "15m", "30m", "1h", "1d", "1wk")
        
        Returns:
            DataFrame with OHLCV data
        """
        if self.source != "yfinance":
            raise ValueError("Use fetch_crypto_data for ccxt source")
        
        logger.info(f"Fetching {symbol} data from {start_date} to {end_date}")
        
        ticker = self.yf.Ticker(symbol)
        df = ticker.history(start=start_date, end=end_date, interval=interval)
        
        if df.empty:
            raise ValueError(f"No data found for {symbol}")
        
        # Standardize column names
        df = self._standardize_columns(df)
        
        # Add metadata
        df.attrs["symbol"] = symbol
        df.attrs["source"] = "yfinance"
        df.attrs["interval"] = interval
        
        logger.info(f"Fetched {len(df)} rows for {symbol}")
        return df
    
    def fetch_crypto_data(
        self,
        symbol: str,
        exchange: str = "kraken",
        start_date: str = None,
        end_date: str = None,
        timeframe: str = "1d",
        limit: int = 1000
    ) -> pd.DataFrame:
        """
        Fetch cryptocurrency data using ccxt.
        
        Args:
            symbol: Trading pair (e.g., "BTC/USDT", "ETH/USDT")
            exchange: Exchange name (e.g., "binance", "coinbase", "kraken")
            start_date: Start date in format "YYYY-MM-DD" (optional)
            end_date: End date in format "YYYY-MM-DD" (optional)
            timeframe: Data timeframe ("1m", "5m", "15m", "1h", "4h", "1d")
            limit: Maximum number of candles to fetch per request
        
        Returns:
            DataFrame with OHLCV data
            
        Note:
            Some exchanges (e.g., Binance) have geographic restrictions.
            If you encounter HTTP 451 errors, try a different exchange
            like "kraken" or "coinbase".
        """
        if self.source != "ccxt":
            raise ValueError("Use fetch_stock_data for yfinance source")
        
        logger.info(f"Fetching {symbol} from {exchange}")
        
        # Get exchange class
        exchange_class = getattr(self.ccxt, exchange)
        exchange_instance = exchange_class({
            'enableRateLimit': True,
        })
        
        # Convert dates to timestamps
        since = None
        if start_date:
            since = exchange_instance.parse8601(f"{start_date}T00:00:00Z")
        
        all_ohlcv = []
        max_iterations = 50  # Safety limit to prevent infinite loops
        iteration = 0
        
        # Fetch data in batches
        while iteration < max_iterations:
            iteration += 1
            try:
                ohlcv = exchange_instance.fetch_ohlcv(
                    symbol,
                    timeframe=timeframe,
                    since=since,
                    limit=limit
                )
            except Exception as e:
                logger.warning(f"Error fetching batch {iteration}: {e}")
                break
            
            if not ohlcv:
                break
            
            all_ohlcv.extend(ohlcv)
            logger.debug(f"Batch {iteration}: fetched {len(ohlcv)} rows, total: {len(all_ohlcv)}")
            
            # Update since for next batch
            since = ohlcv[-1][0] + 1
            
            # Check if we've reached the end date
            if end_date:
                end_timestamp = exchange_instance.parse8601(f"{end_date}T23:59:59Z")
                if since > end_timestamp:
                    break
            
            # Respect rate limits - if we got fewer than limit, we've reached the end
            if len(ohlcv) < limit:
                break
            
            # Small delay to respect rate limits
            import time
            time.sleep(exchange_instance.rateLimit / 1000)
        
        if not all_ohlcv:
            raise ValueError(f"No data found for {symbol}")
        
        # Convert to DataFrame
        df = pd.DataFrame(
            all_ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        
        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        
        # Filter by end_date if specified
        if end_date:
            df = df[df.index <= end_date]
        
        # Add metadata
        df.attrs["symbol"] = symbol
        df.attrs["source"] = f"ccxt:{exchange}"
        df.attrs["interval"] = timeframe
        
        logger.info(f"Fetched {len(df)} rows for {symbol}")
        return df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize column names to lowercase.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with standardized column names
        """
        column_mapping = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
            "Dividends": "dividends",
            "Stock Splits": "stock_splits",
        }
        
        df = df.rename(columns=column_mapping)
        
        # Keep only OHLCV columns
        ohlcv_cols = ["open", "high", "low", "close", "volume"]
        available_cols = [col for col in ohlcv_cols if col in df.columns]
        
        return df[available_cols]
    
    def fetch(
        self,
        symbol: str,
        start_date: str,
        end_date: str,
        **kwargs
    ) -> pd.DataFrame:
        """
        Unified fetch method that routes to the appropriate source.
        
        Args:
            symbol: Symbol to fetch
            start_date: Start date
            end_date: End date
            **kwargs: Additional arguments for the specific fetcher
        
        Returns:
            DataFrame with OHLCV data
        """
        if self.source == "yfinance":
            return self.fetch_stock_data(symbol, start_date, end_date, **kwargs)
        elif self.source == "ccxt":
            return self.fetch_crypto_data(symbol, start_date=start_date, end_date=end_date, **kwargs)
        else:
            raise ValueError(f"Unknown source: {self.source}")


def fetch_sample_data(
    symbol: str = "SPY",
    years: int = 5,
    source: str = "yfinance"
) -> pd.DataFrame:
    """
    Convenience function to fetch sample data for testing.
    
    Args:
        symbol: Symbol to fetch
        years: Number of years of historical data
        source: Data source
    
    Returns:
        DataFrame with OHLCV data
    """
    end_date = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() - timedelta(days=years * 365)).strftime("%Y-%m-%d")
    
    fetcher = DataFetcher(source=source)
    return fetcher.fetch(symbol, start_date, end_date)

