"""
Data module for HFT Market Simulator.

This module provides functionality for:
- Fetching market data from various sources (yfinance, ccxt)
- Data cleaning and validation
- Feature preprocessing
- Train/test splitting
- Walk-forward validation
"""

from .fetcher import DataFetcher
from .preprocessor import DataPreprocessor
from .validator import DataValidator
from .splitter import DataSplitter

__all__ = [
    "DataFetcher",
    "DataPreprocessor",
    "DataValidator",
    "DataSplitter",
]

