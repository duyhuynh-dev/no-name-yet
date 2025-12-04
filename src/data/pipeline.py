"""
Data Pipeline module - orchestrates the complete data workflow.

This module provides a unified interface for:
- Fetching data from various sources
- Validating and cleaning data
- Preprocessing and feature engineering
- Splitting data for training/testing
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Tuple
from pathlib import Path
import logging
import json

from .fetcher import DataFetcher
from .validator import DataValidator
from .preprocessor import DataPreprocessor
from .splitter import DataSplitter

logger = logging.getLogger(__name__)


class DataPipeline:
    """
    Complete data pipeline for HFT Market Simulator.
    
    Orchestrates:
    1. Data fetching (yfinance/ccxt)
    2. Validation and cleaning
    3. Feature engineering
    4. Train/test splitting
    """
    
    def __init__(
        self,
        source: str = "yfinance",
        lookback_window: int = 30,
        scaler_type: str = "standard",
        train_start: str = "2018-01-01",
        train_end: str = "2022-12-31",
        test_start: str = "2023-01-01",
        test_end: str = "2023-12-31",
        data_dir: str = "data"
    ):
        """
        Initialize the data pipeline.
        
        Args:
            source: Data source ("yfinance" or "ccxt")
            lookback_window: Window size for rolling calculations
            scaler_type: Type of scaler ("standard", "minmax", "none")
            train_start: Start date for training data
            train_end: End date for training data
            test_start: Start date for test data
            test_end: End date for test data
            data_dir: Directory to store data files
        """
        self.source = source
        self.lookback_window = lookback_window
        self.data_dir = Path(data_dir)
        
        # Initialize components
        self.fetcher = DataFetcher(source=source)
        self.validator = DataValidator()
        self.preprocessor = DataPreprocessor(
            scaler_type=scaler_type,
            lookback_window=lookback_window
        )
        self.splitter = DataSplitter(
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end
        )
        
        # Storage for processed data
        self.raw_data: Optional[pd.DataFrame] = None
        self.cleaned_data: Optional[pd.DataFrame] = None
        self.processed_data: Optional[pd.DataFrame] = None
        self.split_data: Optional[Dict[str, pd.DataFrame]] = None
        
        # Metadata
        self.metadata: Dict = {}
    
    def fetch(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        Fetch raw data from the source.
        
        Args:
            symbol: Symbol to fetch
            start_date: Start date (defaults to train_start)
            end_date: End date (defaults to test_end)
            **kwargs: Additional arguments for the fetcher
        
        Returns:
            Raw DataFrame with OHLCV data
        """
        if start_date is None:
            start_date = self.splitter.train_start.strftime("%Y-%m-%d")
        if end_date is None:
            end_date = self.splitter.test_end.strftime("%Y-%m-%d")
        
        logger.info(f"Fetching {symbol} from {start_date} to {end_date}")
        
        self.raw_data = self.fetcher.fetch(symbol, start_date, end_date, **kwargs)
        
        # Store metadata
        self.metadata["symbol"] = symbol
        self.metadata["source"] = self.source
        self.metadata["start_date"] = start_date
        self.metadata["end_date"] = end_date
        self.metadata["raw_rows"] = len(self.raw_data)
        
        return self.raw_data
    
    def validate_and_clean(
        self,
        df: Optional[pd.DataFrame] = None,
        handle_missing: str = "ffill",
        handle_outliers: str = "clip"
    ) -> Tuple[pd.DataFrame, Dict]:
        """
        Validate and clean the data.
        
        Args:
            df: Input DataFrame (uses raw_data if None)
            handle_missing: Method for handling missing values
            handle_outliers: Method for handling outliers
        
        Returns:
            Tuple of (cleaned DataFrame, validation results)
        """
        if df is None:
            df = self.raw_data
        
        if df is None:
            raise ValueError("No data to validate. Call fetch() first.")
        
        # Validate
        validation_results = self.validator.validate(df)
        
        # Clean
        self.cleaned_data = self.validator.clean(
            df,
            handle_missing=handle_missing,
            handle_outliers=handle_outliers
        )
        
        # Update metadata
        self.metadata["validation"] = validation_results
        self.metadata["cleaned_rows"] = len(self.cleaned_data)
        
        return self.cleaned_data, validation_results
    
    def preprocess(
        self,
        df: Optional[pd.DataFrame] = None,
        add_features: bool = True,
        fit_scaler: bool = True,
        feature_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Preprocess the data (add features, scale).
        
        Args:
            df: Input DataFrame (uses cleaned_data if None)
            add_features: Whether to add derived features
            fit_scaler: Whether to fit the scaler (True for training data)
            feature_columns: Columns to scale
        
        Returns:
            Preprocessed DataFrame
        """
        if df is None:
            df = self.cleaned_data
        
        if df is None:
            raise ValueError("No data to preprocess. Call validate_and_clean() first.")
        
        # Add features
        if add_features:
            df = self.preprocessor.add_returns(df)
            df = self.preprocessor.add_volatility(df)
            df = self.preprocessor.add_rolling_stats(df)
            df = self.preprocessor.add_price_features(df)
        
        # Drop NaN values created by rolling calculations
        df = df.dropna()
        
        # Fit scaler if needed
        if fit_scaler:
            self.preprocessor.fit(df, feature_columns)
        
        self.processed_data = df
        self.metadata["processed_rows"] = len(self.processed_data)
        self.metadata["features"] = list(self.processed_data.columns)
        
        return self.processed_data
    
    def split(
        self,
        df: Optional[pd.DataFrame] = None,
        include_validation: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Split data into train/validation/test sets.
        
        Args:
            df: Input DataFrame (uses processed_data if None)
            include_validation: Whether to include validation set
        
        Returns:
            Dictionary with "train", "validation", "test" DataFrames
        """
        if df is None:
            df = self.processed_data
        
        if df is None:
            raise ValueError("No data to split. Call preprocess() first.")
        
        self.split_data = self.splitter.split(df, include_validation=include_validation)
        
        # Check if splits have enough data
        min_samples = 10  # Minimum samples required
        for name, data in self.split_data.items():
            if len(data) < min_samples:
                logger.warning(
                    f"{name} set has only {len(data)} samples (minimum recommended: {min_samples}). "
                    f"Consider adjusting date ranges or fetching more data."
                )
        
        # Update metadata
        self.metadata["split_info"] = {
            name: {
                "rows": len(data),
                "start": data.index.min().strftime("%Y-%m-%d") if len(data) > 0 else None,
                "end": data.index.max().strftime("%Y-%m-%d") if len(data) > 0 else None
            }
            for name, data in self.split_data.items()
        }
        
        return self.split_data
    
    def run(
        self,
        symbol: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        Run the complete pipeline.
        
        Args:
            symbol: Symbol to fetch
            start_date: Start date
            end_date: End date
            **kwargs: Additional arguments for the fetcher
        
        Returns:
            Dictionary with "train", "validation", "test" DataFrames
        """
        logger.info(f"Running pipeline for {symbol}")
        
        # Step 1: Fetch data
        self.fetch(symbol, start_date, end_date, **kwargs)
        
        # Step 2: Validate and clean
        self.validate_and_clean()
        
        # Step 3: Preprocess
        self.preprocess()
        
        # Step 4: Split
        self.split()
        
        logger.info(f"Pipeline complete for {symbol}")
        return self.split_data
    
    def save(self, path: Optional[str] = None, symbol: Optional[str] = None):
        """
        Save processed data to disk.
        
        Args:
            path: Directory to save to (defaults to data_dir/processed)
            symbol: Symbol name for filename
        """
        if path is None:
            path = self.data_dir / "processed"
        else:
            path = Path(path)
        
        path.mkdir(parents=True, exist_ok=True)
        
        if symbol is None:
            symbol = self.metadata.get("symbol", "data")
        
        # Save split data
        if self.split_data is not None:
            for name, df in self.split_data.items():
                filepath = path / f"{symbol}_{name}.parquet"
                df.to_parquet(filepath)
                logger.info(f"Saved {name} data to {filepath}")
        
        # Save metadata
        metadata_path = path / f"{symbol}_metadata.json"
        with open(metadata_path, "w") as f:
            # Convert non-serializable items
            metadata_to_save = {
                k: v for k, v in self.metadata.items()
                if k != "validation"  # Skip validation dict with numpy types
            }
            json.dump(metadata_to_save, f, indent=2, default=str)
        logger.info(f"Saved metadata to {metadata_path}")
    
    def load(self, path: str, symbol: str) -> Dict[str, pd.DataFrame]:
        """
        Load processed data from disk.
        
        Args:
            path: Directory to load from
            symbol: Symbol name
        
        Returns:
            Dictionary with loaded DataFrames
        """
        path = Path(path)
        
        self.split_data = {}
        for name in ["train", "validation", "test"]:
            filepath = path / f"{symbol}_{name}.parquet"
            if filepath.exists():
                self.split_data[name] = pd.read_parquet(filepath)
                logger.info(f"Loaded {name} data from {filepath}")
        
        # Load metadata
        metadata_path = path / f"{symbol}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, "r") as f:
                self.metadata = json.load(f)
        
        return self.split_data
    
    def get_summary(self) -> Dict:
        """
        Get a summary of the pipeline state.
        
        Returns:
            Dictionary with pipeline summary
        """
        return {
            "source": self.source,
            "metadata": self.metadata,
            "has_raw_data": self.raw_data is not None,
            "has_cleaned_data": self.cleaned_data is not None,
            "has_processed_data": self.processed_data is not None,
            "has_split_data": self.split_data is not None,
            "scaler_fitted": self.preprocessor._is_fitted
        }


def download_and_prepare_data(
    symbol: str = "SPY",
    source: str = "yfinance",
    save: bool = True
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to download and prepare data.
    
    Args:
        symbol: Symbol to fetch
        source: Data source
        save: Whether to save the data
    
    Returns:
        Dictionary with train/validation/test DataFrames
    """
    pipeline = DataPipeline(source=source)
    data = pipeline.run(symbol)
    
    if save:
        pipeline.save(symbol=symbol)
    
    return data

