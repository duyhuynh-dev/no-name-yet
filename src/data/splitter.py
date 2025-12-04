"""
Data Splitter module for train/test splitting and walk-forward validation.

Provides functionality for:
- Time-based train/test splits
- Walk-forward validation
- Cross-validation for time series
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Optional, Generator, Dict
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class DataSplitter:
    """
    Handles data splitting for time series data.
    
    Features:
    - Time-based splits (no data leakage)
    - Walk-forward validation
    - Expanding window validation
    """
    
    def __init__(
        self,
        train_start: str = "2018-01-01",
        train_end: str = "2022-12-31",
        test_start: str = "2023-01-01",
        test_end: str = "2023-12-31",
        validation_ratio: float = 0.1
    ):
        """
        Initialize the splitter.
        
        Args:
            train_start: Start date for training data
            train_end: End date for training data
            test_start: Start date for test data
            test_end: End date for test data
            validation_ratio: Ratio of training data to use for validation
        """
        # Store as strings, convert when needed with proper timezone handling
        self._train_start = train_start
        self._train_end = train_end
        self._test_start = test_start
        self._test_end = test_end
        self.validation_ratio = validation_ratio
    
    def _get_datetime(self, date_str: str, tz=None) -> pd.Timestamp:
        """Convert date string to timestamp with optional timezone."""
        ts = pd.to_datetime(date_str)
        if tz is not None:
            ts = ts.tz_localize(tz)
        return ts
    
    @property
    def train_start(self):
        return pd.to_datetime(self._train_start)
    
    @property
    def train_end(self):
        return pd.to_datetime(self._train_end)
    
    @property
    def test_start(self):
        return pd.to_datetime(self._test_start)
    
    @property
    def test_end(self):
        return pd.to_datetime(self._test_end)
    
    def split(
        self,
        df: pd.DataFrame,
        include_validation: bool = True
    ) -> Dict[str, pd.DataFrame]:
        """
        Split data into train, validation, and test sets.
        
        Args:
            df: Input DataFrame with datetime index
            include_validation: Whether to split out a validation set
        
        Returns:
            Dictionary with "train", "validation" (optional), and "test" DataFrames
        """
        # Ensure datetime index
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex")
        
        # Sort by index
        df = df.sort_index()
        
        # Handle timezone-aware indices
        tz = df.index.tz
        train_start = self._get_datetime(self._train_start, tz)
        train_end = self._get_datetime(self._train_end, tz)
        test_start = self._get_datetime(self._test_start, tz)
        test_end = self._get_datetime(self._test_end, tz)
        
        # Split by date
        train_mask = (df.index >= train_start) & (df.index <= train_end)
        test_mask = (df.index >= test_start) & (df.index <= test_end)
        
        train_df = df[train_mask].copy()
        test_df = df[test_mask].copy()
        
        result = {"train": train_df, "test": test_df}
        
        if include_validation and self.validation_ratio > 0:
            # Split validation from the end of training data
            val_size = int(len(train_df) * self.validation_ratio)
            result["train"] = train_df.iloc[:-val_size].copy()
            result["validation"] = train_df.iloc[-val_size:].copy()
        
        # Log split info
        for name, data in result.items():
            if len(data) > 0:
                logger.info(
                    f"{name}: {len(data)} samples, "
                    f"{data.index.min().date()} to {data.index.max().date()}"
                )
            else:
                logger.warning(f"{name}: No data in specified date range")
        
        return result
    
    def walk_forward_split(
        self,
        df: pd.DataFrame,
        n_splits: int = 5,
        train_size: Optional[int] = None,
        test_size: Optional[int] = None,
        gap: int = 0
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """
        Generate walk-forward validation splits.
        
        Walk-forward validation trains on historical data and tests on future data,
        moving the window forward for each split.
        
        Args:
            df: Input DataFrame
            n_splits: Number of splits
            train_size: Fixed training size (if None, uses expanding window)
            test_size: Size of each test set (if None, splits evenly)
            gap: Number of samples to skip between train and test (to avoid lookahead)
        
        Yields:
            Tuple of (train_df, test_df) for each split
        """
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame must have a DatetimeIndex")
        
        df = df.sort_index()
        n_samples = len(df)
        
        # Calculate test size if not specified
        if test_size is None:
            test_size = n_samples // (n_splits + 1)
        
        logger.info(f"Walk-forward validation: {n_splits} splits, test_size={test_size}")
        
        for i in range(n_splits):
            # Calculate indices
            test_end_idx = n_samples - (n_splits - i - 1) * test_size
            test_start_idx = test_end_idx - test_size
            train_end_idx = test_start_idx - gap
            
            if train_size is not None:
                train_start_idx = max(0, train_end_idx - train_size)
            else:
                train_start_idx = 0  # Expanding window
            
            # Split data
            train_df = df.iloc[train_start_idx:train_end_idx].copy()
            test_df = df.iloc[test_start_idx:test_end_idx].copy()
            
            logger.info(
                f"Split {i+1}/{n_splits}: "
                f"train={len(train_df)} ({train_df.index.min().date()} to {train_df.index.max().date()}), "
                f"test={len(test_df)} ({test_df.index.min().date()} to {test_df.index.max().date()})"
            )
            
            yield train_df, test_df
    
    def expanding_window_split(
        self,
        df: pd.DataFrame,
        initial_train_size: int,
        test_size: int,
        step_size: Optional[int] = None
    ) -> Generator[Tuple[pd.DataFrame, pd.DataFrame], None, None]:
        """
        Generate expanding window validation splits.
        
        Training window expands with each split while test window moves forward.
        
        Args:
            df: Input DataFrame
            initial_train_size: Initial size of training set
            test_size: Size of each test set
            step_size: How much to move forward each split (default: test_size)
        
        Yields:
            Tuple of (train_df, test_df) for each split
        """
        if step_size is None:
            step_size = test_size
        
        df = df.sort_index()
        n_samples = len(df)
        
        train_end = initial_train_size
        split_num = 0
        
        while train_end + test_size <= n_samples:
            train_df = df.iloc[:train_end].copy()
            test_df = df.iloc[train_end:train_end + test_size].copy()
            
            split_num += 1
            logger.info(
                f"Split {split_num}: "
                f"train={len(train_df)}, test={len(test_df)}"
            )
            
            yield train_df, test_df
            
            train_end += step_size
    
    def get_split_info(self, df: pd.DataFrame) -> Dict:
        """
        Get information about the data split without actually splitting.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Dictionary with split information
        """
        df = df.sort_index()
        
        # Handle timezone-aware indices
        tz = df.index.tz
        train_start = self._get_datetime(self._train_start, tz)
        train_end = self._get_datetime(self._train_end, tz)
        test_start = self._get_datetime(self._test_start, tz)
        test_end = self._get_datetime(self._test_end, tz)
        
        train_mask = (df.index >= train_start) & (df.index <= train_end)
        test_mask = (df.index >= test_start) & (df.index <= test_end)
        
        train_count = train_mask.sum()
        test_count = test_mask.sum()
        
        val_count = int(train_count * self.validation_ratio)
        actual_train_count = train_count - val_count
        
        return {
            "total_samples": len(df),
            "date_range": {
                "start": df.index.min().strftime("%Y-%m-%d"),
                "end": df.index.max().strftime("%Y-%m-%d")
            },
            "train": {
                "samples": actual_train_count,
                "start": self.train_start.strftime("%Y-%m-%d"),
                "end": self.train_end.strftime("%Y-%m-%d")
            },
            "validation": {
                "samples": val_count,
                "ratio": self.validation_ratio
            },
            "test": {
                "samples": test_count,
                "start": self.test_start.strftime("%Y-%m-%d"),
                "end": self.test_end.strftime("%Y-%m-%d")
            },
            "train_test_ratio": round(actual_train_count / max(test_count, 1), 2)
        }


def create_train_test_split(
    df: pd.DataFrame,
    test_ratio: float = 0.2,
    validation_ratio: float = 0.1
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function for simple time-based split.
    
    Args:
        df: Input DataFrame with datetime index
        test_ratio: Ratio of data to use for testing
        validation_ratio: Ratio of training data to use for validation
    
    Returns:
        Dictionary with "train", "validation", and "test" DataFrames
    """
    df = df.sort_index()
    n = len(df)
    
    test_size = int(n * test_ratio)
    train_val_size = n - test_size
    val_size = int(train_val_size * validation_ratio)
    train_size = train_val_size - val_size
    
    return {
        "train": df.iloc[:train_size].copy(),
        "validation": df.iloc[train_size:train_size + val_size].copy(),
        "test": df.iloc[-test_size:].copy()
    }

