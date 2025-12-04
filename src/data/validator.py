"""
Data Validator module for validating and cleaning market data.

Provides functionality for:
- Missing value detection and handling
- Outlier detection and handling
- Data quality validation
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class DataValidator:
    """
    Validates and cleans market data.
    
    Handles:
    - Missing values
    - Outliers
    - Data integrity checks
    """
    
    def __init__(
        self,
        outlier_method: str = "zscore",
        outlier_threshold: float = 3.0,
        missing_threshold: float = 0.05
    ):
        """
        Initialize the validator.
        
        Args:
            outlier_method: Method for outlier detection ("zscore", "iqr", "percentile")
            outlier_threshold: Threshold for outlier detection
            missing_threshold: Maximum allowed fraction of missing values (0-1)
        """
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold
        self.missing_threshold = missing_threshold
    
    def validate(self, df: pd.DataFrame) -> Dict:
        """
        Perform comprehensive validation on the data.
        
        Args:
            df: Input DataFrame with OHLCV data
        
        Returns:
            Dictionary with validation results
        """
        results = {
            "is_valid": True,
            "total_rows": len(df),
            "missing_values": {},
            "outliers": {},
            "issues": [],
            "warnings": []
        }
        
        # Check for empty DataFrame
        if df.empty:
            results["is_valid"] = False
            results["issues"].append("DataFrame is empty")
            return results
        
        # Check required columns
        required_cols = ["open", "high", "low", "close", "volume"]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            results["is_valid"] = False
            results["issues"].append(f"Missing required columns: {missing_cols}")
            return results
        
        # Check for missing values
        for col in required_cols:
            missing_count = df[col].isna().sum()
            missing_pct = missing_count / len(df)
            results["missing_values"][col] = {
                "count": int(missing_count),
                "percentage": round(missing_pct * 100, 2)
            }
            if missing_pct > self.missing_threshold:
                results["warnings"].append(
                    f"Column '{col}' has {missing_pct*100:.2f}% missing values"
                )
        
        # Check for outliers in price columns
        price_cols = ["open", "high", "low", "close"]
        for col in price_cols:
            outlier_mask = self._detect_outliers(df[col].dropna())
            outlier_count = outlier_mask.sum()
            results["outliers"][col] = {
                "count": int(outlier_count),
                "percentage": round(outlier_count / len(df) * 100, 2)
            }
        
        # Check for negative values
        for col in required_cols:
            neg_count = (df[col] < 0).sum()
            if neg_count > 0:
                results["issues"].append(f"Column '{col}' has {neg_count} negative values")
                results["is_valid"] = False
        
        # Check OHLC relationships
        invalid_ohlc = self._check_ohlc_relationships(df)
        if invalid_ohlc > 0:
            results["warnings"].append(
                f"{invalid_ohlc} rows have invalid OHLC relationships"
            )
        
        # Check for duplicate timestamps
        if df.index.duplicated().any():
            dup_count = df.index.duplicated().sum()
            results["warnings"].append(f"{dup_count} duplicate timestamps found")
        
        # Check if data is sorted
        if not df.index.is_monotonic_increasing:
            results["warnings"].append("Data is not sorted by timestamp")
        
        logger.info(f"Validation complete: {len(results['issues'])} issues, {len(results['warnings'])} warnings")
        return results
    
    def _detect_outliers(self, series: pd.Series) -> pd.Series:
        """
        Detect outliers using the specified method.
        
        Args:
            series: Input Series
        
        Returns:
            Boolean Series indicating outliers
        """
        if self.outlier_method == "zscore":
            z_scores = np.abs((series - series.mean()) / series.std())
            return z_scores > self.outlier_threshold
        
        elif self.outlier_method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - self.outlier_threshold * iqr
            upper_bound = q3 + self.outlier_threshold * iqr
            return (series < lower_bound) | (series > upper_bound)
        
        elif self.outlier_method == "percentile":
            lower = series.quantile(0.01)
            upper = series.quantile(0.99)
            return (series < lower) | (series > upper)
        
        else:
            raise ValueError(f"Unknown outlier method: {self.outlier_method}")
    
    def _check_ohlc_relationships(self, df: pd.DataFrame) -> int:
        """
        Check that OHLC values maintain proper relationships.
        
        High should be >= Open, Close, Low
        Low should be <= Open, Close, High
        
        Args:
            df: Input DataFrame
        
        Returns:
            Number of rows with invalid OHLC relationships
        """
        invalid_high = (df["high"] < df["open"]) | (df["high"] < df["close"]) | (df["high"] < df["low"])
        invalid_low = (df["low"] > df["open"]) | (df["low"] > df["close"]) | (df["low"] > df["high"])
        
        return (invalid_high | invalid_low).sum()
    
    def clean(
        self,
        df: pd.DataFrame,
        handle_missing: str = "ffill",
        handle_outliers: str = "clip",
        fix_ohlc: bool = True
    ) -> pd.DataFrame:
        """
        Clean the data by handling missing values and outliers.
        
        Args:
            df: Input DataFrame
            handle_missing: Method for handling missing values ("drop", "ffill", "bfill", "interpolate")
            handle_outliers: Method for handling outliers ("clip", "remove", "none")
            fix_ohlc: Whether to fix invalid OHLC relationships
        
        Returns:
            Cleaned DataFrame
        """
        df = df.copy()
        
        # Sort by index if not sorted
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()
        
        # Remove duplicates
        if df.index.duplicated().any():
            df = df[~df.index.duplicated(keep="first")]
            logger.info("Removed duplicate timestamps")
        
        # Handle missing values
        if handle_missing == "drop":
            df = df.dropna()
        elif handle_missing == "ffill":
            df = df.ffill()
            df = df.bfill()  # Handle any remaining NaNs at the start
        elif handle_missing == "bfill":
            df = df.bfill()
            df = df.ffill()
        elif handle_missing == "interpolate":
            df = df.interpolate(method="time")
            df = df.ffill().bfill()
        
        # Handle outliers in price columns
        if handle_outliers != "none":
            price_cols = ["open", "high", "low", "close"]
            for col in price_cols:
                if handle_outliers == "clip":
                    df[col] = self._clip_outliers(df[col])
                elif handle_outliers == "remove":
                    outlier_mask = self._detect_outliers(df[col])
                    df.loc[outlier_mask, col] = np.nan
                    df = df.ffill().bfill()
        
        # Fix OHLC relationships
        if fix_ohlc:
            df = self._fix_ohlc_relationships(df)
        
        # Ensure no negative values
        for col in ["open", "high", "low", "close", "volume"]:
            if col in df.columns:
                df[col] = df[col].clip(lower=0)
        
        logger.info(f"Cleaning complete: {len(df)} rows remaining")
        return df
    
    def _clip_outliers(self, series: pd.Series) -> pd.Series:
        """
        Clip outliers to the threshold boundaries.
        
        Args:
            series: Input Series
        
        Returns:
            Series with outliers clipped
        """
        if self.outlier_method == "zscore":
            mean = series.mean()
            std = series.std()
            lower = mean - self.outlier_threshold * std
            upper = mean + self.outlier_threshold * std
        elif self.outlier_method == "iqr":
            q1 = series.quantile(0.25)
            q3 = series.quantile(0.75)
            iqr = q3 - q1
            lower = q1 - self.outlier_threshold * iqr
            upper = q3 + self.outlier_threshold * iqr
        else:
            lower = series.quantile(0.01)
            upper = series.quantile(0.99)
        
        return series.clip(lower=lower, upper=upper)
    
    def _fix_ohlc_relationships(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fix invalid OHLC relationships.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with fixed OHLC relationships
        """
        df = df.copy()
        
        # High should be the maximum of OHLC
        df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
        
        # Low should be the minimum of OHLC
        df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
        
        return df


def validate_and_clean(
    df: pd.DataFrame,
    verbose: bool = True
) -> Tuple[pd.DataFrame, Dict]:
    """
    Convenience function to validate and clean data.
    
    Args:
        df: Input DataFrame
        verbose: Whether to print validation results
    
    Returns:
        Tuple of (cleaned DataFrame, validation results)
    """
    validator = DataValidator()
    
    # Validate
    results = validator.validate(df)
    
    if verbose:
        print(f"Validation Results:")
        print(f"  Total rows: {results['total_rows']}")
        print(f"  Is valid: {results['is_valid']}")
        if results["issues"]:
            print(f"  Issues: {results['issues']}")
        if results["warnings"]:
            print(f"  Warnings: {results['warnings']}")
    
    # Clean
    df_cleaned = validator.clean(df)
    
    return df_cleaned, results

