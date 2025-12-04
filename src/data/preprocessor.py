"""
Data Preprocessor module for feature engineering and normalization.

Provides functionality for:
- Feature normalization/standardization
- Return calculations
- Volatility calculations
- Rolling statistics
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import logging

logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Preprocesses market data for RL training.
    
    Features:
    - Price normalization
    - Return calculations
    - Rolling statistics
    - Feature scaling
    """
    
    def __init__(
        self,
        scaler_type: str = "standard",
        lookback_window: int = 30
    ):
        """
        Initialize the preprocessor.
        
        Args:
            scaler_type: Type of scaler ("standard", "minmax", "none")
            lookback_window: Window size for rolling calculations
        """
        self.scaler_type = scaler_type
        self.lookback_window = lookback_window
        self.scalers: Dict[str, object] = {}
        self._is_fitted = False
    
    def _get_scaler(self):
        """Get a new scaler instance."""
        if self.scaler_type == "standard":
            return StandardScaler()
        elif self.scaler_type == "minmax":
            return MinMaxScaler()
        else:
            return None
    
    def add_returns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add return columns to the DataFrame.
        
        Args:
            df: Input DataFrame with OHLCV data
        
        Returns:
            DataFrame with return columns added
        """
        df = df.copy()
        
        # Simple returns
        df["returns"] = df["close"].pct_change()
        
        # Log returns
        df["log_returns"] = np.log(df["close"] / df["close"].shift(1))
        
        # Intraday return (close vs open)
        df["intraday_return"] = (df["close"] - df["open"]) / df["open"]
        
        # Overnight return (open vs previous close)
        df["overnight_return"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
        
        return df
    
    def add_volatility(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility metrics to the DataFrame.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with volatility columns added
        """
        df = df.copy()
        
        # Ensure returns exist
        if "returns" not in df.columns:
            df = self.add_returns(df)
        
        # Rolling volatility (standard deviation of returns)
        df["volatility"] = df["returns"].rolling(window=self.lookback_window).std()
        
        # True Range
        df["tr"] = np.maximum(
            df["high"] - df["low"],
            np.maximum(
                np.abs(df["high"] - df["close"].shift(1)),
                np.abs(df["low"] - df["close"].shift(1))
            )
        )
        
        # Average True Range (ATR)
        df["atr"] = df["tr"].rolling(window=14).mean()
        
        # Normalized ATR (ATR / Close)
        df["atr_pct"] = df["atr"] / df["close"]
        
        return df
    
    def add_rolling_stats(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add rolling statistics to the DataFrame.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with rolling statistics added
        """
        df = df.copy()
        
        # Rolling mean
        df["close_ma"] = df["close"].rolling(window=self.lookback_window).mean()
        
        # Rolling standard deviation
        df["close_std"] = df["close"].rolling(window=self.lookback_window).std()
        
        # Price relative to moving average
        df["close_ma_ratio"] = df["close"] / df["close_ma"]
        
        # Rolling min/max
        df["close_rolling_min"] = df["close"].rolling(window=self.lookback_window).min()
        df["close_rolling_max"] = df["close"].rolling(window=self.lookback_window).max()
        
        # Position in rolling range (0-1)
        df["close_range_position"] = (df["close"] - df["close_rolling_min"]) / (
            df["close_rolling_max"] - df["close_rolling_min"]
        )
        
        # Volume rolling mean
        df["volume_ma"] = df["volume"].rolling(window=self.lookback_window).mean()
        df["volume_ma_ratio"] = df["volume"] / df["volume_ma"]
        
        return df
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price-based features.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with price features added
        """
        df = df.copy()
        
        # Spread (High - Low)
        df["spread"] = df["high"] - df["low"]
        df["spread_pct"] = df["spread"] / df["close"]
        
        # Body (Close - Open)
        df["body"] = df["close"] - df["open"]
        df["body_pct"] = df["body"] / df["open"]
        
        # Upper shadow
        df["upper_shadow"] = df["high"] - np.maximum(df["open"], df["close"])
        
        # Lower shadow
        df["lower_shadow"] = np.minimum(df["open"], df["close"]) - df["low"]
        
        return df
    
    def fit(self, df: pd.DataFrame, feature_columns: Optional[List[str]] = None):
        """
        Fit the scaler(s) on the training data.
        
        Args:
            df: Training DataFrame
            feature_columns: Columns to scale (if None, scales all numeric columns)
        """
        if self.scaler_type == "none":
            self._is_fitted = True
            return
        
        if feature_columns is None:
            feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        self.feature_columns = feature_columns
        
        for col in feature_columns:
            if col in df.columns:
                scaler = self._get_scaler()
                # Reshape for sklearn
                values = df[col].dropna().values.reshape(-1, 1)
                scaler.fit(values)
                self.scalers[col] = scaler
        
        self._is_fitted = True
        logger.info(f"Fitted scalers for {len(self.scalers)} columns")
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform the data using fitted scalers.
        
        Args:
            df: Input DataFrame
        
        Returns:
            Scaled DataFrame
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        if self.scaler_type == "none":
            return df.copy()
        
        df = df.copy()
        
        for col, scaler in self.scalers.items():
            if col in df.columns:
                # Handle NaN values
                mask = ~df[col].isna()
                values = df.loc[mask, col].values.reshape(-1, 1)
                df.loc[mask, col] = scaler.transform(values).flatten()
        
        return df
    
    def fit_transform(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        Fit and transform the data.
        
        Args:
            df: Input DataFrame
            feature_columns: Columns to scale
        
        Returns:
            Scaled DataFrame
        """
        self.fit(df, feature_columns)
        return self.transform(df)
    
    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Inverse transform the scaled data.
        
        Args:
            df: Scaled DataFrame
        
        Returns:
            Original scale DataFrame
        """
        if not self._is_fitted:
            raise ValueError("Preprocessor must be fitted before inverse_transform")
        
        if self.scaler_type == "none":
            return df.copy()
        
        df = df.copy()
        
        for col, scaler in self.scalers.items():
            if col in df.columns:
                mask = ~df[col].isna()
                values = df.loc[mask, col].values.reshape(-1, 1)
                df.loc[mask, col] = scaler.inverse_transform(values).flatten()
        
        return df
    
    def preprocess(
        self,
        df: pd.DataFrame,
        add_features: bool = True,
        scale: bool = True,
        drop_na: bool = True
    ) -> pd.DataFrame:
        """
        Full preprocessing pipeline.
        
        Args:
            df: Input DataFrame with OHLCV data
            add_features: Whether to add derived features
            scale: Whether to scale the data
            drop_na: Whether to drop rows with NaN values
        
        Returns:
            Preprocessed DataFrame
        """
        df = df.copy()
        
        if add_features:
            # Add all features
            df = self.add_returns(df)
            df = self.add_volatility(df)
            df = self.add_rolling_stats(df)
            df = self.add_price_features(df)
        
        if drop_na:
            df = df.dropna()
        
        if scale and self._is_fitted:
            df = self.transform(df)
        
        logger.info(f"Preprocessing complete: {len(df)} rows, {len(df.columns)} columns")
        return df


def create_feature_matrix(
    df: pd.DataFrame,
    window_size: int = 30,
    feature_columns: Optional[List[str]] = None
) -> np.ndarray:
    """
    Create a feature matrix with lookback window.
    
    Args:
        df: Preprocessed DataFrame
        window_size: Number of timesteps to include
        feature_columns: Columns to include in the matrix
    
    Returns:
        3D numpy array of shape (samples, window_size, features)
    """
    if feature_columns is None:
        feature_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    data = df[feature_columns].values
    
    # Create sliding windows
    samples = []
    for i in range(window_size, len(data)):
        samples.append(data[i - window_size:i])
    
    return np.array(samples)

