"""
Indicator Features module for comprehensive feature engineering.

This module provides a unified interface to add multiple technical
indicators to OHLCV data for use in ML/RL models.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import logging

from .technical import TechnicalIndicators

logger = logging.getLogger(__name__)


class IndicatorFeatures:
    """
    Feature engineering using technical indicators.
    
    Provides easy-to-use methods to add comprehensive indicator features
    to market data for ML/RL training.
    """
    
    # Default indicator configurations
    DEFAULT_RSI_PERIODS = [7, 14, 21]
    DEFAULT_MA_PERIODS = [5, 10, 20, 50]
    DEFAULT_BB_PERIOD = 20
    DEFAULT_MACD_PARAMS = (12, 26, 9)
    DEFAULT_ATR_PERIOD = 14
    DEFAULT_ADX_PERIOD = 14
    
    def __init__(
        self,
        rsi_periods: List[int] = None,
        ma_periods: List[int] = None,
        bb_period: int = None,
        macd_params: tuple = None,
        atr_period: int = None,
        adx_period: int = None
    ):
        """
        Initialize feature generator with configurable periods.
        
        Args:
            rsi_periods: List of RSI periods to calculate
            ma_periods: List of moving average periods
            bb_period: Bollinger Bands period
            macd_params: Tuple of (fast, slow, signal) for MACD
            atr_period: ATR period
            adx_period: ADX period
        """
        self.rsi_periods = rsi_periods or self.DEFAULT_RSI_PERIODS
        self.ma_periods = ma_periods or self.DEFAULT_MA_PERIODS
        self.bb_period = bb_period or self.DEFAULT_BB_PERIOD
        self.macd_params = macd_params or self.DEFAULT_MACD_PARAMS
        self.atr_period = atr_period or self.DEFAULT_ATR_PERIOD
        self.adx_period = adx_period or self.DEFAULT_ADX_PERIOD
        
        self.ti = TechnicalIndicators()
    
    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum indicators to DataFrame.
        
        Adds:
        - RSI (multiple periods)
        - Stochastic Oscillator
        - Williams %R
        - CCI
        - MFI (if volume available)
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with momentum indicators added
        """
        df = df.copy()
        
        # RSI with multiple periods
        for period in self.rsi_periods:
            df[f"rsi_{period}"] = self.ti.rsi(df["close"], period)
        
        # Stochastic
        stoch_k, stoch_d = self.ti.stochastic(df["high"], df["low"], df["close"])
        df["stoch_k"] = stoch_k
        df["stoch_d"] = stoch_d
        
        # Williams %R
        df["willr_14"] = self.ti.williams_r(df["high"], df["low"], df["close"])
        
        # CCI
        df["cci_20"] = self.ti.cci(df["high"], df["low"], df["close"])
        
        # MFI (requires volume)
        if "volume" in df.columns:
            df["mfi_14"] = self.ti.mfi(df["high"], df["low"], df["close"], df["volume"])
        
        logger.info("Added momentum indicators")
        return df
    
    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend indicators to DataFrame.
        
        Adds:
        - MACD (line, signal, histogram)
        - ADX
        - +DI / -DI
        - Aroon
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with trend indicators added
        """
        df = df.copy()
        
        # MACD
        macd, signal, hist = self.ti.macd(
            df["close"],
            self.macd_params[0],
            self.macd_params[1],
            self.macd_params[2]
        )
        df["macd"] = macd
        df["macd_signal"] = signal
        df["macd_hist"] = hist
        
        # Normalize MACD by price for cross-asset comparability
        df["macd_pct"] = macd / df["close"] * 100
        
        # ADX
        df["adx"] = self.ti.adx(df["high"], df["low"], df["close"], self.adx_period)
        
        # Directional indicators
        df["plus_di"] = self.ti.plus_di(df["high"], df["low"], df["close"], self.adx_period)
        df["minus_di"] = self.ti.minus_di(df["high"], df["low"], df["close"], self.adx_period)
        df["di_diff"] = df["plus_di"] - df["minus_di"]
        
        # Aroon
        aroon_down, aroon_up = self.ti.aroon(df["high"], df["low"])
        df["aroon_up"] = aroon_up
        df["aroon_down"] = aroon_down
        df["aroon_osc"] = aroon_up - aroon_down
        
        logger.info("Added trend indicators")
        return df
    
    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility indicators to DataFrame.
        
        Adds:
        - Bollinger Bands (upper, middle, lower, width, %B)
        - ATR / NATR
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with volatility indicators added
        """
        df = df.copy()
        
        # Bollinger Bands
        bb_upper, bb_middle, bb_lower = self.ti.bollinger_bands(
            df["close"], self.bb_period
        )
        df["bb_upper"] = bb_upper
        df["bb_middle"] = bb_middle
        df["bb_lower"] = bb_lower
        
        # Bollinger Band Width (volatility measure)
        df["bb_width"] = (bb_upper - bb_lower) / bb_middle
        
        # %B (position within bands)
        df["bb_pct_b"] = (df["close"] - bb_lower) / (bb_upper - bb_lower)
        
        # ATR
        df["atr"] = self.ti.atr(df["high"], df["low"], df["close"], self.atr_period)
        
        # NATR (Normalized ATR)
        df["natr"] = self.ti.natr(df["high"], df["low"], df["close"], self.atr_period)
        
        logger.info("Added volatility indicators")
        return df
    
    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume indicators to DataFrame.
        
        Adds:
        - OBV
        - A/D Line
        - Chaikin Oscillator
        - Volume moving averages
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with volume indicators added
        """
        df = df.copy()
        
        if "volume" not in df.columns:
            logger.warning("No volume data available, skipping volume indicators")
            return df
        
        # OBV
        df["obv"] = self.ti.obv(df["close"], df["volume"])
        
        # Normalized OBV (rate of change)
        df["obv_roc"] = df["obv"].pct_change(periods=5) * 100
        
        # A/D Line
        df["ad"] = self.ti.ad(df["high"], df["low"], df["close"], df["volume"])
        
        # Chaikin Oscillator
        df["adosc"] = self.ti.adosc(df["high"], df["low"], df["close"], df["volume"])
        
        # Volume moving averages
        for period in [5, 10, 20]:
            df[f"volume_sma_{period}"] = self.ti.sma(df["volume"], period)
        
        # Volume ratio (current vs average)
        df["volume_ratio"] = df["volume"] / df["volume_sma_20"]
        
        logger.info("Added volume indicators")
        return df
    
    def add_moving_averages(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add moving average indicators to DataFrame.
        
        Adds:
        - SMA (multiple periods)
        - EMA (multiple periods)
        - Price relative to MAs
        - MA crossover signals
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with moving average indicators added
        """
        df = df.copy()
        
        # SMA
        for period in self.ma_periods:
            df[f"sma_{period}"] = self.ti.sma(df["close"], period)
            df[f"close_sma_{period}_ratio"] = df["close"] / df[f"sma_{period}"]
        
        # EMA
        for period in self.ma_periods:
            df[f"ema_{period}"] = self.ti.ema(df["close"], period)
            df[f"close_ema_{period}_ratio"] = df["close"] / df[f"ema_{period}"]
        
        # MA crossover signals
        if len(self.ma_periods) >= 2:
            short_ma = f"sma_{self.ma_periods[0]}"
            long_ma = f"sma_{self.ma_periods[-1]}"
            df["ma_cross"] = (df[short_ma] > df[long_ma]).astype(int)
            df["ma_diff"] = (df[short_ma] - df[long_ma]) / df[long_ma] * 100
        
        logger.info("Added moving average indicators")
        return df
    
    def add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add price-based features.
        
        Adds:
        - Returns (various periods)
        - Log returns
        - Price momentum
        - High/Low ratios
        
        Args:
            df: DataFrame with OHLCV data
        
        Returns:
            DataFrame with price features added
        """
        df = df.copy()
        
        # Returns at various lookbacks
        for period in [1, 5, 10, 20]:
            df[f"return_{period}d"] = df["close"].pct_change(period)
            df[f"log_return_{period}d"] = np.log(df["close"] / df["close"].shift(period))
        
        # Price momentum (rate of change)
        df["roc_10"] = (df["close"] - df["close"].shift(10)) / df["close"].shift(10) * 100
        df["roc_20"] = (df["close"] - df["close"].shift(20)) / df["close"].shift(20) * 100
        
        # Intraday features
        df["intraday_range"] = (df["high"] - df["low"]) / df["open"]
        df["upper_wick"] = (df["high"] - np.maximum(df["open"], df["close"])) / df["open"]
        df["lower_wick"] = (np.minimum(df["open"], df["close"]) - df["low"]) / df["open"]
        df["body_size"] = np.abs(df["close"] - df["open"]) / df["open"]
        df["body_direction"] = np.sign(df["close"] - df["open"])
        
        # Gap features
        df["gap"] = (df["open"] - df["close"].shift(1)) / df["close"].shift(1)
        
        logger.info("Added price features")
        return df
    
    def add_all_indicators(
        self,
        df: pd.DataFrame,
        include_volume: bool = True,
        drop_na: bool = True
    ) -> pd.DataFrame:
        """
        Add all technical indicators to DataFrame.
        
        This is the main method to use for comprehensive feature engineering.
        
        Args:
            df: DataFrame with OHLCV data
            include_volume: Whether to add volume indicators
            drop_na: Whether to drop rows with NaN values
        
        Returns:
            DataFrame with all indicators added
        """
        logger.info(f"Adding all indicators to {len(df)} rows")
        
        # Add each category of indicators
        df = self.add_momentum_indicators(df)
        df = self.add_trend_indicators(df)
        df = self.add_volatility_indicators(df)
        df = self.add_moving_averages(df)
        df = self.add_price_features(df)
        
        if include_volume and "volume" in df.columns:
            df = self.add_volume_indicators(df)
        
        if drop_na:
            initial_rows = len(df)
            df = df.dropna()
            dropped = initial_rows - len(df)
            logger.info(f"Dropped {dropped} rows with NaN values")
        
        logger.info(f"Feature engineering complete: {len(df)} rows, {len(df.columns)} features")
        return df
    
    def add_state_features(
        self,
        df: pd.DataFrame,
        features: List[str] = None
    ) -> pd.DataFrame:
        """
        Add minimal features for RL state space.
        
        Adds only the essential indicators for the RL agent's observation:
        - RSI
        - MACD
        - Bollinger %B
        - ATR
        - Returns
        
        Args:
            df: DataFrame with OHLCV data
            features: Optional list of specific features to add
        
        Returns:
            DataFrame with state features
        """
        df = df.copy()
        
        if features is None:
            # Default minimal state features
            features = ["rsi", "macd", "bb", "atr", "returns"]
        
        if "rsi" in features:
            df["rsi"] = self.ti.rsi(df["close"], 14)
        
        if "macd" in features:
            macd, signal, hist = self.ti.macd(df["close"])
            df["macd"] = macd / df["close"] * 100  # Normalized
            df["macd_signal"] = signal / df["close"] * 100
            df["macd_hist"] = hist / df["close"] * 100
        
        if "bb" in features:
            bb_upper, bb_middle, bb_lower = self.ti.bollinger_bands(df["close"])
            df["bb_pct_b"] = (df["close"] - bb_lower) / (bb_upper - bb_lower)
            df["bb_width"] = (bb_upper - bb_lower) / bb_middle
        
        if "atr" in features:
            df["atr_pct"] = self.ti.natr(df["high"], df["low"], df["close"])
        
        if "returns" in features:
            df["return_1d"] = df["close"].pct_change()
            df["return_5d"] = df["close"].pct_change(5)
        
        return df.dropna()
    
    def get_feature_names(self) -> Dict[str, List[str]]:
        """
        Get names of features added by each method.
        
        Returns:
            Dictionary mapping category to list of feature names
        """
        return {
            "momentum": [
                *[f"rsi_{p}" for p in self.rsi_periods],
                "stoch_k", "stoch_d", "willr_14", "cci_20", "mfi_14"
            ],
            "trend": [
                "macd", "macd_signal", "macd_hist", "macd_pct",
                "adx", "plus_di", "minus_di", "di_diff",
                "aroon_up", "aroon_down", "aroon_osc"
            ],
            "volatility": [
                "bb_upper", "bb_middle", "bb_lower", "bb_width", "bb_pct_b",
                "atr", "natr"
            ],
            "volume": [
                "obv", "obv_roc", "ad", "adosc",
                "volume_sma_5", "volume_sma_10", "volume_sma_20", "volume_ratio"
            ],
            "moving_averages": [
                *[f"sma_{p}" for p in self.ma_periods],
                *[f"ema_{p}" for p in self.ma_periods],
                *[f"close_sma_{p}_ratio" for p in self.ma_periods],
                *[f"close_ema_{p}_ratio" for p in self.ma_periods],
                "ma_cross", "ma_diff"
            ],
            "price": [
                "return_1d", "return_5d", "return_10d", "return_20d",
                "log_return_1d", "log_return_5d", "log_return_10d", "log_return_20d",
                "roc_10", "roc_20",
                "intraday_range", "upper_wick", "lower_wick", "body_size", "body_direction",
                "gap"
            ]
        }


def add_indicators(
    df: pd.DataFrame,
    minimal: bool = False
) -> pd.DataFrame:
    """
    Convenience function to add indicators to a DataFrame.
    
    Args:
        df: DataFrame with OHLCV data
        minimal: If True, add only minimal state features
    
    Returns:
        DataFrame with indicators added
    """
    features = IndicatorFeatures()
    
    if minimal:
        return features.add_state_features(df)
    else:
        return features.add_all_indicators(df)

