"""
Technical Indicators using TA-Lib.

This module provides a comprehensive set of technical indicators
commonly used in trading and financial analysis.
"""

import pandas as pd
import numpy as np
import talib
from typing import Optional, Tuple, Dict
import logging

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """
    Calculate technical indicators using TA-Lib.
    
    Provides methods for:
    - Momentum indicators (RSI, Stochastic, etc.)
    - Trend indicators (MACD, ADX, etc.)
    - Volatility indicators (Bollinger Bands, ATR, etc.)
    - Volume indicators (OBV, AD, etc.)
    - Moving averages (SMA, EMA, etc.)
    """
    
    # ==================== MOMENTUM INDICATORS ====================
    
    @staticmethod
    def rsi(
        close: pd.Series,
        timeperiod: int = 14
    ) -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).
        
        RSI measures the speed and magnitude of price changes.
        Values above 70 indicate overbought, below 30 indicate oversold.
        
        Args:
            close: Close prices
            timeperiod: RSI period (default: 14)
        
        Returns:
            RSI values (0-100)
        """
        rsi = talib.RSI(close.values, timeperiod=timeperiod)
        return pd.Series(rsi, index=close.index, name=f"rsi_{timeperiod}")
    
    @staticmethod
    def stochastic(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        fastk_period: int = 14,
        slowk_period: int = 3,
        slowd_period: int = 3
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Stochastic Oscillator.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            fastk_period: Fast %K period
            slowk_period: Slow %K period
            slowd_period: Slow %D period
        
        Returns:
            Tuple of (slowk, slowd)
        """
        slowk, slowd = talib.STOCH(
            high.values, low.values, close.values,
            fastk_period=fastk_period,
            slowk_period=slowk_period,
            slowd_period=slowd_period
        )
        return (
            pd.Series(slowk, index=close.index, name="stoch_k"),
            pd.Series(slowd, index=close.index, name="stoch_d")
        )
    
    @staticmethod
    def williams_r(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        timeperiod: int = 14
    ) -> pd.Series:
        """
        Calculate Williams %R.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            timeperiod: Period
        
        Returns:
            Williams %R values (-100 to 0)
        """
        willr = talib.WILLR(high.values, low.values, close.values, timeperiod=timeperiod)
        return pd.Series(willr, index=close.index, name=f"willr_{timeperiod}")
    
    @staticmethod
    def cci(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        timeperiod: int = 20
    ) -> pd.Series:
        """
        Calculate Commodity Channel Index (CCI).
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            timeperiod: Period
        
        Returns:
            CCI values
        """
        cci = talib.CCI(high.values, low.values, close.values, timeperiod=timeperiod)
        return pd.Series(cci, index=close.index, name=f"cci_{timeperiod}")
    
    @staticmethod
    def mfi(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        timeperiod: int = 14
    ) -> pd.Series:
        """
        Calculate Money Flow Index (MFI).
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume
            timeperiod: Period
        
        Returns:
            MFI values (0-100)
        """
        # Convert to float64 for TA-Lib compatibility
        mfi = talib.MFI(
            high.values.astype(np.float64),
            low.values.astype(np.float64),
            close.values.astype(np.float64),
            volume.values.astype(np.float64),
            timeperiod=timeperiod
        )
        return pd.Series(mfi, index=close.index, name=f"mfi_{timeperiod}")
    
    # ==================== TREND INDICATORS ====================
    
    @staticmethod
    def macd(
        close: pd.Series,
        fastperiod: int = 12,
        slowperiod: int = 26,
        signalperiod: int = 9
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Moving Average Convergence Divergence (MACD).
        
        MACD is a trend-following momentum indicator.
        
        Args:
            close: Close prices
            fastperiod: Fast EMA period (default: 12)
            slowperiod: Slow EMA period (default: 26)
            signalperiod: Signal line period (default: 9)
        
        Returns:
            Tuple of (macd, signal, histogram)
        """
        macd, signal, hist = talib.MACD(
            close.values,
            fastperiod=fastperiod,
            slowperiod=slowperiod,
            signalperiod=signalperiod
        )
        return (
            pd.Series(macd, index=close.index, name="macd"),
            pd.Series(signal, index=close.index, name="macd_signal"),
            pd.Series(hist, index=close.index, name="macd_hist")
        )
    
    @staticmethod
    def adx(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        timeperiod: int = 14
    ) -> pd.Series:
        """
        Calculate Average Directional Index (ADX).
        
        ADX measures trend strength (0-100).
        Values above 25 indicate a strong trend.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            timeperiod: Period
        
        Returns:
            ADX values (0-100)
        """
        adx = talib.ADX(high.values, low.values, close.values, timeperiod=timeperiod)
        return pd.Series(adx, index=close.index, name=f"adx_{timeperiod}")
    
    @staticmethod
    def plus_di(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        timeperiod: int = 14
    ) -> pd.Series:
        """Calculate Plus Directional Indicator (+DI)."""
        plus_di = talib.PLUS_DI(high.values, low.values, close.values, timeperiod=timeperiod)
        return pd.Series(plus_di, index=close.index, name=f"plus_di_{timeperiod}")
    
    @staticmethod
    def minus_di(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        timeperiod: int = 14
    ) -> pd.Series:
        """Calculate Minus Directional Indicator (-DI)."""
        minus_di = talib.MINUS_DI(high.values, low.values, close.values, timeperiod=timeperiod)
        return pd.Series(minus_di, index=close.index, name=f"minus_di_{timeperiod}")
    
    @staticmethod
    def aroon(
        high: pd.Series,
        low: pd.Series,
        timeperiod: int = 25
    ) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate Aroon indicator.
        
        Args:
            high: High prices
            low: Low prices
            timeperiod: Period
        
        Returns:
            Tuple of (aroon_down, aroon_up)
        """
        aroon_down, aroon_up = talib.AROON(high.values, low.values, timeperiod=timeperiod)
        return (
            pd.Series(aroon_down, index=high.index, name="aroon_down"),
            pd.Series(aroon_up, index=high.index, name="aroon_up")
        )
    
    # ==================== VOLATILITY INDICATORS ====================
    
    @staticmethod
    def bollinger_bands(
        close: pd.Series,
        timeperiod: int = 20,
        nbdevup: float = 2.0,
        nbdevdn: float = 2.0
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """
        Calculate Bollinger Bands.
        
        Bollinger Bands consist of a middle band (SMA) with upper and lower bands
        based on standard deviation.
        
        Args:
            close: Close prices
            timeperiod: Period for SMA (default: 20)
            nbdevup: Number of standard deviations for upper band
            nbdevdn: Number of standard deviations for lower band
        
        Returns:
            Tuple of (upper_band, middle_band, lower_band)
        """
        upper, middle, lower = talib.BBANDS(
            close.values,
            timeperiod=timeperiod,
            nbdevup=nbdevup,
            nbdevdn=nbdevdn
        )
        return (
            pd.Series(upper, index=close.index, name="bb_upper"),
            pd.Series(middle, index=close.index, name="bb_middle"),
            pd.Series(lower, index=close.index, name="bb_lower")
        )
    
    @staticmethod
    def atr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        timeperiod: int = 14
    ) -> pd.Series:
        """
        Calculate Average True Range (ATR).
        
        ATR measures volatility.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            timeperiod: Period
        
        Returns:
            ATR values
        """
        atr = talib.ATR(high.values, low.values, close.values, timeperiod=timeperiod)
        return pd.Series(atr, index=close.index, name=f"atr_{timeperiod}")
    
    @staticmethod
    def natr(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        timeperiod: int = 14
    ) -> pd.Series:
        """
        Calculate Normalized Average True Range (NATR).
        
        NATR is ATR normalized as a percentage of close price.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            timeperiod: Period
        
        Returns:
            NATR values (percentage)
        """
        natr = talib.NATR(high.values, low.values, close.values, timeperiod=timeperiod)
        return pd.Series(natr, index=close.index, name=f"natr_{timeperiod}")
    
    # ==================== VOLUME INDICATORS ====================
    
    @staticmethod
    def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
        """
        Calculate On-Balance Volume (OBV).
        
        OBV uses volume flow to predict price changes.
        
        Args:
            close: Close prices
            volume: Volume
        
        Returns:
            OBV values
        """
        obv = talib.OBV(
            close.values.astype(np.float64),
            volume.values.astype(np.float64)
        )
        return pd.Series(obv, index=close.index, name="obv")
    
    @staticmethod
    def ad(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series
    ) -> pd.Series:
        """
        Calculate Accumulation/Distribution Line (AD).
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume
        
        Returns:
            AD values
        """
        ad = talib.AD(
            high.values.astype(np.float64),
            low.values.astype(np.float64),
            close.values.astype(np.float64),
            volume.values.astype(np.float64)
        )
        return pd.Series(ad, index=close.index, name="ad")
    
    @staticmethod
    def adosc(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        fastperiod: int = 3,
        slowperiod: int = 10
    ) -> pd.Series:
        """
        Calculate Chaikin A/D Oscillator.
        
        Args:
            high: High prices
            low: Low prices
            close: Close prices
            volume: Volume
            fastperiod: Fast period
            slowperiod: Slow period
        
        Returns:
            ADOSC values
        """
        adosc = talib.ADOSC(
            high.values.astype(np.float64),
            low.values.astype(np.float64),
            close.values.astype(np.float64),
            volume.values.astype(np.float64),
            fastperiod=fastperiod, slowperiod=slowperiod
        )
        return pd.Series(adosc, index=close.index, name="adosc")
    
    # ==================== MOVING AVERAGES ====================
    
    @staticmethod
    def sma(close: pd.Series, timeperiod: int = 20) -> pd.Series:
        """Calculate Simple Moving Average (SMA)."""
        sma = talib.SMA(close.values.astype(np.float64), timeperiod=timeperiod)
        return pd.Series(sma, index=close.index, name=f"sma_{timeperiod}")
    
    @staticmethod
    def ema(close: pd.Series, timeperiod: int = 20) -> pd.Series:
        """Calculate Exponential Moving Average (EMA)."""
        ema = talib.EMA(close.values.astype(np.float64), timeperiod=timeperiod)
        return pd.Series(ema, index=close.index, name=f"ema_{timeperiod}")
    
    @staticmethod
    def wma(close: pd.Series, timeperiod: int = 20) -> pd.Series:
        """Calculate Weighted Moving Average (WMA)."""
        wma = talib.WMA(close.values.astype(np.float64), timeperiod=timeperiod)
        return pd.Series(wma, index=close.index, name=f"wma_{timeperiod}")
    
    @staticmethod
    def dema(close: pd.Series, timeperiod: int = 20) -> pd.Series:
        """Calculate Double Exponential Moving Average (DEMA)."""
        dema = talib.DEMA(close.values.astype(np.float64), timeperiod=timeperiod)
        return pd.Series(dema, index=close.index, name=f"dema_{timeperiod}")
    
    @staticmethod
    def tema(close: pd.Series, timeperiod: int = 20) -> pd.Series:
        """Calculate Triple Exponential Moving Average (TEMA)."""
        tema = talib.TEMA(close.values.astype(np.float64), timeperiod=timeperiod)
        return pd.Series(tema, index=close.index, name=f"tema_{timeperiod}")
    
    # ==================== PATTERN RECOGNITION ====================
    
    @staticmethod
    def cdl_doji(
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> pd.Series:
        """Detect Doji candlestick pattern."""
        doji = talib.CDLDOJI(open_.values, high.values, low.values, close.values)
        return pd.Series(doji, index=close.index, name="cdl_doji")
    
    @staticmethod
    def cdl_hammer(
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> pd.Series:
        """Detect Hammer candlestick pattern."""
        hammer = talib.CDLHAMMER(open_.values, high.values, low.values, close.values)
        return pd.Series(hammer, index=close.index, name="cdl_hammer")
    
    @staticmethod
    def cdl_engulfing(
        open_: pd.Series,
        high: pd.Series,
        low: pd.Series,
        close: pd.Series
    ) -> pd.Series:
        """Detect Engulfing candlestick pattern."""
        engulfing = talib.CDLENGULFING(open_.values, high.values, low.values, close.values)
        return pd.Series(engulfing, index=close.index, name="cdl_engulfing")


# Convenience functions for quick indicator calculation
def calculate_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Quick RSI calculation."""
    return TechnicalIndicators.rsi(close, period)


def calculate_macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Quick MACD calculation."""
    return TechnicalIndicators.macd(close, fast, slow, signal)


def calculate_bollinger_bands(
    close: pd.Series,
    period: int = 20,
    std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Quick Bollinger Bands calculation."""
    return TechnicalIndicators.bollinger_bands(close, period, std, std)

