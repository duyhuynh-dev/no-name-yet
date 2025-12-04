"""
Momentum Trading Agent

Implements trend-following strategies based on price momentum indicators.
Buys when momentum is positive and increasing, sells when momentum is negative.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd

from .base import BaseAgent, AgentSignal, AgentAction


class MomentumAgent(BaseAgent):
    """
    Momentum-based trading agent that follows trends.
    
    Uses multiple momentum indicators:
    - Rate of Change (ROC)
    - Moving Average Crossovers
    - ADX for trend strength
    """
    
    def __init__(
        self,
        name: str = "momentum_agent",
        fast_period: int = 10,
        slow_period: int = 30,
        roc_period: int = 14,
        adx_period: int = 14,
        adx_threshold: float = 25.0,
        **kwargs
    ):
        """
        Initialize the Momentum Agent.
        
        Args:
            name: Agent identifier
            fast_period: Fast moving average period
            slow_period: Slow moving average period
            roc_period: Rate of change period
            adx_period: ADX calculation period
            adx_threshold: Minimum ADX for trend confirmation
        """
        super().__init__(
            name=name,
            description="Trend-following momentum strategy",
            lookback_period=max(slow_period, adx_period) + 10,
            **kwargs
        )
        self.fast_period = fast_period
        self.slow_period = slow_period
        self.roc_period = roc_period
        self.adx_period = adx_period
        self.adx_threshold = adx_threshold
    
    def _calculate_roc(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Rate of Change."""
        return (prices - prices.shift(period)) / prices.shift(period) * 100
    
    def _calculate_adx(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average Directional Index."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Directional Movement
        up_move = high - high.shift(1)
        down_move = low.shift(1) - low
        
        plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
        minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
        
        # Smoothed values
        atr = pd.Series(tr).rolling(period).mean()
        plus_di = 100 * pd.Series(plus_dm).rolling(period).mean() / atr
        minus_di = 100 * pd.Series(minus_dm).rolling(period).mean() / atr
        
        # ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx = dx.rolling(period).mean()
        
        return adx
    
    def generate_signal(
        self,
        data: pd.DataFrame,
        position: int = 0,
        **kwargs
    ) -> AgentSignal:
        """
        Generate momentum-based trading signal.
        
        Args:
            data: OHLCV DataFrame
            position: Current position
            
        Returns:
            AgentSignal with momentum-based recommendation
        """
        if len(data) < self.lookback_period:
            return AgentSignal(
                action=AgentAction.HOLD,
                confidence=0.0,
                strength=0.0,
                metadata={"reason": "Insufficient data"}
            )
        
        close = data['close']
        
        # Calculate indicators
        fast_ma = close.rolling(self.fast_period).mean()
        slow_ma = close.rolling(self.slow_period).mean()
        roc = self._calculate_roc(close, self.roc_period)
        adx = self._calculate_adx(data, self.adx_period)
        
        # Get latest values
        current_price = close.iloc[-1]
        fast_ma_val = fast_ma.iloc[-1]
        slow_ma_val = slow_ma.iloc[-1]
        roc_val = roc.iloc[-1]
        adx_val = adx.iloc[-1]
        
        # Previous values for crossover detection
        prev_fast_ma = fast_ma.iloc[-2]
        prev_slow_ma = slow_ma.iloc[-2]
        
        # Determine trend direction
        bullish_cross = (prev_fast_ma <= prev_slow_ma) and (fast_ma_val > slow_ma_val)
        bearish_cross = (prev_fast_ma >= prev_slow_ma) and (fast_ma_val < slow_ma_val)
        
        # Trend strength from ADX
        strong_trend = adx_val > self.adx_threshold
        
        # Calculate signal strength (-1 to 1)
        ma_diff = (fast_ma_val - slow_ma_val) / slow_ma_val
        strength = np.clip(ma_diff * 10 + roc_val / 10, -1, 1)
        
        # Determine action
        if bullish_cross and strong_trend:
            action = AgentAction.STRONG_BUY
            confidence = min(0.9, 0.5 + adx_val / 100)
        elif fast_ma_val > slow_ma_val and roc_val > 0:
            action = AgentAction.BUY
            confidence = min(0.8, 0.4 + adx_val / 100)
        elif bearish_cross and strong_trend:
            action = AgentAction.STRONG_SELL
            confidence = min(0.9, 0.5 + adx_val / 100)
        elif fast_ma_val < slow_ma_val and roc_val < 0:
            action = AgentAction.SELL
            confidence = min(0.8, 0.4 + adx_val / 100)
        else:
            action = AgentAction.HOLD
            confidence = 0.5
        
        return AgentSignal(
            action=action,
            confidence=confidence,
            strength=strength,
            metadata={
                "fast_ma": fast_ma_val,
                "slow_ma": slow_ma_val,
                "roc": roc_val,
                "adx": adx_val,
                "bullish_cross": bullish_cross,
                "bearish_cross": bearish_cross,
            }
        )
    
    def get_strategy_params(self) -> Dict[str, Any]:
        """Get momentum strategy parameters."""
        return {
            "fast_period": self.fast_period,
            "slow_period": self.slow_period,
            "roc_period": self.roc_period,
            "adx_period": self.adx_period,
            "adx_threshold": self.adx_threshold,
        }

