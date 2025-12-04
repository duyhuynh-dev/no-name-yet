"""
Mean Reversion Trading Agent

Implements mean reversion strategies that profit from price returning to average.
Buys when price is oversold, sells when price is overbought.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd

from .base import BaseAgent, AgentSignal, AgentAction


class MeanReversionAgent(BaseAgent):
    """
    Mean reversion trading agent.
    
    Uses indicators to identify overbought/oversold conditions:
    - Bollinger Bands
    - RSI (Relative Strength Index)
    - Z-Score of price
    """
    
    def __init__(
        self,
        name: str = "mean_reversion_agent",
        bb_period: int = 20,
        bb_std: float = 2.0,
        rsi_period: int = 14,
        rsi_oversold: float = 30.0,
        rsi_overbought: float = 70.0,
        zscore_threshold: float = 2.0,
        **kwargs
    ):
        """
        Initialize the Mean Reversion Agent.
        
        Args:
            name: Agent identifier
            bb_period: Bollinger Bands period
            bb_std: Bollinger Bands standard deviation multiplier
            rsi_period: RSI calculation period
            rsi_oversold: RSI level indicating oversold
            rsi_overbought: RSI level indicating overbought
            zscore_threshold: Z-score threshold for signals
        """
        super().__init__(
            name=name,
            description="Mean reversion strategy using BB and RSI",
            lookback_period=max(bb_period, rsi_period) + 10,
            **kwargs
        )
        self.bb_period = bb_period
        self.bb_std = bb_std
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.zscore_threshold = zscore_threshold
    
    def _calculate_bollinger_bands(
        self, prices: pd.Series, period: int, std_mult: float
    ) -> tuple:
        """Calculate Bollinger Bands."""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + std_mult * std
        lower = sma - std_mult * std
        return sma, upper, lower
    
    def _calculate_rsi(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Relative Strength Index."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        return 100 - (100 / (1 + rs))
    
    def _calculate_zscore(self, prices: pd.Series, period: int) -> pd.Series:
        """Calculate Z-Score of prices."""
        mean = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        return (prices - mean) / (std + 1e-10)
    
    def generate_signal(
        self,
        data: pd.DataFrame,
        position: int = 0,
        **kwargs
    ) -> AgentSignal:
        """
        Generate mean reversion trading signal.
        
        Args:
            data: OHLCV DataFrame
            position: Current position
            
        Returns:
            AgentSignal with mean reversion recommendation
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
        sma, bb_upper, bb_lower = self._calculate_bollinger_bands(
            close, self.bb_period, self.bb_std
        )
        rsi = self._calculate_rsi(close, self.rsi_period)
        zscore = self._calculate_zscore(close, self.bb_period)
        
        # Get latest values
        current_price = close.iloc[-1]
        sma_val = sma.iloc[-1]
        bb_upper_val = bb_upper.iloc[-1]
        bb_lower_val = bb_lower.iloc[-1]
        rsi_val = rsi.iloc[-1]
        zscore_val = zscore.iloc[-1]
        
        # Calculate percent B (position within Bollinger Bands)
        bb_width = bb_upper_val - bb_lower_val
        percent_b = (current_price - bb_lower_val) / (bb_width + 1e-10)
        
        # Signal strength (negative when oversold, positive when overbought)
        # We want to BUY when oversold (strength negative) and SELL when overbought
        strength = -np.clip(zscore_val / self.zscore_threshold, -1, 1)
        
        # Determine action based on multiple confirmations
        oversold_bb = current_price < bb_lower_val
        overbought_bb = current_price > bb_upper_val
        oversold_rsi = rsi_val < self.rsi_oversold
        overbought_rsi = rsi_val > self.rsi_overbought
        extreme_zscore_low = zscore_val < -self.zscore_threshold
        extreme_zscore_high = zscore_val > self.zscore_threshold
        
        # Count confirmations
        buy_signals = sum([oversold_bb, oversold_rsi, extreme_zscore_low])
        sell_signals = sum([overbought_bb, overbought_rsi, extreme_zscore_high])
        
        if buy_signals >= 2:
            action = AgentAction.STRONG_BUY if buy_signals == 3 else AgentAction.BUY
            confidence = 0.5 + buy_signals * 0.15
        elif sell_signals >= 2:
            action = AgentAction.STRONG_SELL if sell_signals == 3 else AgentAction.SELL
            confidence = 0.5 + sell_signals * 0.15
        elif buy_signals == 1 and position <= 0:
            action = AgentAction.BUY
            confidence = 0.5
        elif sell_signals == 1 and position >= 0:
            action = AgentAction.SELL
            confidence = 0.5
        else:
            action = AgentAction.HOLD
            confidence = 0.5
        
        return AgentSignal(
            action=action,
            confidence=min(confidence, 0.95),
            strength=strength,
            metadata={
                "rsi": rsi_val,
                "zscore": zscore_val,
                "percent_b": percent_b,
                "bb_upper": bb_upper_val,
                "bb_lower": bb_lower_val,
                "oversold_bb": oversold_bb,
                "overbought_bb": overbought_bb,
                "oversold_rsi": oversold_rsi,
                "overbought_rsi": overbought_rsi,
            }
        )
    
    def get_strategy_params(self) -> Dict[str, Any]:
        """Get mean reversion strategy parameters."""
        return {
            "bb_period": self.bb_period,
            "bb_std": self.bb_std,
            "rsi_period": self.rsi_period,
            "rsi_oversold": self.rsi_oversold,
            "rsi_overbought": self.rsi_overbought,
            "zscore_threshold": self.zscore_threshold,
        }

