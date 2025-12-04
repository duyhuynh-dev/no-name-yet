"""
Breakout Trading Agent

Implements breakout strategies that detect and trade range breakouts.
Buys when price breaks above resistance, sells when price breaks below support.
"""

from typing import Dict, Any, Tuple
import numpy as np
import pandas as pd

from .base import BaseAgent, AgentSignal, AgentAction


class BreakoutAgent(BaseAgent):
    """
    Breakout trading agent.
    
    Identifies support/resistance levels and trades breakouts:
    - Donchian Channel breakouts
    - Volume-confirmed breakouts
    - ATR-based stop placement
    """
    
    def __init__(
        self,
        name: str = "breakout_agent",
        channel_period: int = 20,
        volume_period: int = 20,
        volume_threshold: float = 1.5,
        atr_period: int = 14,
        breakout_buffer: float = 0.001,
        **kwargs
    ):
        """
        Initialize the Breakout Agent.
        
        Args:
            name: Agent identifier
            channel_period: Period for Donchian Channel
            volume_period: Period for volume average
            volume_threshold: Volume multiplier for confirmation
            atr_period: ATR calculation period
            breakout_buffer: Buffer above/below levels (as decimal)
        """
        super().__init__(
            name=name,
            description="Breakout strategy with volume confirmation",
            lookback_period=max(channel_period, volume_period, atr_period) + 10,
            **kwargs
        )
        self.channel_period = channel_period
        self.volume_period = volume_period
        self.volume_threshold = volume_threshold
        self.atr_period = atr_period
        self.breakout_buffer = breakout_buffer
    
    def _calculate_donchian_channel(
        self, data: pd.DataFrame, period: int
    ) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Donchian Channel (highest high, lowest low)."""
        upper = data['high'].rolling(period).max()
        lower = data['low'].rolling(period).min()
        middle = (upper + lower) / 2
        return upper, lower, middle
    
    def _calculate_atr(self, data: pd.DataFrame, period: int) -> pd.Series:
        """Calculate Average True Range."""
        high = data['high']
        low = data['low']
        close = data['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        return tr.rolling(period).mean()
    
    def _detect_consolidation(
        self, data: pd.DataFrame, period: int
    ) -> Tuple[bool, float]:
        """
        Detect if price is in consolidation (low volatility range).
        
        Returns:
            Tuple of (is_consolidating, range_width_as_pct)
        """
        recent = data.tail(period)
        high = recent['high'].max()
        low = recent['low'].min()
        mid = (high + low) / 2
        range_pct = (high - low) / mid
        
        # Consider consolidation if range is less than 5%
        is_consolidating = range_pct < 0.05
        return is_consolidating, range_pct
    
    def generate_signal(
        self,
        data: pd.DataFrame,
        position: int = 0,
        **kwargs
    ) -> AgentSignal:
        """
        Generate breakout trading signal.
        
        Args:
            data: OHLCV DataFrame
            position: Current position
            
        Returns:
            AgentSignal with breakout recommendation
        """
        if len(data) < self.lookback_period:
            return AgentSignal(
                action=AgentAction.HOLD,
                confidence=0.0,
                strength=0.0,
                metadata={"reason": "Insufficient data"}
            )
        
        # Calculate indicators
        upper, lower, middle = self._calculate_donchian_channel(
            data, self.channel_period
        )
        atr = self._calculate_atr(data, self.atr_period)
        vol_avg = data['volume'].rolling(self.volume_period).mean()
        
        # Get current and previous values
        current_close = data['close'].iloc[-1]
        current_high = data['high'].iloc[-1]
        current_low = data['low'].iloc[-1]
        current_volume = data['volume'].iloc[-1]
        
        # Previous channel values (excluding current bar)
        prev_upper = data['high'].iloc[-self.channel_period-1:-1].max()
        prev_lower = data['low'].iloc[-self.channel_period-1:-1].min()
        
        atr_val = atr.iloc[-1]
        vol_avg_val = vol_avg.iloc[-1]
        
        # Check for breakout conditions
        buffer = prev_upper * self.breakout_buffer
        
        breakout_up = current_close > (prev_upper + buffer)
        breakout_down = current_close < (prev_lower - buffer)
        
        # Volume confirmation
        volume_confirmed = current_volume > (vol_avg_val * self.volume_threshold)
        
        # Check consolidation
        is_consolidating, range_pct = self._detect_consolidation(
            data.iloc[:-1], self.channel_period
        )
        
        # Calculate signal strength
        if breakout_up:
            strength = min(1.0, (current_close - prev_upper) / atr_val)
        elif breakout_down:
            strength = max(-1.0, (current_close - prev_lower) / atr_val)
        else:
            # Distance from channel midpoint
            dist_from_mid = (current_close - middle.iloc[-1]) / atr_val
            strength = np.clip(dist_from_mid * 0.3, -0.5, 0.5)
        
        # Determine action
        if breakout_up and volume_confirmed:
            action = AgentAction.STRONG_BUY
            confidence = 0.85 if is_consolidating else 0.7
        elif breakout_up:
            action = AgentAction.BUY
            confidence = 0.65 if is_consolidating else 0.55
        elif breakout_down and volume_confirmed:
            action = AgentAction.STRONG_SELL
            confidence = 0.85 if is_consolidating else 0.7
        elif breakout_down:
            action = AgentAction.SELL
            confidence = 0.65 if is_consolidating else 0.55
        else:
            action = AgentAction.HOLD
            confidence = 0.5
        
        return AgentSignal(
            action=action,
            confidence=confidence,
            strength=strength,
            metadata={
                "channel_upper": prev_upper,
                "channel_lower": prev_lower,
                "atr": atr_val,
                "volume_ratio": current_volume / vol_avg_val if vol_avg_val > 0 else 0,
                "breakout_up": breakout_up,
                "breakout_down": breakout_down,
                "volume_confirmed": volume_confirmed,
                "is_consolidating": is_consolidating,
                "range_pct": range_pct,
            }
        )
    
    def get_strategy_params(self) -> Dict[str, Any]:
        """Get breakout strategy parameters."""
        return {
            "channel_period": self.channel_period,
            "volume_period": self.volume_period,
            "volume_threshold": self.volume_threshold,
            "atr_period": self.atr_period,
            "breakout_buffer": self.breakout_buffer,
        }

