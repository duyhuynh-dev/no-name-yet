"""
Market Maker Trading Agent

Implements market making strategies that profit from bid-ask spread.
Provides liquidity by placing both buy and sell orders around mid-price.
"""

from typing import Dict, Any
import numpy as np
import pandas as pd

from .base import BaseAgent, AgentSignal, AgentAction


class MarketMakerAgent(BaseAgent):
    """
    Market making agent.
    
    Strategies for capturing spread:
    - Inventory management
    - Volatility-adjusted spread
    - Order book imbalance (simulated)
    """
    
    def __init__(
        self,
        name: str = "market_maker_agent",
        spread_period: int = 20,
        volatility_period: int = 14,
        inventory_limit: float = 0.3,
        base_spread_bps: float = 10.0,
        volatility_mult: float = 2.0,
        **kwargs
    ):
        """
        Initialize the Market Maker Agent.
        
        Args:
            name: Agent identifier
            spread_period: Period for spread calculation
            volatility_period: Period for volatility calculation
            inventory_limit: Max inventory as fraction of capital
            base_spread_bps: Base spread in basis points
            volatility_mult: Multiplier for volatility adjustment
        """
        super().__init__(
            name=name,
            description="Market making with inventory management",
            lookback_period=max(spread_period, volatility_period) + 10,
            **kwargs
        )
        self.spread_period = spread_period
        self.volatility_period = volatility_period
        self.inventory_limit = inventory_limit
        self.base_spread_bps = base_spread_bps
        self.volatility_mult = volatility_mult
        self._inventory = 0.0  # Track inventory position
    
    def _calculate_realized_volatility(
        self, prices: pd.Series, period: int
    ) -> pd.Series:
        """Calculate realized volatility from returns."""
        returns = prices.pct_change()
        return returns.rolling(period).std() * np.sqrt(252)  # Annualized
    
    def _calculate_mid_price(self, data: pd.DataFrame) -> pd.Series:
        """Calculate mid price (average of high and low)."""
        return (data['high'] + data['low']) / 2
    
    def _estimate_order_imbalance(self, data: pd.DataFrame) -> float:
        """
        Estimate order book imbalance from price action.
        
        Positive = more buying pressure
        Negative = more selling pressure
        """
        recent = data.tail(5)
        
        # Use close vs open and volume to estimate imbalance
        price_pressure = (recent['close'] - recent['open']).sum()
        volume_weighted = (
            (recent['close'] - recent['open']) * recent['volume']
        ).sum()
        
        avg_volume = recent['volume'].mean()
        if avg_volume > 0:
            imbalance = volume_weighted / (avg_volume * recent['close'].mean())
        else:
            imbalance = 0
        
        return np.clip(imbalance, -1, 1)
    
    def _calculate_optimal_spread(
        self, volatility: float, inventory: float
    ) -> float:
        """
        Calculate optimal spread based on volatility and inventory.
        
        Higher volatility = wider spread
        Higher inventory = skew quotes to reduce position
        """
        # Base spread adjusted for volatility
        vol_adjustment = self.volatility_mult * volatility * 100  # Convert to bps
        spread_bps = self.base_spread_bps + vol_adjustment
        
        # Inventory adjustment (skew)
        inventory_skew = inventory * 5  # bps per unit of inventory
        
        return spread_bps, inventory_skew
    
    def generate_signal(
        self,
        data: pd.DataFrame,
        position: int = 0,
        **kwargs
    ) -> AgentSignal:
        """
        Generate market making signal.
        
        Args:
            data: OHLCV DataFrame
            position: Current position (-1, 0, 1)
            
        Returns:
            AgentSignal with market making recommendation
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
        volatility = self._calculate_realized_volatility(
            close, self.volatility_period
        )
        mid_price = self._calculate_mid_price(data)
        imbalance = self._estimate_order_imbalance(data)
        
        # Get current values
        current_price = close.iloc[-1]
        current_mid = mid_price.iloc[-1]
        current_vol = volatility.iloc[-1]
        
        # Update inventory tracking
        self._inventory = position * 0.5  # Normalize position
        
        # Calculate optimal spread
        spread_bps, inventory_skew = self._calculate_optimal_spread(
            current_vol, self._inventory
        )
        
        # Calculate fair value (mid price adjusted for imbalance)
        fair_value = current_mid * (1 + imbalance * 0.001)
        
        # Determine bid/ask levels
        half_spread = spread_bps / 10000 / 2
        bid_price = fair_value * (1 - half_spread)
        ask_price = fair_value * (1 + half_spread)
        
        # Apply inventory skew
        skew_factor = inventory_skew / 10000
        bid_price *= (1 - skew_factor)
        ask_price *= (1 - skew_factor)
        
        # Determine action based on current price relative to our quotes
        price_vs_fair = (current_price - fair_value) / fair_value
        
        # Check inventory limits
        inventory_ok = abs(self._inventory) < self.inventory_limit
        
        # Signal strength based on distance from fair value
        strength = np.clip(-price_vs_fair * 10, -1, 1)
        
        # Market making logic
        if current_price < bid_price and inventory_ok:
            # Price below our bid - good to buy
            action = AgentAction.BUY
            confidence = 0.7 + min(0.2, (bid_price - current_price) / current_price * 100)
        elif current_price > ask_price and inventory_ok:
            # Price above our ask - good to sell
            action = AgentAction.SELL
            confidence = 0.7 + min(0.2, (current_price - ask_price) / current_price * 100)
        elif self._inventory > self.inventory_limit * 0.8:
            # Too much long inventory - reduce
            action = AgentAction.SELL
            confidence = 0.6
        elif self._inventory < -self.inventory_limit * 0.8:
            # Too much short inventory - reduce
            action = AgentAction.BUY
            confidence = 0.6
        else:
            action = AgentAction.HOLD
            confidence = 0.5
        
        return AgentSignal(
            action=action,
            confidence=min(confidence, 0.9),
            strength=strength,
            metadata={
                "fair_value": fair_value,
                "bid_price": bid_price,
                "ask_price": ask_price,
                "spread_bps": spread_bps,
                "volatility": current_vol,
                "imbalance": imbalance,
                "inventory": self._inventory,
                "inventory_skew": inventory_skew,
            }
        )
    
    def reset(self) -> None:
        """Reset agent state including inventory."""
        super().reset()
        self._inventory = 0.0
    
    def get_strategy_params(self) -> Dict[str, Any]:
        """Get market maker strategy parameters."""
        return {
            "spread_period": self.spread_period,
            "volatility_period": self.volatility_period,
            "inventory_limit": self.inventory_limit,
            "base_spread_bps": self.base_spread_bps,
            "volatility_mult": self.volatility_mult,
        }

