"""
Position Sizing Module

Implements various position sizing algorithms:
- Kelly Criterion
- Fixed Fractional
- Volatility-based
- Risk Parity
"""

from enum import Enum
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd


class SizingMethod(Enum):
    """Position sizing methods."""
    FIXED = "fixed"
    KELLY = "kelly"
    HALF_KELLY = "half_kelly"
    VOLATILITY = "volatility"
    RISK_PARITY = "risk_parity"
    ATR = "atr"


@dataclass
class PositionSize:
    """Position sizing result."""
    symbol: str
    shares: float
    value: float
    weight: float
    method: SizingMethod
    
    # Risk metrics
    risk_per_share: float = 0.0
    max_loss: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "shares": self.shares,
            "value": self.value,
            "weight": self.weight,
            "method": self.method.value,
            "risk_per_share": self.risk_per_share,
            "max_loss": self.max_loss,
        }


class PositionSizer:
    """
    Position Sizing Calculator.
    
    Determines optimal position sizes based on various algorithms
    and risk parameters.
    """
    
    def __init__(
        self,
        portfolio_value: float,
        max_position_pct: float = 0.1,  # Max 10% per position
        max_portfolio_risk_pct: float = 0.02,  # Max 2% portfolio risk per trade
        kelly_fraction: float = 0.5,  # Half Kelly by default
    ):
        """
        Initialize Position Sizer.
        
        Args:
            portfolio_value: Total portfolio value
            max_position_pct: Maximum position size as % of portfolio
            max_portfolio_risk_pct: Maximum risk per trade as % of portfolio
            kelly_fraction: Fraction of Kelly to use (0.5 = Half Kelly)
        """
        self.portfolio_value = portfolio_value
        self.max_position_pct = max_position_pct
        self.max_portfolio_risk_pct = max_portfolio_risk_pct
        self.kelly_fraction = kelly_fraction
    
    def update_portfolio_value(self, value: float) -> None:
        """Update portfolio value."""
        self.portfolio_value = value
    
    def fixed_size(
        self,
        symbol: str,
        price: float,
        fixed_shares: int = 100,
    ) -> PositionSize:
        """
        Fixed share quantity sizing.
        
        Args:
            symbol: Symbol to size
            price: Current price
            fixed_shares: Number of shares to buy
            
        Returns:
            PositionSize result
        """
        value = fixed_shares * price
        weight = value / self.portfolio_value
        
        return PositionSize(
            symbol=symbol,
            shares=fixed_shares,
            value=value,
            weight=weight,
            method=SizingMethod.FIXED,
        )
    
    def kelly_criterion(
        self,
        symbol: str,
        price: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> PositionSize:
        """
        Kelly Criterion position sizing.
        
        f* = (p * b - q) / b
        where:
        - p = probability of winning
        - q = probability of losing (1 - p)
        - b = ratio of avg win to avg loss
        
        Args:
            symbol: Symbol to size
            price: Current price
            win_rate: Historical win rate (0-1)
            avg_win: Average winning trade amount
            avg_loss: Average losing trade amount (positive number)
            
        Returns:
            PositionSize result
        """
        if avg_loss == 0 or win_rate <= 0 or win_rate >= 1:
            # Invalid parameters, return minimum position
            return self._minimum_position(symbol, price)
        
        # Kelly formula
        b = avg_win / avg_loss  # Win/loss ratio
        p = win_rate
        q = 1 - p
        
        kelly_pct = (p * b - q) / b
        
        # Apply Kelly fraction (e.g., Half Kelly)
        kelly_pct *= self.kelly_fraction
        
        # Clamp to max position size
        kelly_pct = max(0, min(kelly_pct, self.max_position_pct))
        
        # Calculate position
        position_value = self.portfolio_value * kelly_pct
        shares = position_value / price
        
        return PositionSize(
            symbol=symbol,
            shares=int(shares),
            value=int(shares) * price,
            weight=kelly_pct,
            method=SizingMethod.KELLY if self.kelly_fraction == 1.0 else SizingMethod.HALF_KELLY,
            risk_per_share=avg_loss / (avg_win + avg_loss) * price,
            max_loss=int(shares) * avg_loss,
        )
    
    def volatility_based(
        self,
        symbol: str,
        price: float,
        volatility: float,
        target_volatility: float = 0.15,  # 15% annualized
    ) -> PositionSize:
        """
        Volatility-based position sizing.
        
        Sizes position to achieve target volatility contribution.
        
        Args:
            symbol: Symbol to size
            price: Current price
            volatility: Asset's annualized volatility
            target_volatility: Target contribution to portfolio volatility
            
        Returns:
            PositionSize result
        """
        if volatility <= 0:
            return self._minimum_position(symbol, price)
        
        # Position weight to achieve target volatility
        weight = target_volatility / volatility
        
        # Clamp to max
        weight = min(weight, self.max_position_pct)
        
        # Calculate position
        position_value = self.portfolio_value * weight
        shares = position_value / price
        
        return PositionSize(
            symbol=symbol,
            shares=int(shares),
            value=int(shares) * price,
            weight=weight,
            method=SizingMethod.VOLATILITY,
            risk_per_share=price * volatility / np.sqrt(252),  # Daily vol
        )
    
    def atr_based(
        self,
        symbol: str,
        price: float,
        atr: float,
        atr_multiplier: float = 2.0,
    ) -> PositionSize:
        """
        ATR-based position sizing.
        
        Sizes position based on Average True Range for stop placement.
        
        Args:
            symbol: Symbol to size
            price: Current price
            atr: Average True Range
            atr_multiplier: ATR multiplier for stop distance
            
        Returns:
            PositionSize result
        """
        if atr <= 0:
            return self._minimum_position(symbol, price)
        
        # Risk per share (stop distance)
        risk_per_share = atr * atr_multiplier
        
        # Maximum dollar risk
        max_risk = self.portfolio_value * self.max_portfolio_risk_pct
        
        # Position size based on risk
        shares = max_risk / risk_per_share
        
        # Clamp to max position size
        max_shares = (self.portfolio_value * self.max_position_pct) / price
        shares = min(shares, max_shares)
        
        position_value = int(shares) * price
        weight = position_value / self.portfolio_value
        
        return PositionSize(
            symbol=symbol,
            shares=int(shares),
            value=position_value,
            weight=weight,
            method=SizingMethod.ATR,
            risk_per_share=risk_per_share,
            max_loss=int(shares) * risk_per_share,
        )
    
    def risk_parity(
        self,
        symbols: List[str],
        prices: Dict[str, float],
        volatilities: Dict[str, float],
    ) -> Dict[str, PositionSize]:
        """
        Risk Parity position sizing.
        
        Allocates equal risk contribution to each asset.
        
        Args:
            symbols: List of symbols to size
            prices: Dictionary of symbol to price
            volatilities: Dictionary of symbol to volatility
            
        Returns:
            Dictionary of symbol to PositionSize
        """
        # Calculate inverse volatility weights
        inv_vols = {s: 1 / volatilities[s] for s in symbols if volatilities.get(s, 0) > 0}
        total_inv_vol = sum(inv_vols.values())
        
        if total_inv_vol == 0:
            return {}
        
        # Normalize weights
        weights = {s: inv_vols[s] / total_inv_vol for s in inv_vols}
        
        # Scale to respect max position constraint
        max_weight = max(weights.values())
        if max_weight > self.max_position_pct:
            scale = self.max_position_pct / max_weight
            weights = {s: w * scale for s, w in weights.items()}
        
        # Calculate positions
        positions = {}
        for symbol in symbols:
            if symbol not in weights:
                continue
            
            weight = weights[symbol]
            price = prices[symbol]
            position_value = self.portfolio_value * weight
            shares = position_value / price
            
            positions[symbol] = PositionSize(
                symbol=symbol,
                shares=int(shares),
                value=int(shares) * price,
                weight=weight,
                method=SizingMethod.RISK_PARITY,
                risk_per_share=price * volatilities[symbol] / np.sqrt(252),
            )
        
        return positions
    
    def calculate_size(
        self,
        symbol: str,
        price: float,
        method: SizingMethod = SizingMethod.FIXED,
        **kwargs
    ) -> PositionSize:
        """
        Calculate position size using specified method.
        
        Args:
            symbol: Symbol to size
            price: Current price
            method: Sizing method to use
            **kwargs: Method-specific parameters
            
        Returns:
            PositionSize result
        """
        if method == SizingMethod.FIXED:
            return self.fixed_size(symbol, price, kwargs.get("fixed_shares", 100))
        elif method in [SizingMethod.KELLY, SizingMethod.HALF_KELLY]:
            return self.kelly_criterion(
                symbol, price,
                kwargs.get("win_rate", 0.5),
                kwargs.get("avg_win", 100),
                kwargs.get("avg_loss", 100),
            )
        elif method == SizingMethod.VOLATILITY:
            return self.volatility_based(
                symbol, price,
                kwargs.get("volatility", 0.2),
                kwargs.get("target_volatility", 0.15),
            )
        elif method == SizingMethod.ATR:
            return self.atr_based(
                symbol, price,
                kwargs.get("atr", price * 0.02),
                kwargs.get("atr_multiplier", 2.0),
            )
        else:
            raise ValueError(f"Unknown sizing method: {method}")
    
    def _minimum_position(self, symbol: str, price: float) -> PositionSize:
        """Return minimum position size."""
        return PositionSize(
            symbol=symbol,
            shares=1,
            value=price,
            weight=price / self.portfolio_value,
            method=SizingMethod.FIXED,
        )
    
    def calculate_max_shares(
        self,
        symbol: str,
        price: float,
        stop_loss_pct: float = 0.02,
    ) -> int:
        """
        Calculate maximum shares given stop loss.
        
        Args:
            symbol: Symbol
            price: Current price
            stop_loss_pct: Stop loss as percentage of price
            
        Returns:
            Maximum number of shares
        """
        # Risk per share
        risk_per_share = price * stop_loss_pct
        
        # Maximum dollar risk
        max_risk = self.portfolio_value * self.max_portfolio_risk_pct
        
        # Maximum shares
        max_shares_risk = max_risk / risk_per_share
        
        # Maximum shares from position limit
        max_shares_position = (self.portfolio_value * self.max_position_pct) / price
        
        return int(min(max_shares_risk, max_shares_position))

