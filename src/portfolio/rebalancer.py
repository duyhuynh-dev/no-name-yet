"""
Portfolio Rebalancing

Implements various rebalancing strategies with
transaction cost optimization.
"""

import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime


class RebalanceMethod(Enum):
    """Rebalancing methods."""
    CALENDAR = "calendar"  # Fixed schedule
    THRESHOLD = "threshold"  # When drift exceeds threshold
    HYBRID = "hybrid"  # Combination
    TAX_AWARE = "tax_aware"  # Consider tax implications
    COST_AWARE = "cost_aware"  # Minimize transaction costs


@dataclass
class RebalanceResult:
    """Result of rebalancing decision."""
    should_rebalance: bool
    trades: Dict[str, float]  # Symbol -> shares to trade (negative = sell)
    current_weights: Dict[str, float]
    target_weights: Dict[str, float]
    drift: float
    estimated_cost: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "should_rebalance": self.should_rebalance,
            "trades": self.trades,
            "current_weights": self.current_weights,
            "target_weights": self.target_weights,
            "drift": self.drift,
            "estimated_cost": self.estimated_cost,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class Position:
    """Current portfolio position."""
    symbol: str
    shares: float
    cost_basis: float
    current_price: float
    
    @property
    def market_value(self) -> float:
        return self.shares * self.current_price
    
    @property
    def unrealized_pnl(self) -> float:
        return (self.current_price - self.cost_basis) * self.shares
    
    @property
    def unrealized_pnl_pct(self) -> float:
        if self.cost_basis == 0:
            return 0
        return (self.current_price / self.cost_basis - 1) * 100


class Rebalancer:
    """
    Portfolio Rebalancer.
    
    Manages portfolio rebalancing with multiple strategies
    and transaction cost optimization.
    """
    
    def __init__(
        self,
        method: RebalanceMethod = RebalanceMethod.THRESHOLD,
        threshold: float = 0.05,  # 5% drift threshold
        min_trade_value: float = 100,  # Minimum trade value
        commission_rate: float = 0.001,  # 0.1% commission
        slippage_rate: float = 0.001,  # 0.1% slippage
        tax_rate_short: float = 0.35,  # Short-term capital gains
        tax_rate_long: float = 0.15,  # Long-term capital gains
    ):
        """
        Initialize Rebalancer.
        
        Args:
            method: Rebalancing method
            threshold: Drift threshold for rebalancing
            min_trade_value: Minimum trade value to execute
            commission_rate: Commission rate per trade
            slippage_rate: Expected slippage rate
            tax_rate_short: Short-term capital gains tax rate
            tax_rate_long: Long-term capital gains tax rate
        """
        self.method = method
        self.threshold = threshold
        self.min_trade_value = min_trade_value
        self.commission_rate = commission_rate
        self.slippage_rate = slippage_rate
        self.tax_rate_short = tax_rate_short
        self.tax_rate_long = tax_rate_long
        
        self._last_rebalance: Optional[datetime] = None
        self._rebalance_history: List[RebalanceResult] = []
    
    def calculate_drift(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
    ) -> float:
        """
        Calculate portfolio drift from target.
        
        Uses sum of absolute weight differences.
        
        Args:
            current_weights: Current portfolio weights
            target_weights: Target portfolio weights
            
        Returns:
            Total drift
        """
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())
        
        drift = 0.0
        for symbol in all_symbols:
            current = current_weights.get(symbol, 0)
            target = target_weights.get(symbol, 0)
            drift += abs(current - target)
        
        return drift / 2  # Divide by 2 since trades net out
    
    def should_rebalance(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        days_since_last: Optional[int] = None,
    ) -> bool:
        """
        Determine if rebalancing is needed.
        
        Args:
            current_weights: Current weights
            target_weights: Target weights
            days_since_last: Days since last rebalance
            
        Returns:
            True if should rebalance
        """
        drift = self.calculate_drift(current_weights, target_weights)
        
        if self.method == RebalanceMethod.CALENDAR:
            # Monthly rebalancing
            return days_since_last is not None and days_since_last >= 21
        
        elif self.method == RebalanceMethod.THRESHOLD:
            return drift >= self.threshold
        
        elif self.method == RebalanceMethod.HYBRID:
            # Rebalance if threshold exceeded OR monthly
            threshold_triggered = drift >= self.threshold
            calendar_triggered = days_since_last is not None and days_since_last >= 21
            return threshold_triggered or calendar_triggered
        
        elif self.method == RebalanceMethod.COST_AWARE:
            # Only rebalance if benefit > cost
            estimated_cost = self._estimate_trade_cost(
                current_weights, target_weights, 100000  # Assume $100k portfolio
            )
            # Rough estimate of benefit (drift reduction)
            benefit = drift * 0.01 * 100000  # 1% of drift * portfolio
            return benefit > estimated_cost * 2
        
        return drift >= self.threshold
    
    def calculate_trades(
        self,
        positions: Dict[str, Position],
        target_weights: Dict[str, float],
        cash: float = 0,
    ) -> Dict[str, float]:
        """
        Calculate required trades to reach target weights.
        
        Args:
            positions: Current positions
            target_weights: Target weights
            cash: Available cash
            
        Returns:
            Dictionary of symbol to shares to trade
        """
        # Calculate portfolio value
        portfolio_value = sum(p.market_value for p in positions.values()) + cash
        
        if portfolio_value <= 0:
            return {}
        
        trades = {}
        
        all_symbols = set(positions.keys()) | set(target_weights.keys())
        
        for symbol in all_symbols:
            current_value = positions[symbol].market_value if symbol in positions else 0
            current_price = positions[symbol].current_price if symbol in positions else 0
            
            target_value = target_weights.get(symbol, 0) * portfolio_value
            trade_value = target_value - current_value
            
            # Skip small trades
            if abs(trade_value) < self.min_trade_value:
                continue
            
            if current_price > 0:
                trades[symbol] = trade_value / current_price
            elif symbol in target_weights and target_weights[symbol] > 0:
                # New position, need price
                trades[symbol] = 0  # Placeholder
        
        return trades
    
    def _estimate_trade_cost(
        self,
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float,
    ) -> float:
        """Estimate total transaction cost."""
        drift = self.calculate_drift(current_weights, target_weights)
        trade_value = drift * portfolio_value
        
        commission = trade_value * self.commission_rate
        slippage = trade_value * self.slippage_rate
        
        return commission + slippage
    
    def optimize_rebalance(
        self,
        positions: Dict[str, Position],
        target_weights: Dict[str, float],
        cash: float = 0,
    ) -> RebalanceResult:
        """
        Optimize rebalancing considering costs.
        
        Args:
            positions: Current positions
            target_weights: Target weights
            cash: Available cash
            
        Returns:
            RebalanceResult with optimal trades
        """
        portfolio_value = sum(p.market_value for p in positions.values()) + cash
        
        # Current weights
        current_weights = {}
        for symbol, pos in positions.items():
            current_weights[symbol] = pos.market_value / portfolio_value if portfolio_value > 0 else 0
        
        if cash > 0 and portfolio_value > 0:
            current_weights["_CASH"] = cash / portfolio_value
        
        drift = self.calculate_drift(current_weights, target_weights)
        should_rebal = self.should_rebalance(current_weights, target_weights)
        
        if not should_rebal:
            return RebalanceResult(
                should_rebalance=False,
                trades={},
                current_weights=current_weights,
                target_weights=target_weights,
                drift=drift,
                estimated_cost=0,
            )
        
        # Calculate optimal trades
        trades = self.calculate_trades(positions, target_weights, cash)
        
        # Estimate cost
        trade_value = sum(
            abs(shares * positions[s].current_price) 
            for s, shares in trades.items() 
            if s in positions
        )
        estimated_cost = trade_value * (self.commission_rate + self.slippage_rate)
        
        result = RebalanceResult(
            should_rebalance=True,
            trades=trades,
            current_weights=current_weights,
            target_weights=target_weights,
            drift=drift,
            estimated_cost=estimated_cost,
        )
        
        self._last_rebalance = datetime.now()
        self._rebalance_history.append(result)
        
        return result
    
    def tax_aware_rebalance(
        self,
        positions: Dict[str, Position],
        target_weights: Dict[str, float],
        holding_periods: Dict[str, int],  # Days held
        cash: float = 0,
    ) -> RebalanceResult:
        """
        Tax-aware rebalancing.
        
        Minimizes tax impact by preferring to sell losses
        and avoiding short-term gains.
        
        Args:
            positions: Current positions
            target_weights: Target weights
            holding_periods: Days each position has been held
            cash: Available cash
            
        Returns:
            RebalanceResult with tax-optimized trades
        """
        portfolio_value = sum(p.market_value for p in positions.values()) + cash
        
        # Calculate current weights
        current_weights = {
            s: p.market_value / portfolio_value
            for s, p in positions.items()
        }
        
        trades = {}
        
        # Categorize positions
        losses = {}
        short_term_gains = {}
        long_term_gains = {}
        
        for symbol, pos in positions.items():
            if pos.unrealized_pnl < 0:
                losses[symbol] = pos.unrealized_pnl
            elif holding_periods.get(symbol, 0) < 365:
                short_term_gains[symbol] = pos.unrealized_pnl
            else:
                long_term_gains[symbol] = pos.unrealized_pnl
        
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())
        
        for symbol in all_symbols:
            current = current_weights.get(symbol, 0)
            target = target_weights.get(symbol, 0)
            diff = target - current
            
            if abs(diff) < 0.01:  # Less than 1% difference
                continue
            
            if diff < 0:  # Need to sell
                # Prefer selling losses for tax harvesting
                if symbol in losses:
                    # Full sell is tax efficient
                    trade_value = diff * portfolio_value
                elif symbol in short_term_gains:
                    # Avoid short-term gains - partial sell only if necessary
                    trade_value = diff * portfolio_value * 0.5  # Half the trade
                else:
                    # Long-term gains are OK
                    trade_value = diff * portfolio_value
                
                if symbol in positions:
                    trades[symbol] = trade_value / positions[symbol].current_price
            else:
                # Buying - no tax impact
                if symbol in positions:
                    trades[symbol] = diff * portfolio_value / positions[symbol].current_price
        
        drift = self.calculate_drift(current_weights, target_weights)
        
        # Estimate tax cost
        tax_cost = 0
        for symbol, shares in trades.items():
            if shares < 0 and symbol in positions:
                pos = positions[symbol]
                gain = pos.unrealized_pnl * (abs(shares) / pos.shares)
                
                if gain > 0:
                    if holding_periods.get(symbol, 0) < 365:
                        tax_cost += gain * self.tax_rate_short
                    else:
                        tax_cost += gain * self.tax_rate_long
        
        trade_value = sum(
            abs(shares * positions[s].current_price)
            for s, shares in trades.items()
            if s in positions
        )
        estimated_cost = (
            trade_value * (self.commission_rate + self.slippage_rate) +
            tax_cost
        )
        
        return RebalanceResult(
            should_rebalance=len(trades) > 0,
            trades=trades,
            current_weights=current_weights,
            target_weights=target_weights,
            drift=drift,
            estimated_cost=estimated_cost,
        )
    
    def get_rebalance_schedule(
        self,
        portfolio_value: float,
        target_weights: Dict[str, float],
        prices: Dict[str, float],
    ) -> List[Dict[str, Any]]:
        """
        Generate a rebalance execution schedule.
        
        Breaks large trades into smaller chunks for
        reduced market impact.
        
        Args:
            portfolio_value: Total portfolio value
            target_weights: Target weights
            prices: Current prices
            
        Returns:
            List of trade chunks with timing
        """
        schedule = []
        
        # Calculate total trade sizes
        trades = {}
        for symbol, weight in target_weights.items():
            if symbol in prices:
                target_value = weight * portfolio_value
                shares = target_value / prices[symbol]
                trades[symbol] = shares
        
        # Split into chunks (TWAP-like)
        num_chunks = 5
        for i in range(num_chunks):
            chunk = {
                "chunk_id": i + 1,
                "delay_minutes": i * 15,  # 15 minutes apart
                "trades": {
                    symbol: shares / num_chunks
                    for symbol, shares in trades.items()
                },
            }
            schedule.append(chunk)
        
        return schedule
    
    def get_history(self, limit: int = 50) -> List[RebalanceResult]:
        """Get rebalancing history."""
        return self._rebalance_history[-limit:]

