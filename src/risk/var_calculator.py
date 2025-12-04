"""
Value at Risk (VaR) Calculator

Implements multiple VaR calculation methods:
- Historical Simulation
- Parametric (Variance-Covariance)
- Monte Carlo Simulation
"""

from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy import stats


class VaRMethod(Enum):
    """VaR calculation methods."""
    HISTORICAL = "historical"
    PARAMETRIC = "parametric"
    MONTE_CARLO = "monte_carlo"


@dataclass
class VaRResult:
    """VaR calculation result."""
    var: float
    cvar: float  # Conditional VaR (Expected Shortfall)
    confidence_level: float
    method: VaRMethod
    time_horizon_days: int
    portfolio_value: float
    
    # Additional metrics
    var_pct: float = 0.0
    cvar_pct: float = 0.0
    
    def __post_init__(self):
        if self.portfolio_value > 0:
            self.var_pct = (self.var / self.portfolio_value) * 100
            self.cvar_pct = (self.cvar / self.portfolio_value) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "var": self.var,
            "var_pct": self.var_pct,
            "cvar": self.cvar,
            "cvar_pct": self.cvar_pct,
            "confidence_level": self.confidence_level,
            "method": self.method.value,
            "time_horizon_days": self.time_horizon_days,
            "portfolio_value": self.portfolio_value,
        }


class VaRCalculator:
    """
    Value at Risk Calculator.
    
    Calculates VaR and CVaR using multiple methods for portfolios
    and individual positions.
    """
    
    def __init__(
        self,
        confidence_level: float = 0.95,
        time_horizon_days: int = 1,
        monte_carlo_simulations: int = 10000,
    ):
        """
        Initialize VaR Calculator.
        
        Args:
            confidence_level: Confidence level (e.g., 0.95 for 95%)
            time_horizon_days: Risk horizon in days
            monte_carlo_simulations: Number of MC simulations
        """
        self.confidence_level = confidence_level
        self.time_horizon_days = time_horizon_days
        self.mc_simulations = monte_carlo_simulations
    
    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """Calculate log returns from price series."""
        return np.log(prices / prices.shift(1)).dropna()
    
    def historical_var(
        self,
        returns: pd.Series,
        portfolio_value: float,
    ) -> VaRResult:
        """
        Calculate VaR using Historical Simulation.
        
        Uses actual historical returns to estimate potential losses.
        
        Args:
            returns: Historical returns series
            portfolio_value: Current portfolio value
            
        Returns:
            VaRResult with calculated VaR and CVaR
        """
        # Scale returns for time horizon
        scaled_returns = returns * np.sqrt(self.time_horizon_days)
        
        # VaR is the percentile of losses
        var_percentile = 1 - self.confidence_level
        var_return = np.percentile(scaled_returns, var_percentile * 100)
        
        # VaR as positive number (potential loss)
        var = abs(var_return * portfolio_value)
        
        # CVaR (Expected Shortfall) - average of losses beyond VaR
        tail_returns = scaled_returns[scaled_returns <= var_return]
        if len(tail_returns) > 0:
            cvar_return = tail_returns.mean()
            cvar = abs(cvar_return * portfolio_value)
        else:
            cvar = var
        
        return VaRResult(
            var=var,
            cvar=cvar,
            confidence_level=self.confidence_level,
            method=VaRMethod.HISTORICAL,
            time_horizon_days=self.time_horizon_days,
            portfolio_value=portfolio_value,
        )
    
    def parametric_var(
        self,
        returns: pd.Series,
        portfolio_value: float,
    ) -> VaRResult:
        """
        Calculate VaR using Parametric (Variance-Covariance) method.
        
        Assumes returns are normally distributed.
        
        Args:
            returns: Historical returns series
            portfolio_value: Current portfolio value
            
        Returns:
            VaRResult with calculated VaR and CVaR
        """
        # Calculate mean and standard deviation
        mu = returns.mean() * self.time_horizon_days
        sigma = returns.std() * np.sqrt(self.time_horizon_days)
        
        # Z-score for confidence level
        z_score = stats.norm.ppf(1 - self.confidence_level)
        
        # VaR calculation
        var_return = mu + z_score * sigma
        var = abs(var_return * portfolio_value)
        
        # CVaR for normal distribution
        # E[X | X < VaR] = mu - sigma * phi(z) / (1-alpha)
        phi = stats.norm.pdf(z_score)
        cvar_return = mu - sigma * phi / (1 - self.confidence_level)
        cvar = abs(cvar_return * portfolio_value)
        
        return VaRResult(
            var=var,
            cvar=cvar,
            confidence_level=self.confidence_level,
            method=VaRMethod.PARAMETRIC,
            time_horizon_days=self.time_horizon_days,
            portfolio_value=portfolio_value,
        )
    
    def monte_carlo_var(
        self,
        returns: pd.Series,
        portfolio_value: float,
    ) -> VaRResult:
        """
        Calculate VaR using Monte Carlo Simulation.
        
        Simulates many possible future scenarios.
        
        Args:
            returns: Historical returns series
            portfolio_value: Current portfolio value
            
        Returns:
            VaRResult with calculated VaR and CVaR
        """
        # Estimate return parameters
        mu = returns.mean()
        sigma = returns.std()
        
        # Generate simulated returns
        np.random.seed(42)  # For reproducibility
        simulated_returns = np.random.normal(
            mu * self.time_horizon_days,
            sigma * np.sqrt(self.time_horizon_days),
            self.mc_simulations
        )
        
        # Calculate portfolio values
        simulated_values = portfolio_value * (1 + simulated_returns)
        simulated_pnl = simulated_values - portfolio_value
        
        # VaR from simulations
        var_percentile = 1 - self.confidence_level
        var = abs(np.percentile(simulated_pnl, var_percentile * 100))
        
        # CVaR
        tail_pnl = simulated_pnl[simulated_pnl <= -var]
        if len(tail_pnl) > 0:
            cvar = abs(tail_pnl.mean())
        else:
            cvar = var
        
        return VaRResult(
            var=var,
            cvar=cvar,
            confidence_level=self.confidence_level,
            method=VaRMethod.MONTE_CARLO,
            time_horizon_days=self.time_horizon_days,
            portfolio_value=portfolio_value,
        )
    
    def calculate_var(
        self,
        returns: pd.Series,
        portfolio_value: float,
        method: VaRMethod = VaRMethod.HISTORICAL,
    ) -> VaRResult:
        """
        Calculate VaR using specified method.
        
        Args:
            returns: Historical returns series
            portfolio_value: Current portfolio value
            method: VaR calculation method
            
        Returns:
            VaRResult with calculated VaR and CVaR
        """
        if method == VaRMethod.HISTORICAL:
            return self.historical_var(returns, portfolio_value)
        elif method == VaRMethod.PARAMETRIC:
            return self.parametric_var(returns, portfolio_value)
        elif method == VaRMethod.MONTE_CARLO:
            return self.monte_carlo_var(returns, portfolio_value)
        else:
            raise ValueError(f"Unknown VaR method: {method}")
    
    def calculate_all_methods(
        self,
        returns: pd.Series,
        portfolio_value: float,
    ) -> Dict[str, VaRResult]:
        """Calculate VaR using all methods for comparison."""
        return {
            "historical": self.historical_var(returns, portfolio_value),
            "parametric": self.parametric_var(returns, portfolio_value),
            "monte_carlo": self.monte_carlo_var(returns, portfolio_value),
        }
    
    def portfolio_var(
        self,
        positions: Dict[str, float],  # symbol -> value
        returns_df: pd.DataFrame,  # columns = symbols, rows = returns
        method: VaRMethod = VaRMethod.HISTORICAL,
    ) -> VaRResult:
        """
        Calculate portfolio VaR considering correlations.
        
        Args:
            positions: Dictionary of symbol to position value
            returns_df: DataFrame with returns for each symbol
            method: VaR calculation method
            
        Returns:
            VaRResult for the portfolio
        """
        # Calculate weights
        total_value = sum(positions.values())
        weights = np.array([positions.get(col, 0) / total_value 
                          for col in returns_df.columns])
        
        # Portfolio returns
        portfolio_returns = (returns_df * weights).sum(axis=1)
        
        return self.calculate_var(portfolio_returns, total_value, method)
    
    def calculate_component_var(
        self,
        positions: Dict[str, float],
        returns_df: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Calculate component VaR (contribution of each position to total VaR).
        
        Args:
            positions: Dictionary of symbol to position value
            returns_df: DataFrame with returns for each symbol
            
        Returns:
            Dictionary of symbol to component VaR
        """
        total_value = sum(positions.values())
        symbols = list(positions.keys())
        
        # Covariance matrix
        cov_matrix = returns_df[symbols].cov() * self.time_horizon_days
        
        # Weights
        weights = np.array([positions[s] / total_value for s in symbols])
        
        # Portfolio variance
        port_variance = np.dot(weights.T, np.dot(cov_matrix, weights))
        port_std = np.sqrt(port_variance)
        
        # Z-score
        z_score = stats.norm.ppf(self.confidence_level)
        
        # Marginal VaR
        marginal_var = z_score * np.dot(cov_matrix, weights) / port_std
        
        # Component VaR
        component_var = {}
        for i, symbol in enumerate(symbols):
            component_var[symbol] = weights[i] * marginal_var[i] * total_value
        
        return component_var
    
    def calculate_incremental_var(
        self,
        current_positions: Dict[str, float],
        new_position: Tuple[str, float],
        returns_df: pd.DataFrame,
    ) -> float:
        """
        Calculate incremental VaR when adding a new position.
        
        Args:
            current_positions: Current portfolio positions
            new_position: Tuple of (symbol, value) to add
            returns_df: DataFrame with returns for each symbol
            
        Returns:
            Incremental VaR (change in portfolio VaR)
        """
        # Current VaR
        current_var = self.portfolio_var(
            current_positions, returns_df
        ).var
        
        # New positions
        new_positions = current_positions.copy()
        symbol, value = new_position
        new_positions[symbol] = new_positions.get(symbol, 0) + value
        
        # New VaR
        new_var = self.portfolio_var(new_positions, returns_df).var
        
        return new_var - current_var

