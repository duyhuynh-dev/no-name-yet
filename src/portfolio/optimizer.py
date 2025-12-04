"""
Portfolio Optimization

Implements various portfolio optimization methods including
Mean-Variance, Black-Litterman, Risk Parity, and HRP.
"""

import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from scipy.optimize import minimize
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform


class OptimizationMethod(Enum):
    """Portfolio optimization methods."""
    MEAN_VARIANCE = "mean_variance"
    MIN_VARIANCE = "min_variance"
    MAX_SHARPE = "max_sharpe"
    RISK_PARITY = "risk_parity"
    BLACK_LITTERMAN = "black_litterman"
    HRP = "hierarchical_risk_parity"
    EQUAL_WEIGHT = "equal_weight"
    MAX_DIVERSIFICATION = "max_diversification"


@dataclass
class OptimizationResult:
    """Result of portfolio optimization."""
    weights: Dict[str, float]
    method: OptimizationMethod
    
    # Performance metrics
    expected_return: float = 0.0
    expected_volatility: float = 0.0
    sharpe_ratio: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    var_95: float = 0.0
    
    # Diversification
    effective_n: float = 0.0  # Effective number of assets
    concentration: float = 0.0  # HHI concentration
    
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "weights": self.weights,
            "method": self.method.value,
            "expected_return": self.expected_return,
            "expected_volatility": self.expected_volatility,
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "var_95": self.var_95,
            "effective_n": self.effective_n,
            "concentration": self.concentration,
            "timestamp": self.timestamp.isoformat(),
        }


class PortfolioOptimizer:
    """
    Portfolio Optimizer.
    
    Implements multiple optimization methods for portfolio construction.
    """
    
    def __init__(
        self,
        risk_free_rate: float = 0.04,
        max_weight: float = 0.25,
        min_weight: float = 0.0,
        allow_short: bool = False,
    ):
        """
        Initialize Portfolio Optimizer.
        
        Args:
            risk_free_rate: Annual risk-free rate
            max_weight: Maximum weight per asset
            min_weight: Minimum weight per asset
            allow_short: Whether to allow short positions
        """
        self.risk_free_rate = risk_free_rate
        self.max_weight = max_weight
        self.min_weight = min_weight if not allow_short else -max_weight
        self.allow_short = allow_short
    
    def optimize(
        self,
        returns: pd.DataFrame,
        method: OptimizationMethod = OptimizationMethod.MAX_SHARPE,
        views: Optional[Dict[str, float]] = None,
        view_confidences: Optional[Dict[str, float]] = None,
        target_return: Optional[float] = None,
        target_volatility: Optional[float] = None,
    ) -> OptimizationResult:
        """
        Optimize portfolio using specified method.
        
        Args:
            returns: DataFrame of asset returns
            method: Optimization method
            views: Optional market views (for Black-Litterman)
            view_confidences: Confidence in views
            target_return: Target return (for mean-variance)
            target_volatility: Target volatility
            
        Returns:
            OptimizationResult with optimal weights
        """
        if method == OptimizationMethod.MEAN_VARIANCE:
            weights = self._mean_variance(returns, target_return)
        elif method == OptimizationMethod.MIN_VARIANCE:
            weights = self._min_variance(returns)
        elif method == OptimizationMethod.MAX_SHARPE:
            weights = self._max_sharpe(returns)
        elif method == OptimizationMethod.RISK_PARITY:
            weights = self._risk_parity(returns)
        elif method == OptimizationMethod.BLACK_LITTERMAN:
            weights = self._black_litterman(returns, views, view_confidences)
        elif method == OptimizationMethod.HRP:
            weights = self._hrp(returns)
        elif method == OptimizationMethod.EQUAL_WEIGHT:
            weights = self._equal_weight(returns)
        elif method == OptimizationMethod.MAX_DIVERSIFICATION:
            weights = self._max_diversification(returns)
        else:
            weights = self._equal_weight(returns)
        
        # Calculate metrics
        result = self._calculate_metrics(returns, weights, method)
        
        return result
    
    def _mean_variance(
        self,
        returns: pd.DataFrame,
        target_return: Optional[float] = None,
    ) -> Dict[str, float]:
        """Mean-variance optimization (Markowitz)."""
        mu = returns.mean() * 252  # Annualized
        cov = returns.cov() * 252
        n = len(returns.columns)
        
        def objective(w):
            return np.sqrt(w @ cov @ w)
        
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        
        if target_return is not None:
            constraints.append({
                "type": "eq",
                "fun": lambda w: w @ mu - target_return
            })
        
        bounds = [(self.min_weight, self.max_weight) for _ in range(n)]
        
        x0 = np.ones(n) / n
        result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        
        return dict(zip(returns.columns, result.x))
    
    def _min_variance(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Minimum variance portfolio."""
        cov = returns.cov() * 252
        n = len(returns.columns)
        
        def objective(w):
            return w @ cov @ w
        
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(self.min_weight, self.max_weight) for _ in range(n)]
        
        x0 = np.ones(n) / n
        result = minimize(objective, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        
        return dict(zip(returns.columns, result.x))
    
    def _max_sharpe(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Maximum Sharpe ratio portfolio."""
        mu = returns.mean() * 252
        cov = returns.cov() * 252
        n = len(returns.columns)
        rf = self.risk_free_rate
        
        def neg_sharpe(w):
            port_return = w @ mu
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol == 0:
                return 0
            return -(port_return - rf) / port_vol
        
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(self.min_weight, self.max_weight) for _ in range(n)]
        
        x0 = np.ones(n) / n
        result = minimize(neg_sharpe, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        
        return dict(zip(returns.columns, result.x))
    
    def _risk_parity(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Risk parity portfolio (equal risk contribution)."""
        cov = returns.cov() * 252
        n = len(returns.columns)
        
        def risk_budget_objective(w):
            port_vol = np.sqrt(w @ cov @ w)
            if port_vol == 0:
                return 0
            
            # Marginal risk contribution
            mrc = (cov @ w) / port_vol
            # Risk contribution
            rc = w * mrc
            # Target: equal risk contribution
            target_rc = port_vol / n
            
            return np.sum((rc - target_rc) ** 2)
        
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0.01, self.max_weight) for _ in range(n)]  # Must be positive
        
        x0 = np.ones(n) / n
        result = minimize(
            risk_budget_objective, x0, method="SLSQP",
            bounds=bounds, constraints=constraints
        )
        
        return dict(zip(returns.columns, result.x))
    
    def _black_litterman(
        self,
        returns: pd.DataFrame,
        views: Optional[Dict[str, float]] = None,
        confidences: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Black-Litterman model.
        
        Combines market equilibrium with investor views.
        """
        mu = returns.mean() * 252
        cov = returns.cov() * 252
        n = len(returns.columns)
        symbols = returns.columns.tolist()
        
        # Risk aversion coefficient
        delta = (mu.mean() - self.risk_free_rate) / (returns.std().mean() ** 2 * 252)
        delta = max(1.0, min(5.0, delta))  # Clamp to reasonable range
        
        # Market cap weights (use equal weight as proxy)
        market_weights = np.ones(n) / n
        
        # Equilibrium returns
        pi = delta * cov @ market_weights
        
        if views is None or len(views) == 0:
            # No views, use equilibrium
            combined_mu = pi
        else:
            # Build view matrices
            view_symbols = [s for s in views.keys() if s in symbols]
            k = len(view_symbols)
            
            if k == 0:
                combined_mu = pi
            else:
                P = np.zeros((k, n))
                Q = np.zeros(k)
                
                for i, symbol in enumerate(view_symbols):
                    idx = symbols.index(symbol)
                    P[i, idx] = 1
                    Q[i] = views[symbol]
                
                # Uncertainty in views
                tau = 0.05
                
                if confidences:
                    omega_diag = [
                        (1 - confidences.get(s, 0.5)) * 0.1
                        for s in view_symbols
                    ]
                else:
                    omega_diag = [0.05] * k
                
                omega = np.diag(omega_diag)
                
                # Black-Litterman formula
                tau_cov = tau * cov
                
                M1 = np.linalg.inv(tau_cov)
                M2 = P.T @ np.linalg.inv(omega) @ P
                
                combined_mu = np.linalg.inv(M1 + M2) @ (M1 @ pi + P.T @ np.linalg.inv(omega) @ Q)
        
        # Optimize with combined returns
        def neg_utility(w):
            port_return = w @ combined_mu
            port_var = w @ cov @ w
            return -(port_return - 0.5 * delta * port_var)
        
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(self.min_weight, self.max_weight) for _ in range(n)]
        
        x0 = market_weights
        result = minimize(neg_utility, x0, method="SLSQP", bounds=bounds, constraints=constraints)
        
        return dict(zip(symbols, result.x))
    
    def _hrp(self, returns: pd.DataFrame) -> Dict[str, float]:
        """
        Hierarchical Risk Parity.
        
        Uses hierarchical clustering to build diversified portfolios.
        """
        cov = returns.cov()
        corr = returns.corr()
        n = len(returns.columns)
        symbols = returns.columns.tolist()
        
        # Step 1: Tree clustering
        dist = np.sqrt((1 - corr) / 2)
        dist_condensed = squareform(dist.values, checks=False)
        link = linkage(dist_condensed, method="single")
        
        # Step 2: Quasi-diagonalization
        sort_idx = leaves_list(link)
        sorted_symbols = [symbols[i] for i in sort_idx]
        
        # Step 3: Recursive bisection
        def get_cluster_var(cov_matrix, symbols_list):
            """Get variance of cluster using inverse-variance weights."""
            cov_slice = cov_matrix.loc[symbols_list, symbols_list]
            ivp = 1 / np.diag(cov_slice)
            ivp /= ivp.sum()
            return ivp @ cov_slice @ ivp
        
        def recursive_bisection(cov_matrix, sorted_syms):
            weights = pd.Series(1.0, index=sorted_syms)
            clusters = [sorted_syms]
            
            while len(clusters) > 0:
                # Split each cluster
                new_clusters = []
                for cluster in clusters:
                    if len(cluster) > 1:
                        mid = len(cluster) // 2
                        left = cluster[:mid]
                        right = cluster[mid:]
                        
                        # Variance of each half
                        var_left = get_cluster_var(cov_matrix, left)
                        var_right = get_cluster_var(cov_matrix, right)
                        
                        # Allocate inversely to variance
                        alpha = 1 - var_left / (var_left + var_right)
                        
                        weights[left] *= alpha
                        weights[right] *= (1 - alpha)
                        
                        if len(left) > 1:
                            new_clusters.append(left)
                        if len(right) > 1:
                            new_clusters.append(right)
                
                clusters = new_clusters
            
            return weights
        
        weights = recursive_bisection(cov, sorted_symbols)
        
        # Normalize
        weights = weights / weights.sum()
        
        # Apply constraints
        weights = weights.clip(self.min_weight, self.max_weight)
        weights = weights / weights.sum()
        
        return weights.to_dict()
    
    def _equal_weight(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Equal weight portfolio."""
        n = len(returns.columns)
        weight = 1.0 / n
        return {col: weight for col in returns.columns}
    
    def _max_diversification(self, returns: pd.DataFrame) -> Dict[str, float]:
        """Maximum diversification portfolio."""
        cov = returns.cov() * 252
        vol = np.sqrt(np.diag(cov))
        n = len(returns.columns)
        
        def neg_diversification(w):
            port_vol = np.sqrt(w @ cov @ w)
            weighted_vol = w @ vol
            if port_vol == 0:
                return 0
            return -weighted_vol / port_vol
        
        constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]
        bounds = [(0.01, self.max_weight) for _ in range(n)]
        
        x0 = np.ones(n) / n
        result = minimize(
            neg_diversification, x0, method="SLSQP",
            bounds=bounds, constraints=constraints
        )
        
        return dict(zip(returns.columns, result.x))
    
    def _calculate_metrics(
        self,
        returns: pd.DataFrame,
        weights: Dict[str, float],
        method: OptimizationMethod,
    ) -> OptimizationResult:
        """Calculate portfolio metrics."""
        symbols = list(weights.keys())
        w = np.array([weights[s] for s in symbols])
        
        ret = returns[symbols]
        
        # Expected return and volatility (annualized)
        mu = ret.mean() * 252
        cov = ret.cov() * 252
        
        expected_return = w @ mu
        expected_volatility = np.sqrt(w @ cov @ w)
        
        # Sharpe ratio
        sharpe = (expected_return - self.risk_free_rate) / expected_volatility if expected_volatility > 0 else 0
        
        # Portfolio returns for drawdown
        port_returns = (ret * w).sum(axis=1)
        cumulative = (1 + port_returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = abs(drawdown.min())
        
        # VaR 95%
        var_95 = np.percentile(port_returns, 5) * np.sqrt(252)
        
        # Diversification metrics
        effective_n = 1 / (w ** 2).sum()  # Effective number of assets
        concentration = (w ** 2).sum()  # HHI
        
        return OptimizationResult(
            weights=weights,
            method=method,
            expected_return=expected_return,
            expected_volatility=expected_volatility,
            sharpe_ratio=sharpe,
            max_drawdown=max_drawdown,
            var_95=var_95,
            effective_n=effective_n,
            concentration=concentration,
        )
    
    def efficient_frontier(
        self,
        returns: pd.DataFrame,
        n_points: int = 50,
    ) -> List[OptimizationResult]:
        """
        Generate efficient frontier.
        
        Args:
            returns: DataFrame of asset returns
            n_points: Number of points on frontier
            
        Returns:
            List of OptimizationResult for each point
        """
        mu = returns.mean() * 252
        
        min_ret = mu.min()
        max_ret = mu.max()
        
        target_returns = np.linspace(min_ret, max_ret, n_points)
        
        frontier = []
        for target in target_returns:
            try:
                result = self.optimize(
                    returns,
                    method=OptimizationMethod.MEAN_VARIANCE,
                    target_return=target,
                )
                frontier.append(result)
            except Exception:
                continue
        
        return frontier
    
    def compare_methods(
        self,
        returns: pd.DataFrame,
    ) -> Dict[str, OptimizationResult]:
        """
        Compare all optimization methods.
        
        Args:
            returns: DataFrame of asset returns
            
        Returns:
            Dictionary of method to result
        """
        results = {}
        
        for method in OptimizationMethod:
            try:
                result = self.optimize(returns, method=method)
                results[method.value] = result
            except Exception as e:
                print(f"Error with {method.value}: {e}")
        
        return results

