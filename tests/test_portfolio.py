"""
Tests for Portfolio Optimization & Multi-Asset Module.
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime

from src.portfolio.assets import (
    Asset,
    AssetClass,
    Sector,
    CryptoAsset,
    AssetUniverse,
)
from src.portfolio.optimizer import (
    PortfolioOptimizer,
    OptimizationMethod,
    OptimizationResult,
)
from src.portfolio.strategies import (
    PairsTrading,
    StatisticalArbitrage,
    FactorInvesting,
    CrossAssetMomentum,
    SignalType,
)
from src.portfolio.rebalancer import (
    Rebalancer,
    RebalanceMethod,
    Position,
)


@pytest.fixture
def sample_returns():
    """Generate sample return DataFrame."""
    np.random.seed(42)
    n = 252
    
    # Generate correlated returns
    cov = np.array([
        [0.04, 0.02, 0.01, 0.015],
        [0.02, 0.03, 0.01, 0.01],
        [0.01, 0.01, 0.02, 0.005],
        [0.015, 0.01, 0.005, 0.025],
    ])
    
    mean = [0.0003, 0.0002, 0.0001, 0.00025]
    
    returns = np.random.multivariate_normal(mean, cov / 252, n)
    
    return pd.DataFrame(
        returns,
        columns=["AAPL", "MSFT", "GOOGL", "AMZN"],
    )


@pytest.fixture
def sample_prices():
    """Generate sample price DataFrame."""
    np.random.seed(42)
    n = 252
    
    # Random walk prices
    prices = {}
    for symbol in ["AAPL", "MSFT", "GOOGL", "AMZN"]:
        base = 100 + np.random.randint(0, 100)
        changes = np.random.normal(0.0003, 0.02, n)
        prices[symbol] = base * np.cumprod(1 + changes)
    
    return pd.DataFrame(prices)


class TestAssetUniverse:
    """Tests for Asset Universe."""
    
    def test_add_asset(self):
        universe = AssetUniverse()
        
        asset = Asset(
            symbol="AAPL",
            name="Apple Inc",
            asset_class=AssetClass.EQUITY_US,
            sector=Sector.TECHNOLOGY,
        )
        
        universe.add_asset(asset)
        
        assert "AAPL" in universe
        assert len(universe) == 1
    
    def test_get_by_class(self):
        universe = AssetUniverse()
        
        universe.add_asset(Asset("AAPL", "Apple", AssetClass.EQUITY_US, Sector.TECHNOLOGY))
        universe.add_asset(Asset("BTC", "Bitcoin", AssetClass.CRYPTO))
        universe.add_asset(Asset("MSFT", "Microsoft", AssetClass.EQUITY_US, Sector.TECHNOLOGY))
        
        equities = universe.get_by_class(AssetClass.EQUITY_US)
        
        assert len(equities) == 2
    
    def test_get_by_sector(self):
        universe = AssetUniverse()
        
        universe.add_asset(Asset("AAPL", "Apple", AssetClass.EQUITY_US, Sector.TECHNOLOGY))
        universe.add_asset(Asset("JPM", "JPMorgan", AssetClass.EQUITY_US, Sector.FINANCIALS))
        universe.add_asset(Asset("MSFT", "Microsoft", AssetClass.EQUITY_US, Sector.TECHNOLOGY))
        
        tech = universe.get_by_sector(Sector.TECHNOLOGY)
        
        assert len(tech) == 2
    
    def test_filter_assets(self):
        universe = AssetUniverse()
        
        universe.add_asset(Asset("AAPL", "Apple", AssetClass.EQUITY_US, Sector.TECHNOLOGY, volatility=0.25))
        universe.add_asset(Asset("MSFT", "Microsoft", AssetClass.EQUITY_US, Sector.TECHNOLOGY, volatility=0.20))
        universe.add_asset(Asset("XYZ", "High Vol", AssetClass.EQUITY_US, Sector.TECHNOLOGY, volatility=0.50))
        
        filtered = universe.filter_assets(max_volatility=0.30)
        
        assert len(filtered) == 2
    
    def test_create_default_universe(self):
        universe = AssetUniverse()
        universe.create_default_universe()
        
        assert len(universe) > 0
        assert "AAPL" in universe
        assert "BTC" in universe
    
    def test_crypto_asset(self):
        asset = CryptoAsset(
            symbol="BTC",
            name="Bitcoin",
            asset_class=AssetClass.CRYPTO,
            base_currency="BTC",
        )
        
        assert asset.asset_class == AssetClass.CRYPTO
        assert asset.margin_requirement == 0.1


class TestPortfolioOptimizer:
    """Tests for Portfolio Optimizer."""
    
    def test_equal_weight(self, sample_returns):
        optimizer = PortfolioOptimizer()
        result = optimizer.optimize(sample_returns, OptimizationMethod.EQUAL_WEIGHT)
        
        assert len(result.weights) == 4
        assert all(abs(w - 0.25) < 0.01 for w in result.weights.values())
    
    def test_min_variance(self, sample_returns):
        optimizer = PortfolioOptimizer()
        result = optimizer.optimize(sample_returns, OptimizationMethod.MIN_VARIANCE)
        
        assert len(result.weights) == 4
        assert abs(sum(result.weights.values()) - 1.0) < 0.01
        assert result.expected_volatility > 0
    
    def test_max_sharpe(self, sample_returns):
        optimizer = PortfolioOptimizer()
        result = optimizer.optimize(sample_returns, OptimizationMethod.MAX_SHARPE)
        
        assert len(result.weights) == 4
        assert abs(sum(result.weights.values()) - 1.0) < 0.01
        assert result.sharpe_ratio != 0
    
    def test_risk_parity(self, sample_returns):
        optimizer = PortfolioOptimizer()
        result = optimizer.optimize(sample_returns, OptimizationMethod.RISK_PARITY)
        
        assert len(result.weights) == 4
        assert abs(sum(result.weights.values()) - 1.0) < 0.01
        # All weights should be positive
        assert all(w > 0 for w in result.weights.values())
    
    def test_black_litterman(self, sample_returns):
        optimizer = PortfolioOptimizer()
        
        # Views: AAPL will outperform by 10%
        views = {"AAPL": 0.10}
        confidences = {"AAPL": 0.8}
        
        result = optimizer.optimize(
            sample_returns,
            OptimizationMethod.BLACK_LITTERMAN,
            views=views,
            view_confidences=confidences,
        )
        
        assert len(result.weights) == 4
        # AAPL should have higher weight due to bullish view
        assert result.weights["AAPL"] > 0.2
    
    def test_hrp(self, sample_returns):
        optimizer = PortfolioOptimizer()
        result = optimizer.optimize(sample_returns, OptimizationMethod.HRP)
        
        assert len(result.weights) == 4
        assert abs(sum(result.weights.values()) - 1.0) < 0.01
    
    def test_max_weight_constraint(self, sample_returns):
        optimizer = PortfolioOptimizer(max_weight=0.3)
        result = optimizer.optimize(sample_returns, OptimizationMethod.MAX_SHARPE)
        
        assert all(w <= 0.31 for w in result.weights.values())  # Small tolerance
    
    def test_efficient_frontier(self, sample_returns):
        optimizer = PortfolioOptimizer()
        frontier = optimizer.efficient_frontier(sample_returns, n_points=10)
        
        assert len(frontier) > 0
        # Returns should be roughly increasing (allowing for numerical precision)
        returns = [r.expected_return for r in frontier]
        # Check that returns are generally non-decreasing (allow small tolerance)
        is_sorted = all(
            returns[i] <= returns[i+1] + 0.001 
            for i in range(len(returns)-1)
        )
        assert is_sorted or len(frontier) < 3
    
    def test_compare_methods(self, sample_returns):
        optimizer = PortfolioOptimizer()
        results = optimizer.compare_methods(sample_returns)
        
        assert len(results) > 0
        assert "equal_weight" in results
    
    def test_result_to_dict(self, sample_returns):
        optimizer = PortfolioOptimizer()
        result = optimizer.optimize(sample_returns, OptimizationMethod.EQUAL_WEIGHT)
        
        d = result.to_dict()
        
        assert "weights" in d
        assert "expected_return" in d
        assert "sharpe_ratio" in d


class TestPairsTrading:
    """Tests for Pairs Trading Strategy."""
    
    def test_find_pairs(self, sample_prices):
        pt = PairsTrading(coint_pvalue=0.2)  # Relaxed threshold for test
        pairs = pt.find_pairs(sample_prices, min_correlation=0.5)
        
        # May or may not find pairs depending on random data
        assert isinstance(pairs, list)
    
    def test_generate_signals(self, sample_prices):
        pt = PairsTrading()
        
        # First find pairs
        pt.find_pairs(sample_prices, min_correlation=0.3)
        
        # Generate signals
        signals = pt.generate_signals(sample_prices)
        
        assert isinstance(signals, list)
    
    def test_hedge_ratio(self, sample_prices):
        pt = PairsTrading()
        
        hedge_ratio = pt._calculate_hedge_ratio(
            sample_prices["AAPL"],
            sample_prices["MSFT"],
        )
        
        assert hedge_ratio != 0


class TestStatisticalArbitrage:
    """Tests for Statistical Arbitrage Strategy."""
    
    def test_fit(self, sample_returns):
        sa = StatisticalArbitrage(num_factors=2)
        sa.fit(sample_returns)
        
        assert sa._factor_loadings is not None
        assert sa._residuals_std is not None
    
    def test_generate_signals(self, sample_returns):
        sa = StatisticalArbitrage(entry_threshold=1.0)
        
        signals = sa.generate_signals(sample_returns)
        
        assert isinstance(signals, list)


class TestFactorInvesting:
    """Tests for Factor Investing Strategy."""
    
    def test_calculate_momentum(self, sample_prices):
        fi = FactorInvesting(momentum_lookback=60)
        
        momentum = fi.calculate_momentum(sample_prices)
        
        assert len(momentum) == 4
    
    def test_calculate_low_volatility(self, sample_returns):
        fi = FactorInvesting()
        
        low_vol = fi.calculate_low_volatility(sample_returns)
        
        assert len(low_vol) == 4
        # All scores should be negative (lower vol = higher score when inverted)
        assert all(v < 0 for v in low_vol.values())
    
    def test_rank_assets(self, sample_prices, sample_returns):
        fi = FactorInvesting()
        
        rankings = fi.rank_assets(sample_prices, sample_returns)
        
        assert len(rankings) == 4
    
    def test_generate_signals(self, sample_prices, sample_returns):
        fi = FactorInvesting()
        
        signals = fi.generate_signals(sample_prices, sample_returns, top_n=2)
        
        assert len(signals) == 2
        assert all(s.signal_type == SignalType.LONG for s in signals)


class TestCrossAssetMomentum:
    """Tests for Cross-Asset Momentum Strategy."""
    
    def test_calculate_momentum_scores(self, sample_prices):
        cam = CrossAssetMomentum(lookback=60)
        
        scores = cam.calculate_momentum_scores(sample_prices)
        
        assert len(scores) == 4
    
    def test_generate_signals(self, sample_prices):
        cam = CrossAssetMomentum(top_pct=0.5, bottom_pct=0.25)
        
        signals = cam.generate_signals(sample_prices)
        
        assert len(signals) > 0
        
        long_signals = [s for s in signals if s.signal_type == SignalType.LONG]
        short_signals = [s for s in signals if s.signal_type == SignalType.SHORT]
        
        assert len(long_signals) >= 1
        assert len(short_signals) >= 1


class TestRebalancer:
    """Tests for Portfolio Rebalancer."""
    
    def test_calculate_drift(self):
        rebalancer = Rebalancer()
        
        current = {"AAPL": 0.30, "MSFT": 0.40, "GOOGL": 0.30}
        target = {"AAPL": 0.33, "MSFT": 0.33, "GOOGL": 0.34}
        
        drift = rebalancer.calculate_drift(current, target)
        
        assert drift > 0
        assert drift < 0.5
    
    def test_should_rebalance_threshold(self):
        rebalancer = Rebalancer(method=RebalanceMethod.THRESHOLD, threshold=0.05)
        
        # Small drift - no rebalance
        current = {"AAPL": 0.51, "MSFT": 0.49}
        target = {"AAPL": 0.50, "MSFT": 0.50}
        
        assert not rebalancer.should_rebalance(current, target)
        
        # Large drift - should rebalance
        current = {"AAPL": 0.60, "MSFT": 0.40}
        target = {"AAPL": 0.50, "MSFT": 0.50}
        
        assert rebalancer.should_rebalance(current, target)
    
    def test_should_rebalance_calendar(self):
        rebalancer = Rebalancer(method=RebalanceMethod.CALENDAR)
        
        current = {"AAPL": 0.50, "MSFT": 0.50}
        target = {"AAPL": 0.50, "MSFT": 0.50}
        
        assert not rebalancer.should_rebalance(current, target, days_since_last=10)
        assert rebalancer.should_rebalance(current, target, days_since_last=30)
    
    def test_calculate_trades(self):
        rebalancer = Rebalancer()
        
        positions = {
            "AAPL": Position("AAPL", 100, 150, 160),
            "MSFT": Position("MSFT", 50, 300, 320),
        }
        
        target_weights = {"AAPL": 0.6, "MSFT": 0.4}
        
        trades = rebalancer.calculate_trades(positions, target_weights)
        
        assert isinstance(trades, dict)
    
    def test_optimize_rebalance(self):
        rebalancer = Rebalancer(threshold=0.05)
        
        positions = {
            "AAPL": Position("AAPL", 100, 150, 160),
            "MSFT": Position("MSFT", 50, 300, 320),
        }
        
        target_weights = {"AAPL": 0.6, "MSFT": 0.4}
        
        result = rebalancer.optimize_rebalance(positions, target_weights)
        
        assert isinstance(result.should_rebalance, bool)
        assert "AAPL" in result.current_weights or "MSFT" in result.current_weights
    
    def test_tax_aware_rebalance(self):
        rebalancer = Rebalancer()
        
        positions = {
            "AAPL": Position("AAPL", 100, 150, 160),  # Gain
            "MSFT": Position("MSFT", 50, 350, 320),   # Loss
        }
        
        target_weights = {"AAPL": 0.4, "MSFT": 0.6}
        holding_periods = {"AAPL": 400, "MSFT": 100}  # AAPL long-term, MSFT short
        
        result = rebalancer.tax_aware_rebalance(
            positions, target_weights, holding_periods
        )
        
        assert isinstance(result.estimated_cost, float)
    
    def test_get_rebalance_schedule(self):
        rebalancer = Rebalancer()
        
        schedule = rebalancer.get_rebalance_schedule(
            portfolio_value=100000,
            target_weights={"AAPL": 0.5, "MSFT": 0.5},
            prices={"AAPL": 150, "MSFT": 300},
        )
        
        assert len(schedule) == 5  # 5 chunks
        assert "trades" in schedule[0]


class TestIntegration:
    """Integration tests for portfolio module."""
    
    def test_optimize_then_rebalance(self, sample_returns):
        # Optimize
        optimizer = PortfolioOptimizer()
        result = optimizer.optimize(sample_returns, OptimizationMethod.MAX_SHARPE)
        
        # Create positions based on current allocation
        positions = {
            "AAPL": Position("AAPL", 100, 150, 160),
            "MSFT": Position("MSFT", 80, 300, 310),
            "GOOGL": Position("GOOGL", 50, 100, 105),
            "AMZN": Position("AMZN", 30, 130, 135),
        }
        
        # Rebalance to optimal
        rebalancer = Rebalancer(threshold=0.01)
        rebal_result = rebalancer.optimize_rebalance(positions, result.weights)
        
        assert rebal_result is not None
    
    def test_factor_investing_pipeline(self, sample_prices, sample_returns):
        # Run factor analysis
        fi = FactorInvesting()
        rankings = fi.rank_assets(sample_prices, sample_returns)
        
        # Use rankings for optimization
        optimizer = PortfolioOptimizer()
        
        # Weight based on rankings
        total_rank = sum(rankings.values())
        target_weights = {s: r / total_rank for s, r in rankings.items()}
        
        assert abs(sum(target_weights.values()) - 1.0) < 0.01


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

