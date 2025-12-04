"""
Portfolio Optimization & Multi-Asset Module

Provides portfolio construction, optimization, and multi-asset
strategy implementations.
"""

from .optimizer import (
    PortfolioOptimizer,
    OptimizationMethod,
    OptimizationResult,
)
from .assets import (
    AssetClass,
    Asset,
    AssetUniverse,
)
from .strategies import (
    PairsTrading,
    StatisticalArbitrage,
    FactorInvesting,
    CrossAssetMomentum,
)
from .rebalancer import (
    Rebalancer,
    RebalanceMethod,
    RebalanceResult,
)

__all__ = [
    "PortfolioOptimizer",
    "OptimizationMethod",
    "OptimizationResult",
    "AssetClass",
    "Asset",
    "AssetUniverse",
    "PairsTrading",
    "StatisticalArbitrage",
    "FactorInvesting",
    "CrossAssetMomentum",
    "Rebalancer",
    "RebalanceMethod",
    "RebalanceResult",
]

