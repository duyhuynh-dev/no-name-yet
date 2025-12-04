"""
Advanced Trading Strategies

Implements pairs trading, statistical arbitrage,
factor investing, and cross-asset momentum.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from scipy import stats
from statsmodels.tsa.stattools import coint, adfuller


class SignalType(Enum):
    """Trading signal types."""
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"
    CLOSE = "close"


@dataclass
class TradingSignal:
    """Trading signal from a strategy."""
    symbol: str
    signal_type: SignalType
    strength: float  # -1 to 1
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "signal": self.signal_type.value,
            "strength": self.strength,
            "entry_price": self.entry_price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "metadata": self.metadata,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class PairResult:
    """Result of pairs trading analysis."""
    asset1: str
    asset2: str
    correlation: float
    cointegration_pvalue: float
    half_life: float
    spread_zscore: float
    is_cointegrated: bool
    hedge_ratio: float


class PairsTrading:
    """
    Pairs Trading Strategy.
    
    Identifies cointegrated pairs and trades mean-reversion
    of their spread.
    """
    
    def __init__(
        self,
        lookback: int = 60,
        entry_zscore: float = 2.0,
        exit_zscore: float = 0.5,
        stop_zscore: float = 3.0,
        coint_pvalue: float = 0.05,
    ):
        """
        Initialize Pairs Trading.
        
        Args:
            lookback: Lookback period for calculations
            entry_zscore: Z-score threshold for entry
            exit_zscore: Z-score threshold for exit
            stop_zscore: Z-score threshold for stop loss
            coint_pvalue: P-value threshold for cointegration
        """
        self.lookback = lookback
        self.entry_zscore = entry_zscore
        self.exit_zscore = exit_zscore
        self.stop_zscore = stop_zscore
        self.coint_pvalue = coint_pvalue
        
        self._pairs: List[PairResult] = []
        self._active_positions: Dict[Tuple[str, str], Dict] = {}
    
    def find_pairs(
        self,
        prices: pd.DataFrame,
        min_correlation: float = 0.7,
    ) -> List[PairResult]:
        """
        Find cointegrated pairs.
        
        Args:
            prices: DataFrame of asset prices
            min_correlation: Minimum correlation threshold
            
        Returns:
            List of PairResult for cointegrated pairs
        """
        symbols = prices.columns.tolist()
        pairs = []
        
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                # Check correlation
                corr = prices[sym1].corr(prices[sym2])
                if abs(corr) < min_correlation:
                    continue
                
                # Check cointegration
                try:
                    _, pvalue, _ = coint(prices[sym1], prices[sym2])
                except Exception:
                    continue
                
                is_coint = pvalue < self.coint_pvalue
                
                # Calculate hedge ratio
                hedge_ratio = self._calculate_hedge_ratio(
                    prices[sym1], prices[sym2]
                )
                
                # Calculate spread
                spread = prices[sym1] - hedge_ratio * prices[sym2]
                
                # Half-life of mean reversion
                half_life = self._calculate_half_life(spread)
                
                # Current z-score
                zscore = (spread.iloc[-1] - spread.mean()) / spread.std()
                
                pair = PairResult(
                    asset1=sym1,
                    asset2=sym2,
                    correlation=corr,
                    cointegration_pvalue=pvalue,
                    half_life=half_life,
                    spread_zscore=zscore,
                    is_cointegrated=is_coint,
                    hedge_ratio=hedge_ratio,
                )
                
                pairs.append(pair)
        
        # Sort by cointegration p-value
        pairs.sort(key=lambda p: p.cointegration_pvalue)
        
        self._pairs = [p for p in pairs if p.is_cointegrated]
        return self._pairs
    
    def _calculate_hedge_ratio(
        self,
        y: pd.Series,
        x: pd.Series,
    ) -> float:
        """Calculate hedge ratio using OLS."""
        slope, _, _, _, _ = stats.linregress(x, y)
        return slope
    
    def _calculate_half_life(self, spread: pd.Series) -> float:
        """Calculate half-life of mean reversion."""
        spread_lag = spread.shift(1).dropna()
        spread_diff = spread.diff().dropna()
        
        # Align lengths
        min_len = min(len(spread_lag), len(spread_diff))
        spread_lag = spread_lag.iloc[-min_len:]
        spread_diff = spread_diff.iloc[-min_len:]
        
        if len(spread_lag) < 10:
            return float('inf')
        
        try:
            slope, _, _, _, _ = stats.linregress(spread_lag, spread_diff)
            if slope >= 0:
                return float('inf')
            return -np.log(2) / slope
        except Exception:
            return float('inf')
    
    def generate_signals(
        self,
        prices: pd.DataFrame,
    ) -> List[TradingSignal]:
        """
        Generate trading signals for pairs.
        
        Args:
            prices: Current prices
            
        Returns:
            List of trading signals
        """
        signals = []
        
        for pair in self._pairs:
            if pair.asset1 not in prices.columns or pair.asset2 not in prices.columns:
                continue
            
            # Calculate current spread and z-score
            spread = prices[pair.asset1] - pair.hedge_ratio * prices[pair.asset2]
            zscore = (spread.iloc[-1] - spread.iloc[-self.lookback:].mean()) / spread.iloc[-self.lookback:].std()
            
            pair_key = (pair.asset1, pair.asset2)
            
            if pair_key in self._active_positions:
                # Check for exit
                if abs(zscore) < self.exit_zscore:
                    signals.append(TradingSignal(
                        symbol=pair.asset1,
                        signal_type=SignalType.CLOSE,
                        strength=0,
                        metadata={"pair": pair.asset2, "zscore": zscore},
                    ))
                    signals.append(TradingSignal(
                        symbol=pair.asset2,
                        signal_type=SignalType.CLOSE,
                        strength=0,
                        metadata={"pair": pair.asset1, "zscore": zscore},
                    ))
                    del self._active_positions[pair_key]
                    
                # Check for stop loss
                elif abs(zscore) > self.stop_zscore:
                    signals.append(TradingSignal(
                        symbol=pair.asset1,
                        signal_type=SignalType.CLOSE,
                        strength=0,
                        metadata={"pair": pair.asset2, "stop_loss": True},
                    ))
                    signals.append(TradingSignal(
                        symbol=pair.asset2,
                        signal_type=SignalType.CLOSE,
                        strength=0,
                        metadata={"pair": pair.asset1, "stop_loss": True},
                    ))
                    del self._active_positions[pair_key]
            else:
                # Check for entry
                if zscore > self.entry_zscore:
                    # Short asset1, long asset2
                    signals.append(TradingSignal(
                        symbol=pair.asset1,
                        signal_type=SignalType.SHORT,
                        strength=min(1.0, zscore / self.stop_zscore),
                        metadata={"pair": pair.asset2, "hedge_ratio": pair.hedge_ratio},
                    ))
                    signals.append(TradingSignal(
                        symbol=pair.asset2,
                        signal_type=SignalType.LONG,
                        strength=min(1.0, zscore / self.stop_zscore),
                        metadata={"pair": pair.asset1, "hedge_ratio": 1/pair.hedge_ratio},
                    ))
                    self._active_positions[pair_key] = {"direction": "short_spread"}
                    
                elif zscore < -self.entry_zscore:
                    # Long asset1, short asset2
                    signals.append(TradingSignal(
                        symbol=pair.asset1,
                        signal_type=SignalType.LONG,
                        strength=min(1.0, abs(zscore) / self.stop_zscore),
                        metadata={"pair": pair.asset2, "hedge_ratio": pair.hedge_ratio},
                    ))
                    signals.append(TradingSignal(
                        symbol=pair.asset2,
                        signal_type=SignalType.SHORT,
                        strength=min(1.0, abs(zscore) / self.stop_zscore),
                        metadata={"pair": pair.asset1, "hedge_ratio": 1/pair.hedge_ratio},
                    ))
                    self._active_positions[pair_key] = {"direction": "long_spread"}
        
        return signals


class StatisticalArbitrage:
    """
    Statistical Arbitrage Strategy.
    
    Exploits statistical mispricings across multiple assets.
    """
    
    def __init__(
        self,
        lookback: int = 20,
        entry_threshold: float = 1.5,
        exit_threshold: float = 0.5,
        num_factors: int = 5,
    ):
        """
        Initialize Statistical Arbitrage.
        
        Args:
            lookback: Lookback period
            entry_threshold: Entry threshold (std devs)
            exit_threshold: Exit threshold (std devs)
            num_factors: Number of PCA factors
        """
        self.lookback = lookback
        self.entry_threshold = entry_threshold
        self.exit_threshold = exit_threshold
        self.num_factors = num_factors
        
        self._factor_loadings: Optional[np.ndarray] = None
        self._residuals_mean: Optional[np.ndarray] = None
        self._residuals_std: Optional[np.ndarray] = None
    
    def fit(self, returns: pd.DataFrame) -> None:
        """
        Fit factor model using PCA.
        
        Args:
            returns: DataFrame of asset returns
        """
        # Standardize returns
        ret_std = (returns - returns.mean()) / returns.std()
        
        # PCA
        from numpy.linalg import svd
        U, S, Vt = svd(ret_std.values, full_matrices=False)
        
        # Keep top factors
        n_factors = min(self.num_factors, len(S))
        self._factor_loadings = Vt[:n_factors].T
        
        # Factor returns
        factor_returns = ret_std.values @ self._factor_loadings
        
        # Residuals
        reconstructed = factor_returns @ self._factor_loadings.T
        residuals = ret_std.values - reconstructed
        
        self._residuals_mean = residuals.mean(axis=0)
        self._residuals_std = residuals.std(axis=0)
        self._symbols = returns.columns.tolist()
    
    def generate_signals(
        self,
        returns: pd.DataFrame,
    ) -> List[TradingSignal]:
        """
        Generate signals based on residual deviation.
        
        Args:
            returns: Recent returns
            
        Returns:
            List of trading signals
        """
        if self._factor_loadings is None:
            self.fit(returns)
        
        signals = []
        
        # Calculate current residuals
        ret_std = (returns - returns.mean()) / returns.std()
        recent = ret_std.iloc[-self.lookback:].mean()
        
        factor_returns = recent.values @ self._factor_loadings
        reconstructed = factor_returns @ self._factor_loadings.T
        residuals = recent.values - reconstructed
        
        # Z-scores of residuals
        zscores = (residuals - self._residuals_mean) / self._residuals_std
        
        for i, symbol in enumerate(self._symbols):
            if symbol not in returns.columns:
                continue
            
            zscore = zscores[i]
            
            if zscore > self.entry_threshold:
                signals.append(TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.SHORT,
                    strength=min(1.0, zscore / 3),
                    metadata={"zscore": zscore, "strategy": "stat_arb"},
                ))
            elif zscore < -self.entry_threshold:
                signals.append(TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.LONG,
                    strength=min(1.0, abs(zscore) / 3),
                    metadata={"zscore": zscore, "strategy": "stat_arb"},
                ))
        
        return signals


class Factor(Enum):
    """Investment factors."""
    MOMENTUM = "momentum"
    VALUE = "value"
    SIZE = "size"
    QUALITY = "quality"
    LOW_VOLATILITY = "low_volatility"
    DIVIDEND = "dividend"


class FactorInvesting:
    """
    Factor Investing Strategy.
    
    Implements momentum, value, quality, and other factors.
    """
    
    def __init__(
        self,
        momentum_lookback: int = 252,
        momentum_skip: int = 21,
        rebalance_frequency: int = 21,
    ):
        """
        Initialize Factor Investing.
        
        Args:
            momentum_lookback: Lookback for momentum calculation
            momentum_skip: Days to skip for momentum (avoid reversal)
            rebalance_frequency: Days between rebalancing
        """
        self.momentum_lookback = momentum_lookback
        self.momentum_skip = momentum_skip
        self.rebalance_frequency = rebalance_frequency
        
        self._factor_scores: Dict[str, Dict[Factor, float]] = {}
        self._last_rebalance: Optional[datetime] = None
    
    def calculate_momentum(
        self,
        prices: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Calculate momentum scores.
        
        Uses 12-month momentum with 1-month skip.
        """
        if len(prices) < self.momentum_lookback:
            return {}
        
        # Price change excluding most recent month
        start_idx = -self.momentum_lookback
        end_idx = -self.momentum_skip if self.momentum_skip > 0 else None
        
        momentum = {}
        for symbol in prices.columns:
            start_price = prices[symbol].iloc[start_idx]
            end_price = prices[symbol].iloc[end_idx] if end_idx else prices[symbol].iloc[-1]
            
            if start_price > 0:
                momentum[symbol] = (end_price / start_price) - 1
        
        return momentum
    
    def calculate_value(
        self,
        fundamentals: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """
        Calculate value scores.
        
        Uses P/E, P/B, and other value metrics.
        """
        value_scores = {}
        
        for symbol, data in fundamentals.items():
            pe = data.get("pe_ratio", float('inf'))
            pb = data.get("pb_ratio", float('inf'))
            
            # Lower is better for value
            if pe > 0 and pb > 0:
                value_scores[symbol] = -1 / (0.5 * pe + 0.5 * pb)
        
        return value_scores
    
    def calculate_quality(
        self,
        fundamentals: Dict[str, Dict[str, float]],
    ) -> Dict[str, float]:
        """
        Calculate quality scores.
        
        Uses ROE, ROA, and debt ratios.
        """
        quality_scores = {}
        
        for symbol, data in fundamentals.items():
            roe = data.get("roe", 0)
            roa = data.get("roa", 0)
            debt_equity = data.get("debt_equity", float('inf'))
            
            # Higher ROE/ROA, lower debt is better
            if debt_equity > 0:
                quality_scores[symbol] = 0.4 * roe + 0.3 * roa - 0.3 * (debt_equity / 100)
        
        return quality_scores
    
    def calculate_low_volatility(
        self,
        returns: pd.DataFrame,
        window: int = 60,
    ) -> Dict[str, float]:
        """
        Calculate low volatility scores.
        
        Lower volatility = higher score.
        """
        volatility = returns.iloc[-window:].std() * np.sqrt(252)
        
        # Invert so lower vol = higher score
        return {symbol: -vol for symbol, vol in volatility.items()}
    
    def rank_assets(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        fundamentals: Optional[Dict[str, Dict[str, float]]] = None,
        factor_weights: Optional[Dict[Factor, float]] = None,
    ) -> Dict[str, float]:
        """
        Rank assets by combined factor scores.
        
        Args:
            prices: Price data
            returns: Return data
            fundamentals: Fundamental data per asset
            factor_weights: Weight for each factor
            
        Returns:
            Combined scores per asset
        """
        if factor_weights is None:
            factor_weights = {
                Factor.MOMENTUM: 0.4,
                Factor.LOW_VOLATILITY: 0.3,
                Factor.QUALITY: 0.2,
                Factor.VALUE: 0.1,
            }
        
        # Calculate factor scores
        momentum = self.calculate_momentum(prices)
        low_vol = self.calculate_low_volatility(returns)
        
        quality = {}
        value = {}
        if fundamentals:
            quality = self.calculate_quality(fundamentals)
            value = self.calculate_value(fundamentals)
        
        # Combine scores
        all_symbols = set(momentum.keys()) | set(low_vol.keys())
        combined = {}
        
        for symbol in all_symbols:
            score = 0
            
            if Factor.MOMENTUM in factor_weights and symbol in momentum:
                score += factor_weights[Factor.MOMENTUM] * self._normalize_score(momentum, symbol)
            
            if Factor.LOW_VOLATILITY in factor_weights and symbol in low_vol:
                score += factor_weights[Factor.LOW_VOLATILITY] * self._normalize_score(low_vol, symbol)
            
            if Factor.QUALITY in factor_weights and symbol in quality:
                score += factor_weights[Factor.QUALITY] * self._normalize_score(quality, symbol)
            
            if Factor.VALUE in factor_weights and symbol in value:
                score += factor_weights[Factor.VALUE] * self._normalize_score(value, symbol)
            
            combined[symbol] = score
        
        return combined
    
    def _normalize_score(
        self,
        scores: Dict[str, float],
        symbol: str,
    ) -> float:
        """Normalize score to [0, 1] range."""
        values = list(scores.values())
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:
            return 0.5
        
        return (scores[symbol] - min_val) / (max_val - min_val)
    
    def generate_signals(
        self,
        prices: pd.DataFrame,
        returns: pd.DataFrame,
        top_n: int = 10,
        bottom_n: int = 0,
    ) -> List[TradingSignal]:
        """
        Generate signals based on factor rankings.
        
        Args:
            prices: Price data
            returns: Return data
            top_n: Number of top assets to go long
            bottom_n: Number of bottom assets to go short
            
        Returns:
            List of trading signals
        """
        rankings = self.rank_assets(prices, returns)
        sorted_assets = sorted(rankings.items(), key=lambda x: x[1], reverse=True)
        
        signals = []
        
        # Long top N
        for symbol, score in sorted_assets[:top_n]:
            signals.append(TradingSignal(
                symbol=symbol,
                signal_type=SignalType.LONG,
                strength=score,
                metadata={"rank": sorted_assets.index((symbol, score)) + 1},
            ))
        
        # Short bottom N
        if bottom_n > 0:
            for symbol, score in sorted_assets[-bottom_n:]:
                signals.append(TradingSignal(
                    symbol=symbol,
                    signal_type=SignalType.SHORT,
                    strength=1 - score,
                    metadata={"rank": len(sorted_assets) - sorted_assets.index((symbol, score))},
                ))
        
        return signals


class CrossAssetMomentum:
    """
    Cross-Asset Momentum Strategy.
    
    Applies momentum across different asset classes.
    """
    
    def __init__(
        self,
        lookback: int = 63,  # ~3 months
        holding_period: int = 21,  # ~1 month
        top_pct: float = 0.2,
        bottom_pct: float = 0.2,
    ):
        """
        Initialize Cross-Asset Momentum.
        
        Args:
            lookback: Momentum lookback period
            holding_period: Holding period for signals
            top_pct: Top percentile to go long
            bottom_pct: Bottom percentile to go short
        """
        self.lookback = lookback
        self.holding_period = holding_period
        self.top_pct = top_pct
        self.bottom_pct = bottom_pct
    
    def calculate_momentum_scores(
        self,
        prices: pd.DataFrame,
    ) -> Dict[str, float]:
        """
        Calculate risk-adjusted momentum scores.
        
        Args:
            prices: Price data for all assets
            
        Returns:
            Momentum scores per asset
        """
        if len(prices) < self.lookback:
            return {}
        
        returns = prices.pct_change()
        
        scores = {}
        for symbol in prices.columns:
            recent_returns = returns[symbol].iloc[-self.lookback:]
            
            # Risk-adjusted momentum (Sharpe-like)
            mean_ret = recent_returns.mean()
            std_ret = recent_returns.std()
            
            if std_ret > 0:
                scores[symbol] = mean_ret / std_ret
            else:
                scores[symbol] = 0
        
        return scores
    
    def generate_signals(
        self,
        prices: pd.DataFrame,
        asset_classes: Optional[Dict[str, str]] = None,
    ) -> List[TradingSignal]:
        """
        Generate cross-asset momentum signals.
        
        Args:
            prices: Price data
            asset_classes: Optional mapping of symbol to asset class
            
        Returns:
            List of trading signals
        """
        scores = self.calculate_momentum_scores(prices)
        
        if not scores:
            return []
        
        sorted_assets = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        n_assets = len(sorted_assets)
        
        top_n = max(1, int(n_assets * self.top_pct))
        bottom_n = max(1, int(n_assets * self.bottom_pct))
        
        signals = []
        
        # Long top momentum
        for symbol, score in sorted_assets[:top_n]:
            signals.append(TradingSignal(
                symbol=symbol,
                signal_type=SignalType.LONG,
                strength=min(1.0, score / 2),
                metadata={
                    "momentum_score": score,
                    "asset_class": asset_classes.get(symbol) if asset_classes else None,
                },
            ))
        
        # Short bottom momentum
        for symbol, score in sorted_assets[-bottom_n:]:
            signals.append(TradingSignal(
                symbol=symbol,
                signal_type=SignalType.SHORT,
                strength=min(1.0, abs(score) / 2),
                metadata={
                    "momentum_score": score,
                    "asset_class": asset_classes.get(symbol) if asset_classes else None,
                },
            ))
        
        return signals

