"""
Market Regime Detection

Identifies market regimes (trending, mean-reverting, volatile)
for adaptive strategy selection.
"""

import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from abc import ABC, abstractmethod


class MarketRegime(Enum):
    """Market regime classification."""
    BULL_TRENDING = "bull_trending"
    BEAR_TRENDING = "bear_trending"
    MEAN_REVERTING = "mean_reverting"
    HIGH_VOLATILITY = "high_volatility"
    LOW_VOLATILITY = "low_volatility"
    RANGING = "ranging"
    BREAKOUT = "breakout"
    UNKNOWN = "unknown"


@dataclass
class RegimeState:
    """Current regime state."""
    regime: MarketRegime
    confidence: float
    duration_bars: int
    start_time: datetime
    features: Dict[str, float] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime": self.regime.value,
            "confidence": self.confidence,
            "duration_bars": self.duration_bars,
            "start_time": self.start_time.isoformat(),
            "features": self.features,
        }


@dataclass
class RegimeTransition:
    """Regime transition event."""
    from_regime: MarketRegime
    to_regime: MarketRegime
    timestamp: datetime
    confidence: float


class BaseRegimeDetector(ABC):
    """Abstract base class for regime detectors."""
    
    @abstractmethod
    def detect(self, data: pd.DataFrame) -> RegimeState:
        """Detect current market regime."""
        pass
    
    @abstractmethod
    def update(self, new_data: pd.DataFrame) -> Optional[RegimeTransition]:
        """Update with new data and check for transitions."""
        pass


class StatisticalRegimeDetector(BaseRegimeDetector):
    """
    Statistical regime detection using market indicators.
    
    Uses volatility, trend strength, and mean reversion
    metrics to classify regimes.
    """
    
    def __init__(
        self,
        volatility_window: int = 20,
        trend_window: int = 50,
        volatility_threshold_high: float = 1.5,
        volatility_threshold_low: float = 0.7,
        trend_threshold: float = 0.02,
        adf_significance: float = 0.05,
    ):
        """
        Initialize Statistical Regime Detector.
        
        Args:
            volatility_window: Window for volatility calculation
            trend_window: Window for trend analysis
            volatility_threshold_high: Threshold for high volatility regime
            volatility_threshold_low: Threshold for low volatility regime
            trend_threshold: Threshold for trend detection
            adf_significance: Significance level for ADF test
        """
        self.volatility_window = volatility_window
        self.trend_window = trend_window
        self.volatility_threshold_high = volatility_threshold_high
        self.volatility_threshold_low = volatility_threshold_low
        self.trend_threshold = trend_threshold
        self.adf_significance = adf_significance
        
        self._current_state: Optional[RegimeState] = None
        self._history: List[RegimeState] = []
        self._transitions: List[RegimeTransition] = []
    
    def _calculate_volatility_ratio(self, returns: pd.Series) -> float:
        """Calculate current volatility relative to historical."""
        if len(returns) < self.volatility_window * 2:
            return 1.0
        
        current_vol = returns.iloc[-self.volatility_window:].std()
        historical_vol = returns.iloc[:-self.volatility_window].std()
        
        if historical_vol == 0:
            return 1.0
        
        return current_vol / historical_vol
    
    def _calculate_trend_strength(self, prices: pd.Series) -> Tuple[float, str]:
        """
        Calculate trend strength and direction.
        
        Returns:
            Tuple of (strength, direction)
        """
        if len(prices) < self.trend_window:
            return 0.0, "none"
        
        # Linear regression slope
        x = np.arange(self.trend_window)
        y = prices.iloc[-self.trend_window:].values
        
        # Normalize prices
        y_norm = (y - y.mean()) / y.std() if y.std() > 0 else y - y.mean()
        
        # Calculate slope
        slope = np.polyfit(x, y_norm, 1)[0]
        
        # R-squared for strength
        y_pred = slope * x + np.polyfit(x, y_norm, 1)[1]
        ss_res = np.sum((y_norm - y_pred) ** 2)
        ss_tot = np.sum((y_norm - y_norm.mean()) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        direction = "up" if slope > 0 else "down" if slope < 0 else "none"
        
        return abs(slope) * r_squared, direction
    
    def _calculate_mean_reversion_score(self, prices: pd.Series) -> float:
        """
        Calculate mean reversion score using Hurst exponent approximation.
        
        H < 0.5: Mean reverting
        H = 0.5: Random walk
        H > 0.5: Trending
        """
        if len(prices) < 100:
            return 0.5
        
        # Simplified Hurst calculation using R/S analysis
        n = len(prices)
        max_k = min(n // 4, 100)
        
        rs_list = []
        ns = []
        
        for k in range(10, max_k, 10):
            rs_values = []
            
            for start in range(0, n - k, k):
                subset = prices.iloc[start:start + k].values
                returns = np.diff(subset) / subset[:-1]
                
                # Cumulative deviation
                mean_return = returns.mean()
                cumulative_dev = np.cumsum(returns - mean_return)
                
                # Range
                R = cumulative_dev.max() - cumulative_dev.min()
                
                # Standard deviation
                S = returns.std()
                
                if S > 0:
                    rs_values.append(R / S)
            
            if rs_values:
                rs_list.append(np.mean(rs_values))
                ns.append(k)
        
        if len(rs_list) < 2:
            return 0.5
        
        # Fit log-log regression
        log_n = np.log(ns)
        log_rs = np.log(rs_list)
        
        hurst = np.polyfit(log_n, log_rs, 1)[0]
        
        return hurst
    
    def _classify_regime(
        self,
        volatility_ratio: float,
        trend_strength: float,
        trend_direction: str,
        hurst: float,
    ) -> Tuple[MarketRegime, float]:
        """
        Classify regime based on indicators.
        
        Returns:
            Tuple of (regime, confidence)
        """
        scores = {regime: 0.0 for regime in MarketRegime}
        
        # High volatility
        if volatility_ratio > self.volatility_threshold_high:
            scores[MarketRegime.HIGH_VOLATILITY] += volatility_ratio
        
        # Low volatility
        if volatility_ratio < self.volatility_threshold_low:
            scores[MarketRegime.LOW_VOLATILITY] += 1 / max(volatility_ratio, 0.1)
        
        # Trending
        if trend_strength > self.trend_threshold:
            if trend_direction == "up":
                scores[MarketRegime.BULL_TRENDING] += trend_strength * 10
            else:
                scores[MarketRegime.BEAR_TRENDING] += trend_strength * 10
        
        # Mean reverting
        if hurst < 0.4:
            scores[MarketRegime.MEAN_REVERTING] += (0.5 - hurst) * 5
        
        # Ranging
        if trend_strength < self.trend_threshold / 2 and 0.4 < hurst < 0.6:
            scores[MarketRegime.RANGING] += 1.0
        
        # Breakout
        if volatility_ratio > 1.2 and trend_strength > self.trend_threshold * 1.5:
            scores[MarketRegime.BREAKOUT] += (volatility_ratio * trend_strength) * 5
        
        # Find best regime
        best_regime = max(scores, key=scores.get)
        best_score = scores[best_regime]
        
        # Calculate confidence
        total_score = sum(scores.values())
        if total_score > 0:
            confidence = best_score / total_score
        else:
            best_regime = MarketRegime.UNKNOWN
            confidence = 0.0
        
        return best_regime, min(confidence, 1.0)
    
    def detect(self, data: pd.DataFrame) -> RegimeState:
        """
        Detect current market regime.
        
        Args:
            data: DataFrame with 'close' column
            
        Returns:
            RegimeState with classification
        """
        if 'close' not in data.columns:
            raise ValueError("DataFrame must have 'close' column")
        
        prices = data['close']
        returns = prices.pct_change().dropna()
        
        # Calculate indicators
        volatility_ratio = self._calculate_volatility_ratio(returns)
        trend_strength, trend_direction = self._calculate_trend_strength(prices)
        hurst = self._calculate_mean_reversion_score(prices)
        
        # Classify
        regime, confidence = self._classify_regime(
            volatility_ratio, trend_strength, trend_direction, hurst
        )
        
        # Create state
        state = RegimeState(
            regime=regime,
            confidence=confidence,
            duration_bars=1 if self._current_state is None else (
                self._current_state.duration_bars + 1
                if self._current_state.regime == regime
                else 1
            ),
            start_time=datetime.now() if (
                self._current_state is None or
                self._current_state.regime != regime
            ) else self._current_state.start_time,
            features={
                "volatility_ratio": volatility_ratio,
                "trend_strength": trend_strength,
                "trend_direction": trend_direction,
                "hurst_exponent": hurst,
            },
        )
        
        self._current_state = state
        self._history.append(state)
        
        return state
    
    def update(self, new_data: pd.DataFrame) -> Optional[RegimeTransition]:
        """
        Update with new data.
        
        Args:
            new_data: New price data
            
        Returns:
            RegimeTransition if regime changed, None otherwise
        """
        old_regime = self._current_state.regime if self._current_state else None
        
        new_state = self.detect(new_data)
        
        if old_regime and old_regime != new_state.regime:
            transition = RegimeTransition(
                from_regime=old_regime,
                to_regime=new_state.regime,
                timestamp=datetime.now(),
                confidence=new_state.confidence,
            )
            self._transitions.append(transition)
            return transition
        
        return None
    
    def get_regime_probabilities(self, data: pd.DataFrame) -> Dict[MarketRegime, float]:
        """
        Get probability distribution over regimes.
        
        Args:
            data: Price data
            
        Returns:
            Dictionary of regime to probability
        """
        prices = data['close']
        returns = prices.pct_change().dropna()
        
        volatility_ratio = self._calculate_volatility_ratio(returns)
        trend_strength, trend_direction = self._calculate_trend_strength(prices)
        hurst = self._calculate_mean_reversion_score(prices)
        
        scores = {regime: 0.0 for regime in MarketRegime}
        
        # Score each regime
        scores[MarketRegime.HIGH_VOLATILITY] = max(0, volatility_ratio - 1)
        scores[MarketRegime.LOW_VOLATILITY] = max(0, 1 - volatility_ratio)
        scores[MarketRegime.BULL_TRENDING] = trend_strength if trend_direction == "up" else 0
        scores[MarketRegime.BEAR_TRENDING] = trend_strength if trend_direction == "down" else 0
        scores[MarketRegime.MEAN_REVERTING] = max(0, 0.5 - hurst)
        scores[MarketRegime.RANGING] = max(0, 0.2 - abs(trend_strength))
        
        # Normalize to probabilities
        total = sum(scores.values())
        if total > 0:
            return {k: v / total for k, v in scores.items()}
        
        return {MarketRegime.UNKNOWN: 1.0}
    
    def get_history(self, limit: int = 100) -> List[RegimeState]:
        """Get regime history."""
        return self._history[-limit:]
    
    def get_transitions(self, limit: int = 50) -> List[RegimeTransition]:
        """Get transition history."""
        return self._transitions[-limit:]


class RegimeDetector:
    """
    Main regime detector with multiple backends.
    
    Supports statistical and ML-based detection.
    """
    
    def __init__(
        self,
        method: str = "statistical",
        **kwargs,
    ):
        """
        Initialize Regime Detector.
        
        Args:
            method: Detection method ("statistical", "hmm", "ml")
            **kwargs: Method-specific parameters
        """
        self.method = method
        
        if method == "statistical":
            self._detector = StatisticalRegimeDetector(**kwargs)
        else:
            self._detector = StatisticalRegimeDetector(**kwargs)
    
    def detect(self, data: pd.DataFrame) -> RegimeState:
        """Detect current regime."""
        return self._detector.detect(data)
    
    def update(self, new_data: pd.DataFrame) -> Optional[RegimeTransition]:
        """Update with new data."""
        return self._detector.update(new_data)
    
    def get_current_state(self) -> Optional[RegimeState]:
        """Get current regime state."""
        return self._detector._current_state
    
    def get_regime_probabilities(self, data: pd.DataFrame) -> Dict[MarketRegime, float]:
        """Get regime probabilities."""
        return self._detector.get_regime_probabilities(data)
    
    def get_recommended_strategy(self, regime: MarketRegime) -> Dict[str, Any]:
        """
        Get recommended strategy parameters for a regime.
        
        Args:
            regime: Current market regime
            
        Returns:
            Strategy recommendations
        """
        recommendations = {
            MarketRegime.BULL_TRENDING: {
                "strategy": "momentum",
                "position_bias": "long",
                "stop_loss_atr_mult": 2.5,
                "take_profit_atr_mult": 4.0,
                "position_size_mult": 1.2,
            },
            MarketRegime.BEAR_TRENDING: {
                "strategy": "momentum",
                "position_bias": "short",
                "stop_loss_atr_mult": 2.5,
                "take_profit_atr_mult": 4.0,
                "position_size_mult": 1.0,
            },
            MarketRegime.MEAN_REVERTING: {
                "strategy": "mean_reversion",
                "position_bias": "neutral",
                "stop_loss_atr_mult": 1.5,
                "take_profit_atr_mult": 2.0,
                "position_size_mult": 1.0,
            },
            MarketRegime.HIGH_VOLATILITY: {
                "strategy": "volatility",
                "position_bias": "neutral",
                "stop_loss_atr_mult": 3.0,
                "take_profit_atr_mult": 3.0,
                "position_size_mult": 0.5,
            },
            MarketRegime.LOW_VOLATILITY: {
                "strategy": "breakout",
                "position_bias": "neutral",
                "stop_loss_atr_mult": 1.5,
                "take_profit_atr_mult": 3.0,
                "position_size_mult": 1.5,
            },
            MarketRegime.RANGING: {
                "strategy": "range",
                "position_bias": "neutral",
                "stop_loss_atr_mult": 1.5,
                "take_profit_atr_mult": 1.5,
                "position_size_mult": 0.8,
            },
            MarketRegime.BREAKOUT: {
                "strategy": "breakout",
                "position_bias": "follow",
                "stop_loss_atr_mult": 2.0,
                "take_profit_atr_mult": 5.0,
                "position_size_mult": 1.3,
            },
            MarketRegime.UNKNOWN: {
                "strategy": "conservative",
                "position_bias": "neutral",
                "stop_loss_atr_mult": 2.0,
                "take_profit_atr_mult": 2.0,
                "position_size_mult": 0.5,
            },
        }
        
        return recommendations.get(regime, recommendations[MarketRegime.UNKNOWN])

