"""
Base Agent Interface

Provides the abstract base class that all specialized trading agents must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd


class AgentAction(Enum):
    """Possible trading actions."""
    HOLD = 0
    BUY = 1
    SELL = 2
    STRONG_BUY = 3
    STRONG_SELL = 4


@dataclass
class AgentSignal:
    """
    Trading signal produced by an agent.
    
    Attributes:
        action: The recommended action
        confidence: Confidence level (0.0 to 1.0)
        strength: Signal strength (-1.0 to 1.0, negative = bearish, positive = bullish)
        metadata: Additional information about the signal
    """
    action: AgentAction
    confidence: float
    strength: float
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        # Clamp values
        self.confidence = np.clip(self.confidence, 0.0, 1.0)
        self.strength = np.clip(self.strength, -1.0, 1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert signal to dictionary."""
        return {
            "action": self.action.name,
            "action_id": self.action.value,
            "confidence": self.confidence,
            "strength": self.strength,
            "metadata": self.metadata,
        }


class BaseAgent(ABC):
    """
    Abstract base class for all trading agents.
    
    Each specialized agent implements a specific trading strategy and produces
    signals that can be used individually or combined in an ensemble.
    """
    
    def __init__(
        self,
        name: str,
        description: str = "",
        lookback_period: int = 20,
        **kwargs
    ):
        """
        Initialize the base agent.
        
        Args:
            name: Unique identifier for the agent
            description: Human-readable description of the strategy
            lookback_period: Number of historical bars to consider
            **kwargs: Additional agent-specific parameters
        """
        self.name = name
        self.description = description
        self.lookback_period = lookback_period
        self.params = kwargs
        self._is_trained = False
        self._performance_history: List[float] = []
        self._signal_history: List[AgentSignal] = []
    
    @abstractmethod
    def generate_signal(
        self,
        data: pd.DataFrame,
        position: int = 0,
        **kwargs
    ) -> AgentSignal:
        """
        Generate a trading signal based on market data.
        
        Args:
            data: OHLCV DataFrame with at least 'open', 'high', 'low', 'close', 'volume'
            position: Current position (-1 = short, 0 = flat, 1 = long)
            **kwargs: Additional context
            
        Returns:
            AgentSignal with the recommended action
        """
        pass
    
    @abstractmethod
    def get_strategy_params(self) -> Dict[str, Any]:
        """
        Get the strategy-specific parameters.
        
        Returns:
            Dictionary of parameter names and values
        """
        pass
    
    def update_performance(self, pnl: float) -> None:
        """
        Update the agent's performance history.
        
        Args:
            pnl: Profit/loss from the last trade
        """
        self._performance_history.append(pnl)
    
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Calculate performance metrics.
        
        Returns:
            Dictionary with performance statistics
        """
        if not self._performance_history:
            return {
                "total_pnl": 0.0,
                "win_rate": 0.0,
                "avg_pnl": 0.0,
                "sharpe": 0.0,
                "num_trades": 0,
            }
        
        pnl_array = np.array(self._performance_history)
        wins = pnl_array[pnl_array > 0]
        
        return {
            "total_pnl": float(np.sum(pnl_array)),
            "win_rate": float(len(wins) / len(pnl_array)) if len(pnl_array) > 0 else 0.0,
            "avg_pnl": float(np.mean(pnl_array)),
            "sharpe": float(np.mean(pnl_array) / np.std(pnl_array)) if np.std(pnl_array) > 0 else 0.0,
            "num_trades": len(pnl_array),
        }
    
    def reset(self) -> None:
        """Reset the agent's state."""
        self._performance_history = []
        self._signal_history = []
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}', lookback={self.lookback_period})"

