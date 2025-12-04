"""
Multi-Agent Trading System

This module provides specialized trading agents that can be used individually
or combined in an ensemble for improved trading decisions.
"""

from .base import BaseAgent, AgentAction, AgentSignal
from .momentum import MomentumAgent
from .mean_reversion import MeanReversionAgent
from .breakout import BreakoutAgent
from .market_maker import MarketMakerAgent
from .registry import AgentRegistry
from .ensemble import EnsembleAgent

__all__ = [
    "BaseAgent",
    "AgentAction",
    "AgentSignal",
    "MomentumAgent",
    "MeanReversionAgent",
    "BreakoutAgent",
    "MarketMakerAgent",
    "AgentRegistry",
    "EnsembleAgent",
]

