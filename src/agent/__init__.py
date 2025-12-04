"""
Agent module for HFT Market Simulator.

This module provides:
- Custom LSTM-based policy networks
- PPO agent configuration
- Training utilities
- Model evaluation
"""

from .networks import LSTMFeatureExtractor, TradingPolicy
from .trainer import PPOTrainer
from .evaluator import ModelEvaluator

__all__ = [
    "LSTMFeatureExtractor",
    "TradingPolicy",
    "PPOTrainer",
    "ModelEvaluator",
]

