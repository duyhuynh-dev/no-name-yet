"""
Advanced AI/ML Models Module

Provides sophisticated machine learning models for trading:
- Transformer-based time series models
- Attention mechanisms
- Sentiment analysis with LLM
- Regime detection
"""

from .transformer import TransformerPredictor, TemporalFusionTransformer
from .attention import AttentionLayer, MultiHeadAttention
from .sentiment import SentimentAnalyzer, NewsProcessor
from .regime import RegimeDetector, MarketRegime
from .advanced_rl import SACAgent, TD3Agent

__all__ = [
    "TransformerPredictor",
    "TemporalFusionTransformer",
    "AttentionLayer",
    "MultiHeadAttention",
    "SentimentAnalyzer",
    "NewsProcessor",
    "RegimeDetector",
    "MarketRegime",
    "SACAgent",
    "TD3Agent",
]

