"""
Tests for Advanced AI/ML Models Module.
"""

import pytest
import torch
import numpy as np
import pandas as pd
from datetime import datetime

from src.models.attention import (
    AttentionLayer,
    MultiHeadAttention,
    TemporalAttention,
    FeatureAttention,
)
from src.models.transformer import (
    TransformerPredictor,
    TransformerConfig,
    TemporalFusionTransformer,
    PositionalEncoding,
)
from src.models.sentiment import (
    SentimentAnalyzer,
    SentimentLabel,
    NewsProcessor,
    SentimentSignalGenerator,
    NewsItem,
)
from src.models.regime import (
    RegimeDetector,
    MarketRegime,
    StatisticalRegimeDetector,
)
from src.models.advanced_rl import (
    SACAgent,
    TD3Agent,
    ReplayBuffer,
)


@pytest.fixture
def sample_sequence():
    """Generate sample sequence data."""
    batch_size = 4
    seq_len = 30
    input_dim = 16
    return torch.randn(batch_size, seq_len, input_dim)


@pytest.fixture
def sample_price_df():
    """Generate sample price DataFrame."""
    np.random.seed(42)
    n = 300
    
    prices = [100.0]
    for _ in range(n - 1):
        change = np.random.normal(0.0005, 0.02)
        prices.append(prices[-1] * (1 + change))
    
    return pd.DataFrame({
        "close": prices,
        "volume": np.random.randint(1000, 10000, n),
    })


class TestAttentionLayers:
    """Tests for attention mechanisms."""
    
    def test_attention_layer(self, sample_sequence):
        d_model = sample_sequence.size(-1)
        attention = AttentionLayer(d_model)
        
        output, weights = attention(sample_sequence, sample_sequence, sample_sequence)
        
        assert output.shape == sample_sequence.shape
        assert weights.shape[1] == sample_sequence.size(1)
        assert weights.shape[2] == sample_sequence.size(1)
    
    def test_multi_head_attention(self, sample_sequence):
        d_model = sample_sequence.size(-1)
        num_heads = 4
        
        mha = MultiHeadAttention(d_model, num_heads)
        output, weights = mha(sample_sequence, sample_sequence, sample_sequence)
        
        assert output.shape == sample_sequence.shape
    
    def test_temporal_attention(self, sample_sequence):
        input_dim = sample_sequence.size(-1)
        temporal_attn = TemporalAttention(input_dim)
        
        output, weights = temporal_attn(sample_sequence)
        
        assert output.shape == (sample_sequence.size(0), input_dim)
        assert weights.shape == (sample_sequence.size(0), sample_sequence.size(1))
        # Weights should sum to 1
        assert torch.allclose(weights.sum(dim=1), torch.ones(sample_sequence.size(0)), atol=1e-5)
    
    def test_feature_attention(self, sample_sequence):
        num_features = sample_sequence.size(-1)
        feature_attn = FeatureAttention(num_features)
        
        output, weights = feature_attn(sample_sequence)
        
        assert output.shape == sample_sequence.shape
        assert weights.shape == (sample_sequence.size(0), num_features)


class TestTransformerModels:
    """Tests for Transformer models."""
    
    def test_positional_encoding(self):
        d_model = 64
        max_len = 100
        
        pe = PositionalEncoding(d_model, max_len)
        x = torch.zeros(2, 50, d_model)
        
        output = pe(x)
        
        assert output.shape == x.shape
        # Positional encodings should add variation
        assert not torch.allclose(output, x)
    
    def test_transformer_predictor(self, sample_sequence):
        config = TransformerConfig(
            input_dim=sample_sequence.size(-1),
            d_model=32,
            num_heads=4,
            num_encoder_layers=2,
            d_ff=64,
            output_dim=3,
        )
        
        model = TransformerPredictor(config)
        output = model(sample_sequence)
        
        assert output.shape == (sample_sequence.size(0), 3)
    
    def test_transformer_with_attention(self, sample_sequence):
        config = TransformerConfig(
            input_dim=sample_sequence.size(-1),
            d_model=32,
            num_heads=4,
            num_encoder_layers=2,
        )
        
        model = TransformerPredictor(config)
        output, attention = model(sample_sequence, return_attention=True)
        
        assert len(attention) == 2  # 2 encoder layers
    
    def test_transformer_predict(self, sample_sequence):
        config = TransformerConfig(
            input_dim=sample_sequence.size(-1),
            d_model=32,
            num_heads=4,
            output_dim=3,
        )
        
        model = TransformerPredictor(config)
        
        proba = model.predict_proba(sample_sequence)
        assert proba.shape == (sample_sequence.size(0), 3)
        assert torch.allclose(proba.sum(dim=1), torch.ones(sample_sequence.size(0)), atol=1e-5)
        
        pred = model.predict(sample_sequence)
        assert pred.shape == (sample_sequence.size(0),)
        assert all(p in [0, 1, 2] for p in pred.tolist())
    
    def test_temporal_fusion_transformer(self, sample_sequence):
        num_features = sample_sequence.size(-1)
        
        tft = TemporalFusionTransformer(
            num_features=num_features,
            d_model=32,
            num_heads=2,
            num_quantiles=3,
            prediction_horizon=1,
        )
        
        output = tft(sample_sequence)
        
        assert output.shape == (sample_sequence.size(0), 1, 3)


class TestSentimentAnalysis:
    """Tests for sentiment analysis."""
    
    def test_sentiment_analyzer_bullish(self):
        analyzer = SentimentAnalyzer()
        
        text = "Strong earnings beat expectations, stock rallies on positive momentum"
        result = analyzer.analyze(text)
        
        assert result.sentiment in [SentimentLabel.BULLISH, SentimentLabel.VERY_BULLISH]
        assert result.confidence > 0
        assert len(result.keywords) > 0
    
    def test_sentiment_analyzer_bearish(self):
        analyzer = SentimentAnalyzer()
        
        text = "Stock plunges on weak guidance, fears of recession grow"
        result = analyzer.analyze(text)
        
        assert result.sentiment in [SentimentLabel.BEARISH, SentimentLabel.VERY_BEARISH]
    
    def test_sentiment_analyzer_neutral(self):
        analyzer = SentimentAnalyzer()
        
        text = "Company announces regular quarterly dividend"
        result = analyzer.analyze(text)
        
        assert result.sentiment == SentimentLabel.NEUTRAL or result.confidence < 0.5
    
    def test_sentiment_batch(self):
        analyzer = SentimentAnalyzer()
        
        texts = [
            "Bullish outlook for tech stocks",
            "Market crashes on economic fears",
            "Company reports stable earnings",
        ]
        
        results = analyzer.analyze_batch(texts)
        
        assert len(results) == 3
    
    def test_aggregate_sentiment(self):
        analyzer = SentimentAnalyzer()
        
        texts = [
            "Strong growth momentum",
            "Bullish sentiment continues",
            "Weak performance this quarter",
        ]
        
        results = analyzer.analyze_batch(texts)
        sentiment, score = analyzer.get_aggregate_sentiment(results)
        
        assert isinstance(sentiment, SentimentLabel)
        assert -1 <= score <= 1
    
    def test_news_processor(self):
        processor = NewsProcessor()
        
        news = processor.process_news(
            title="AAPL Stock Surges on Strong iPhone Sales",
            content="Apple Inc reported record iPhone sales...",
            source="Test News",
            url="https://example.com/news",
        )
        
        assert "AAPL" in news.symbols
        assert news.sentiment is not None
    
    def test_sentiment_signal_generator(self):
        generator = SentimentSignalGenerator()
        
        signal = generator.generate_signal(
            news_sentiment=0.5,
            social_sentiment=0.3,
            news_count=10,
            social_count=50,
        )
        
        assert signal["action"] in ["buy", "sell", "hold"]
        assert 0 <= signal["confidence"] <= 1


class TestRegimeDetection:
    """Tests for market regime detection."""
    
    def test_statistical_detector(self, sample_price_df):
        detector = StatisticalRegimeDetector()
        
        state = detector.detect(sample_price_df)
        
        assert state.regime in MarketRegime
        assert 0 <= state.confidence <= 1
        assert state.duration_bars > 0
    
    def test_regime_detector_wrapper(self, sample_price_df):
        detector = RegimeDetector(method="statistical")
        
        state = detector.detect(sample_price_df)
        
        assert state.regime in MarketRegime
    
    def test_regime_probabilities(self, sample_price_df):
        detector = RegimeDetector()
        
        probs = detector.get_regime_probabilities(sample_price_df)
        
        assert sum(probs.values()) == pytest.approx(1.0, rel=0.01)
    
    def test_regime_update(self, sample_price_df):
        detector = RegimeDetector()
        
        # First detection
        state1 = detector.detect(sample_price_df)
        
        # Update with same data
        transition = detector.update(sample_price_df)
        
        # May or may not have transition
        assert detector.get_current_state() is not None
    
    def test_recommended_strategy(self):
        detector = RegimeDetector()
        
        for regime in MarketRegime:
            strategy = detector.get_recommended_strategy(regime)
            
            assert "strategy" in strategy
            assert "position_bias" in strategy
            assert "stop_loss_atr_mult" in strategy
    
    def test_regime_state_to_dict(self, sample_price_df):
        detector = RegimeDetector()
        state = detector.detect(sample_price_df)
        
        d = state.to_dict()
        
        assert "regime" in d
        assert "confidence" in d
        assert "features" in d


class TestAdvancedRL:
    """Tests for advanced RL algorithms."""
    
    def test_replay_buffer(self):
        buffer = ReplayBuffer(capacity=100)
        
        for i in range(50):
            buffer.push(
                state=np.random.randn(10),
                action=np.random.randn(2),
                reward=np.random.randn(),
                next_state=np.random.randn(10),
                done=False,
            )
        
        assert len(buffer) == 50
        
        batch = buffer.sample(10)
        assert len(batch) == 10
    
    def test_sac_agent_init(self):
        agent = SACAgent(
            state_dim=10,
            action_dim=2,
            hidden_dims=[32, 32],
            buffer_size=1000,
            batch_size=32,
        )
        
        assert agent.action_dim == 2
        assert agent.training_steps == 0
    
    def test_sac_select_action(self):
        agent = SACAgent(
            state_dim=10,
            action_dim=2,
            hidden_dims=[32, 32],
        )
        
        state = np.random.randn(10)
        
        action_stochastic = agent.select_action(state, deterministic=False)
        action_deterministic = agent.select_action(state, deterministic=True)
        
        assert action_stochastic.shape == (2,)
        assert action_deterministic.shape == (2,)
        assert np.all(np.abs(action_stochastic) <= 1)
        assert np.all(np.abs(action_deterministic) <= 1)
    
    def test_sac_update(self):
        agent = SACAgent(
            state_dim=10,
            action_dim=2,
            hidden_dims=[32, 32],
            buffer_size=1000,
            batch_size=32,
        )
        
        # Fill buffer
        for _ in range(100):
            agent.store_transition(
                state=np.random.randn(10),
                action=np.random.randn(2),
                reward=np.random.randn(),
                next_state=np.random.randn(10),
                done=False,
            )
        
        metrics = agent.update()
        
        assert "critic_loss" in metrics
        assert "actor_loss" in metrics
        assert "alpha" in metrics
    
    def test_td3_agent_init(self):
        agent = TD3Agent(
            state_dim=10,
            action_dim=2,
            hidden_dims=[32, 32],
            buffer_size=1000,
            batch_size=32,
        )
        
        assert agent.action_dim == 2
        assert agent.training_steps == 0
    
    def test_td3_select_action(self):
        agent = TD3Agent(
            state_dim=10,
            action_dim=2,
            hidden_dims=[32, 32],
        )
        
        state = np.random.randn(10)
        
        action_noisy = agent.select_action(state, add_noise=True)
        action_clean = agent.select_action(state, add_noise=False)
        
        assert action_noisy.shape == (2,)
        assert action_clean.shape == (2,)
    
    def test_td3_update(self):
        agent = TD3Agent(
            state_dim=10,
            action_dim=2,
            hidden_dims=[32, 32],
            buffer_size=1000,
            batch_size=32,
            policy_delay=2,
        )
        
        # Fill buffer
        for _ in range(100):
            agent.store_transition(
                state=np.random.randn(10),
                action=np.random.randn(2),
                reward=np.random.randn(),
                next_state=np.random.randn(10),
                done=False,
            )
        
        # Multiple updates to trigger policy update
        for _ in range(3):
            metrics = agent.update()
        
        assert "critic_loss" in metrics


class TestIntegration:
    """Integration tests for models."""
    
    def test_transformer_with_regime(self, sample_price_df, sample_sequence):
        # Detect regime
        detector = RegimeDetector()
        state = detector.detect(sample_price_df)
        
        # Get strategy recommendation
        strategy = detector.get_recommended_strategy(state.regime)
        
        # Use transformer for prediction
        config = TransformerConfig(
            input_dim=sample_sequence.size(-1),
            d_model=32,
            num_heads=4,
            output_dim=3,
        )
        model = TransformerPredictor(config)
        
        prediction = model.predict(sample_sequence)
        
        # Combine regime info with prediction
        assert state.regime in MarketRegime
        assert prediction.shape[0] == sample_sequence.size(0)
    
    def test_sentiment_with_signal(self):
        analyzer = SentimentAnalyzer()
        generator = SentimentSignalGenerator()
        
        news_texts = [
            "Strong quarterly earnings beat expectations",
            "Tech sector shows bullish momentum",
        ]
        
        results = analyzer.analyze_batch(news_texts)
        _, news_score = analyzer.get_aggregate_sentiment(results)
        
        signal = generator.generate_signal(
            news_sentiment=news_score,
            social_sentiment=0.0,
            news_count=len(results),
            social_count=0,
        )
        
        assert signal["action"] in ["buy", "sell", "hold"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

