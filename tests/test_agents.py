"""
Tests for the Multi-Agent Trading System.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.agents import (
    BaseAgent,
    AgentAction,
    AgentSignal,
    MomentumAgent,
    MeanReversionAgent,
    BreakoutAgent,
    MarketMakerAgent,
    AgentRegistry,
    EnsembleAgent,
)
from src.agents.ensemble import VotingMethod


@pytest.fixture
def sample_data():
    """Generate sample OHLCV data for testing."""
    np.random.seed(42)
    n = 100
    
    # Generate trending data
    base_price = 100
    prices = [base_price]
    for i in range(n - 1):
        change = np.random.normal(0.001, 0.02)  # Slight upward drift
        prices.append(prices[-1] * (1 + change))
    
    prices = np.array(prices)
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, n)),
        'high': prices * (1 + np.random.uniform(0, 0.02, n)),
        'low': prices * (1 - np.random.uniform(0, 0.02, n)),
        'close': prices,
        'volume': np.random.uniform(1e6, 2e6, n),
    })
    
    data.index = pd.date_range(start='2024-01-01', periods=n, freq='1h')
    return data


@pytest.fixture
def mean_reverting_data():
    """Generate mean-reverting data for testing."""
    np.random.seed(123)
    n = 100
    
    # Generate oscillating data around mean
    mean_price = 100
    prices = []
    price = mean_price
    
    for i in range(n):
        # Mean reversion force
        reversion = (mean_price - price) * 0.1
        noise = np.random.normal(0, 2)
        price = price + reversion + noise
        prices.append(price)
    
    prices = np.array(prices)
    
    data = pd.DataFrame({
        'open': prices * (1 + np.random.uniform(-0.005, 0.005, n)),
        'high': prices * (1 + np.random.uniform(0, 0.01, n)),
        'low': prices * (1 - np.random.uniform(0, 0.01, n)),
        'close': prices,
        'volume': np.random.uniform(1e6, 2e6, n),
    })
    
    data.index = pd.date_range(start='2024-01-01', periods=n, freq='1h')
    return data


class TestAgentSignal:
    """Tests for AgentSignal dataclass."""
    
    def test_signal_creation(self):
        signal = AgentSignal(
            action=AgentAction.BUY,
            confidence=0.8,
            strength=0.5
        )
        assert signal.action == AgentAction.BUY
        assert signal.confidence == 0.8
        assert signal.strength == 0.5
    
    def test_signal_clamping(self):
        """Test that confidence and strength are clamped."""
        signal = AgentSignal(
            action=AgentAction.SELL,
            confidence=1.5,  # Should be clamped to 1.0
            strength=-2.0   # Should be clamped to -1.0
        )
        assert signal.confidence == 1.0
        assert signal.strength == -1.0
    
    def test_signal_to_dict(self):
        signal = AgentSignal(
            action=AgentAction.HOLD,
            confidence=0.5,
            strength=0.0,
            metadata={"test": True}
        )
        d = signal.to_dict()
        assert d["action"] == "HOLD"
        assert d["action_id"] == 0
        assert d["confidence"] == 0.5
        assert d["metadata"]["test"] is True


class TestMomentumAgent:
    """Tests for MomentumAgent."""
    
    def test_initialization(self):
        agent = MomentumAgent(name="test_momentum")
        assert agent.name == "test_momentum"
        assert agent.fast_period == 10
        assert agent.slow_period == 30
    
    def test_generate_signal(self, sample_data):
        agent = MomentumAgent()
        signal = agent.generate_signal(sample_data)
        
        assert isinstance(signal, AgentSignal)
        assert signal.action in AgentAction
        assert 0 <= signal.confidence <= 1
        assert -1 <= signal.strength <= 1
    
    def test_insufficient_data(self):
        agent = MomentumAgent()
        small_data = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [99, 100],
            'close': [101, 102],
            'volume': [1e6, 1e6],
        })
        
        signal = agent.generate_signal(small_data)
        assert signal.action == AgentAction.HOLD
        assert signal.confidence == 0.0
    
    def test_strategy_params(self):
        agent = MomentumAgent(fast_period=5, slow_period=20)
        params = agent.get_strategy_params()
        
        assert params["fast_period"] == 5
        assert params["slow_period"] == 20


class TestMeanReversionAgent:
    """Tests for MeanReversionAgent."""
    
    def test_initialization(self):
        agent = MeanReversionAgent(name="test_mr")
        assert agent.name == "test_mr"
        assert agent.bb_period == 20
        assert agent.rsi_period == 14
    
    def test_generate_signal(self, mean_reverting_data):
        agent = MeanReversionAgent()
        signal = agent.generate_signal(mean_reverting_data)
        
        assert isinstance(signal, AgentSignal)
        assert signal.action in AgentAction
        assert "rsi" in signal.metadata
        assert "zscore" in signal.metadata
    
    def test_oversold_detection(self):
        """Test that agent detects oversold conditions."""
        agent = MeanReversionAgent(rsi_oversold=30)
        
        # Create oversold data (declining prices)
        n = 50
        prices = np.linspace(100, 70, n)
        data = pd.DataFrame({
            'open': prices + 1,
            'high': prices + 2,
            'low': prices - 1,
            'close': prices,
            'volume': np.ones(n) * 1e6,
        })
        
        signal = agent.generate_signal(data)
        # Should detect oversold
        assert signal.metadata.get("oversold_rsi") or signal.metadata.get("oversold_bb")


class TestBreakoutAgent:
    """Tests for BreakoutAgent."""
    
    def test_initialization(self):
        agent = BreakoutAgent(name="test_breakout")
        assert agent.name == "test_breakout"
        assert agent.channel_period == 20
    
    def test_generate_signal(self, sample_data):
        agent = BreakoutAgent()
        signal = agent.generate_signal(sample_data)
        
        assert isinstance(signal, AgentSignal)
        assert "channel_upper" in signal.metadata
        assert "channel_lower" in signal.metadata
    
    def test_breakout_detection(self):
        """Test breakout detection."""
        agent = BreakoutAgent(channel_period=10)
        
        # Create data with breakout
        n = 30
        # Range-bound then breakout
        prices = np.concatenate([
            np.ones(20) * 100 + np.random.normal(0, 1, 20),
            np.linspace(100, 110, 10)  # Breakout up
        ])
        
        data = pd.DataFrame({
            'open': prices - 0.5,
            'high': prices + 1,
            'low': prices - 1,
            'close': prices,
            'volume': np.concatenate([np.ones(20) * 1e6, np.ones(10) * 2e6]),
        })
        
        signal = agent.generate_signal(data)
        # Should detect breakout
        assert "breakout_up" in signal.metadata


class TestMarketMakerAgent:
    """Tests for MarketMakerAgent."""
    
    def test_initialization(self):
        agent = MarketMakerAgent(name="test_mm")
        assert agent.name == "test_mm"
        assert agent.base_spread_bps == 10.0
    
    def test_generate_signal(self, sample_data):
        agent = MarketMakerAgent()
        signal = agent.generate_signal(sample_data)
        
        assert isinstance(signal, AgentSignal)
        assert "fair_value" in signal.metadata
        assert "bid_price" in signal.metadata
        assert "ask_price" in signal.metadata
    
    def test_inventory_tracking(self, sample_data):
        agent = MarketMakerAgent()
        
        # Test with different positions
        signal_flat = agent.generate_signal(sample_data, position=0)
        signal_long = agent.generate_signal(sample_data, position=1)
        signal_short = agent.generate_signal(sample_data, position=-1)
        
        # Inventory should affect signals
        assert signal_long.metadata["inventory"] != signal_short.metadata["inventory"]


class TestAgentRegistry:
    """Tests for AgentRegistry."""
    
    def test_register_agent(self):
        registry = AgentRegistry()
        agent = MomentumAgent(name="mom1")
        
        registry.register(agent, initial_weight=1.5)
        
        assert "mom1" in registry
        assert len(registry) == 1
        assert registry.get_weight("mom1") == 1.5
    
    def test_duplicate_registration(self):
        registry = AgentRegistry()
        agent = MomentumAgent(name="mom1")
        
        registry.register(agent)
        with pytest.raises(ValueError):
            registry.register(agent)  # Same name
    
    def test_unregister_agent(self):
        registry = AgentRegistry()
        agent = MomentumAgent(name="mom1")
        
        registry.register(agent)
        removed = registry.unregister("mom1")
        
        assert removed == agent
        assert "mom1" not in registry
    
    def test_get_signals(self, sample_data):
        registry = AgentRegistry()
        registry.register(MomentumAgent(name="mom"))
        registry.register(MeanReversionAgent(name="mr"))
        
        signals = registry.get_signals(sample_data)
        
        assert "mom" in signals
        assert "mr" in signals
        assert all(isinstance(s, AgentSignal) for s in signals.values())
    
    def test_rankings(self):
        registry = AgentRegistry()
        
        mom = MomentumAgent(name="mom")
        mr = MeanReversionAgent(name="mr")
        
        # Add some performance history
        mom._performance_history = [100, 50, -20, 80]
        mr._performance_history = [-10, 20, 30, 40]
        
        registry.register(mom)
        registry.register(mr)
        
        rankings = registry.get_rankings()
        
        assert len(rankings) == 2
        assert all("total_pnl" in r for r in rankings)


class TestEnsembleAgent:
    """Tests for EnsembleAgent."""
    
    def test_initialization(self):
        ensemble = EnsembleAgent(name="test_ensemble")
        assert ensemble.name == "test_ensemble"
        assert ensemble.voting_method == VotingMethod.WEIGHTED
    
    def test_add_remove_agents(self):
        ensemble = EnsembleAgent()
        mom = MomentumAgent(name="mom")
        mr = MeanReversionAgent(name="mr")
        
        ensemble.add_agent(mom)
        ensemble.add_agent(mr)
        
        assert len(ensemble.registry) == 2
        
        ensemble.remove_agent("mom")
        assert len(ensemble.registry) == 1
    
    def test_generate_signal(self, sample_data):
        ensemble = EnsembleAgent()
        ensemble.add_agent(MomentumAgent(name="mom"))
        ensemble.add_agent(MeanReversionAgent(name="mr"))
        ensemble.add_agent(BreakoutAgent(name="bo"))
        
        signal = ensemble.generate_signal(sample_data)
        
        assert isinstance(signal, AgentSignal)
        assert "individual_signals" in signal.metadata
        assert "num_agents" in signal.metadata
        assert signal.metadata["num_agents"] == 3
    
    def test_voting_methods(self, sample_data):
        """Test different voting methods produce signals."""
        for method in VotingMethod:
            ensemble = EnsembleAgent(voting_method=method)
            ensemble.add_agent(MomentumAgent(name="mom"))
            ensemble.add_agent(MeanReversionAgent(name="mr"))
            
            signal = ensemble.generate_signal(sample_data)
            
            assert isinstance(signal, AgentSignal)
            assert signal.metadata["voting_method"] == method.value
    
    def test_empty_ensemble(self, sample_data):
        ensemble = EnsembleAgent()
        signal = ensemble.generate_signal(sample_data)
        
        assert signal.action == AgentAction.HOLD
        assert signal.confidence == 0.0
    
    def test_get_last_signals(self, sample_data):
        ensemble = EnsembleAgent()
        ensemble.add_agent(MomentumAgent(name="mom"))
        ensemble.add_agent(MeanReversionAgent(name="mr"))
        
        # Generate signal first
        ensemble.generate_signal(sample_data)
        
        last_signals = ensemble.get_last_signals()
        assert "mom" in last_signals
        assert "mr" in last_signals


class TestAgentPerformance:
    """Tests for agent performance tracking."""
    
    def test_performance_update(self):
        agent = MomentumAgent(name="test")
        
        agent.update_performance(100)
        agent.update_performance(-50)
        agent.update_performance(75)
        
        metrics = agent.get_performance_metrics()
        
        assert metrics["num_trades"] == 3
        assert metrics["total_pnl"] == 125
        assert 0 <= metrics["win_rate"] <= 1
    
    def test_reset(self):
        agent = MomentumAgent(name="test")
        agent.update_performance(100)
        
        agent.reset()
        
        metrics = agent.get_performance_metrics()
        assert metrics["num_trades"] == 0
        assert metrics["total_pnl"] == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

