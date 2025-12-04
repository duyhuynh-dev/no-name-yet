"""
Unit tests for Reward Functions.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env.rewards import (
    SimplePnLReward,
    RiskAdjustedReward,
    SharpeReward,
    DifferentialSharpeReward,
    create_reward_function,
)


class TestSimplePnLReward:
    """Tests for SimplePnLReward."""
    
    def test_initialization(self):
        """Test reward function initialization."""
        reward_fn = SimplePnLReward(scale=100.0)
        assert reward_fn.scale == 100.0
    
    def test_reset(self):
        """Test reset (should be no-op for simple PnL)."""
        reward_fn = SimplePnLReward()
        reward_fn.reset()  # Should not raise
    
    def test_positive_pnl(self):
        """Test reward for positive PnL."""
        reward_fn = SimplePnLReward(scale=1.0)
        
        reward = reward_fn.calculate(
            pnl=100.0,
            position=1,
            action=0,
            info={},
        )
        
        assert reward == 100.0, "Positive PnL should give positive reward"
    
    def test_negative_pnl(self):
        """Test reward for negative PnL."""
        reward_fn = SimplePnLReward(scale=1.0)
        
        reward = reward_fn.calculate(
            pnl=-50.0,
            position=1,
            action=0,
            info={},
        )
        
        assert reward == -50.0, "Negative PnL should give negative reward"
    
    def test_no_pnl(self):
        """Test reward for no PnL."""
        reward_fn = SimplePnLReward(scale=1.0)
        
        reward = reward_fn.calculate(
            pnl=0.0,
            position=0,
            action=0,
            info={},
        )
        
        assert reward == 0.0, "No PnL should give zero reward"
    
    def test_scaling(self):
        """Test reward scaling."""
        reward_fn = SimplePnLReward(scale=10.0)
        
        reward = reward_fn.calculate(pnl=5.0, position=1, action=0, info={})
        
        assert reward == 50.0, "Reward should be scaled"


class TestRiskAdjustedReward:
    """Tests for RiskAdjustedReward."""
    
    def test_initialization(self):
        """Test reward function initialization."""
        reward_fn = RiskAdjustedReward(
            volatility_penalty=0.1,
            drawdown_penalty=0.1,
        )
        assert reward_fn.volatility_penalty == 0.1
        assert reward_fn.drawdown_penalty == 0.1
    
    def test_reset(self):
        """Test reset clears state."""
        reward_fn = RiskAdjustedReward()
        reward_fn.returns_history = [0.01, 0.02, -0.01]
        reward_fn.peak_value = 15000.0
        
        reward_fn.reset()
        
        assert len(reward_fn.returns_history) == 0
        assert reward_fn.peak_value == 0.0
    
    def test_basic_reward(self):
        """Test basic reward calculation."""
        reward_fn = RiskAdjustedReward(
            volatility_penalty=0.0,
            drawdown_penalty=0.0,
            scale=1.0,
        )
        reward_fn.reset()
        
        # First call sets up initial state
        reward = reward_fn.calculate(
            pnl=100.0,
            position=1,
            action=0,
            info={'portfolio_value': 10100.0},
        )
        
        assert isinstance(reward, float)
    
    def test_reward_clipping(self):
        """Test rewards are clipped."""
        reward_fn = RiskAdjustedReward(scale=1000.0, clip_reward=10.0)
        reward_fn.reset()
        
        # Large PnL
        reward = reward_fn.calculate(
            pnl=1000.0,  # Large profit
            position=1,
            action=0,
            info={'portfolio_value': 20000.0},
        )
        
        assert -10.0 <= reward <= 10.0, "Reward should be clipped"


class TestSharpeReward:
    """Tests for SharpeReward."""
    
    def test_initialization(self):
        """Test reward function initialization."""
        reward_fn = SharpeReward(window_size=20)
        assert reward_fn.window_size == 20
    
    def test_reset(self):
        """Test reset clears state."""
        reward_fn = SharpeReward()
        reward_fn.returns_history = [0.01, 0.02]
        reward_fn.reset()
        assert len(reward_fn.returns_history) == 0
    
    def test_reward_calculation(self):
        """Test Sharpe reward is calculated."""
        reward_fn = SharpeReward(window_size=10, scale=1.0)
        reward_fn.reset()
        
        rewards = []
        for i in range(15):
            reward = reward_fn.calculate(
                pnl=10.0,  # Consistent positive
                position=1,
                action=0,
                info={'portfolio_value': 10000 + i * 10},
            )
            rewards.append(reward)
        
        assert len(rewards) == 15


class TestDifferentialSharpeReward:
    """Tests for DifferentialSharpeReward."""
    
    def test_initialization(self):
        """Test reward function initialization."""
        reward_fn = DifferentialSharpeReward(eta=0.01)
        assert reward_fn.eta == 0.01
    
    def test_reset(self):
        """Test reset is callable."""
        reward_fn = DifferentialSharpeReward()
        reward_fn.reset()  # Should not raise
        # Note: Some implementations may not reset A_t/B_t
    
    def test_reward_calculation(self):
        """Test differential Sharpe reward."""
        reward_fn = DifferentialSharpeReward(eta=0.1, scale=1.0)
        reward_fn.reset()
        
        # Several steps
        for pnl in [10, 20, 30, 40]:
            reward = reward_fn.calculate(
                pnl=float(pnl),
                position=1,
                action=0,
                info={'portfolio_value': 10000 + pnl},
            )
        
        # Running stats should be updated
        assert isinstance(reward, float)


class TestCreateRewardFunction:
    """Tests for reward function factory."""
    
    def test_create_simple(self):
        """Test creating simple reward function."""
        reward_fn = create_reward_function('simple', scale=100.0)
        assert isinstance(reward_fn, SimplePnLReward)
    
    def test_create_risk_adjusted(self):
        """Test creating risk-adjusted reward function."""
        reward_fn = create_reward_function('risk_adjusted')
        assert isinstance(reward_fn, RiskAdjustedReward)
    
    def test_create_sharpe(self):
        """Test creating Sharpe reward function."""
        reward_fn = create_reward_function('sharpe')
        assert isinstance(reward_fn, SharpeReward)
    
    def test_create_diff_sharpe(self):
        """Test creating differential Sharpe reward function."""
        reward_fn = create_reward_function('diff_sharpe')
        assert isinstance(reward_fn, DifferentialSharpeReward)
    
    def test_invalid_type(self):
        """Test invalid reward type raises error."""
        with pytest.raises(ValueError):
            create_reward_function('invalid_type')


class TestRewardFunctionIntegration:
    """Integration tests for reward functions with environment."""
    
    def test_reward_in_env(self, trading_env):
        """Test reward function works in environment."""
        trading_env.reset()
        
        total_reward = 0
        for _ in range(20):
            action = trading_env.action_space.sample()
            _, reward, terminated, truncated, _ = trading_env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        # Should have some non-zero reward
        assert isinstance(total_reward, (int, float))
        assert not np.isnan(total_reward)
        assert not np.isinf(total_reward)
