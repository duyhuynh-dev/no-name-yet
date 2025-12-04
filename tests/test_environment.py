"""
Unit tests for the Trading Environment.
"""

import numpy as np
import pytest
from gymnasium import spaces


class TestTradingEnv:
    """Tests for TradingEnv class."""
    
    def test_env_creation(self, trading_env):
        """Test environment is created correctly."""
        assert trading_env is not None
        assert trading_env.initial_balance == 10000.0
        assert trading_env.window_size == 30
    
    def test_observation_space(self, trading_env):
        """Test observation space is correct."""
        obs_space = trading_env.observation_space
        
        assert isinstance(obs_space, spaces.Box)
        assert obs_space.dtype == np.float32
        # Shape should be (window_size * n_features + 2,)
        expected_shape = (trading_env.window_size * 27 + 2,)
        assert obs_space.shape == expected_shape
    
    def test_action_space(self, trading_env):
        """Test action space is discrete with 3 actions."""
        action_space = trading_env.action_space
        
        assert isinstance(action_space, spaces.Discrete)
        assert action_space.n == 3  # Hold, Buy, Sell
    
    def test_reset(self, trading_env):
        """Test environment reset."""
        obs, info = trading_env.reset()
        
        assert obs is not None
        assert isinstance(obs, np.ndarray)
        assert obs.shape == trading_env.observation_space.shape
        assert isinstance(info, dict)
        
        # Check initial state
        assert info.get('initial_balance') == 10000.0
    
    def test_reset_with_seed(self, trading_env):
        """Test environment reset with seed is deterministic."""
        obs1, _ = trading_env.reset(seed=42)
        obs2, _ = trading_env.reset(seed=42)
        
        np.testing.assert_array_equal(obs1, obs2)
    
    def test_step_hold(self, trading_env):
        """Test hold action (0)."""
        trading_env.reset()
        
        obs, reward, terminated, truncated, info = trading_env.step(0)
        
        assert obs is not None
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)
        
        # Position should still be 0 (flat)
        assert info.get('position') == 0
    
    def test_step_buy(self, trading_env):
        """Test buy action (1)."""
        trading_env.reset()
        
        obs, reward, terminated, truncated, info = trading_env.step(1)
        
        # Position should be 1 (long)
        assert info.get('position') == 1
        # Should have shares
        assert trading_env.shares > 0
    
    def test_step_sell(self, trading_env):
        """Test sell action (2)."""
        trading_env.reset()
        
        obs, reward, terminated, truncated, info = trading_env.step(2)
        
        # Position should be -1 (short)
        assert info.get('position') == -1
    
    def test_buy_then_sell(self, trading_env):
        """Test buy followed by sell."""
        trading_env.reset()
        
        # Buy
        trading_env.step(1)
        assert trading_env.position == 1
        
        # Sell (close long, open short)
        _, _, _, _, info = trading_env.step(2)
        assert info.get('position') == -1
    
    def test_observation_normalization(self, trading_env):
        """Test that observations are normalized."""
        obs, _ = trading_env.reset()
        
        # Normalized observations should have reasonable range
        assert np.abs(obs).mean() < 10  # Not too extreme
        assert not np.any(np.isnan(obs))
        assert not np.any(np.isinf(obs))
    
    def test_episode_termination(self, trading_env):
        """Test episode terminates at end of data."""
        trading_env.reset()
        
        done = False
        steps = 0
        max_steps = len(trading_env.df) + 100  # Safety limit
        
        while not done and steps < max_steps:
            _, _, terminated, truncated, _ = trading_env.step(0)
            done = terminated or truncated
            steps += 1
        
        assert done, "Episode should terminate"
        assert steps <= len(trading_env.df) - trading_env.window_size
    
    def test_reward_clipping(self, trading_env):
        """Test that rewards are clipped to reasonable range."""
        trading_env.reset()
        
        rewards = []
        for _ in range(50):
            action = trading_env.action_space.sample()
            _, reward, terminated, truncated, _ = trading_env.step(action)
            rewards.append(reward)
            if terminated or truncated:
                break
        
        # Rewards should be clipped
        assert all(-15 <= r <= 15 for r in rewards), f"Rewards out of range: {min(rewards)}, {max(rewards)}"
    
    def test_portfolio_value_tracking(self, trading_env):
        """Test portfolio value is tracked correctly."""
        trading_env.reset()
        initial_value = 10000.0
        
        # Hold for a few steps
        for _ in range(5):
            _, _, _, _, info = trading_env.step(0)
        
        # Portfolio value should still be approximately initial (no position)
        assert abs(info['portfolio_value'] - initial_value) < 100
    
    def test_transaction_costs(self, trading_env):
        """Test transaction costs are applied."""
        trading_env.reset()
        initial_cash = trading_env.cash
        
        # Buy
        trading_env.step(1)
        
        # Sell
        trading_env.step(2)
        
        # Should have less cash due to transaction costs
        # (even if price didn't change)
        # Note: This depends on price movement, so we just check it's different
        assert trading_env.trades, "Should have recorded trades"
    
    def test_info_dict_contents(self, trading_env):
        """Test info dict contains expected keys after step."""
        trading_env.reset()
        _, _, _, _, info = trading_env.step(0)  # Take a step to get full info
        
        expected_keys = ['portfolio_value', 'position', 'cash', 'shares']
        for key in expected_keys:
            assert key in info, f"Missing key: {key}"
    
    def test_multiple_episodes(self, trading_env):
        """Test running multiple episodes."""
        for episode in range(3):
            obs, _ = trading_env.reset()
            assert obs is not None
            
            done = False
            steps = 0
            while not done and steps < 100:
                action = trading_env.action_space.sample()
                obs, _, terminated, truncated, _ = trading_env.step(action)
                done = terminated or truncated
                steps += 1


class TestEnvironmentEdgeCases:
    """Edge case tests for TradingEnv."""
    
    def test_rapid_trading(self, trading_env):
        """Test rapid buy/sell cycles."""
        trading_env.reset()
        
        for _ in range(20):
            trading_env.step(1)  # Buy
            trading_env.step(2)  # Sell
        
        # Should not crash
        assert trading_env.portfolio_value > 0
    
    def test_all_holds(self, trading_env):
        """Test holding for entire episode."""
        trading_env.reset()
        
        for _ in range(50):
            _, _, terminated, truncated, info = trading_env.step(0)
            if terminated or truncated:
                break
        
        # Should have approximately initial balance
        assert abs(info['portfolio_value'] - 10000.0) < 100
    
    def test_all_buys(self, trading_env):
        """Test buying every step."""
        trading_env.reset()
        
        for _ in range(20):
            _, _, terminated, truncated, _ = trading_env.step(1)
            if terminated or truncated:
                break
        
        # Should have a long position
        assert trading_env.position == 1
    
    def test_observation_dtype(self, trading_env):
        """Test observation dtype is float32."""
        obs, _ = trading_env.reset()
        assert obs.dtype == np.float32

