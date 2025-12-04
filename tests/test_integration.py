"""
Integration tests for the HFT RL Trading System.
"""

import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestDataPipelineIntegration:
    """Integration tests for data pipeline."""
    
    def test_data_loading_and_preprocessing(self, sample_ohlcv_data):
        """Test data can be loaded and preprocessed."""
        from src.data.preprocessor import DataPreprocessor
        
        preprocessor = DataPreprocessor()
        processed = preprocessor.fit_transform(sample_ohlcv_data)
        
        assert processed is not None
        assert len(processed) > 0
        assert not processed.isnull().any().any()
    
    def test_data_validation(self, sample_ohlcv_data):
        """Test data validation works."""
        from src.data.validator import DataValidator
        
        validator = DataValidator()
        result = validator.validate(sample_ohlcv_data)
        
        # Result could be tuple or single value depending on implementation
        if isinstance(result, tuple):
            is_valid = result[0]
        else:
            is_valid = result
        
        assert is_valid, "Validation failed"
    
    def test_data_splitting(self, sample_features_data):
        """Test data splitting works."""
        from src.data.splitter import DataSplitter
        
        # Use the actual constructor arguments
        splitter = DataSplitter()
        
        train, val, test = splitter.split(sample_features_data)
        
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0
        # Total should be <= original (some data may be used for gaps)
        assert len(train) + len(val) + len(test) <= len(sample_features_data)


class TestEnvironmentAgentIntegration:
    """Integration tests for environment-agent interaction."""
    
    def test_env_compatible_with_sb3(self, trading_env):
        """Test environment is compatible with Stable-Baselines3."""
        from stable_baselines3.common.env_checker import check_env
        
        # This will raise an exception if env is not compatible
        try:
            check_env(trading_env, warn=True)
        except Exception as e:
            pytest.fail(f"Environment not compatible with SB3: {e}")
    
    def test_env_with_random_policy(self, trading_env):
        """Test environment with random policy."""
        obs, _ = trading_env.reset()
        
        total_reward = 0
        done = False
        steps = 0
        
        while not done and steps < 100:
            action = trading_env.action_space.sample()
            obs, reward, terminated, truncated, info = trading_env.step(action)
            done = terminated or truncated
            total_reward += reward
            steps += 1
        
        assert steps > 0
        assert isinstance(total_reward, (int, float))
    
    def test_env_reset_consistency(self, trading_env):
        """Test environment reset is consistent."""
        obs1, info1 = trading_env.reset(seed=42)
        
        # Run some steps
        for _ in range(10):
            trading_env.step(1)
        
        # Reset again with same seed
        obs2, info2 = trading_env.reset(seed=42)
        
        np.testing.assert_array_almost_equal(obs1, obs2)
        assert info1['initial_balance'] == info2['initial_balance']


class TestBacktesterIntegration:
    """Integration tests for backtester."""
    
    def test_backtester_with_mock_model(self, trading_env):
        """Test backtester with a simple mock model."""
        from src.mlops.backtester import Backtester
        
        class MockModel:
            """Simple mock model that always holds."""
            def predict(self, obs, deterministic=True):
                return 0, None  # Always hold
        
        backtester = Backtester(
            transaction_cost=0.001,
            slippage=0.0005,
        )
        
        model = MockModel()
        result = backtester.run(model, trading_env)
        
        assert result is not None
        assert hasattr(result, 'total_return')
        assert hasattr(result, 'equity_curve')
        assert len(result.equity_curve) > 0
    
    def test_backtester_metrics(self, trading_env):
        """Test backtester calculates metrics correctly."""
        from src.mlops.backtester import Backtester
        
        class RandomModel:
            """Random model for testing."""
            def predict(self, obs, deterministic=True):
                return np.random.randint(0, 3), None
        
        backtester = Backtester()
        model = RandomModel()
        result = backtester.run(model, trading_env)
        
        # Check metrics are valid
        assert not np.isnan(result.total_return)
        assert not np.isnan(result.max_drawdown)
        assert 0 <= result.max_drawdown <= 1
        assert result.initial_balance == 10000.0


class TestMLflowIntegration:
    """Integration tests for MLflow tracking."""
    
    def test_experiment_tracker_initialization(self):
        """Test experiment tracker can be initialized."""
        from src.mlops.experiment_tracker import ExperimentTracker
        
        tracker = ExperimentTracker(experiment_name="test_experiment")
        
        assert tracker.experiment_id is not None
    
    def test_experiment_tracker_logging(self, tmp_path):
        """Test experiment tracker can log metrics."""
        import os
        from src.mlops.experiment_tracker import ExperimentTracker
        
        # Use temporary directory for mlruns
        os.environ['MLFLOW_TRACKING_URI'] = str(tmp_path / 'mlruns')
        
        tracker = ExperimentTracker(
            experiment_name="test_logging",
            tracking_uri=str(tmp_path / 'mlruns'),
        )
        
        tracker.start_run(run_name="test_run")
        tracker.log_params({"learning_rate": 0.001, "batch_size": 64})
        tracker.log_metric("test_metric", 0.5, step=1)
        tracker.end_run()
        
        # Run should complete without errors


class TestModelServiceIntegration:
    """Integration tests for model service."""
    
    def test_model_service_preprocessing(self):
        """Test model service preprocessing."""
        from api.model_service import ModelService
        
        service = ModelService(window_size=30, n_features=27)
        
        # Create sample OHLCV data
        n = 50
        ohlcv_data = {
            'open': [100.0 + i * 0.1 for i in range(n)],
            'high': [101.0 + i * 0.1 for i in range(n)],
            'low': [99.0 + i * 0.1 for i in range(n)],
            'close': [100.5 + i * 0.1 for i in range(n)],
            'volume': [1000000.0] * n,
        }
        
        state = service.preprocess_state(ohlcv_data)
        
        assert state is not None
        assert state.shape == (812,)  # window_size * n_features + 2
        assert state.dtype == np.float32
        assert not np.any(np.isnan(state))


class TestEndToEndPipeline:
    """End-to-end pipeline tests."""
    
    def test_data_to_environment_pipeline(self, sample_features_data):
        """Test data flows correctly to environment."""
        from src.env import TradingEnv
        
        # Create environment with sample data
        env = TradingEnv(
            df=sample_features_data,
            window_size=30,
            initial_balance=10000.0,
        )
        
        # Run episode
        obs, info = env.reset()
        assert obs is not None
        
        done = False
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        # Episode completed
        assert info['portfolio_value'] > 0
    
    def test_full_backtest_pipeline(self, sample_features_data):
        """Test full backtest pipeline."""
        from src.env import TradingEnv
        from src.mlops.backtester import Backtester, BacktestResult
        
        # Create environment
        env = TradingEnv(
            df=sample_features_data,
            window_size=30,
            initial_balance=10000.0,
            random_start=False,
        )
        
        # Create mock model
        class BuyAndHoldModel:
            """Buy and hold strategy."""
            def __init__(self):
                self.bought = False
            
            def predict(self, obs, deterministic=True):
                if not self.bought:
                    self.bought = True
                    return 1, None  # Buy
                return 0, None  # Hold
        
        # Run backtest
        backtester = Backtester()
        model = BuyAndHoldModel()
        result = backtester.run(model, env)
        
        assert isinstance(result, BacktestResult)
        assert len(result.equity_curve) > 1
        assert result.num_trades >= 1

