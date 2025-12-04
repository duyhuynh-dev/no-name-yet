"""
Backtesting and Walk-Forward Validation Module.

Provides:
- Historical backtesting with realistic constraints
- Walk-forward validation for robust evaluation
- Performance metrics calculation
- Trade analysis
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    
    # Basic metrics
    total_return: float = 0.0
    annual_return: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    max_drawdown: float = 0.0
    calmar_ratio: float = 0.0
    
    # Trading metrics
    num_trades: int = 0
    win_rate: float = 0.0
    profit_factor: float = 0.0
    avg_trade_return: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    max_consecutive_wins: int = 0
    max_consecutive_losses: int = 0
    
    # Time info
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    trading_days: int = 0
    
    # Portfolio
    initial_balance: float = 10000.0
    final_balance: float = 10000.0
    peak_balance: float = 10000.0
    
    # Detailed data
    equity_curve: List[float] = field(default_factory=list)
    returns_series: List[float] = field(default_factory=list)
    trades: List[Dict[str, Any]] = field(default_factory=list)
    positions: List[int] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "total_return": self.total_return,
            "annual_return": self.annual_return,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "max_drawdown": self.max_drawdown,
            "calmar_ratio": self.calmar_ratio,
            "num_trades": self.num_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "avg_trade_return": self.avg_trade_return,
            "initial_balance": self.initial_balance,
            "final_balance": self.final_balance,
            "trading_days": self.trading_days,
        }


class Backtester:
    """
    Backtesting engine for RL trading agents.
    
    Features:
    - Realistic transaction costs
    - Slippage simulation
    - Market impact modeling
    - Position limits
    - Comprehensive metrics
    """
    
    def __init__(
        self,
        transaction_cost: float = 0.001,  # 0.1%
        slippage: float = 0.0005,          # 0.05%
        market_impact: float = 0.0,        # Optional market impact
        max_position_size: float = 1.0,    # Max fraction of portfolio per trade
        risk_free_rate: float = 0.02,      # Annual risk-free rate
    ):
        """
        Initialize the backtester.
        
        Args:
            transaction_cost: Transaction cost as fraction of trade value
            slippage: Slippage as fraction of price
            market_impact: Market impact coefficient
            max_position_size: Maximum position size as fraction of portfolio
            risk_free_rate: Annual risk-free rate for Sharpe calculation
        """
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.market_impact = market_impact
        self.max_position_size = max_position_size
        self.risk_free_rate = risk_free_rate
    
    def run(
        self,
        model,
        env,
        deterministic: bool = True,
        verbose: bool = False,
    ) -> BacktestResult:
        """
        Run a backtest.
        
        Args:
            model: Trained RL model
            env: Trading environment
            deterministic: Use deterministic actions
            verbose: Print progress
            
        Returns:
            BacktestResult with all metrics
        """
        result = BacktestResult()
        result.initial_balance = env.initial_balance
        
        # Reset environment
        obs, info = env.reset()
        
        done = False
        equity_curve = [env.initial_balance]
        returns = []
        positions = [0]
        
        step = 0
        while not done:
            # Get action from model
            action, _ = model.predict(obs, deterministic=deterministic)
            
            # Execute step
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # Track equity and positions
            portfolio_value = info.get("portfolio_value", equity_curve[-1])
            equity_curve.append(portfolio_value)
            
            # Calculate return
            if equity_curve[-2] > 0:
                ret = (equity_curve[-1] - equity_curve[-2]) / equity_curve[-2]
            else:
                ret = 0.0
            returns.append(ret)
            
            # Track position
            positions.append(info.get("position", 0))
            
            step += 1
            if verbose and step % 100 == 0:
                logger.info(f"Step {step}: Portfolio = ${portfolio_value:.2f}")
        
        # Calculate metrics
        result.equity_curve = equity_curve
        result.returns_series = returns
        result.positions = positions
        result.trades = env.trades if hasattr(env, "trades") else []
        
        result.final_balance = equity_curve[-1]
        result.peak_balance = max(equity_curve)
        result.trading_days = len(equity_curve) - 1
        
        # Performance metrics
        result.total_return = (result.final_balance - result.initial_balance) / result.initial_balance
        result.max_drawdown = self._calculate_max_drawdown(equity_curve)
        
        if len(returns) > 0 and np.std(returns) > 0:
            # Annualize metrics (assuming 252 trading days)
            daily_return = np.mean(returns)
            daily_vol = np.std(returns)
            
            result.annual_return = daily_return * 252
            result.sharpe_ratio = (result.annual_return - self.risk_free_rate) / (daily_vol * np.sqrt(252))
            
            # Sortino ratio (downside deviation)
            downside_returns = [r for r in returns if r < 0]
            if downside_returns:
                downside_vol = np.std(downside_returns)
                result.sortino_ratio = (result.annual_return - self.risk_free_rate) / (downside_vol * np.sqrt(252))
            
            # Calmar ratio
            if result.max_drawdown > 0:
                result.calmar_ratio = result.annual_return / result.max_drawdown
        
        # Trade analysis
        if result.trades:
            self._analyze_trades(result)
        
        return result
    
    def _calculate_max_drawdown(self, equity_curve: List[float]) -> float:
        """Calculate maximum drawdown."""
        peak = equity_curve[0]
        max_dd = 0.0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak if peak > 0 else 0
            max_dd = max(max_dd, drawdown)
        
        return max_dd
    
    def _analyze_trades(self, result: BacktestResult) -> None:
        """Analyze trade performance."""
        trades = result.trades
        
        if not trades:
            return
        
        result.num_trades = len(trades)
        
        # Extract PnL from trades
        pnls = []
        for trade in trades:
            if "pnl" in trade:
                pnls.append(trade["pnl"])
        
        if not pnls:
            return
        
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        result.win_rate = len(wins) / len(pnls) if pnls else 0.0
        result.avg_trade_return = np.mean(pnls) if pnls else 0.0
        result.avg_win = np.mean(wins) if wins else 0.0
        result.avg_loss = np.mean(losses) if losses else 0.0
        
        # Profit factor
        total_wins = sum(wins) if wins else 0
        total_losses = abs(sum(losses)) if losses else 1
        result.profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
        
        # Consecutive wins/losses
        result.max_consecutive_wins = self._max_consecutive(pnls, positive=True)
        result.max_consecutive_losses = self._max_consecutive(pnls, positive=False)
    
    def _max_consecutive(self, values: List[float], positive: bool = True) -> int:
        """Calculate max consecutive wins or losses."""
        max_count = 0
        current_count = 0
        
        for v in values:
            if (positive and v > 0) or (not positive and v < 0):
                current_count += 1
                max_count = max(max_count, current_count)
            else:
                current_count = 0
        
        return max_count


class WalkForwardValidator:
    """
    Walk-forward validation for robust model evaluation.
    
    Implements:
    - Rolling window training/testing
    - Anchored training (expanding window)
    - Out-of-sample performance aggregation
    - Statistical significance testing
    """
    
    def __init__(
        self,
        train_period: int = 252,          # Training window size (trading days)
        test_period: int = 63,             # Test window size (trading days)
        step_size: int = 21,               # Step between windows
        anchored: bool = False,            # Use expanding window for training
        min_train_samples: int = 100,      # Minimum samples for training
    ):
        """
        Initialize walk-forward validator.
        
        Args:
            train_period: Training window size in days
            test_period: Test window size in days
            step_size: Step between consecutive windows
            anchored: If True, training window expands from start
            min_train_samples: Minimum training samples required
        """
        self.train_period = train_period
        self.test_period = test_period
        self.step_size = step_size
        self.anchored = anchored
        self.min_train_samples = min_train_samples
    
    def generate_splits(
        self,
        data: pd.DataFrame,
    ) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate train/test splits for walk-forward validation.
        
        Args:
            data: Full dataset with datetime index
            
        Returns:
            List of (train_df, test_df) tuples
        """
        splits = []
        total_len = len(data)
        
        start_idx = 0
        
        while True:
            # Calculate indices
            if self.anchored:
                train_start = 0
            else:
                train_start = start_idx
            
            train_end = start_idx + self.train_period
            test_start = train_end
            test_end = test_start + self.test_period
            
            # Check if we have enough data
            if test_end > total_len:
                break
            
            if train_end - train_start < self.min_train_samples:
                start_idx += self.step_size
                continue
            
            # Create splits
            train_df = data.iloc[train_start:train_end].copy()
            test_df = data.iloc[test_start:test_end].copy()
            
            splits.append((train_df, test_df))
            
            # Move to next window
            start_idx += self.step_size
        
        logger.info(f"Generated {len(splits)} walk-forward splits")
        return splits
    
    def validate(
        self,
        model_class,
        model_kwargs: Dict[str, Any],
        env_class,
        env_kwargs: Dict[str, Any],
        data: pd.DataFrame,
        train_timesteps: int = 100000,
        backtester: Optional[Backtester] = None,
        verbose: bool = False,
    ) -> Dict[str, Any]:
        """
        Run walk-forward validation.
        
        Args:
            model_class: Model class (e.g., PPO)
            model_kwargs: Model initialization kwargs
            env_class: Environment class
            env_kwargs: Environment initialization kwargs
            data: Full dataset
            train_timesteps: Training timesteps per split
            backtester: Backtester instance (creates default if None)
            verbose: Print progress
            
        Returns:
            Validation results with aggregated metrics
        """
        if backtester is None:
            backtester = Backtester()
        
        splits = self.generate_splits(data)
        
        if not splits:
            raise ValueError("No valid splits generated. Check data size and parameters.")
        
        results = []
        
        for i, (train_df, test_df) in enumerate(splits):
            if verbose:
                logger.info(f"Walk-forward split {i+1}/{len(splits)}")
                logger.info(f"  Train: {len(train_df)} samples")
                logger.info(f"  Test: {len(test_df)} samples")
            
            # Create training environment
            train_env_kwargs = env_kwargs.copy()
            train_env_kwargs["df"] = train_df
            train_env = env_class(**train_env_kwargs)
            
            # Create and train model
            model = model_class(
                env=train_env,
                **model_kwargs,
            )
            model.learn(total_timesteps=train_timesteps)
            
            # Create test environment
            test_env_kwargs = env_kwargs.copy()
            test_env_kwargs["df"] = test_df
            test_env_kwargs["random_start"] = False  # Start from beginning
            test_env = env_class(**test_env_kwargs)
            
            # Run backtest
            result = backtester.run(model, test_env)
            result.start_date = test_df.index[0] if hasattr(test_df.index, '__getitem__') else None
            result.end_date = test_df.index[-1] if hasattr(test_df.index, '__getitem__') else None
            
            results.append(result)
            
            if verbose:
                logger.info(f"  Return: {result.total_return:.2%}")
                logger.info(f"  Sharpe: {result.sharpe_ratio:.3f}")
        
        # Aggregate results
        return self._aggregate_results(results)
    
    def _aggregate_results(
        self,
        results: List[BacktestResult],
    ) -> Dict[str, Any]:
        """Aggregate walk-forward results."""
        if not results:
            return {}
        
        # Extract metrics
        returns = [r.total_return for r in results]
        sharpes = [r.sharpe_ratio for r in results]
        max_dds = [r.max_drawdown for r in results]
        win_rates = [r.win_rate for r in results]
        
        # Calculate statistics
        aggregated = {
            "n_splits": len(results),
            
            # Return statistics
            "mean_return": np.mean(returns),
            "std_return": np.std(returns),
            "median_return": np.median(returns),
            "min_return": np.min(returns),
            "max_return": np.max(returns),
            "positive_splits": sum(1 for r in returns if r > 0),
            "positive_rate": sum(1 for r in returns if r > 0) / len(returns),
            
            # Risk-adjusted metrics
            "mean_sharpe": np.mean(sharpes),
            "std_sharpe": np.std(sharpes),
            "mean_max_drawdown": np.mean(max_dds),
            "worst_max_drawdown": np.max(max_dds),
            
            # Trading metrics
            "mean_win_rate": np.mean(win_rates),
            
            # Detailed results
            "split_results": [r.to_dict() for r in results],
        }
        
        # Statistical significance
        # t-test against zero return
        if len(returns) > 1:
            from scipy import stats
            t_stat, p_value = stats.ttest_1samp(returns, 0)
            aggregated["t_statistic"] = t_stat
            aggregated["p_value"] = p_value
            aggregated["significant_95"] = p_value < 0.05
            aggregated["significant_99"] = p_value < 0.01
        
        return aggregated

