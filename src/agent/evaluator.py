"""
Model Evaluator for Trading Agents.

Provides utilities for evaluating trained models:
- Backtesting
- Performance metrics (Sharpe, Sortino, Max Drawdown)
- Visualization
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import json
import logging

from stable_baselines3 import PPO

logger = logging.getLogger(__name__)


class ModelEvaluator:
    """
    Evaluator for trading models.
    
    Provides comprehensive evaluation including:
    - Episode statistics
    - Risk metrics
    - Trade analysis
    - Performance visualization
    """
    
    def __init__(self, model: PPO, env):
        """
        Initialize the evaluator.
        
        Args:
            model: Trained PPO model
            env: Trading environment
        """
        self.model = model
        self.env = env
        self.results: List[Dict] = []
    
    def evaluate(
        self,
        n_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False,
        verbose: bool = True
    ) -> Dict:
        """
        Evaluate the model over multiple episodes.
        
        Args:
            n_episodes: Number of episodes to run
            deterministic: Whether to use deterministic policy
            render: Whether to render episodes
            verbose: Whether to print progress
        
        Returns:
            Dictionary with evaluation results
        """
        episode_rewards = []
        episode_lengths = []
        episode_pnls = []
        episode_returns = []
        episode_sharpes = []
        episode_max_drawdowns = []
        episode_n_trades = []
        episode_win_rates = []
        all_trades = []
        
        for episode in range(n_episodes):
            obs, info = self.env.reset()
            done = False
            episode_reward = 0
            step = 0
            
            while not done:
                action, _ = self.model.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated
                episode_reward += reward
                step += 1
                
                if render:
                    self.env.render()
            
            # Get episode stats
            stats = self.env.get_episode_stats()
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(step)
            episode_pnls.append(stats['total_pnl'])
            episode_returns.append(stats['total_return'])
            episode_sharpes.append(stats['sharpe_ratio'])
            episode_max_drawdowns.append(stats['max_drawdown'])
            episode_n_trades.append(stats['n_trades'])
            episode_win_rates.append(stats['win_rate'])
            all_trades.extend(self.env.trades)
            
            if verbose:
                logger.info(
                    f"Episode {episode + 1}/{n_episodes}: "
                    f"reward={episode_reward:.2f}, "
                    f"PnL=${stats['total_pnl']:.2f}, "
                    f"return={stats['total_return']*100:.2f}%"
                )
        
        # Aggregate results
        results = {
            # Reward statistics
            "mean_reward": np.mean(episode_rewards),
            "std_reward": np.std(episode_rewards),
            "min_reward": np.min(episode_rewards),
            "max_reward": np.max(episode_rewards),
            
            # Episode length
            "mean_episode_length": np.mean(episode_lengths),
            
            # PnL statistics
            "mean_pnl": np.mean(episode_pnls),
            "std_pnl": np.std(episode_pnls),
            "total_pnl": np.sum(episode_pnls),
            
            # Return statistics
            "mean_return": np.mean(episode_returns),
            "std_return": np.std(episode_returns),
            
            # Risk metrics
            "mean_sharpe": np.mean(episode_sharpes),
            "mean_max_drawdown": np.mean(episode_max_drawdowns),
            "worst_max_drawdown": np.max(episode_max_drawdowns),
            
            # Trading statistics
            "mean_n_trades": np.mean(episode_n_trades),
            "total_trades": sum(episode_n_trades),
            "mean_win_rate": np.mean(episode_win_rates),
            
            # Raw data
            "episode_rewards": episode_rewards,
            "episode_pnls": episode_pnls,
            "episode_returns": episode_returns,
            "n_episodes": n_episodes,
        }
        
        # Trade analysis
        if all_trades:
            trade_pnls = [t.get('pnl', 0) for t in all_trades if 'pnl' in t]
            if trade_pnls:
                results["trade_mean_pnl"] = np.mean(trade_pnls)
                results["trade_std_pnl"] = np.std(trade_pnls)
                results["profitable_trades"] = sum(1 for p in trade_pnls if p > 0)
                results["losing_trades"] = sum(1 for p in trade_pnls if p < 0)
        
        self.results.append(results)
        
        if verbose:
            self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict):
        """Print evaluation summary."""
        print("\n" + "=" * 60)
        print("EVALUATION SUMMARY")
        print("=" * 60)
        print(f"Episodes: {results['n_episodes']}")
        print(f"\nReward Statistics:")
        print(f"  Mean: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Range: [{results['min_reward']:.2f}, {results['max_reward']:.2f}]")
        print(f"\nPnL Statistics:")
        print(f"  Mean: ${results['mean_pnl']:.2f}")
        print(f"  Total: ${results['total_pnl']:.2f}")
        print(f"\nReturn Statistics:")
        print(f"  Mean: {results['mean_return']*100:.2f}%")
        print(f"\nRisk Metrics:")
        print(f"  Mean Sharpe: {results['mean_sharpe']:.4f}")
        print(f"  Mean Max Drawdown: {results['mean_max_drawdown']*100:.2f}%")
        print(f"  Worst Max Drawdown: {results['worst_max_drawdown']*100:.2f}%")
        print(f"\nTrading Statistics:")
        print(f"  Mean Trades/Episode: {results['mean_n_trades']:.1f}")
        print(f"  Mean Win Rate: {results['mean_win_rate']*100:.1f}%")
        print("=" * 60)
    
    def backtest(
        self,
        df: pd.DataFrame,
        deterministic: bool = True,
        verbose: bool = True
    ) -> Dict:
        """
        Run a backtest on historical data.
        
        Args:
            df: DataFrame with market data
            deterministic: Whether to use deterministic policy
            verbose: Whether to print progress
        
        Returns:
            Backtest results
        """
        from src.env import TradingEnv
        
        # Create environment with the data
        backtest_env = TradingEnv(
            df=df,
            window_size=30,
            initial_balance=10000.0,
            random_start=False,  # Start from beginning for backtest
        )
        
        # Run episode
        obs, info = backtest_env.reset()
        done = False
        
        portfolio_values = [backtest_env.portfolio_value]
        positions = [backtest_env.position]
        actions = []
        rewards = []
        
        while not done:
            action, _ = self.model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = backtest_env.step(action)
            done = terminated or truncated
            
            portfolio_values.append(info['portfolio_value'])
            positions.append(info['position'])
            actions.append(action)
            rewards.append(reward)
        
        # Calculate metrics
        returns = np.diff(portfolio_values) / portfolio_values[:-1]
        
        results = {
            "portfolio_values": portfolio_values,
            "positions": positions,
            "actions": actions,
            "rewards": rewards,
            "returns": returns.tolist(),
            
            # Performance metrics
            "total_return": (portfolio_values[-1] - portfolio_values[0]) / portfolio_values[0],
            "total_pnl": portfolio_values[-1] - portfolio_values[0],
            "sharpe_ratio": self._calculate_sharpe(returns),
            "sortino_ratio": self._calculate_sortino(returns),
            "max_drawdown": self._calculate_max_drawdown(portfolio_values),
            "calmar_ratio": self._calculate_calmar(returns, portfolio_values),
            
            # Trade statistics
            "n_trades": len(backtest_env.trades),
            "trades": backtest_env.trades,
            
            # Episode stats
            **backtest_env.get_episode_stats(),
        }
        
        if verbose:
            print("\n" + "=" * 60)
            print("BACKTEST RESULTS")
            print("=" * 60)
            print(f"Total Return: {results['total_return']*100:.2f}%")
            print(f"Total PnL: ${results['total_pnl']:.2f}")
            print(f"Sharpe Ratio: {results['sharpe_ratio']:.4f}")
            print(f"Sortino Ratio: {results['sortino_ratio']:.4f}")
            print(f"Max Drawdown: {results['max_drawdown']*100:.2f}%")
            print(f"Calmar Ratio: {results['calmar_ratio']:.4f}")
            print(f"Number of Trades: {results['n_trades']}")
            print("=" * 60)
        
        return results
    
    def _calculate_sharpe(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate Sharpe ratio."""
        excess_returns = returns - risk_free_rate
        if np.std(excess_returns) == 0:
            return 0.0
        return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(252)  # Annualized
    
    def _calculate_sortino(self, returns: np.ndarray, risk_free_rate: float = 0.0) -> float:
        """Calculate Sortino ratio (only penalizes downside volatility)."""
        excess_returns = returns - risk_free_rate
        downside_returns = excess_returns[excess_returns < 0]
        if len(downside_returns) == 0 or np.std(downside_returns) == 0:
            return 0.0
        return np.mean(excess_returns) / np.std(downside_returns) * np.sqrt(252)
    
    def _calculate_max_drawdown(self, portfolio_values: List[float]) -> float:
        """Calculate maximum drawdown."""
        peak = portfolio_values[0]
        max_dd = 0.0
        
        for value in portfolio_values:
            if value > peak:
                peak = value
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def _calculate_calmar(self, returns: np.ndarray, portfolio_values: List[float]) -> float:
        """Calculate Calmar ratio (return / max drawdown)."""
        max_dd = self._calculate_max_drawdown(portfolio_values)
        if max_dd == 0:
            return 0.0
        annual_return = np.mean(returns) * 252
        return annual_return / max_dd
    
    def compare_models(
        self,
        models: Dict[str, PPO],
        n_episodes: int = 10,
        deterministic: bool = True
    ) -> pd.DataFrame:
        """
        Compare multiple models.
        
        Args:
            models: Dictionary of model_name -> PPO model
            n_episodes: Number of episodes per model
            deterministic: Whether to use deterministic policy
        
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for name, model in models.items():
            logger.info(f"Evaluating {name}...")
            self.model = model
            eval_results = self.evaluate(n_episodes=n_episodes, deterministic=deterministic, verbose=False)
            eval_results['model_name'] = name
            results.append(eval_results)
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame([
            {
                'Model': r['model_name'],
                'Mean Reward': r['mean_reward'],
                'Mean PnL': r['mean_pnl'],
                'Mean Return (%)': r['mean_return'] * 100,
                'Sharpe': r['mean_sharpe'],
                'Max Drawdown (%)': r['mean_max_drawdown'] * 100,
                'Win Rate (%)': r['mean_win_rate'] * 100,
                'Trades/Episode': r['mean_n_trades'],
            }
            for r in results
        ])
        
        return comparison_df
    
    def save_results(self, path: str):
        """Save evaluation results to file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'w') as f:
            json.dump(self.results, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        
        logger.info(f"Results saved to {path}")


def evaluate_model(
    model: PPO,
    env,
    n_episodes: int = 10,
    deterministic: bool = True
) -> Dict:
    """
    Convenience function to evaluate a model.
    
    Args:
        model: PPO model
        env: Trading environment
        n_episodes: Number of episodes
        deterministic: Whether to use deterministic policy
    
    Returns:
        Evaluation results
    """
    evaluator = ModelEvaluator(model, env)
    return evaluator.evaluate(n_episodes=n_episodes, deterministic=deterministic)

