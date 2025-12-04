#!/usr/bin/env python
"""
Training script with MLflow experiment tracking.

Usage:
    python scripts/train_with_mlflow.py --symbol SPY --timesteps 100000
    python scripts/train_with_mlflow.py --symbol SPY --timesteps 500000 --extractor lstm
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np

from src.env import TradingEnv
from src.agent import PPOTrainer, ModelEvaluator
from src.mlops import ExperimentTracker, Backtester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train PPO agent with MLflow tracking")
    
    # Data arguments
    parser.add_argument("--symbol", type=str, default="SPY", help="Trading symbol")
    parser.add_argument("--data-dir", type=str, default="data/processed", help="Data directory")
    
    # Training arguments
    parser.add_argument("--timesteps", type=int, default=100000, help="Total training timesteps")
    parser.add_argument("--extractor", type=str, default="mlp", choices=["mlp", "lstm"], help="Feature extractor")
    parser.add_argument("--learning-rate", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--n-steps", type=int, default=2048, help="Steps per rollout")
    parser.add_argument("--batch-size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--n-epochs", type=int, default=10, help="PPO epochs")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    
    # Network arguments
    parser.add_argument("--features-dim", type=int, default=128, help="Feature dimension")
    parser.add_argument("--lstm-hidden", type=int, default=128, help="LSTM hidden size")
    parser.add_argument("--lstm-layers", type=int, default=2, help="LSTM layers")
    
    # Environment arguments
    parser.add_argument("--window-size", type=int, default=30, help="Observation window")
    parser.add_argument("--initial-balance", type=float, default=10000.0, help="Initial balance")
    
    # Evaluation arguments
    parser.add_argument("--eval-episodes", type=int, default=5, help="Evaluation episodes")
    
    # MLflow arguments
    parser.add_argument("--experiment-name", type=str, default="hft-rl-trading", help="MLflow experiment name")
    parser.add_argument("--run-name", type=str, default=None, help="Run name")
    parser.add_argument("--tags", type=str, nargs="*", default=[], help="Tags (key=value)")
    
    # Output arguments
    parser.add_argument("--model-dir", type=str, default="models", help="Model save directory")
    
    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()
    
    # Load data
    train_path = Path(args.data_dir) / f"{args.symbol}_train.parquet"
    val_path = Path(args.data_dir) / f"{args.symbol}_val.parquet"
    test_path = Path(args.data_dir) / f"{args.symbol}_test.parquet"
    
    logger.info(f"Loading data from {train_path}")
    train_df = pd.read_parquet(train_path)
    val_df = pd.read_parquet(val_path) if val_path.exists() else None
    test_df = pd.read_parquet(test_path) if test_path.exists() else None
    
    n_features = len(train_df.columns)
    logger.info(f"Loaded {len(train_df)} training samples, {n_features} features")
    
    # Initialize MLflow tracker
    tracker = ExperimentTracker(experiment_name=args.experiment_name)
    
    # Parse tags
    tags = {}
    for tag in args.tags:
        if "=" in tag:
            key, value = tag.split("=", 1)
            tags[key] = value
    
    # Add default tags
    tags["symbol"] = args.symbol
    tags["extractor"] = args.extractor
    
    # Generate run name
    run_name = args.run_name or f"{args.extractor}_{args.symbol}_{args.timesteps // 1000}k"
    
    # Start MLflow run
    tracker.start_run(run_name=run_name, tags=tags)
    
    try:
        # Log hyperparameters
        params = {
            "symbol": args.symbol,
            "total_timesteps": args.timesteps,
            "extractor_type": args.extractor,
            "learning_rate": args.learning_rate,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "features_dim": args.features_dim,
            "window_size": args.window_size,
            "initial_balance": args.initial_balance,
            "n_features": n_features,
        }
        
        if args.extractor == "lstm":
            params["lstm_hidden_size"] = args.lstm_hidden
            params["lstm_num_layers"] = args.lstm_layers
        
        tracker.log_params(params)
        logger.info("Logged hyperparameters to MLflow")
        
        # Create training environment
        train_env = TradingEnv(
            df=train_df,
            window_size=args.window_size,
            initial_balance=args.initial_balance,
            normalize_obs=True,
            random_start=True,
        )
        
        # Create validation environment
        eval_env = None
        if val_df is not None:
            eval_env = TradingEnv(
                df=val_df,
                window_size=args.window_size,
                initial_balance=args.initial_balance,
                normalize_obs=True,
                random_start=False,
            )
        
        # Create trainer
        trainer = PPOTrainer(
            env=train_env,
            model_dir=args.model_dir,
            log_dir="logs",
            device="auto",
        )
        
        # Create model
        logger.info(f"Creating PPO model with {args.extractor} extractor...")
        trainer.create_model(
            extractor_type=args.extractor,
            learning_rate=args.learning_rate,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            features_dim=args.features_dim,
            lstm_hidden_size=args.lstm_hidden if args.extractor == "lstm" else 128,
            lstm_num_layers=args.lstm_layers if args.extractor == "lstm" else 1,
            window_size=args.window_size,
            n_features=n_features,
            verbose=1,
        )
        
        # Training callback to log to MLflow
        class MLflowCallback:
            def __init__(self, tracker, log_freq=1000):
                self.tracker = tracker
                self.log_freq = log_freq
                self.n_calls = 0
                self.episode_rewards = []
                self.episode_lengths = []
            
            def __call__(self, locals_dict, globals_dict):
                self.n_calls += 1
                
                # Extract episode info
                if "infos" in locals_dict:
                    for info in locals_dict["infos"]:
                        if "episode" in info:
                            self.episode_rewards.append(info["episode"]["r"])
                            self.episode_lengths.append(info["episode"]["l"])
                
                # Log at frequency
                if self.n_calls % self.log_freq == 0 and self.episode_rewards:
                    recent_rewards = self.episode_rewards[-100:]
                    recent_lengths = self.episode_lengths[-100:]
                    
                    self.tracker.log_metrics({
                        "train/mean_reward": np.mean(recent_rewards),
                        "train/mean_length": np.mean(recent_lengths),
                        "train/episodes": len(self.episode_rewards),
                    }, step=self.n_calls)
                
                return True
        
        callback = MLflowCallback(tracker, log_freq=2048)
        
        # Train
        logger.info(f"Starting training for {args.timesteps} timesteps...")
        model = trainer.train(
            total_timesteps=args.timesteps,
            run_name=run_name,
            eval_env=eval_env,
            eval_freq=10000 if eval_env else 0,
            callback=callback,
        )
        
        # Evaluate on training data
        logger.info("Evaluating on training data...")
        train_evaluator = ModelEvaluator(model, train_env)
        train_results = train_evaluator.evaluate(n_episodes=args.eval_episodes)
        tracker.log_evaluation_results(train_results, prefix="train_eval")
        
        # Evaluate on validation data
        if val_df is not None:
            logger.info("Evaluating on validation data...")
            val_evaluator = ModelEvaluator(model, eval_env)
            val_results = val_evaluator.evaluate(n_episodes=args.eval_episodes)
            tracker.log_evaluation_results(val_results, prefix="val_eval")
        
        # Evaluate on test data
        if test_df is not None:
            logger.info("Evaluating on test data...")
            test_env = TradingEnv(
                df=test_df,
                window_size=args.window_size,
                initial_balance=args.initial_balance,
                normalize_obs=True,
                random_start=False,
            )
            test_evaluator = ModelEvaluator(model, test_env)
            test_results = test_evaluator.evaluate(n_episodes=args.eval_episodes)
            tracker.log_evaluation_results(test_results, prefix="test_eval")
            
            # Run backtest
            logger.info("Running backtest on test data...")
            backtester = Backtester(
                transaction_cost=0.001,
                slippage=0.0005,
            )
            backtest_result = backtester.run(model, test_env)
            
            tracker.log_metrics({
                "backtest/total_return": backtest_result.total_return,
                "backtest/sharpe_ratio": backtest_result.sharpe_ratio,
                "backtest/max_drawdown": backtest_result.max_drawdown,
                "backtest/win_rate": backtest_result.win_rate,
                "backtest/num_trades": backtest_result.num_trades,
            })
        
        # Save model artifact
        model_path = Path(args.model_dir) / run_name / "final_model.zip"
        if model_path.exists():
            tracker.log_artifact(str(model_path), artifact_path="model")
        
        # Log final summary
        final_metrics = {
            "final/mean_reward": train_results["mean_reward"],
            "final/mean_pnl": train_results["mean_pnl"],
            "final/mean_return": train_results["mean_return"],
        }
        if test_df is not None:
            final_metrics["final/test_return"] = backtest_result.total_return
            final_metrics["final/test_sharpe"] = backtest_result.sharpe_ratio
        
        tracker.log_metrics(final_metrics)
        
        # Print summary
        print("\n" + "=" * 60)
        print("TRAINING COMPLETE WITH MLFLOW")
        print("=" * 60)
        print(f"Run ID: {tracker.run_id}")
        print(f"Run Name: {run_name}")
        print(f"Total Timesteps: {args.timesteps}")
        print(f"Mean Reward: {train_results['mean_reward']:.2f}")
        print(f"Mean PnL: ${train_results['mean_pnl']:.2f}")
        if test_df is not None:
            print(f"Test Return: {backtest_result.total_return:.2%}")
            print(f"Test Sharpe: {backtest_result.sharpe_ratio:.3f}")
        print("=" * 60)
        print(f"\nView in MLflow UI: mlflow ui --backend-store-uri ./mlruns")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        tracker.end_run(status="FAILED")
        raise
    
    finally:
        tracker.end_run()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

