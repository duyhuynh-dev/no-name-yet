#!/usr/bin/env python
"""
Training script for PPO trading agent.

Usage:
    python scripts/train_agent.py --symbol SPY --timesteps 100000
    python scripts/train_agent.py --symbol SPY --timesteps 100000 --extractor lstm
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import logging

from src.env import TradingEnv
from src.agent import PPOTrainer, ModelEvaluator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Train PPO trading agent')
    
    # Data arguments
    parser.add_argument('--symbol', type=str, default='SPY',
                        help='Symbol to train on')
    parser.add_argument('--data-path', type=str, default=None,
                        help='Path to training data (parquet file)')
    
    # Training arguments
    parser.add_argument('--timesteps', type=int, default=100000,
                        help='Total training timesteps')
    parser.add_argument('--learning-rate', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--n-steps', type=int, default=2048,
                        help='Steps per rollout')
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--n-epochs', type=int, default=10,
                        help='Epochs per update')
    parser.add_argument('--gamma', type=float, default=0.99,
                        help='Discount factor')
    parser.add_argument('--clip-range', type=float, default=0.2,
                        help='PPO clip range')
    parser.add_argument('--ent-coef', type=float, default=0.01,
                        help='Entropy coefficient')
    
    # Network arguments
    parser.add_argument('--extractor', type=str, default='mlp',
                        choices=['lstm', 'mlp', 'attention'],
                        help='Feature extractor type')
    parser.add_argument('--features-dim', type=int, default=128,
                        help='Feature dimension')
    parser.add_argument('--lstm-hidden', type=int, default=128,
                        help='LSTM hidden size')
    parser.add_argument('--lstm-layers', type=int, default=2,
                        help='Number of LSTM layers')
    
    # Environment arguments
    parser.add_argument('--window-size', type=int, default=30,
                        help='Observation window size')
    parser.add_argument('--initial-balance', type=float, default=10000.0,
                        help='Initial balance')
    parser.add_argument('--transaction-cost', type=float, default=0.001,
                        help='Transaction cost (fraction)')
    
    # Output arguments
    parser.add_argument('--model-dir', type=str, default='models',
                        help='Directory to save models')
    parser.add_argument('--run-name', type=str, default=None,
                        help='Name for training run')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed')
    
    # Evaluation arguments
    parser.add_argument('--eval-episodes', type=int, default=5,
                        help='Number of evaluation episodes')
    parser.add_argument('--eval-freq', type=int, default=10000,
                        help='Evaluation frequency')
    
    args = parser.parse_args()
    
    # Load data
    if args.data_path:
        data_path = args.data_path
    else:
        data_path = f'data/processed/{args.symbol}_train.parquet'
    
    logger.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    
    # Get number of features
    n_features = len(df.columns)
    
    # Create training environment
    logger.info("Creating training environment...")
    train_env = TradingEnv(
        df=df,
        window_size=args.window_size,
        initial_balance=args.initial_balance,
        transaction_cost=args.transaction_cost,
        reward_type='risk_adjusted',
        normalize_obs=True,
        random_start=True,
    )
    
    # Create evaluation environment (no random start)
    eval_env = TradingEnv(
        df=df,
        window_size=args.window_size,
        initial_balance=args.initial_balance,
        transaction_cost=args.transaction_cost,
        reward_type='risk_adjusted',
        normalize_obs=True,
        random_start=False,
    )
    
    # Create trainer
    logger.info("Creating PPO trainer...")
    trainer = PPOTrainer(
        env=train_env,
        model_dir=args.model_dir,
        log_dir='logs',
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
        clip_range=args.clip_range,
        ent_coef=args.ent_coef,
        features_dim=args.features_dim,
        lstm_hidden_size=args.lstm_hidden,
        lstm_num_layers=args.lstm_layers,
        window_size=args.window_size,
        n_features=n_features,
        seed=args.seed,
    )
    
    # Train
    logger.info(f"Starting training for {args.timesteps} timesteps...")
    model = trainer.train(
        total_timesteps=args.timesteps,
        eval_env=eval_env,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.eval_episodes,
        run_name=args.run_name,
    )
    
    # Final evaluation
    logger.info("Running final evaluation...")
    evaluator = ModelEvaluator(model, eval_env)
    results = evaluator.evaluate(n_episodes=args.eval_episodes)
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total timesteps: {args.timesteps}")
    print(f"Mean reward: {results['mean_reward']:.2f}")
    print(f"Mean PnL: ${results['mean_pnl']:.2f}")
    print(f"Mean return: {results['mean_return']*100:.2f}%")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())

