"""
Training Pipeline for PPO Agent.

Provides utilities for training PPO agents on the trading environment.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Callable, List, Any
import numpy as np
import pandas as pd
import torch
import logging

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
    CallbackList,
)
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from .networks import create_policy_kwargs

logger = logging.getLogger(__name__)


class TradingCallback(BaseCallback):
    """
    Custom callback for logging trading-specific metrics.
    """
    
    def __init__(
        self,
        log_freq: int = 1000,
        verbose: int = 1,
        custom_callback: Optional[Callable] = None
    ):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.portfolio_values: List[float] = []
        self.n_trades: List[int] = []
        self.custom_callback = custom_callback
    
    def _on_step(self) -> bool:
        # Log episode info when available
        if len(self.model.ep_info_buffer) > 0:
            ep_info = self.model.ep_info_buffer[-1]
            self.episode_rewards.append(ep_info.get('r', 0))
            self.episode_lengths.append(ep_info.get('l', 0))
        
        # Log every log_freq steps
        if self.n_calls % self.log_freq == 0 and self.verbose > 0:
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-100:])
                logger.info(
                    f"Step {self.n_calls}: "
                    f"mean_reward={mean_reward:.2f}, "
                    f"episodes={len(self.episode_rewards)}"
                )
        
        # Call custom callback if provided
        if self.custom_callback is not None:
            self.custom_callback(self.locals, self.globals)
        
        return True
    
    def _on_training_end(self):
        logger.info(f"Training complete: {len(self.episode_rewards)} episodes")


class PPOTrainer:
    """
    Trainer class for PPO agent on trading environment.
    
    Handles:
    - Model initialization with custom networks
    - Training with callbacks
    - Checkpoint saving/loading
    - Hyperparameter configuration
    """
    
    def __init__(
        self,
        env,
        model_dir: str = "models",
        log_dir: str = "logs",
        device: str = "auto",
    ):
        """
        Initialize the trainer.
        
        Args:
            env: Trading environment (or environment factory)
            model_dir: Directory to save models
            log_dir: Directory for tensorboard logs
            device: Device to use ("auto", "cpu", "cuda", "mps")
        """
        self.env = env
        self.model_dir = Path(model_dir)
        self.log_dir = Path(log_dir)
        self.device = self._get_device(device)
        
        # Create directories
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.model: Optional[PPO] = None
        self.training_history: List[Dict] = []
        
        logger.info(f"PPOTrainer initialized: device={self.device}")
    
    def _get_device(self, device: str) -> str:
        """Determine the best device to use."""
        if device == "auto":
            if torch.cuda.is_available():
                return "cuda"
            elif torch.backends.mps.is_available():
                return "mps"
            else:
                return "cpu"
        return device
    
    def create_model(
        self,
        policy: str = "MlpPolicy",
        extractor_type: str = "lstm",
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: Optional[float] = None,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        features_dim: int = 128,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        window_size: int = 30,
        n_features: int = 27,
        verbose: int = 1,
        seed: Optional[int] = None,
        **kwargs
    ) -> PPO:
        """
        Create a PPO model with custom configuration.
        
        Args:
            policy: Policy type ("MlpPolicy" for custom feature extractors)
            extractor_type: Type of feature extractor ("lstm", "mlp", "attention")
            learning_rate: Learning rate
            n_steps: Number of steps per rollout
            batch_size: Minibatch size
            n_epochs: Number of epochs per update
            gamma: Discount factor
            gae_lambda: GAE lambda
            clip_range: PPO clipping parameter
            clip_range_vf: Value function clipping (None to disable)
            ent_coef: Entropy coefficient
            vf_coef: Value function coefficient
            max_grad_norm: Max gradient norm for clipping
            features_dim: Feature extractor output dimension
            lstm_hidden_size: LSTM hidden size
            lstm_num_layers: Number of LSTM layers
            window_size: Observation window size
            n_features: Number of features
            verbose: Verbosity level
            seed: Random seed
            **kwargs: Additional PPO kwargs
        
        Returns:
            PPO model
        """
        # Create policy kwargs
        policy_kwargs = create_policy_kwargs(
            extractor_type=extractor_type,
            features_dim=features_dim,
            lstm_hidden_size=lstm_hidden_size,
            lstm_num_layers=lstm_num_layers,
            window_size=window_size,
            n_features=n_features,
        )
        
        # Create model
        self.model = PPO(
            policy=policy,
            env=self.env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            tensorboard_log=str(self.log_dir),
            seed=seed,
            device=self.device,
            **kwargs
        )
        
        # Store config
        self.config = {
            "policy": policy,
            "extractor_type": extractor_type,
            "learning_rate": learning_rate,
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": n_epochs,
            "gamma": gamma,
            "gae_lambda": gae_lambda,
            "clip_range": clip_range,
            "ent_coef": ent_coef,
            "vf_coef": vf_coef,
            "features_dim": features_dim,
            "lstm_hidden_size": lstm_hidden_size,
            "lstm_num_layers": lstm_num_layers,
            "window_size": window_size,
            "n_features": n_features,
            "device": self.device,
        }
        
        logger.info(f"PPO model created with {extractor_type} feature extractor")
        return self.model
    
    def train(
        self,
        total_timesteps: int,
        eval_env=None,
        eval_freq: int = 10000,
        n_eval_episodes: int = 5,
        save_freq: int = 50000,
        log_freq: int = 1000,
        run_name: Optional[str] = None,
        reset_num_timesteps: bool = True,
        callback=None,
    ) -> PPO:
        """
        Train the PPO model.
        
        Args:
            total_timesteps: Total timesteps to train
            eval_env: Environment for evaluation (optional)
            eval_freq: Evaluation frequency
            n_eval_episodes: Number of evaluation episodes
            save_freq: Checkpoint save frequency
            log_freq: Logging frequency
            run_name: Name for this training run
            reset_num_timesteps: Whether to reset timestep counter
            callback: Additional callback function (receives locals, globals)
        
        Returns:
            Trained PPO model
        """
        if self.model is None:
            raise ValueError("Model not created. Call create_model() first.")
        
        # Generate run name
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        run_dir = self.model_dir / run_name
        run_dir.mkdir(parents=True, exist_ok=True)
        
        # Create callbacks
        callbacks = []
        
        # Checkpoint callback
        checkpoint_callback = CheckpointCallback(
            save_freq=save_freq,
            save_path=str(run_dir / "checkpoints"),
            name_prefix="ppo_trading",
            save_replay_buffer=False,
            save_vecnormalize=True,
        )
        callbacks.append(checkpoint_callback)
        
        # Evaluation callback
        if eval_env is not None:
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=str(run_dir / "best_model"),
                log_path=str(run_dir / "eval_logs"),
                eval_freq=eval_freq,
                n_eval_episodes=n_eval_episodes,
                deterministic=True,
            )
            callbacks.append(eval_callback)
        
        # Custom trading callback
        trading_callback = TradingCallback(log_freq=log_freq, custom_callback=callback)
        callbacks.append(trading_callback)
        
        combined_callback = CallbackList(callbacks)
        
        # Save config
        config_path = run_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)
        
        logger.info(f"Starting training: {total_timesteps} timesteps, run={run_name}")
        
        # Train
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=combined_callback,
            reset_num_timesteps=reset_num_timesteps,
            tb_log_name=run_name,
        )
        
        # Save final model
        final_model_path = run_dir / "final_model"
        self.model.save(str(final_model_path))
        logger.info(f"Final model saved to {final_model_path}")
        
        # Save training history
        history = {
            "total_timesteps": total_timesteps,
            "run_name": run_name,
            "episode_rewards": trading_callback.episode_rewards,
            "episode_lengths": trading_callback.episode_lengths,
            "config": self.config,
        }
        self.training_history.append(history)
        
        history_path = run_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2, default=lambda x: float(x) if isinstance(x, np.floating) else x)
        
        return self.model
    
    def load(self, path: str, env=None) -> PPO:
        """
        Load a trained model.
        
        Args:
            path: Path to model file
            env: Environment (optional, uses self.env if not provided)
        
        Returns:
            Loaded PPO model
        """
        if env is None:
            env = self.env
        
        self.model = PPO.load(path, env=env, device=self.device)
        logger.info(f"Model loaded from {path}")
        
        return self.model
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True
    ) -> tuple:
        """
        Make a prediction.
        
        Args:
            observation: Current observation
            deterministic: Whether to use deterministic policy
        
        Returns:
            Tuple of (action, state)
        """
        if self.model is None:
            raise ValueError("Model not loaded")
        
        return self.model.predict(observation, deterministic=deterministic)


def train_ppo_agent(
    env,
    total_timesteps: int = 100000,
    eval_env=None,
    extractor_type: str = "lstm",
    learning_rate: float = 3e-4,
    n_steps: int = 2048,
    batch_size: int = 64,
    model_dir: str = "models",
    run_name: Optional[str] = None,
    **kwargs
) -> PPO:
    """
    Convenience function to train a PPO agent.
    
    Args:
        env: Trading environment
        total_timesteps: Total timesteps to train
        eval_env: Evaluation environment
        extractor_type: Feature extractor type
        learning_rate: Learning rate
        n_steps: Steps per rollout
        batch_size: Batch size
        model_dir: Directory for saving models
        run_name: Name for training run
        **kwargs: Additional arguments
    
    Returns:
        Trained PPO model
    """
    trainer = PPOTrainer(env, model_dir=model_dir)
    
    trainer.create_model(
        extractor_type=extractor_type,
        learning_rate=learning_rate,
        n_steps=n_steps,
        batch_size=batch_size,
        **kwargs
    )
    
    model = trainer.train(
        total_timesteps=total_timesteps,
        eval_env=eval_env,
        run_name=run_name,
    )
    
    return model

