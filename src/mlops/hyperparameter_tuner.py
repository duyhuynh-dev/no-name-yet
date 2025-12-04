"""
Hyperparameter Tuning Module.

Provides:
- Grid search
- Random search
- Optuna integration (optional)
- MLflow integration for tracking
"""

import logging
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple
import random

import numpy as np
import pandas as pd

from .experiment_tracker import ExperimentTracker

logger = logging.getLogger(__name__)


class HyperparameterTuner:
    """
    Hyperparameter tuner with MLflow tracking.
    
    Supports:
    - Grid search
    - Random search
    - Custom search strategies
    """
    
    def __init__(
        self,
        param_space: Dict[str, List[Any]],
        objective_fn: Callable[[Dict[str, Any]], float],
        tracker: Optional[ExperimentTracker] = None,
        n_trials: Optional[int] = None,
        search_strategy: str = "random",
        maximize: bool = True,
    ):
        """
        Initialize the tuner.
        
        Args:
            param_space: Dictionary of parameter_name -> list of values
            objective_fn: Function that takes params dict and returns metric
            tracker: MLflow tracker for logging
            n_trials: Number of trials (for random search)
            search_strategy: "grid" or "random"
            maximize: If True, maximize objective; else minimize
        """
        self.param_space = param_space
        self.objective_fn = objective_fn
        self.tracker = tracker
        self.n_trials = n_trials
        self.search_strategy = search_strategy
        self.maximize = maximize
        
        self.results: List[Dict[str, Any]] = []
        self.best_params: Optional[Dict[str, Any]] = None
        self.best_score: Optional[float] = None
    
    def _generate_grid_params(self) -> List[Dict[str, Any]]:
        """Generate all parameter combinations for grid search."""
        keys = list(self.param_space.keys())
        values = list(self.param_space.values())
        
        combinations = []
        for combo in product(*values):
            combinations.append(dict(zip(keys, combo)))
        
        return combinations
    
    def _generate_random_params(self, n: int) -> List[Dict[str, Any]]:
        """Generate random parameter combinations."""
        combinations = []
        
        for _ in range(n):
            params = {}
            for key, values in self.param_space.items():
                params[key] = random.choice(values)
            combinations.append(params)
        
        return combinations
    
    def run(self, verbose: bool = True) -> Dict[str, Any]:
        """
        Run hyperparameter search.
        
        Args:
            verbose: Print progress
            
        Returns:
            Dictionary with best params, score, and all results
        """
        # Generate parameter combinations
        if self.search_strategy == "grid":
            param_combinations = self._generate_grid_params()
            if verbose:
                logger.info(f"Grid search: {len(param_combinations)} combinations")
        else:
            n_trials = self.n_trials or 20
            param_combinations = self._generate_random_params(n_trials)
            if verbose:
                logger.info(f"Random search: {n_trials} trials")
        
        # Run trials
        for i, params in enumerate(param_combinations):
            if verbose:
                logger.info(f"Trial {i+1}/{len(param_combinations)}: {params}")
            
            try:
                # Start MLflow run for this trial
                if self.tracker:
                    run_name = f"trial_{i+1}"
                    self.tracker.start_run(
                        run_name=run_name,
                        tags={"trial": str(i+1), "search_type": self.search_strategy}
                    )
                    self.tracker.log_params(params)
                
                # Run objective function
                score = self.objective_fn(params)
                
                # Log result
                result = {
                    "trial": i + 1,
                    "params": params.copy(),
                    "score": score,
                }
                self.results.append(result)
                
                # Log to MLflow
                if self.tracker:
                    self.tracker.log_metric("objective_score", score)
                    self.tracker.end_run()
                
                # Update best
                if self.best_score is None:
                    self.best_score = score
                    self.best_params = params.copy()
                elif (self.maximize and score > self.best_score) or \
                     (not self.maximize and score < self.best_score):
                    self.best_score = score
                    self.best_params = params.copy()
                
                if verbose:
                    logger.info(f"  Score: {score:.4f} (best: {self.best_score:.4f})")
                
            except Exception as e:
                logger.error(f"Trial {i+1} failed: {e}")
                if self.tracker:
                    self.tracker.end_run(status="FAILED")
                continue
        
        return {
            "best_params": self.best_params,
            "best_score": self.best_score,
            "all_results": self.results,
            "n_trials": len(self.results),
        }
    
    def get_results_df(self) -> pd.DataFrame:
        """Get results as DataFrame."""
        if not self.results:
            return pd.DataFrame()
        
        rows = []
        for result in self.results:
            row = {"trial": result["trial"], "score": result["score"]}
            row.update(result["params"])
            rows.append(row)
        
        return pd.DataFrame(rows)


def create_training_objective(
    env_class,
    env_kwargs: Dict[str, Any],
    model_class,
    train_timesteps: int = 50000,
    eval_episodes: int = 3,
    metric: str = "mean_return",
) -> Callable[[Dict[str, Any]], float]:
    """
    Create an objective function for hyperparameter tuning.
    
    Args:
        env_class: Environment class
        env_kwargs: Base environment kwargs
        model_class: Model class (e.g., PPO)
        train_timesteps: Training timesteps per trial
        eval_episodes: Evaluation episodes
        metric: Metric to optimize
        
    Returns:
        Objective function
    """
    def objective(params: Dict[str, Any]) -> float:
        # Create environment
        env = env_class(**env_kwargs)
        
        # Extract model params
        model_params = {
            "env": env,
            "verbose": 0,
        }
        
        # Map param names to model kwargs
        if "learning_rate" in params:
            model_params["learning_rate"] = params["learning_rate"]
        if "n_steps" in params:
            model_params["n_steps"] = params["n_steps"]
        if "batch_size" in params:
            model_params["batch_size"] = params["batch_size"]
        if "n_epochs" in params:
            model_params["n_epochs"] = params["n_epochs"]
        if "gamma" in params:
            model_params["gamma"] = params["gamma"]
        if "gae_lambda" in params:
            model_params["gae_lambda"] = params["gae_lambda"]
        
        # Create and train model
        model = model_class(**model_params)
        model.learn(total_timesteps=train_timesteps)
        
        # Evaluate
        from src.agent import ModelEvaluator
        evaluator = ModelEvaluator(model, env)
        results = evaluator.evaluate(n_episodes=eval_episodes, verbose=False)
        
        return results.get(metric, 0.0)
    
    return objective


# Default hyperparameter spaces
DEFAULT_PPO_SPACE = {
    "learning_rate": [1e-4, 3e-4, 5e-4, 1e-3],
    "n_steps": [512, 1024, 2048, 4096],
    "batch_size": [32, 64, 128, 256],
    "n_epochs": [3, 5, 10, 15],
    "gamma": [0.95, 0.99, 0.995],
    "gae_lambda": [0.9, 0.95, 0.98],
}

SMALL_PPO_SPACE = {
    "learning_rate": [1e-4, 3e-4, 1e-3],
    "n_steps": [1024, 2048],
    "batch_size": [64, 128],
    "gamma": [0.99, 0.995],
}

