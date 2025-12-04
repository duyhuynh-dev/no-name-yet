"""
MLflow Experiment Tracking Module.

Provides integration with MLflow for:
- Logging hyperparameters
- Tracking metrics over training
- Saving model artifacts
- Comparing experiments
"""

import os
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class ExperimentTracker:
    """
    MLflow-based experiment tracker for RL trading agents.
    
    Features:
    - Automatic experiment creation
    - Hyperparameter logging
    - Training metrics tracking
    - Model artifact storage
    - Experiment comparison
    """
    
    def __init__(
        self,
        experiment_name: str = "hft-rl-trading",
        tracking_uri: Optional[str] = None,
        artifact_location: Optional[str] = None,
    ):
        """
        Initialize the experiment tracker.
        
        Args:
            experiment_name: Name of the MLflow experiment
            tracking_uri: MLflow tracking server URI (default: local ./mlruns)
            artifact_location: Location to store artifacts
        """
        self.experiment_name = experiment_name
        
        # Set tracking URI
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        else:
            # Default to local tracking
            mlflow.set_tracking_uri("./mlruns")
        
        self.tracking_uri = mlflow.get_tracking_uri()
        
        # Get or create experiment
        self.client = MlflowClient()
        experiment = self.client.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            self.experiment_id = self.client.create_experiment(
                name=experiment_name,
                artifact_location=artifact_location,
            )
            logger.info(f"Created new experiment: {experiment_name} (id: {self.experiment_id})")
        else:
            self.experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment: {experiment_name} (id: {self.experiment_id})")
        
        mlflow.set_experiment(experiment_name)
        
        self.active_run = None
        self.run_id = None
    
    def start_run(
        self,
        run_name: Optional[str] = None,
        tags: Optional[Dict[str, str]] = None,
        description: Optional[str] = None,
    ) -> str:
        """
        Start a new MLflow run.
        
        Args:
            run_name: Name for this run
            tags: Additional tags to add
            description: Run description
            
        Returns:
            Run ID
        """
        if run_name is None:
            run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # End any existing run
        if self.active_run:
            self.end_run()
        
        # Start new run
        self.active_run = mlflow.start_run(
            run_name=run_name,
            experiment_id=self.experiment_id,
            description=description,
        )
        self.run_id = self.active_run.info.run_id
        
        # Add default tags
        default_tags = {
            "framework": "stable-baselines3",
            "algorithm": "PPO",
            "project": "hft-rl-simulator",
        }
        if tags:
            default_tags.update(tags)
        
        mlflow.set_tags(default_tags)
        
        logger.info(f"Started run: {run_name} (id: {self.run_id})")
        return self.run_id
    
    def end_run(self, status: str = "FINISHED") -> None:
        """End the current run."""
        if self.active_run:
            mlflow.end_run(status=status)
            logger.info(f"Ended run: {self.run_id} (status: {status})")
            self.active_run = None
            self.run_id = None
    
    def log_params(self, params: Dict[str, Any]) -> None:
        """
        Log hyperparameters.
        
        Args:
            params: Dictionary of parameters to log
        """
        if not self.active_run:
            raise RuntimeError("No active run. Call start_run() first.")
        
        # Flatten nested dicts and convert values to strings
        flat_params = self._flatten_dict(params)
        
        # MLflow has a limit on param value length
        for key, value in flat_params.items():
            str_value = str(value)
            if len(str_value) > 500:
                str_value = str_value[:500] + "..."
            mlflow.log_param(key, str_value)
        
        logger.debug(f"Logged {len(flat_params)} parameters")
    
    def log_metric(
        self,
        key: str,
        value: float,
        step: Optional[int] = None,
    ) -> None:
        """
        Log a single metric.
        
        Args:
            key: Metric name
            value: Metric value
            step: Step/iteration number
        """
        if not self.active_run:
            raise RuntimeError("No active run. Call start_run() first.")
        
        # Handle NaN/Inf values
        if np.isnan(value) or np.isinf(value):
            logger.warning(f"Skipping invalid metric value for {key}: {value}")
            return
        
        mlflow.log_metric(key, value, step=step)
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
    ) -> None:
        """
        Log multiple metrics at once.
        
        Args:
            metrics: Dictionary of metric_name -> value
            step: Step/iteration number
        """
        if not self.active_run:
            raise RuntimeError("No active run. Call start_run() first.")
        
        # Filter out invalid values
        valid_metrics = {}
        for key, value in metrics.items():
            if np.isnan(value) or np.isinf(value):
                logger.warning(f"Skipping invalid metric value for {key}: {value}")
            else:
                valid_metrics[key] = value
        
        mlflow.log_metrics(valid_metrics, step=step)
    
    def log_training_progress(
        self,
        timestep: int,
        episode: int,
        mean_reward: float,
        mean_length: float,
        loss: Optional[float] = None,
        value_loss: Optional[float] = None,
        policy_loss: Optional[float] = None,
        entropy: Optional[float] = None,
        learning_rate: Optional[float] = None,
        explained_variance: Optional[float] = None,
        **kwargs,
    ) -> None:
        """
        Log training progress metrics.
        
        Args:
            timestep: Current timestep
            episode: Current episode
            mean_reward: Mean episode reward
            mean_length: Mean episode length
            loss: Total loss
            value_loss: Value function loss
            policy_loss: Policy gradient loss
            entropy: Policy entropy
            learning_rate: Current learning rate
            explained_variance: Explained variance
            **kwargs: Additional metrics
        """
        metrics = {
            "train/episode": episode,
            "train/mean_reward": mean_reward,
            "train/mean_length": mean_length,
        }
        
        if loss is not None:
            metrics["train/loss"] = loss
        if value_loss is not None:
            metrics["train/value_loss"] = value_loss
        if policy_loss is not None:
            metrics["train/policy_loss"] = policy_loss
        if entropy is not None:
            metrics["train/entropy"] = entropy
        if learning_rate is not None:
            metrics["train/learning_rate"] = learning_rate
        if explained_variance is not None:
            metrics["train/explained_variance"] = explained_variance
        
        # Add any additional metrics
        for key, value in kwargs.items():
            if isinstance(value, (int, float)):
                metrics[f"train/{key}"] = value
        
        self.log_metrics(metrics, step=timestep)
    
    def log_evaluation_results(
        self,
        results: Dict[str, Any],
        prefix: str = "eval",
        step: Optional[int] = None,
    ) -> None:
        """
        Log evaluation results.
        
        Args:
            results: Evaluation results dictionary
            prefix: Metric name prefix
            step: Step number
        """
        metrics = {}
        
        # Extract numeric values
        for key, value in results.items():
            if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                metrics[f"{prefix}/{key}"] = value
            elif isinstance(value, np.ndarray) and value.size == 1:
                metrics[f"{prefix}/{key}"] = float(value)
        
        if metrics:
            self.log_metrics(metrics, step=step)
    
    def log_artifact(
        self,
        local_path: str,
        artifact_path: Optional[str] = None,
    ) -> None:
        """
        Log a file or directory as an artifact.
        
        Args:
            local_path: Path to file or directory
            artifact_path: Destination path in artifact store
        """
        if not self.active_run:
            raise RuntimeError("No active run. Call start_run() first.")
        
        if os.path.exists(local_path):
            mlflow.log_artifact(local_path, artifact_path)
            logger.info(f"Logged artifact: {local_path}")
        else:
            logger.warning(f"Artifact not found: {local_path}")
    
    def log_model(
        self,
        model,
        artifact_path: str = "model",
        registered_model_name: Optional[str] = None,
    ) -> None:
        """
        Log a trained model.
        
        Args:
            model: The model to log (Stable-Baselines3 model)
            artifact_path: Path in artifact store
            registered_model_name: Name to register model under
        """
        if not self.active_run:
            raise RuntimeError("No active run. Call start_run() first.")
        
        # Save model to temporary location
        temp_path = f"temp_model_{self.run_id}"
        model.save(temp_path)
        
        # Log as artifact
        mlflow.log_artifact(f"{temp_path}.zip", artifact_path)
        
        # Clean up
        os.remove(f"{temp_path}.zip")
        
        logger.info(f"Logged model to {artifact_path}")
    
    def log_figure(
        self,
        figure,
        artifact_path: str,
    ) -> None:
        """
        Log a matplotlib figure.
        
        Args:
            figure: Matplotlib figure
            artifact_path: Path to save in artifacts
        """
        if not self.active_run:
            raise RuntimeError("No active run. Call start_run() first.")
        
        mlflow.log_figure(figure, artifact_path)
        logger.debug(f"Logged figure: {artifact_path}")
    
    def log_dict(
        self,
        dictionary: Dict[str, Any],
        artifact_path: str,
    ) -> None:
        """
        Log a dictionary as JSON artifact.
        
        Args:
            dictionary: Dictionary to log
            artifact_path: Path to save (should end in .json)
        """
        if not self.active_run:
            raise RuntimeError("No active run. Call start_run() first.")
        
        mlflow.log_dict(dictionary, artifact_path)
    
    def get_run_metrics(self, run_id: Optional[str] = None) -> Dict[str, List[float]]:
        """
        Get all metrics for a run.
        
        Args:
            run_id: Run ID (uses current run if None)
            
        Returns:
            Dictionary of metric_name -> list of values
        """
        if run_id is None:
            run_id = self.run_id
        
        if run_id is None:
            raise ValueError("No run ID specified and no active run.")
        
        run = self.client.get_run(run_id)
        metrics = {}
        
        for key in run.data.metrics:
            history = self.client.get_metric_history(run_id, key)
            metrics[key] = [m.value for m in history]
        
        return metrics
    
    def compare_runs(
        self,
        run_ids: List[str],
        metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """
        Compare metrics across multiple runs.
        
        Args:
            run_ids: List of run IDs to compare
            metrics: Specific metrics to compare (all if None)
            
        Returns:
            DataFrame with comparison
        """
        data = []
        
        for run_id in run_ids:
            run = self.client.get_run(run_id)
            row = {
                "run_id": run_id,
                "run_name": run.info.run_name,
                "status": run.info.status,
                "start_time": run.info.start_time,
            }
            
            # Add parameters
            for key, value in run.data.params.items():
                row[f"param.{key}"] = value
            
            # Add metrics
            for key, value in run.data.metrics.items():
                if metrics is None or key in metrics:
                    row[f"metric.{key}"] = value
            
            data.append(row)
        
        return pd.DataFrame(data)
    
    def get_best_run(
        self,
        metric: str = "eval/mean_reward",
        ascending: bool = False,
    ) -> Optional[Dict[str, Any]]:
        """
        Get the best run based on a metric.
        
        Args:
            metric: Metric to optimize
            ascending: If True, lower is better
            
        Returns:
            Best run info or None
        """
        runs = self.client.search_runs(
            experiment_ids=[self.experiment_id],
            filter_string="",
            order_by=[f"metrics.{metric} {'ASC' if ascending else 'DESC'}"],
            max_results=1,
        )
        
        if runs:
            run = runs[0]
            return {
                "run_id": run.info.run_id,
                "run_name": run.info.run_name,
                "params": run.data.params,
                "metrics": run.data.metrics,
            }
        
        return None
    
    def _flatten_dict(
        self,
        d: Dict[str, Any],
        parent_key: str = "",
        sep: str = ".",
    ) -> Dict[str, Any]:
        """Flatten a nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.active_run:
            status = "FAILED" if exc_type else "FINISHED"
            self.end_run(status=status)
        return False


class TrainingCallback:
    """
    Callback for logging training progress to MLflow.
    
    Can be used with Stable-Baselines3's callback system.
    """
    
    def __init__(
        self,
        tracker: ExperimentTracker,
        log_freq: int = 1000,
        eval_freq: int = 10000,
        eval_env=None,
        n_eval_episodes: int = 5,
    ):
        """
        Initialize the callback.
        
        Args:
            tracker: ExperimentTracker instance
            log_freq: How often to log training metrics
            eval_freq: How often to run evaluation
            eval_env: Environment for evaluation
            n_eval_episodes: Number of evaluation episodes
        """
        self.tracker = tracker
        self.log_freq = log_freq
        self.eval_freq = eval_freq
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        
        self.n_calls = 0
        self.episode_rewards = []
        self.episode_lengths = []
    
    def on_step(self, locals_dict: Dict[str, Any]) -> bool:
        """
        Called at each training step.
        
        Args:
            locals_dict: Local variables from training
            
        Returns:
            True to continue training
        """
        self.n_calls += 1
        
        # Log at specified frequency
        if self.n_calls % self.log_freq == 0:
            metrics = {}
            
            # Extract relevant metrics
            if "infos" in locals_dict:
                for info in locals_dict["infos"]:
                    if "episode" in info:
                        self.episode_rewards.append(info["episode"]["r"])
                        self.episode_lengths.append(info["episode"]["l"])
            
            if self.episode_rewards:
                metrics["mean_reward"] = np.mean(self.episode_rewards[-100:])
                metrics["mean_length"] = np.mean(self.episode_lengths[-100:])
            
            if metrics:
                self.tracker.log_metrics(
                    {f"train/{k}": v for k, v in metrics.items()},
                    step=self.n_calls,
                )
        
        return True
    
    def on_training_end(self) -> None:
        """Called at the end of training."""
        logger.info(f"Training ended after {self.n_calls} steps")

