"""
Model service for loading and running inference.
"""

import os
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from stable_baselines3 import PPO

logger = logging.getLogger(__name__)


class ModelService:
    """
    Service for managing and running the trading model.
    
    Features:
    - Model loading and caching
    - Fast inference (<50ms target)
    - State preprocessing
    - Action interpretation
    """
    
    ACTION_MAP = {0: "hold", 1: "buy", 2: "sell"}
    
    def __init__(
        self,
        model_path: Optional[str] = None,
        device: str = "auto",
        window_size: int = 30,
        n_features: int = 27,
    ):
        """
        Initialize the model service.
        
        Args:
            model_path: Path to the trained model
            device: Device to run inference on
            window_size: Observation window size
            n_features: Number of features
        """
        self.model_path = model_path
        self.window_size = window_size
        self.n_features = n_features
        
        # Determine device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif torch.backends.mps.is_available():
                self.device = "cpu"  # Use CPU for inference (faster for small batches)
            else:
                self.device = "cpu"
        else:
            self.device = device
        
        self.model: Optional[PPO] = None
        self.model_name: Optional[str] = None
        self.loaded_at: Optional[datetime] = None
        
        # Feature normalization stats (will be loaded with model)
        self.feature_means: Optional[np.ndarray] = None
        self.feature_stds: Optional[np.ndarray] = None
        
        # Load model if path provided
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a trained model.
        
        Args:
            model_path: Path to model file (.zip)
            
        Returns:
            True if successful
        """
        try:
            path = Path(model_path)
            
            # Handle both .zip and directory paths
            if path.is_dir():
                # Look for final_model.zip in directory
                zip_path = path / "final_model.zip"
                if zip_path.exists():
                    model_path = str(zip_path)
                else:
                    # Try without .zip extension
                    model_path = str(path / "final_model")
            
            logger.info(f"Loading model from {model_path}")
            
            # Load with CPU for fast inference
            self.model = PPO.load(model_path, device=self.device)
            self.model_name = Path(model_path).stem
            self.loaded_at = datetime.now()
            
            logger.info(f"Model loaded successfully: {self.model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None
    
    def preprocess_state(
        self,
        ohlcv_data: Dict[str, List[float]],
        position: int = 0,
        cash: float = 10000.0,
        shares: float = 0.0,
        indicators: Optional[Dict[str, List[float]]] = None,
    ) -> np.ndarray:
        """
        Preprocess market state for model input.
        
        Args:
            ohlcv_data: Dict with open, high, low, close, volume lists
            position: Current position (-1, 0, 1)
            cash: Available cash
            shares: Shares held
            indicators: Optional pre-calculated indicators
            
        Returns:
            Preprocessed observation array matching model's expected shape (812,)
        """
        # Create DataFrame from OHLCV
        df = pd.DataFrame({
            'open': ohlcv_data['open'],
            'high': ohlcv_data['high'],
            'low': ohlcv_data['low'],
            'close': ohlcv_data['close'],
            'volume': ohlcv_data['volume'],
        })
        
        # Calculate all 27 features to match training data
        # Basic OHLCV (5 features)
        
        # Returns and ranges
        df['returns'] = df['close'].pct_change().fillna(0)
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1)).fillna(0)
        df['hl_range'] = (df['high'] - df['low']) / (df['close'] + 1e-10)
        df['oc_range'] = (df['close'] - df['open']) / (df['open'] + 1e-10)
        
        # Rolling statistics
        df['volatility_5'] = df['returns'].rolling(5).std().fillna(0)
        df['volatility_20'] = df['returns'].rolling(20).std().fillna(0)
        df['mean_return_5'] = df['returns'].rolling(5).mean().fillna(0)
        df['mean_return_20'] = df['returns'].rolling(20).mean().fillna(0)
        
        # Price momentum
        df['momentum_5'] = (df['close'] / df['close'].shift(5) - 1).fillna(0)
        df['momentum_10'] = (df['close'] / df['close'].shift(10) - 1).fillna(0)
        
        # Moving averages relative to price
        df['sma_5_ratio'] = (df['close'] / df['close'].rolling(5).mean() - 1).fillna(0)
        df['sma_10_ratio'] = (df['close'] / df['close'].rolling(10).mean() - 1).fillna(0)
        df['sma_20_ratio'] = (df['close'] / df['close'].rolling(20).mean() - 1).fillna(0)
        
        # Volume features
        df['volume_sma'] = df['volume'].rolling(20).mean().fillna(df['volume'].mean())
        df['volume_ratio'] = (df['volume'] / (df['volume_sma'] + 1e-10)).fillna(1)
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = (100 - (100 / (1 + rs))).fillna(50) / 100  # Normalize to 0-1
        
        # MACD
        ema12 = df['close'].ewm(span=12).mean()
        ema26 = df['close'].ewm(span=26).mean()
        df['macd'] = ((ema12 - ema26) / (df['close'] + 1e-10)).fillna(0)
        df['macd_signal'] = (df['macd'].ewm(span=9).mean()).fillna(0)
        df['macd_hist'] = (df['macd'] - df['macd_signal']).fillna(0)
        
        # Bollinger Bands
        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        df['bb_upper'] = ((sma20 + 2 * std20) / df['close'] - 1).fillna(0)
        df['bb_lower'] = ((sma20 - 2 * std20) / df['close'] - 1).fillna(0)
        df['bb_width'] = ((4 * std20) / (sma20 + 1e-10)).fillna(0)
        
        # Fill any remaining NaN
        df = df.fillna(0)
        
        # Select exactly n_features columns (27)
        feature_cols = df.columns.tolist()
        
        # Ensure we have exactly n_features
        if len(feature_cols) < self.n_features:
            # Add padding columns
            for i in range(self.n_features - len(feature_cols)):
                df[f'pad_{i}'] = 0
        elif len(feature_cols) > self.n_features:
            # Trim to n_features
            feature_cols = feature_cols[:self.n_features]
            df = df[feature_cols]
        
        # Get the most recent window
        if len(df) >= self.window_size:
            window_data = df.iloc[-self.window_size:]
        else:
            # Pad with zeros if not enough data
            pad_rows = self.window_size - len(df)
            padding = pd.DataFrame(
                np.zeros((pad_rows, len(df.columns))),
                columns=df.columns
            )
            window_data = pd.concat([padding, df], ignore_index=True)
        
        # Flatten the window: shape = (window_size * n_features,)
        obs = window_data.values.flatten()
        
        # Add portfolio state (2 values to match training)
        current_price = df['close'].iloc[-1] if len(df) > 0 else 100.0
        portfolio_value = cash + shares * current_price
        portfolio_state = np.array([
            position,  # Current position
            portfolio_value / 10000.0 - 1,  # Normalized portfolio value
        ])
        
        # Combine: shape = (window_size * n_features + 2,) = (30 * 27 + 2,) = (812,)
        full_obs = np.concatenate([obs, portfolio_state])
        
        # Verify shape
        expected_shape = self.window_size * self.n_features + 2
        if len(full_obs) != expected_shape:
            # Adjust to match expected shape
            if len(full_obs) < expected_shape:
                full_obs = np.pad(full_obs, (0, expected_shape - len(full_obs)))
            else:
                full_obs = full_obs[:expected_shape]
        
        # Normalize (Z-score)
        obs_mean = np.mean(full_obs)
        obs_std = np.std(full_obs) + 1e-8
        normalized_obs = (full_obs - obs_mean) / obs_std
        
        return normalized_obs.astype(np.float32)
    
    def predict(
        self,
        state: np.ndarray,
        deterministic: bool = True,
    ) -> Tuple[int, np.ndarray, float]:
        """
        Make a prediction.
        
        Args:
            state: Preprocessed state array
            deterministic: Use deterministic policy
            
        Returns:
            Tuple of (action, probabilities, latency_ms)
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")
        
        start_time = time.perf_counter()
        
        # Get action and probabilities
        action, _ = self.model.predict(state, deterministic=deterministic)
        
        # Get action probabilities
        obs_tensor = torch.tensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # Get the action distribution
            features = self.model.policy.extract_features(obs_tensor)
            if hasattr(self.model.policy, 'mlp_extractor'):
                latent_pi, _ = self.model.policy.mlp_extractor(features)
            else:
                latent_pi = features
            
            action_logits = self.model.policy.action_net(latent_pi)
            probs = torch.softmax(action_logits, dim=-1).cpu().numpy()[0]
        
        latency_ms = (time.perf_counter() - start_time) * 1000
        
        return int(action), probs, latency_ms
    
    def get_action_name(self, action_id: int) -> str:
        """Get action name from ID."""
        return self.ACTION_MAP.get(action_id, "unknown")
    
    def get_model_info(self) -> Dict:
        """Get model information."""
        if not self.is_loaded():
            return {"loaded": False}
        
        return {
            "loaded": True,
            "name": self.model_name,
            "device": self.device,
            "window_size": self.window_size,
            "n_features": self.n_features,
            "loaded_at": self.loaded_at.isoformat() if self.loaded_at else None,
        }

