"""
Custom Neural Network Architectures for PPO.

Implements LSTM-based feature extractors for financial time series.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Type, Optional
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
import logging

logger = logging.getLogger(__name__)


class LSTMFeatureExtractor(BaseFeaturesExtractor):
    """
    LSTM-based feature extractor for processing financial time series.
    
    Architecture:
        Input -> LSTM -> Dense -> Output
    
    The LSTM captures temporal patterns in the market data,
    while the dense layers extract higher-level features.
    """
    
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        lstm_hidden_size: int = 128,
        lstm_num_layers: int = 2,
        dropout: float = 0.1,
        window_size: int = 30,
        n_features: int = 27,
    ):
        """
        Initialize the LSTM feature extractor.
        
        Args:
            observation_space: Gymnasium observation space
            features_dim: Output dimension of the feature extractor
            lstm_hidden_size: Hidden size of LSTM layers
            lstm_num_layers: Number of LSTM layers
            dropout: Dropout rate
            window_size: Number of timesteps in the window
            n_features: Number of features per timestep
        """
        super().__init__(observation_space, features_dim)
        
        # Infer dimensions from observation space
        # Observation shape: (window_size * n_features + 2,)
        # The +2 is for position and unrealized PnL
        obs_dim = observation_space.shape[0]
        
        # Calculate window_size and n_features if not provided
        # obs_dim = window_size * n_features + 2
        self.extra_features = 2  # position and unrealized PnL
        self.window_size = window_size
        self.n_features = n_features
        
        # Verify dimensions match
        expected_dim = window_size * n_features + self.extra_features
        if obs_dim != expected_dim:
            # Try to infer from observation dimension
            self.window_size = 30  # default
            self.n_features = (obs_dim - self.extra_features) // self.window_size
            logger.warning(
                f"Observation dim {obs_dim} doesn't match expected {expected_dim}. "
                f"Inferred n_features={self.n_features}"
            )
        
        # LSTM for temporal features
        self.lstm = nn.LSTM(
            input_size=self.n_features,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            dropout=dropout if lstm_num_layers > 1 else 0,
            bidirectional=False,
        )
        
        # Dense layers after LSTM
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size + self.extra_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, features_dim),
            nn.ReLU(),
        )
        
        logger.info(
            f"LSTMFeatureExtractor initialized: "
            f"window={self.window_size}, features={self.n_features}, "
            f"lstm_hidden={lstm_hidden_size}, output_dim={features_dim}"
        )
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the feature extractor.
        
        Args:
            observations: Batch of observations (batch_size, obs_dim)
        
        Returns:
            Extracted features (batch_size, features_dim)
        """
        batch_size = observations.shape[0]
        
        # Split observation into window data and extra features
        window_data = observations[:, :-self.extra_features]
        extra = observations[:, -self.extra_features:]
        
        # Reshape window data for LSTM: (batch, seq_len, features)
        window_data = window_data.view(batch_size, self.window_size, self.n_features)
        
        # Pass through LSTM
        lstm_out, (h_n, c_n) = self.lstm(window_data)
        
        # Use the last hidden state
        lstm_features = h_n[-1]  # (batch_size, lstm_hidden_size)
        
        # Concatenate with extra features
        combined = torch.cat([lstm_features, extra], dim=1)
        
        # Pass through dense layers
        features = self.fc(combined)
        
        return features


class MLPFeatureExtractor(BaseFeaturesExtractor):
    """
    Simple MLP feature extractor (baseline).
    
    Can be used as a simpler alternative to LSTM.
    """
    
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        hidden_sizes: List[int] = [256, 256],
        dropout: float = 0.1,
    ):
        """
        Initialize MLP feature extractor.
        
        Args:
            observation_space: Gymnasium observation space
            features_dim: Output dimension
            hidden_sizes: List of hidden layer sizes
            dropout: Dropout rate
        """
        super().__init__(observation_space, features_dim)
        
        obs_dim = observation_space.shape[0]
        
        layers = []
        prev_size = obs_dim
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, features_dim))
        layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)
        
        logger.info(f"MLPFeatureExtractor initialized: obs_dim={obs_dim}, output_dim={features_dim}")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.network(observations)


class AttentionFeatureExtractor(BaseFeaturesExtractor):
    """
    Attention-based feature extractor for financial time series.
    
    Uses self-attention to weight the importance of different timesteps.
    """
    
    def __init__(
        self,
        observation_space: spaces.Box,
        features_dim: int = 128,
        n_heads: int = 4,
        dropout: float = 0.1,
        window_size: int = 30,
        n_features: int = 27,
    ):
        """
        Initialize attention feature extractor.
        
        Args:
            observation_space: Gymnasium observation space
            features_dim: Output dimension
            n_heads: Number of attention heads
            dropout: Dropout rate
            window_size: Number of timesteps
            n_features: Number of features per timestep
        """
        super().__init__(observation_space, features_dim)
        
        self.extra_features = 2
        self.window_size = window_size
        self.n_features = n_features
        
        # Project input features to a common dimension
        self.input_proj = nn.Linear(n_features, 64)
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=64,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        
        # Output projection
        self.fc = nn.Sequential(
            nn.Linear(64 + self.extra_features, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, features_dim),
            nn.ReLU(),
        )
        
        logger.info(f"AttentionFeatureExtractor initialized: n_heads={n_heads}, output_dim={features_dim}")
    
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch_size = observations.shape[0]
        
        # Split observation
        window_data = observations[:, :-self.extra_features]
        extra = observations[:, -self.extra_features:]
        
        # Reshape: (batch, seq_len, features)
        window_data = window_data.view(batch_size, self.window_size, self.n_features)
        
        # Project input
        x = self.input_proj(window_data)  # (batch, seq_len, 64)
        
        # Self-attention
        attn_out, _ = self.attention(x, x, x)
        
        # Global average pooling over sequence
        pooled = attn_out.mean(dim=1)  # (batch, 64)
        
        # Combine with extra features
        combined = torch.cat([pooled, extra], dim=1)
        
        return self.fc(combined)


class TradingPolicy(ActorCriticPolicy):
    """
    Custom Actor-Critic policy for trading.
    
    Uses LSTM-based feature extraction for both actor and critic.
    """
    
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule,
        net_arch: Optional[Dict[str, List[int]]] = None,
        activation_fn: Type[nn.Module] = nn.Tanh,
        features_extractor_class: Type[BaseFeaturesExtractor] = LSTMFeatureExtractor,
        features_extractor_kwargs: Optional[Dict] = None,
        *args,
        **kwargs,
    ):
        """
        Initialize trading policy.
        
        Args:
            observation_space: Gymnasium observation space
            action_space: Gymnasium action space
            lr_schedule: Learning rate schedule
            net_arch: Network architecture for policy and value networks
            activation_fn: Activation function
            features_extractor_class: Feature extractor class
            features_extractor_kwargs: Feature extractor kwargs
        """
        # Default network architecture
        if net_arch is None:
            net_arch = dict(pi=[64, 64], vf=[64, 64])
        
        # Default feature extractor kwargs
        if features_extractor_kwargs is None:
            features_extractor_kwargs = dict(
                features_dim=128,
                lstm_hidden_size=128,
                lstm_num_layers=2,
                dropout=0.1,
            )
        
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            net_arch=net_arch,
            activation_fn=activation_fn,
            features_extractor_class=features_extractor_class,
            features_extractor_kwargs=features_extractor_kwargs,
            *args,
            **kwargs,
        )


def create_policy_kwargs(
    extractor_type: str = "lstm",
    features_dim: int = 128,
    lstm_hidden_size: int = 128,
    lstm_num_layers: int = 2,
    n_heads: int = 4,
    dropout: float = 0.1,
    window_size: int = 30,
    n_features: int = 27,
) -> Dict:
    """
    Create policy kwargs for Stable-Baselines3.
    
    Args:
        extractor_type: Type of feature extractor ("lstm", "mlp", "attention")
        features_dim: Output dimension of feature extractor
        lstm_hidden_size: LSTM hidden size (for LSTM extractor)
        lstm_num_layers: Number of LSTM layers (for LSTM extractor)
        n_heads: Number of attention heads (for attention extractor)
        dropout: Dropout rate
        window_size: Window size for reshaping
        n_features: Number of features per timestep
    
    Returns:
        Dictionary of policy kwargs for SB3
    """
    extractor_classes = {
        "lstm": LSTMFeatureExtractor,
        "mlp": MLPFeatureExtractor,
        "attention": AttentionFeatureExtractor,
    }
    
    if extractor_type not in extractor_classes:
        raise ValueError(f"Unknown extractor type: {extractor_type}")
    
    extractor_kwargs = {
        "features_dim": features_dim,
    }
    
    if extractor_type == "lstm":
        extractor_kwargs.update({
            "lstm_hidden_size": lstm_hidden_size,
            "lstm_num_layers": lstm_num_layers,
            "dropout": dropout,
            "window_size": window_size,
            "n_features": n_features,
        })
    elif extractor_type == "attention":
        extractor_kwargs.update({
            "n_heads": n_heads,
            "dropout": dropout,
            "window_size": window_size,
            "n_features": n_features,
        })
    elif extractor_type == "mlp":
        extractor_kwargs.update({
            "hidden_sizes": [256, 256],
            "dropout": dropout,
        })
    
    return {
        "features_extractor_class": extractor_classes[extractor_type],
        "features_extractor_kwargs": extractor_kwargs,
        "net_arch": dict(pi=[64, 64], vf=[64, 64]),
        "activation_fn": nn.ReLU,
    }

