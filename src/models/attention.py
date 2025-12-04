"""
Attention Mechanisms for Financial Time Series

Implements various attention mechanisms for feature importance
and temporal pattern recognition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any


class AttentionLayer(nn.Module):
    """
    Basic scaled dot-product attention layer.
    
    Computes attention weights to focus on important features/timesteps.
    """
    
    def __init__(
        self,
        d_model: int,
        dropout: float = 0.1,
    ):
        """
        Initialize Attention Layer.
        
        Args:
            d_model: Model dimension
            dropout: Dropout rate
        """
        super().__init__()
        self.d_model = d_model
        self.scale = np.sqrt(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Learnable query, key, value projections
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            query: Query tensor (batch, seq_len, d_model)
            key: Key tensor (batch, seq_len, d_model)
            value: Value tensor (batch, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Project to Q, K, V
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Output projection
        output = self.W_o(context)
        
        return output, attention_weights


class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention mechanism.
    
    Allows the model to attend to information from different
    representation subspaces at different positions.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        """
        Initialize Multi-Head Attention.
        
        Args:
            d_model: Model dimension
            num_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.scale = np.sqrt(self.d_k)
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            query: Query tensor (batch, seq_len, d_model)
            key: Key tensor (batch, seq_len, d_model)
            value: Value tensor (batch, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1) == 0, float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads and project
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(context)
        
        # Average attention weights across heads for visualization
        avg_attention = attention_weights.mean(dim=1)
        
        return output, avg_attention


class TemporalAttention(nn.Module):
    """
    Temporal attention for time series.
    
    Learns which time steps are most important for prediction.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
    ):
        """
        Initialize Temporal Attention.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension for attention
        """
        super().__init__()
        
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            
        Returns:
            Tuple of (weighted_output, attention_weights)
        """
        # Compute attention scores
        attention_scores = self.attention_net(x)  # (batch, seq_len, 1)
        attention_weights = F.softmax(attention_scores, dim=1)
        
        # Weighted sum
        weighted_output = (x * attention_weights).sum(dim=1)
        
        return weighted_output, attention_weights.squeeze(-1)


class FeatureAttention(nn.Module):
    """
    Feature attention for selecting important features.
    
    Learns which features are most relevant for the task.
    """
    
    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 32,
    ):
        """
        Initialize Feature Attention.
        
        Args:
            num_features: Number of input features
            hidden_dim: Hidden dimension
        """
        super().__init__()
        
        self.attention_net = nn.Sequential(
            nn.Linear(num_features, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_features),
            nn.Sigmoid(),
        )
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, num_features)
            
        Returns:
            Tuple of (weighted_features, feature_weights)
        """
        # Global average over time
        global_feat = x.mean(dim=1)  # (batch, num_features)
        
        # Compute feature importance
        feature_weights = self.attention_net(global_feat)  # (batch, num_features)
        
        # Apply weights
        weighted_features = x * feature_weights.unsqueeze(1)
        
        return weighted_features, feature_weights


def get_attention_weights(
    model: nn.Module,
    x: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """
    Extract attention weights from a model for visualization.
    
    Args:
        model: Model with attention layers
        x: Input tensor
        
    Returns:
        Dictionary of layer names to attention weights
    """
    attention_weights = {}
    
    def hook_fn(name):
        def hook(module, input, output):
            if isinstance(output, tuple) and len(output) == 2:
                attention_weights[name] = output[1].detach()
        return hook
    
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (AttentionLayer, MultiHeadAttention, TemporalAttention)):
            hooks.append(module.register_forward_hook(hook_fn(name)))
    
    with torch.no_grad():
        model(x)
    
    for hook in hooks:
        hook.remove()
    
    return attention_weights

