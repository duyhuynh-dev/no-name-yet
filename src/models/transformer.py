"""
Transformer Models for Financial Time Series

Implements transformer-based architectures optimized for
market prediction and trading signal generation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass

from .attention import MultiHeadAttention


@dataclass
class TransformerConfig:
    """Configuration for Transformer models."""
    input_dim: int = 64
    d_model: int = 128
    num_heads: int = 8
    num_encoder_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    max_seq_len: int = 256
    output_dim: int = 3  # Buy, Hold, Sell


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for time series.
    
    Adds positional information to the input embeddings.
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 5000,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


class LearnedPositionalEncoding(nn.Module):
    """
    Learned positional encoding.
    
    More flexible than sinusoidal for financial time series.
    """
    
    def __init__(
        self,
        d_model: int,
        max_len: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe = nn.Embedding(max_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0)
        x = x + self.pe(positions)
        return self.dropout(x)


class TransformerEncoderLayer(nn.Module):
    """
    Single Transformer encoder layer.
    
    Consists of multi-head attention and feed-forward network.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with residual connections.
        
        Args:
            x: Input tensor (batch, seq_len, d_model)
            mask: Optional attention mask
            
        Returns:
            Tuple of (output, attention_weights)
        """
        # Self-attention with residual
        attn_out, attn_weights = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feed-forward with residual
        ff_out = self.feed_forward(x)
        x = self.norm2(x + ff_out)
        
        return x, attn_weights


class TransformerPredictor(nn.Module):
    """
    Transformer-based predictor for trading signals.
    
    Uses self-attention to capture long-range dependencies
    in financial time series.
    """
    
    def __init__(
        self,
        config: Optional[TransformerConfig] = None,
        **kwargs,
    ):
        """
        Initialize Transformer Predictor.
        
        Args:
            config: Transformer configuration
            **kwargs: Override config parameters
        """
        super().__init__()
        
        self.config = config or TransformerConfig(**kwargs)
        
        # Input projection
        self.input_projection = nn.Linear(self.config.input_dim, self.config.d_model)
        
        # Positional encoding
        self.positional_encoding = LearnedPositionalEncoding(
            self.config.d_model,
            self.config.max_seq_len,
            self.config.dropout,
        )
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                self.config.d_model,
                self.config.num_heads,
                self.config.d_ff,
                self.config.dropout,
            )
            for _ in range(self.config.num_encoder_layers)
        ])
        
        # Output head
        self.output_head = nn.Sequential(
            nn.Linear(self.config.d_model, self.config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(self.config.d_model // 2, self.config.output_dim),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier uniform."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, seq_len, input_dim)
            mask: Optional attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            Output logits (batch, output_dim)
        """
        # Project input
        x = self.input_projection(x)
        
        # Add positional encoding
        x = self.positional_encoding(x)
        
        # Pass through encoder layers
        attention_weights = []
        for layer in self.encoder_layers:
            x, attn = layer(x, mask)
            attention_weights.append(attn)
        
        # Use last timestep for prediction (causal)
        x = x[:, -1, :]
        
        # Output projection
        output = self.output_head(x)
        
        if return_attention:
            return output, attention_weights
        return output
    
    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability distribution over actions."""
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get predicted action."""
        logits = self.forward(x)
        return torch.argmax(logits, dim=-1)


class TemporalFusionTransformer(nn.Module):
    """
    Temporal Fusion Transformer for multi-horizon prediction.
    
    Based on the TFT architecture, optimized for financial forecasting.
    Handles static, known, and unknown inputs separately.
    """
    
    def __init__(
        self,
        num_features: int,
        num_static_features: int = 0,
        d_model: int = 128,
        num_heads: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        d_ff: int = 256,
        dropout: float = 0.1,
        num_quantiles: int = 3,  # 10%, 50%, 90%
        prediction_horizon: int = 1,
    ):
        """
        Initialize Temporal Fusion Transformer.
        
        Args:
            num_features: Number of time-varying features
            num_static_features: Number of static features (e.g., asset type)
            d_model: Model dimension
            num_heads: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            num_quantiles: Number of quantiles to predict
            prediction_horizon: Number of steps to predict
        """
        super().__init__()
        
        self.d_model = d_model
        self.num_quantiles = num_quantiles
        self.prediction_horizon = prediction_horizon
        
        # Feature embedding
        self.feature_embedding = nn.Linear(num_features, d_model)
        
        # Static feature processing
        if num_static_features > 0:
            self.static_embedding = nn.Sequential(
                nn.Linear(num_static_features, d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model),
            )
            self.static_enrichment = nn.Linear(d_model * 2, d_model)
        else:
            self.static_embedding = None
        
        # LSTM encoder for local patterns
        self.lstm_encoder = nn.LSTM(
            d_model, d_model, num_layers=1, batch_first=True, dropout=dropout
        )
        
        # Attention layers
        self.attention_layers = nn.ModuleList([
            InterpretableMultiHeadAttention(d_model, num_heads, dropout)
            for _ in range(num_encoder_layers)
        ])
        
        # Gated residual network
        self.grn = GatedResidualNetwork(d_model, dropout)
        
        # Output layer
        self.output_layer = nn.Linear(d_model, num_quantiles * prediction_horizon)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        static: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Time-varying input (batch, seq_len, num_features)
            static: Optional static features (batch, num_static_features)
            
        Returns:
            Quantile predictions (batch, prediction_horizon, num_quantiles)
        """
        batch_size = x.size(0)
        
        # Feature embedding
        x = self.feature_embedding(x)
        
        # Static enrichment
        if self.static_embedding is not None and static is not None:
            static_emb = self.static_embedding(static).unsqueeze(1)
            static_emb = static_emb.expand(-1, x.size(1), -1)
            x = self.static_enrichment(torch.cat([x, static_emb], dim=-1))
        
        # LSTM encoding
        lstm_out, _ = self.lstm_encoder(x)
        
        # Self-attention
        for attention in self.attention_layers:
            lstm_out, _ = attention(lstm_out, lstm_out, lstm_out)
        
        # Gated residual
        out = self.grn(lstm_out[:, -1, :])
        
        # Output projection
        out = self.output_layer(out)
        out = out.view(batch_size, self.prediction_horizon, self.num_quantiles)
        
        return out


class VariableSelectionNetwork(nn.Module):
    """Variable selection for TFT."""
    
    def __init__(
        self,
        num_features: int,
        d_model: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.num_features = num_features
        
        # Per-variable processing
        self.variable_grns = nn.ModuleList([
            GatedResidualNetwork(1, dropout, output_dim=d_model)
            for _ in range(num_features)
        ])
        
        # Variable selection weights
        self.softmax_layer = nn.Sequential(
            nn.Linear(num_features * d_model, num_features),
            nn.Softmax(dim=-1),
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with variable selection."""
        batch_size, seq_len, _ = x.shape
        
        # Process each variable
        var_outputs = []
        for i, grn in enumerate(self.variable_grns):
            var_input = x[:, :, i:i+1]
            var_out = grn(var_input)
            var_outputs.append(var_out)
        
        # Stack and compute selection weights
        stacked = torch.stack(var_outputs, dim=-1)  # (batch, seq, d_model, num_features)
        
        # Compute importance weights
        flat = stacked.view(batch_size, seq_len, -1)
        weights = self.softmax_layer(flat)  # (batch, seq, num_features)
        
        # Weighted combination
        weighted_sum = (stacked * weights.unsqueeze(2)).sum(dim=-1)
        
        return weighted_sum, weights.mean(dim=1)  # Average weights over time


class GatedResidualNetwork(nn.Module):
    """Gated residual network for TFT."""
    
    def __init__(
        self,
        input_dim: int,
        dropout: float = 0.1,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        
        output_dim = output_dim or input_dim
        hidden_dim = output_dim
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(hidden_dim, output_dim)
        
        self.layer_norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
        
        if input_dim != output_dim:
            self.skip = nn.Linear(input_dim, output_dim)
        else:
            self.skip = None
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with gating."""
        # Skip connection
        if self.skip is not None:
            skip = self.skip(x)
        else:
            skip = x
        
        # Main path
        h = F.elu(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h)
        
        # Gate
        gate = torch.sigmoid(self.gate(F.elu(self.fc1(x))))
        h = gate * h
        
        # Add and norm
        out = self.layer_norm(skip + h)
        
        return out


class InterpretableMultiHeadAttention(nn.Module):
    """Interpretable multi-head attention for TFT."""
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.layer_norm = nn.LayerNorm(d_model)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with residual."""
        attn_out, attn_weights = self.attention(query, key, value)
        out = self.layer_norm(query + attn_out)
        return out, attn_weights

