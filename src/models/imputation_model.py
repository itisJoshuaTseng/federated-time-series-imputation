"""
Imputation Backbone Model (Transformer-based Encoder).

Processes time series with missing values, producing rich temporal
representations for the RL agents to condition their imputation policies on.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)"""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MissingAwareAttention(nn.Module):
    """
    Multi-head attention with missing-value-aware masking.

    Masks out contributions from missing entries so the transformer
    attends only to observed values.
    """

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x:    (B, T, d_model)
            mask: (B, T) binary mask where 0 = should be ignored
        """
        # Convert mask to attention format: True = ignore
        if mask is not None:
            # (B, T) → key_padding_mask
            key_padding_mask = (mask.sum(dim=-1) == 0) if mask.dim() == 3 else (mask < 0.5)
        else:
            key_padding_mask = None

        out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        return out


class ImputationTransformer(nn.Module):
    """
    Transformer encoder for time series with missing values.

    Input: concatenation of [observed_values, missing_mask, temporal_features]
    Output: contextualized representations (B, T, d_model) for each time step.

    The encoder captures:
      - Temporal dependencies across time steps
      - Cross-variable relationships
      - Missing pattern information
    """

    def __init__(
        self,
        num_features: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 3,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        max_len: int = 512,
    ):
        super().__init__()
        self.num_features = num_features
        self.d_model = d_model

        # Input projection: [values, mask, delta_t] → d_model
        # values (D) + mask (D) + delta since last obs (D) = 3D
        input_dim = num_features * 3
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)

        # Transformer encoder layers with missing-aware attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,  # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            norm=nn.LayerNorm(d_model),
        )

        # Output projection: produce per-feature representation
        self.output_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
        )

        self._init_weights()

    def _init_weights(self):
        """Xavier initialization for better convergence."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def compute_delta_t(self, mask: torch.Tensor) -> torch.Tensor:
        """
        Compute time since last observation for each feature.

        Args:
            mask: (B, T, D) binary observation mask

        Returns:
            delta: (B, T, D) normalized time since last obs
        """
        B, T, D = mask.shape
        delta = torch.zeros_like(mask)

        for t in range(1, T):
            # If observed at t-1, delta = 1; else delta = prev_delta + 1
            delta[:, t] = torch.where(
                mask[:, t - 1] > 0.5,
                torch.ones(B, D, device=mask.device),
                delta[:, t - 1] + 1,
            )

        # Normalize
        delta = delta / T
        return delta

    def forward(
        self,
        observed: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode time series with missing values.

        Args:
            observed: (B, T, D) observed values (0 where missing)
            mask:     (B, T, D) binary observation mask

        Returns:
            h: (B, T, d_model) contextualized representations
        """
        B, T, D = observed.shape

        # Compute time gaps
        delta_t = self.compute_delta_t(mask)

        # Concatenate input channels
        x = torch.cat([observed, mask, delta_t], dim=-1)  # (B, T, 3D)

        # Project to d_model
        x = self.input_proj(x)  # (B, T, d_model)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create attention mask (mask out fully-missing time steps)
        src_key_padding_mask = (mask.sum(dim=-1) == 0)  # (B, T)

        # Transformer encoding
        h = self.transformer(x, src_key_padding_mask=src_key_padding_mask)

        # Output projection
        h = self.output_proj(h)

        return h  # (B, T, d_model)


class FeatureGroupEncoder(nn.Module):
    """
    Per-agent feature group encoder.

    Takes the global transformer representation and extracts
    agent-specific features for its assigned feature group.
    """

    def __init__(
        self,
        d_model: int,
        group_size: int,
        hidden_dim: int = 64,
    ):
        super().__init__()
        self.group_size = group_size
        self.d_model = d_model

        self.feature_attn = nn.Sequential(
            nn.Linear(d_model + group_size * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

    def forward(
        self,
        global_repr: torch.Tensor,
        observed: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            global_repr: (B, d_model) transformer output at current time step
            observed:    (B, group_size) observed values for agent's features
            mask:        (B, group_size) masks for agent's features

        Returns:
            agent_repr: (B, hidden_dim) agent-specific representation
        """
        x = torch.cat([global_repr, observed, mask], dim=-1)
        return self.feature_attn(x)
