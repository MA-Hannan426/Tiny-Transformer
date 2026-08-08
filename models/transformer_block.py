"""
Transformer Decoder Block.

This module combines self-attention and feed-forward layers
using pre-layer normalization and residual connections.
"""

import torch
import torch.nn as nn

from .attention import MultiHeadSelfAttention
from .feedforward import FeedForwardNetwork
from .normalization import CustomLayerNorm


class TransformerBlock(nn.Module):
    """
    Decoder-only Transformer Block.

    Architecture:

        Input
          │
          ▼
      LayerNorm
          │
          ▼
    Multi-Head Attention
          │
          ▼
    Residual Addition
          │
          ▼
      LayerNorm
          │
          ▼
     Feed Forward
          │
          ▼
    Residual Addition
          │
          ▼
        Output
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        context_length: int,
        expansion_factor: int = 4,
        dropout: float = 0.1,
        use_bias: bool = True,
    ) -> None:

        super().__init__()

        # First normalization layer
        self.attention_norm = CustomLayerNorm(
            embedding_dim,
            use_bias
        )

        # Multi-head attention
        self.self_attention = MultiHeadSelfAttention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            context_length=context_length,
            dropout=dropout,
            use_bias=use_bias,
        )

        # Second normalization layer
        self.feedforward_norm = CustomLayerNorm(
            embedding_dim,
            use_bias
        )

        # Feed-forward network
        self.feed_forward = FeedForwardNetwork(
            embedding_dim=embedding_dim,
            expansion_factor=expansion_factor,
            dropout=dropout,
            use_bias=use_bias,
        )

    ##################################################################
    # Helper Methods
    ##################################################################

    def apply_attention(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply pre-normalized self-attention.
        """

        normalized_states = self.attention_norm(
            hidden_states
        )

        attention_output = self.self_attention(
            normalized_states
        )

        return attention_output

    def apply_feed_forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Apply pre-normalized feed-forward network.
        """

        normalized_states = self.feedforward_norm(
            hidden_states
        )

        feedforward_output = self.feed_forward(
            normalized_states
        )

        return feedforward_output

    ##################################################################
    # Forward Pass
    ##################################################################

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Execute one Transformer decoder block.
        """

        # ---------- Attention ----------
        attention_output = self.apply_attention(
            hidden_states
        )

        hidden_states = (
            hidden_states + attention_output
        )

        # ---------- Feed Forward ----------
        feedforward_output = self.apply_feed_forward(
            hidden_states
        )

        hidden_states = (
            hidden_states + feedforward_output
        )

        return hidden_states