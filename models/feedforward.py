"""
Feed Forward Network used inside Transformer blocks.
"""

import torch
import torch.nn as nn


class FeedForwardNetwork(nn.Module):
    """
    Position-wise Feed Forward Network.

    Expands the embedding dimension, applies a non-linear
    activation, then projects back to the original dimension.
    """

    def __init__(
        self,
        embedding_dim: int,
        expansion_factor: int = 4,
        dropout: float = 0.1,
        use_bias: bool = True
    ) -> None:

        super().__init__()

        hidden_dimension = embedding_dim * expansion_factor

        self.expand = nn.Linear(
            embedding_dim,
            hidden_dimension,
            bias=use_bias
        )

        self.activation = nn.GELU()

        self.project = nn.Linear(
            hidden_dimension,
            embedding_dim,
            bias=use_bias
        )

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass of the feed-forward network.
        """

        hidden_states = self.expand(hidden_states)

        hidden_states = self.activation(hidden_states)

        hidden_states = self.project(hidden_states)

        hidden_states = self.dropout(hidden_states)

        return hidden_states