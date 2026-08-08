"""
Normalization layers for TinyTransformerLM.

This module contains normalization layers used throughout the
Transformer architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CustomLayerNorm(nn.Module):
    """
    Layer Normalization with optional bias.

    Parameters
    ----------
    embedding_dim : int
        Size of the embedding dimension.

    use_bias : bool
        Whether to learn an additive bias.
    """

    def __init__(
        self,
        embedding_dim: int,
        use_bias: bool = True
    ) -> None:

        super().__init__()

        self.weight = nn.Parameter(torch.ones(embedding_dim))

        self.bias = (
            nn.Parameter(torch.zeros(embedding_dim))
            if use_bias
            else None
        )

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Normalize hidden representations.

        Parameters
        ----------
        hidden_states : Tensor
            Input tensor of shape (batch, sequence, embedding)

        Returns
        -------
        Tensor
            Normalized tensor.
        """

        return F.layer_norm(
            hidden_states,
            normalized_shape=self.weight.shape,
            weight=self.weight,
            bias=self.bias,
            eps=1e-5
        )