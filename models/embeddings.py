"""
Embedding layers used by TinyTransformerLM.

This module converts token IDs into dense vector
representations and adds positional information.
"""

import torch
import torch.nn as nn


class TransformerEmbedding(nn.Module):
    """
    Token + Positional Embedding Layer.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        context_length: int,
        dropout: float
    ) -> None:

        super().__init__()

        # Token embeddings
        self.token_embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim
        )

        # Position embeddings
        self.position_embedding = nn.Embedding(
            num_embeddings=context_length,
            embedding_dim=embedding_dim
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert token IDs into contextual embeddings.

        Parameters
        ----------
        input_ids : Tensor
            Shape: (batch_size, sequence_length)

        Returns
        -------
        Tensor
            Shape: (batch_size, sequence_length, embedding_dim)
        """

        batch_size, sequence_length = input_ids.shape

        positions = torch.arange(
            sequence_length,
            device=input_ids.device
        )

        token_vectors = self.token_embedding(input_ids)

        position_vectors = self.position_embedding(positions)

        embeddings = token_vectors + position_vectors

        return self.dropout(embeddings)