"""
Multi-Head Causal Self-Attention.

This module implements masked multi-head self-attention
used in the decoder-only Transformer architecture.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadSelfAttention(nn.Module):
    """
    Decoder-only Multi-Head Self Attention.
    """

    def __init__(
        self,
        embedding_dim: int,
        num_heads: int,
        context_length: int,
        dropout: float = 0.1,
        use_bias: bool = True
    ) -> None:

        super().__init__()

        if embedding_dim % num_heads != 0:
            raise ValueError(
                "Embedding dimension must be divisible by number of heads."
            )

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dimension = embedding_dim // num_heads
        self.context_length = context_length

        # Combined QKV projection
        self.qkv_projection = nn.Linear(
            embedding_dim,
            embedding_dim * 3,
            bias=use_bias
        )

        # Output projection
        self.output_projection = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=use_bias
        )

        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

        # Flash Attention support
        self.use_flash_attention = hasattr(
            F,
            "scaled_dot_product_attention"
        )

        if not self.use_flash_attention:

            causal_mask = torch.tril(
                torch.ones(context_length, context_length)
            )

            self.register_buffer(
                "causal_mask",
                causal_mask.view(
                    1,
                    1,
                    context_length,
                    context_length
                )
            )

    ####################################################################
    # Helper Functions
    ####################################################################

    def create_qkv_projections(
        self,
        hidden_states: torch.Tensor
    ):
        """
        Create Query, Key and Value tensors.
        """

        queries, keys, values = self.qkv_projection(
            hidden_states
        ).chunk(
            3,
            dim=-1
        )

        return queries, keys, values

    def reshape_for_multi_head_attention(
        self,
        tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert

        (B, T, C)

        into

        (B, Heads, T, HeadDim)
        """

        batch_size, sequence_length, _ = tensor.shape

        tensor = tensor.view(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dimension
        )

        return tensor.transpose(1, 2)

    def compute_attention_scores(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute scaled dot-product attention scores.
        """

        scores = (
            queries @ keys.transpose(-2, -1)
        )

        scores = scores / math.sqrt(
            self.head_dimension
        )

        return scores

    def apply_causal_mask(
        self,
        attention_scores: torch.Tensor,
        sequence_length: int
    ) -> torch.Tensor:
        """
        Prevent attention to future tokens.
        """

        attention_scores = attention_scores.masked_fill(
            self.causal_mask[
                :,
                :,
                :sequence_length,
                :sequence_length
            ] == 0,
            float("-inf")
        )

        return attention_scores

    def compute_attention_output(
        self,
        attention_weights: torch.Tensor,
        values: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute weighted sum of Value vectors.
        """

        return attention_weights @ values

    def merge_attention_heads(
        self,
        attention_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert

        (B, Heads, T, HeadDim)

        back into

        (B, T, EmbeddingDim)
        """

        batch_size, _, sequence_length, _ = attention_output.shape

        attention_output = attention_output.transpose(
            1,
            2
        ).contiguous()

        attention_output = attention_output.view(
            batch_size,
            sequence_length,
            self.embedding_dim
        )

        return attention_output

    ####################################################################
    # Forward Pass
    ####################################################################

    def forward(
        self,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:

        batch_size, sequence_length, _ = hidden_states.shape

        queries, keys, values = self.create_qkv_projections(
            hidden_states
        )

        queries = self.reshape_for_multi_head_attention(
            queries
        )

        keys = self.reshape_for_multi_head_attention(
            keys
        )

        values = self.reshape_for_multi_head_attention(
            values
        )

        # ---------- Flash Attention ----------

        if self.use_flash_attention:

            attention_output = F.scaled_dot_product_attention(
                queries,
                keys,
                values,
                attn_mask=None,
                dropout_p=self.attention_dropout.p
                if self.training
                else 0.0,
                is_causal=True
            )

        # ---------- Manual Attention ----------

        else:

            attention_scores = self.compute_attention_scores(
                queries,
                keys
            )

            attention_scores = self.apply_causal_mask(
                attention_scores,
                sequence_length
            )

            attention_weights = F.softmax(
                attention_scores,
                dim=-1
            )

            attention_weights = self.attention_dropout(
                attention_weights
            )

            attention_output = self.compute_attention_output(
                attention_weights,
                values
            )

        attention_output = self.merge_attention_heads(
            attention_output
        )

        attention_output = self.output_projection(
            attention_output
        )

        attention_output = self.output_dropout(
            attention_output
        )

        return attention_output