"""
Unit tests for MultiHeadSelfAttention.
"""

import unittest
import torch

from configs import ModelConfig
from models.attention import MultiHeadSelfAttention


class TestAttention(unittest.TestCase):

    def setUp(self):

        self.config = ModelConfig()

        self.attention = MultiHeadSelfAttention(

            embedding_dim=self.config.embedding_dim,

            num_heads=self.config.num_heads,

            context_length=self.config.context_length,

            dropout=self.config.dropout,

            use_bias=self.config.use_bias,
        )

    def test_output_shape(self):

        hidden_states = torch.randn(
            2,
            64,
            self.config.embedding_dim,
        )

        output = self.attention(
            hidden_states
        )

        self.assertEqual(
            output.shape,
            hidden_states.shape,
        )


if __name__ == "__main__":

    unittest.main()