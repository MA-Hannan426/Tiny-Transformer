"""
Unit tests for TransformerBlock.
"""

import unittest
import torch

from configs import ModelConfig
from models.transformer_block import TransformerBlock


class TestTransformerBlock(unittest.TestCase):

    def setUp(self):

        self.config = ModelConfig()

        self.block = TransformerBlock(

            embedding_dim=self.config.embedding_dim,

            num_heads=self.config.num_heads,

            context_length=self.config.context_length,

            expansion_factor=self.config.expansion_factor,

            dropout=self.config.dropout,

            use_bias=self.config.use_bias,
        )

    def test_output_shape(self):

        x = torch.randn(
            2,
            64,
            self.config.embedding_dim,
        )

        y = self.block(x)

        self.assertEqual(
            x.shape,
            y.shape,
        )


if __name__ == "__main__":

    unittest.main()