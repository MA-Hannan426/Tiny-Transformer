"""
Unit tests for FeedForwardNetwork.
"""

import unittest
import torch

from configs import ModelConfig
from models.feedforward import FeedForwardNetwork


class TestFeedForward(unittest.TestCase):

    def setUp(self):

        self.config = ModelConfig()

        self.network = FeedForwardNetwork(

            embedding_dim=self.config.embedding_dim,

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

        y = self.network(x)

        self.assertEqual(
            x.shape,
            y.shape,
        )


if __name__ == "__main__":

    unittest.main()