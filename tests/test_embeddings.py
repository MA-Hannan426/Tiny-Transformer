"""
Unit tests for TransformerEmbedding.
"""

import unittest
import torch

from configs import ModelConfig
from models.embeddings import TransformerEmbedding


class TestTransformerEmbedding(unittest.TestCase):

    def setUp(self):

        self.config = ModelConfig()

        self.embedding = TransformerEmbedding(
            vocab_size=self.config.vocab_size,
            embedding_dim=self.config.embedding_dim,
            context_length=self.config.context_length,
            dropout=self.config.dropout,
        )

    def test_output_shape(self):

        batch_size = 2
        sequence_length = 32

        input_ids = torch.randint(
            0,
            self.config.vocab_size,
            (batch_size, sequence_length),
        )

        output = self.embedding(input_ids)

        self.assertEqual(
            output.shape,
            (
                batch_size,
                sequence_length,
                self.config.embedding_dim,
            ),
        )


if __name__ == "__main__":

    unittest.main()