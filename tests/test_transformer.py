"""
Unit tests for TransformerLanguageModel.
"""

import unittest
import torch

from configs import ModelConfig
from models.transformer import TransformerLanguageModel


class TestTransformer(unittest.TestCase):

    def setUp(self):

        self.config = ModelConfig()

        self.model = TransformerLanguageModel(
            self.config
        )

    def test_forward(self):

        inputs = torch.randint(

            0,

            self.config.vocab_size,

            (
                2,
                self.config.context_length,
            ),
        )

        logits, loss = self.model(inputs)

        self.assertEqual(

            logits.shape,

            (

                2,

                self.config.context_length,

                self.config.vocab_size,

            ),
        )

        self.assertIsNone(loss)


if __name__ == "__main__":

    unittest.main()