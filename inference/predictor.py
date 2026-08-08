"""
Prediction utilities for TinyTransformerLM.

Provides a reusable interface for loading a trained
model and generating text.
"""

import torch

from configs import (
    ModelConfig,
    GenerationConfig,
)

from data.tokenizer import GPTTokenizer
from models.transformer import TransformerLanguageModel


class Predictor:
    """
    Reusable prediction interface.
    """

    def __init__(
        self,
        checkpoint_path,
        device=None,
    ):

        self.device = (
            device
            if device is not None
            else (
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )
        )

        self.model_config = ModelConfig()

        self.generation_config = GenerationConfig()

        self.tokenizer = GPTTokenizer()

        self.model = TransformerLanguageModel(
            self.model_config
        )

        self.load_model(
            checkpoint_path
        )

    ##########################################################
    # Load Model
    ##########################################################

    def load_model(
        self,
        checkpoint_path,
    ):

        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.to(self.device)

        self.model.eval()

    ##########################################################
    # Encode
    ##########################################################

    def encode(
        self,
        text,
    ):

        tokens = self.tokenizer.encode(text)

        return torch.tensor(
            [tokens],
            dtype=torch.long,
            device=self.device,
        )

    ##########################################################
    # Decode
    ##########################################################

    def decode(
        self,
        token_ids,
    ):

        return self.tokenizer.decode(
            token_ids.tolist()
        )

    ##########################################################
    # Predict
    ##########################################################

    @torch.no_grad()
    def predict(
        self,
        prompt,
        max_new_tokens=None,
        temperature=None,
        top_k=None,
    ):

        input_ids = self.encode(
            prompt
        )

        output_ids = self.model.generate(

            input_ids,

            max_new_tokens=(
                max_new_tokens
                or self.generation_config.max_new_tokens
            ),

            temperature=(
                temperature
                or self.generation_config.temperature
            ),

            top_k=(
                top_k
                or self.generation_config.top_k
            ),
        )

        return self.decode(
            output_ids[0]
        )

    ##########################################################
    # Batch Prediction
    ##########################################################

    @torch.no_grad()
    def batch_predict(
        self,
        prompts,
    ):

        outputs = []

        for prompt in prompts:

            outputs.append(
                self.predict(prompt)
            )

        return outputs