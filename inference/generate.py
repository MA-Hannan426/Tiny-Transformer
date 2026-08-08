"""
Generate text using a trained TinyTransformerLM model.
"""

import torch

from configs import (
    ModelConfig,
    GenerationConfig,
)

from models.transformer import (
    TransformerLanguageModel,
)

from data.tokenizer import GPTTokenizer


class TextGenerator:

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
    # Encode Prompt
    ##########################################################

    def encode_prompt(
        self,
        prompt,
    ):

        tokens = self.tokenizer.encode(prompt)

        return torch.tensor(
            [tokens],
            dtype=torch.long,
            device=self.device,
        )

    ##########################################################
    # Decode Output
    ##########################################################

    def decode_tokens(
        self,
        token_ids,
    ):

        return self.tokenizer.decode(
            token_ids.tolist()
        )

    ##########################################################
    # Generate
    ##########################################################

    @torch.no_grad()
    def generate(
        self,
        prompt,
    ):

        input_ids = self.encode_prompt(
            prompt
        )

        output_ids = self.model.generate(

            input_ids,

            max_new_tokens=self.generation_config.max_new_tokens,

            temperature=self.generation_config.temperature,

            top_k=self.generation_config.top_k,
        )

        return self.decode_tokens(
            output_ids[0]
        )