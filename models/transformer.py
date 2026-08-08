"""
TinyTransformerLM

Complete decoder-only Transformer language model.
"""

from __future__ import annotations

import inspect

import torch
import torch.nn as nn
import torch.nn.functional as F

from .embeddings import TransformerEmbedding
from .normalization import CustomLayerNorm
from .transformer_block import TransformerBlock


class TransformerLanguageModel(nn.Module):
    """
    Decoder-only Transformer Language Model.
    """

    def __init__(self, config):

        super().__init__()

        self.config = config

        self.embedding_layer = TransformerEmbedding(
            vocab_size=config.vocab_size,
            embedding_dim=config.embedding_dim,
            context_length=config.context_length,
            dropout=config.dropout,
        )

        self.transformer_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    embedding_dim=config.embedding_dim,
                    num_heads=config.num_heads,
                    context_length=config.context_length,
                    expansion_factor=config.expansion_factor,
                    dropout=config.dropout,
                    use_bias=config.use_bias,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.final_norm = CustomLayerNorm(
            embedding_dim=config.embedding_dim,
            use_bias=config.use_bias,
        )

        self.language_model_head = nn.Linear(
            config.embedding_dim,
            config.vocab_size,
            bias=False,
        )

        # Weight tying
        self.language_model_head.weight = (
            self.embedding_layer.token_embedding.weight
        )

        self.initialize_weights()


    ##########################################################
    # Initialization
    ##########################################################

    def initialize_weights(self):

        """
        Initialize model parameters.
        """

        for module in self.modules():

            if isinstance(module, nn.Linear):

                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.02,
                )

                if module.bias is not None:

                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Embedding):

                nn.init.normal_(
                    module.weight,
                    mean=0.0,
                    std=0.02,
                )

    ##########################################################
    # Embedding
    ##########################################################

    def build_embeddings(
        self,
        input_ids,
    ):

        return self.embedding_layer(input_ids)

    ##########################################################
    # Transformer Encoder Stack
    ##########################################################

    def run_transformer_blocks(
        self,
        hidden_states,
    ):

        for block in self.transformer_blocks:

            hidden_states = block(hidden_states)

        return hidden_states

    ##########################################################
    # Output Projection
    ##########################################################

    def project_to_vocabulary(
        self,
        hidden_states,
    ):

        hidden_states = self.final_norm(
            hidden_states
        )

        logits = self.language_model_head(
            hidden_states
        )

        return logits

    ##########################################################
    # Loss
    ##########################################################

    def compute_loss(
        self,
        logits,
        targets,
    ):

        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            targets.view(-1),
        )

    ##########################################################
    # Forward
    ##########################################################

    def forward(
        self,
        input_ids,
        targets=None,
    ):

        hidden_states = self.build_embeddings(
            input_ids
        )

        hidden_states = self.run_transformer_blocks(
            hidden_states
        )

        logits = self.project_to_vocabulary(
            hidden_states
        )

        loss = None

        if targets is not None:

            loss = self.compute_loss(
                logits,
                targets,
            )

        return logits, loss

    ##########################################################
    # Generation
    ##########################################################

    @torch.no_grad()
    def generate(
        self,
        input_ids,
        max_new_tokens,
        temperature=1.0,
        top_k=None,
    ):

        self.eval()

        for _ in range(max_new_tokens):

            context = input_ids[
                :,
                -self.config.context_length:,
            ]

            logits, _ = self(context)

            logits = logits[:, -1, :]

            logits = logits / temperature

            if top_k is not None:

                values, _ = torch.topk(
                    logits,
                    top_k,
                )

                logits[
                    logits < values[:, [-1]]
                ] = float("-inf")

            probabilities = F.softmax(
                logits,
                dim=-1,
            )

            next_token = torch.multinomial(
                probabilities,
                num_samples=1,
            )

            input_ids = torch.cat(
                (
                    input_ids,
                    next_token,
                ),
                dim=1,
            )

        return input_ids

    ##########################################################
    # Optimizer
    ##########################################################

    def configure_optimizer(
        self,
        training_config,
        device_type="cuda",
    ):

        parameters = [
            parameter
            for parameter in self.parameters()
            if parameter.requires_grad
        ]

        use_fused = (
            device_type == "cuda"
            and "fused" in inspect.signature(
                torch.optim.AdamW
            ).parameters
        )

        optimizer = torch.optim.AdamW(
            parameters,
            lr=training_config.learning_rate,
            betas=(
                training_config.beta1,
                training_config.beta2,
            ),
            eps=training_config.epsilon,
            weight_decay=training_config.weight_decay,
            fused=use_fused,
        )

        return optimizer

    ##########################################################
    # Utilities
    ##########################################################

    def count_parameters(
        self,
        trainable_only=True,
    ):

        if trainable_only:

            return sum(
                parameter.numel()
                for parameter in self.parameters()
                if parameter.requires_grad
            )

        return sum(
            parameter.numel()
            for parameter in self.parameters()
        )