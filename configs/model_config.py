"""
Model configuration for TinyTransformerLM.

Defines all hyperparameters related to the Transformer architecture.
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    # Vocabulary
    vocab_size: int = 50257

    # Context Window
    context_length: int = 128

    # Transformer Architecture
    num_layers: int = 6
    num_heads: int = 6
    embedding_dim: int = 384

    # Feed Forward Network
    expansion_factor: int = 4

    # Regularization
    dropout: float = 0.1
    use_bias: bool = True