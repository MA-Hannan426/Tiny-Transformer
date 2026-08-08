"""
Training configuration for TinyTransformerLM.
"""

from dataclasses import dataclass


@dataclass
class TrainingConfig:

    # Training
    max_iterations: int = 20000
    batch_size: int = 32

    # Optimization
    learning_rate: float = 1e-4
    minimum_learning_rate: float = 5e-5

    # Warmup
    warmup_steps: int = 1000

    # Gradient
    gradient_accumulation_steps: int = 32
    gradient_clip: float = 0.5

    # Optimizer
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    epsilon: float = 1e-9

    # Evaluation
    evaluation_interval: int = 500

    # Randomness
    seed: int = 42