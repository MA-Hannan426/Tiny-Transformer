"""
Configuration used during text generation.
"""

from dataclasses import dataclass


@dataclass
class GenerationConfig:

    max_new_tokens: int = 200

    temperature: float = 0.8

    top_k: int = 40

    do_sample: bool = True