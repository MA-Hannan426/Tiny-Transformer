"""
Evaluation metrics for TinyTransformerLM.
"""

import math
import torch


class EvaluationMetrics:
    """
    Computes common evaluation metrics for language models.
    """

    ##########################################################
    # Average Loss
    ##########################################################

    @staticmethod
    def average_loss(losses):
        """
        Compute the average loss.
        """

        if len(losses) == 0:
            return 0.0

        return sum(losses) / len(losses)

    ##########################################################
    # Perplexity
    ##########################################################

    @staticmethod
    def perplexity(loss):

        """
        Compute perplexity from cross-entropy loss.
        """

        return math.exp(loss)

    ##########################################################
    # Token Accuracy
    ##########################################################

    @staticmethod
    def token_accuracy(
        logits,
        targets,
    ):
        """
        Compute token-level accuracy.
        """

        predictions = torch.argmax(
            logits,
            dim=-1,
        )

        correct = (
            predictions == targets
        ).float()

        return correct.mean().item()

    ##########################################################
    # Metric Summary
    ##########################################################

    @staticmethod
    def summarize(
        loss,
        perplexity,
        accuracy,
    ):

        print("\nEvaluation Results")
        print("-" * 40)

        print(f"Loss        : {loss:.4f}")
        print(f"Perplexity  : {perplexity:.4f}")
        print(f"Accuracy    : {accuracy:.4f}")

        print("-" * 40)