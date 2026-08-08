"""
Evaluation script for TinyTransformerLM.
"""

import torch

from configs import (
    ModelConfig,
    TrainingConfig,
)

from models.transformer import (
    TransformerLanguageModel,
)

from data.batch_loader import (
    BatchLoader,
)

from evaluation.metrics import (
    EvaluationMetrics,
)


class Evaluator:

    """
    Evaluate a trained language model.
    """

    def __init__(
        self,
        model,
        batch_loader,
        device="cpu",
    ):

        self.model = model

        self.batch_loader = batch_loader

        self.device = device

    ##########################################################
    # Evaluate
    ##########################################################

    @torch.no_grad()
    def evaluate(
        self,
        num_batches=20,
    ):

        self.model.eval()

        losses = []

        accuracies = []

        for _ in range(num_batches):

            inputs, targets = (
                self.batch_loader.get_batch(
                    "validation"
                )
            )

            logits, loss = self.model(
                inputs,
                targets,
            )

            losses.append(
                loss.item()
            )

            accuracies.append(

                EvaluationMetrics.token_accuracy(
                    logits,
                    targets,
                )
            )

        average_loss = (
            EvaluationMetrics.average_loss(
                losses
            )
        )

        perplexity = (
            EvaluationMetrics.perplexity(
                average_loss
            )
        )

        accuracy = (
            sum(accuracies)
            / len(accuracies)
        )

        EvaluationMetrics.summarize(
            average_loss,
            perplexity,
            accuracy,
        )

        return {

            "loss": average_loss,

            "perplexity": perplexity,

            "accuracy": accuracy,
        }