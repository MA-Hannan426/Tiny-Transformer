"""
Optimizer utilities.

Creates and configures optimizers used for training.
"""

import inspect
import torch


class OptimizerFactory:
    """
    Factory class for creating optimizers.
    """

    @staticmethod
    def create_adamw(
        model,
        training_config,
        device_type="cuda",
    ):
        """
        Create an AdamW optimizer.

        Parameters
        ----------
        model : nn.Module
            Model to optimize.

        training_config : TrainingConfig
            Training configuration.

        device_type : str
            "cuda" or "cpu".

        Returns
        -------
        torch.optim.AdamW
        """

        parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ]

        supports_fused = (
            device_type == "cuda"
            and "fused"
            in inspect.signature(
                torch.optim.AdamW
            ).parameters
        )

        optimizer = torch.optim.AdamW(
            params=parameters,
            lr=training_config.learning_rate,
            betas=(
                training_config.beta1,
                training_config.beta2,
            ),
            eps=training_config.epsilon,
            weight_decay=training_config.weight_decay,
            fused=supports_fused,
        )

        return optimizer