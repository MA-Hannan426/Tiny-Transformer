"""
Learning rate scheduler.

Implements linear warmup followed by cosine learning rate decay.
"""

import math


class LearningRateScheduler:
    """
    Linear warmup + cosine decay scheduler.
    """

    def __init__(self, training_config):

        self.learning_rate = training_config.learning_rate

        self.minimum_learning_rate = (
            training_config.minimum_learning_rate
        )

        self.warmup_steps = (
            training_config.warmup_steps
        )

        self.max_iterations = (
            training_config.max_iterations
        )

    ############################################################
    # Warmup
    ############################################################

    def warmup_learning_rate(
        self,
        current_iteration,
    ):
        """
        Compute learning rate during warmup.
        """

        return (
            self.learning_rate
            * (current_iteration + 1)
            / self.warmup_steps
        )

    ############################################################
    # Cosine Decay
    ############################################################

    def cosine_decay_learning_rate(
        self,
        current_iteration,
    ):
        """
        Compute cosine-decayed learning rate.
        """

        decay_progress = (
            current_iteration - self.warmup_steps
        ) / (
            self.max_iterations
            - self.warmup_steps
        )

        decay_progress = min(
            max(decay_progress, 0.0),
            1.0,
        )

        cosine_factor = (
            0.5
            * (
                1.0
                + math.cos(
                    math.pi * decay_progress
                )
            )
        )

        return (
            self.minimum_learning_rate
            + cosine_factor
            * (
                self.learning_rate
                - self.minimum_learning_rate
            )
        )

    ############################################################
    # Public API
    ############################################################

    def get_learning_rate(
        self,
        current_iteration,
    ):
        """
        Return the learning rate for the current iteration.
        """

        # Warmup
        if current_iteration < self.warmup_steps:

            return self.warmup_learning_rate(
                current_iteration
            )

        # Finished training
        if current_iteration >= self.max_iterations:

            return self.minimum_learning_rate

        # Cosine decay
        return self.cosine_decay_learning_rate(
            current_iteration
        )

    ############################################################
    # Optimizer Update
    ############################################################

    def update_optimizer(
        self,
        optimizer,
        current_iteration,
    ):
        """
        Update optimizer learning rate.
        """

        learning_rate = self.get_learning_rate(
            current_iteration
        )

        for parameter_group in optimizer.param_groups:

            parameter_group["lr"] = learning_rate

        return learning_rate