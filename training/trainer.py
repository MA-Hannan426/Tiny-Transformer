"""
Trainer module.

Coordinates the complete training process.
"""

import time
import torch

from training.optimizer import OptimizerFactory
from training.scheduler import LearningRateScheduler
from training.checkpoint import CheckpointManager


class Trainer:

    """
    Complete training engine for TinyTransformerLM.
    """

    def __init__(
        self,
        model,
        batch_loader,
        training_config,
        device="cpu",
    ):

        self.model = model

        self.batch_loader = batch_loader

        self.training_config = training_config

        self.device = device

        self.optimizer = (
            OptimizerFactory.create_adamw(
                model=model,
                training_config=training_config,
                device_type=device,
            )
        )

        self.scheduler = (
            LearningRateScheduler(
                training_config
            )
        )

        self.checkpoint_manager = (
            CheckpointManager()
        )

        self.best_validation_loss = float("inf")

    ##########################################################
    # Train Step
    ##########################################################

    def train_step(self):

        self.model.train()

        inputs, targets = (
            self.batch_loader.get_batch("train")
        )

        logits, loss = self.model(
            inputs,
            targets,
        )

        self.optimizer.zero_grad(
            set_to_none=True
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.training_config.gradient_clip,
        )

        self.optimizer.step()

        return loss.item()

    ##########################################################
    # Validation
    ##########################################################

    @torch.no_grad()
    def validation_step(self):

        self.model.eval()

        inputs, targets = (
            self.batch_loader.get_batch(
                "validation"
            )
        )

        _, loss = self.model(
            inputs,
            targets,
        )

        return loss.item()

    ##########################################################
    # Estimate Loss
    ##########################################################

    @torch.no_grad()
    def estimate_loss(self):

        train_loss = self.train_step()

        validation_loss = (
            self.validation_step()
        )

        return {
            "train": train_loss,
            "validation": validation_loss,
        }

    ##########################################################
    # Save
    ##########################################################

    def save_checkpoint(
        self,
        iteration,
        validation_loss,
    ):

        self.checkpoint_manager.save(

            model=self.model,

            optimizer=self.optimizer,

            scheduler=self.scheduler,

            iteration=iteration,

            validation_loss=validation_loss,
        )

        if validation_loss < self.best_validation_loss:

            self.best_validation_loss = (
                validation_loss
            )

            self.checkpoint_manager.save_best(

                model=self.model,

                optimizer=self.optimizer,

                scheduler=self.scheduler,

                iteration=iteration,

                validation_loss=validation_loss,
            )

    ##########################################################
    # Load
    ##########################################################

    def load_checkpoint(self):

        checkpoint = (
            self.checkpoint_manager.load(
                model=self.model,
                optimizer=self.optimizer,
                scheduler=self.scheduler,
                map_location=self.device,
            )
        )

        self.best_validation_loss = (
            checkpoint["validation_loss"]
        )

        return checkpoint["iteration"]

    ##########################################################
    # Training Loop
    ##########################################################

    def train(self):

        print("Training started...\n")

        start_time = time.time()

        for iteration in range(
            self.training_config.max_iterations
        ):

            learning_rate = (
                self.scheduler.update_optimizer(
                    self.optimizer,
                    iteration,
                )
            )

            train_loss = self.train_step()

            if (
                iteration %
                self.training_config.evaluation_interval
                == 0
            ):

                validation_loss = (
                    self.validation_step()
                )

                elapsed = (
                    time.time()
                    - start_time
                )

                print(
                    f"[{iteration:6d}] "
                    f"Train={train_loss:.4f} "
                    f"Valid={validation_loss:.4f} "
                    f"LR={learning_rate:.6f} "
                    f"Time={elapsed:.1f}s"
                )

                self.save_checkpoint(
                    iteration,
                    validation_loss,
                )

        print("\nTraining completed.")