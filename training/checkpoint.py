"""
Checkpoint management utilities.

Handles saving and loading training checkpoints.
"""

from pathlib import Path
import torch


class CheckpointManager:
    """
    Save and load model checkpoints.
    """

    def __init__(
        self,
        checkpoint_directory="checkpoints",
    ):

        self.checkpoint_directory = Path(
            checkpoint_directory
        )

        self.checkpoint_directory.mkdir(
            parents=True,
            exist_ok=True,
        )

    ############################################################
    # Save Checkpoint
    ############################################################

    def save(
        self,
        model,
        optimizer,
        iteration,
        validation_loss,
        scheduler=None,
        filename="latest_checkpoint.pt",
    ):
        """
        Save training checkpoint.
        """

        checkpoint = {

            "iteration": iteration,

            "validation_loss": validation_loss,

            "model_state_dict":
                model.state_dict(),

            "optimizer_state_dict":
                optimizer.state_dict(),
        }

        if scheduler is not None:

            checkpoint[
                "scheduler_state_dict"
            ] = scheduler.__dict__

        checkpoint_path = (
            self.checkpoint_directory
            / filename
        )

        torch.save(
            checkpoint,
            checkpoint_path,
        )

        print(
            f"Checkpoint saved: {checkpoint_path}"
        )

    ############################################################
    # Load Checkpoint
    ############################################################

    def load(
        self,
        model,
        optimizer=None,
        scheduler=None,
        filename="latest_checkpoint.pt",
        map_location="cpu",
    ):
        """
        Load training checkpoint.
        """

        checkpoint_path = (
            self.checkpoint_directory
            / filename
        )

        if not checkpoint_path.exists():

            raise FileNotFoundError(
                f"No checkpoint found at {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
        )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        if optimizer is not None:

            optimizer.load_state_dict(
                checkpoint[
                    "optimizer_state_dict"
                ]
            )

        if (
            scheduler is not None
            and
            "scheduler_state_dict"
            in checkpoint
        ):

            scheduler.__dict__.update(
                checkpoint[
                    "scheduler_state_dict"
                ]
            )

        print(
            f"Checkpoint loaded: {checkpoint_path}"
        )

        return checkpoint

    ############################################################
    # Save Best Model
    ############################################################

    def save_best(
        self,
        model,
        optimizer,
        iteration,
        validation_loss,
        scheduler=None,
    ):
        """
        Save the best-performing model.
        """

        self.save(
            model=model,
            optimizer=optimizer,
            iteration=iteration,
            validation_loss=validation_loss,
            scheduler=scheduler,
            filename="best_model.pt",
        )

    ############################################################
    # Save Epoch Checkpoint
    ############################################################

    def save_epoch(
        self,
        model,
        optimizer,
        iteration,
        validation_loss,
        scheduler=None,
    ):
        """
        Save checkpoint for current iteration.
        """

        filename = (
            f"checkpoint_{iteration}.pt"
        )

        self.save(
            model=model,
            optimizer=optimizer,
            iteration=iteration,
            validation_loss=validation_loss,
            scheduler=scheduler,
            filename=filename,
        )

    ############################################################
    # Latest Checkpoint
    ############################################################

    def latest_checkpoint(self):
        """
        Return latest checkpoint file.
        """

        checkpoints = sorted(
            self.checkpoint_directory.glob(
                "*.pt"
            )
        )

        if len(checkpoints) == 0:

            return None

        return checkpoints[-1]

    ############################################################
    # List Checkpoints
    ############################################################

    def list_checkpoints(self):
        """
        Print all available checkpoints.
        """

        checkpoints = sorted(
            self.checkpoint_directory.glob(
                "*.pt"
            )
        )

        if not checkpoints:

            print(
                "No checkpoints found."
            )

            return

        print(
            "\nAvailable Checkpoints:\n"
        )

        for checkpoint in checkpoints:

            print(checkpoint.name)