"""
Batch loading utilities.

Provides random mini-batches for language model training.
"""

from pathlib import Path
import numpy as np
import torch


class BatchLoader:
    """
    Loads random batches from binary token files.
    """

    def __init__(
        self,
        data_directory,
        context_length,
        batch_size,
        device="cpu"
    ):

        self.data_directory = Path(data_directory)

        self.context_length = context_length

        self.batch_size = batch_size

        self.device = device

        self.datasets = {
            "train": np.memmap(
                self.data_directory / "train.bin",
                dtype=np.uint16,
                mode="r"
            ),

            "validation": np.memmap(
                self.data_directory / "validation.bin",
                dtype=np.uint16,
                mode="r"
            )
        }

    def get_batch(self, split="train"):
        """
        Sample one random batch.

        Parameters
        ----------
        split : str
            train or validation

        Returns
        -------
        input_tokens
        target_tokens
        """

        token_data = self.datasets[split]

        start_positions = torch.randint(
            0,
            len(token_data) - self.context_length - 1,
            (self.batch_size,)
        )

        input_tokens = torch.stack([
            torch.from_numpy(
                token_data[pos:pos+self.context_length].astype(np.int64)
            )
            for pos in start_positions
        ])

        target_tokens = torch.stack([
            torch.from_numpy(
                token_data[pos+1:pos+self.context_length+1].astype(np.int64)
            )
            for pos in start_positions
        ])

        return (
            input_tokens.to(self.device),
            target_tokens.to(self.device)
        )

    def describe(self):

        for split, data in self.datasets.items():

            print(f"{split}")

            print(f"Tokens : {len(data):,}")

            print(f"Size   : {data.nbytes / (1024**2):.2f} MB\n")

    def batches_per_epoch(self):

        total_tokens = len(self.datasets["train"])

        return total_tokens // (
            self.batch_size *
            self.context_length
        )