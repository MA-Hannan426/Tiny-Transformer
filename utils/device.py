"""
Device utilities.
"""

import torch


class DeviceManager:
    """
    Handles device selection.
    """

    @staticmethod
    def get_device():

        return (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    @staticmethod
    def print_device():

        device = DeviceManager.get_device()

        print(f"Using device: {device}")

        return device