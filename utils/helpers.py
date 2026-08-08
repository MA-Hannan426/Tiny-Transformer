"""
General helper functions.
"""

import os


class Helpers:

    """
    Miscellaneous utilities.
    """

    @staticmethod
    def ensure_directory(path):

        os.makedirs(
            path,
            exist_ok=True,
        )

    @staticmethod
    def format_time(seconds):

        hours = int(seconds // 3600)

        minutes = int(
            (seconds % 3600) // 60
        )

        seconds = int(seconds % 60)

        return (
            f"{hours:02d}:"
            f"{minutes:02d}:"
            f"{seconds:02d}"
        )

    @staticmethod
    def count_trainable_parameters(model):

        return sum(

            parameter.numel()

            for parameter in model.parameters()

            if parameter.requires_grad

        )