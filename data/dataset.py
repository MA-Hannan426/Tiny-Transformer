"""
Dataset loading utilities.

Responsible for downloading and loading datasets
from Hugging Face.
"""

from datasets import load_dataset


class TinyStoriesDataset:
    """
    Loads the TinyStories dataset from Hugging Face.
    """

    def __init__(self):

        self.dataset_name = "roneneldan/TinyStories"

    def load(self):
        """
        Download and return the dataset.

        Returns
        -------
        DatasetDict
            Hugging Face DatasetDict
        """

        print(f"Loading dataset: {self.dataset_name}")

        dataset = load_dataset(self.dataset_name)

        print("Dataset loaded successfully.")

        return dataset