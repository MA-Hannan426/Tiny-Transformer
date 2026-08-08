"""
Data preprocessing utilities.

This module converts raw text into token IDs and
stores the processed dataset in binary format.
"""

from pathlib import Path
import numpy as np
from tqdm.auto import tqdm


class DataPreprocessor:
    """
    Preprocesses text datasets for language model training.
    """

    def __init__(self, tokenizer):

        self.tokenizer = tokenizer

    def tokenize_sample(self, sample):
        """
        Convert one text sample into token IDs.
        """

        token_ids = self.tokenizer.encode(sample["text"])

        return {
            "token_ids": token_ids,
            "sequence_length": len(token_ids)
        }

    def tokenize_dataset(self, dataset):
        """
        Tokenize every split in the dataset.
        """

        return dataset.map(
            self.tokenize_sample,
            remove_columns=["text"],
            desc="Tokenizing dataset",
            num_proc=8
        )

    def save_binary_dataset(
        self,
        tokenized_dataset,
        output_directory
    ):
        """
        Save every dataset split as a binary file.
        """

        output_directory = Path(output_directory)
        output_directory.mkdir(parents=True, exist_ok=True)

        for split_name, split_data in tokenized_dataset.items():

            total_tokens = np.sum(
                split_data["sequence_length"],
                dtype=np.uint64
            )

            file_path = output_directory / f"{split_name}.bin"

            binary_array = np.memmap(
                file_path,
                dtype=np.uint16,
                mode="w+",
                shape=(total_tokens,)
            )

            write_position = 0
            number_of_batches = 1024

            for batch_index in tqdm(
                range(number_of_batches),
                desc=f"Saving {split_name}"
            ):

                batch = (
                    split_data
                    .shard(
                        num_shards=number_of_batches,
                        index=batch_index,
                        contiguous=True
                    )
                    .with_format("numpy")
                )

                token_batch = np.concatenate(batch["token_ids"])

                binary_array[
                    write_position:
                    write_position + len(token_batch)
                ] = token_batch

                write_position += len(token_batch)

            binary_array.flush()


    def dataset_statistics(self, tokenized_dataset):

        for split_name, split_data in tokenized_dataset.items():

            lengths = split_data["sequence_length"]

            print(f"\n{split_name.upper()}")

            print(f"Samples : {len(split_data):,}")

            print(f"Tokens  : {sum(lengths):,}")

            print(f"Average : {np.mean(lengths):.2f}")

            print(f"Maximum : {np.max(lengths)}")

            print(f"Minimum : {np.min(lengths)}")