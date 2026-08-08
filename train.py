"""
Training entry point for TinyTransformerLM.
"""

import torch

from configs import (
    ModelConfig,
    TrainingConfig,
)

from data.batch_loader import BatchLoader

from models.transformer import (
    TransformerLanguageModel,
)

from training.trainer import Trainer


def main():

    ##########################################################
    # Device
    ##########################################################

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nUsing device: {device}\n")

    ##########################################################
    # Configuration
    ##########################################################

    model_config = ModelConfig()

    training_config = TrainingConfig()

    ##########################################################
    # Batch Loader
    ##########################################################

    batch_loader = BatchLoader(

        data_directory="data/processed",

        context_length=model_config.context_length,

        batch_size=training_config.batch_size,

        device=device,
    )

    ##########################################################
    # Model
    ##########################################################

    model = TransformerLanguageModel(
        model_config
    )

    model.to(device)

    print(
        f"Trainable Parameters: "
        f"{model.count_parameters():,}"
    )

    ##########################################################
    # Trainer
    ##########################################################

    trainer = Trainer(

        model=model,

        batch_loader=batch_loader,

        training_config=training_config,

        device=device,
    )

    ##########################################################
    # Train
    ##########################################################

    trainer.train()


if __name__ == "__main__":

    main()