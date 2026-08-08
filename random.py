from configs import ModelConfig, TrainingConfig, GenerationConfig

model_config = ModelConfig()
training_config = TrainingConfig()
generation_config = GenerationConfig()


# data
from data import TinyStoriesDataset
from data import GPTTokenizer

# preprocessor

from data import TinyStoriesDataset
from data import GPTTokenizer
from data.preprocessing import DataPreprocessor

# tokenizer

dataset = TinyStoriesDataset().load()

tokenizer = GPTTokenizer()

preprocessor = DataPreprocessor(tokenizer)

tokenized_dataset = preprocessor.tokenize_dataset(dataset)

preprocessor.save_binary_dataset(
    tokenized_dataset,
    "data/processed"
)


# batch loader
from data.batch_loader import BatchLoader

loader = BatchLoader(
    data_directory="data/processed",
    context_length=128,
    batch_size=32,
    device="cuda"
)

inputs, targets = loader.get_batch("train")


# models - normalization.py
import torch

from models import CustomLayerNorm

layer = CustomLayerNorm(384)

sample = torch.randn(2, 128, 384)

output = layer(sample)

print(output.shape)


