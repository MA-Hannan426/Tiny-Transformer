<<<<<<< HEAD
# Tiny-Transformer
Tiny transformer is a small language model build from scratch
=======
# TinyTransformerLM

A modular decoder-only Transformer Language Model implemented in PyTorch.

---

## Features

- Modular Transformer implementation
- Multi-Head Self Attention
- Feed Forward Network
- Layer Normalization
- Positional Embeddings
- AdamW Optimizer
- Cosine Learning Rate Scheduler
- Checkpoint Management
- Evaluation Pipeline
- Text Generation
- Interactive Chat Interface
- Unit Tests

---

## Project Structure

TinyTransformerLM/

├── configs/

├── data/

├── models/

├── training/

├── inference/

├── evaluation/

├── utils/

├── tests/

├── train.py

└── README.md

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/TinyTransformerLM.git

cd TinyTransformerLM

pip install -r requirements.txt
```

---

## Training

```bash
python train.py
```

---

## Evaluate

```bash
python evaluation/evaluate.py
```

---

## Chat

```bash
python inference/chat.py
```

---

## Generate Text

```python
from inference.predictor import Predictor

predictor = Predictor(
    "checkpoints/best_model.pt"
)

text = predictor.predict(
    "Once upon a time"
)

print(text)
```

---

## Architecture

```
Input Tokens
      │
      ▼
Token Embedding
      │
      ▼
Position Embedding
      │
      ▼
Transformer Blocks
      │
      ▼
LayerNorm
      │
      ▼
Linear Head
      │
      ▼
Vocabulary Logits
```

---

## Model Pipeline

```
Training Data
      │
      ▼
Batch Loader
      │
      ▼
TransformerLanguageModel
      │
      ▼
Optimizer
      │
      ▼
Scheduler
      │
      ▼
Checkpoint
```

---

## Future Improvements

- Flash Attention
- Mixed Precision Training (AMP)
- Distributed Data Parallel (DDP)
- Beam Search
- Top-p Sampling
- ONNX Export
- Hugging Face Integration

---

## License

MIT License
>>>>>>> 764d50e (Final commit)
