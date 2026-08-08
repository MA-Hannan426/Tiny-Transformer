# 🧠 TinyTransformerLM

### A Decoder-Only Transformer Language Model Built from Scratch in PyTorch

TinyTransformerLM is a compact **decoder-only Transformer language model implemented from scratch using PyTorch**.

The project recreates the core architecture and training workflow behind modern autoregressive language models without relying on high-level Transformer model implementations.

It includes custom implementations of:

* Multi-head causal self-attention
* Token and positional embeddings
* Transformer decoder blocks
* Layer normalization
* Feed-forward networks
* Weight tying
* Autoregressive text generation
* Custom training loops
* AdamW optimization
* Learning-rate warmup and cosine decay
* Gradient clipping
* Checkpoint management
* Evaluation metrics
* Interactive text generation

The model is designed as an educational and engineering project for understanding the internal mechanics of decoder-only language models.

---

## ✨ Why This Project?

Modern language models are often consumed through high-level APIs, making it easy to use them without understanding what happens underneath.

TinyTransformer takes the opposite approach.

Instead of using a pre-built Transformer implementation, the project builds the core architecture manually:

```text
Input Tokens
      ↓
Token Embeddings
      +
Position Embeddings
      ↓
Transformer Blocks
      │
      ├── LayerNorm
      ├── Causal Multi-Head Attention
      ├── Residual Connection
      ├── LayerNorm
      ├── Feed-Forward Network
      └── Residual Connection
      │
      ↓
Final LayerNorm
      ↓
Language Model Head
      ↓
Vocabulary Logits
      ↓
Next-Token Prediction
```

The goal is not to reproduce a production-scale LLM, but to provide a clean implementation that makes the architecture and training process understandable.

---

# 🚀 Key Features

### Transformer Architecture

* Decoder-only architecture
* Multi-head self-attention
* Causal attention masking
* Token embeddings
* Learned positional embeddings
* Pre-LayerNorm architecture
* Residual connections
* GELU activation
* Feed-forward expansion
* Final normalization
* Weight tying between token embeddings and output projection

### Attention

* Combined Query/Key/Value projection
* Scaled dot-product attention
* Causal masking
* Multi-head attention
* PyTorch scaled dot-product attention support
* Manual attention fallback

### Training

* Custom PyTorch training loop
* AdamW optimizer
* Weight decay
* Gradient clipping
* Learning-rate warmup
* Cosine learning-rate decay
* Validation loss monitoring
* Best-model checkpointing
* Training checkpoints
* Resume-from-checkpoint support

### Data

* TinyStories dataset
* GPT-2 tokenizer through `tiktoken`
* Parallel tokenization
* Binary token storage
* NumPy memory-mapped datasets
* Random context-window sampling

### Inference

* Autoregressive generation
* Temperature sampling
* Top-k sampling
* Context-window truncation
* Reusable prediction interface
* Interactive CLI chat

### Evaluation

* Cross-entropy loss
* Perplexity
* Token-level accuracy
* Validation evaluation loop

### Testing

Unit tests cover:

* Attention
* Embeddings
* Feed-forward network
* Transformer blocks
* Transformer model
* Training utilities

---

# 🏗️ Model Architecture

The model follows a GPT-style decoder-only architecture.

```text
                     Input Token IDs
                            │
                            ▼
                 ┌────────────────────┐
                 │ Token Embedding    │
                 └─────────┬──────────┘
                           │
                           +
                           │
                 ┌─────────▼──────────┐
                 │ Position Embedding │
                 └─────────┬──────────┘
                           │
                           ▼
              ┌──────────────────────────┐
              │   Transformer Block 1    │
              │                          │
              │  LayerNorm               │
              │      ↓                   │
              │  Causal Self-Attention  │
              │      ↓                   │
              │  Residual                │
              │      ↓                   │
              │  LayerNorm               │
              │      ↓                   │
              │  Feed Forward + GELU     │
              │      ↓                   │
              │  Residual                │
              └────────────┬─────────────┘
                           │
                           ▼
                    ... repeated ...
                           │
                           ▼
              ┌──────────────────────────┐
              │   Transformer Block 6    │
              └────────────┬─────────────┘
                           │
                           ▼
                    Final LayerNorm
                           │
                           ▼
                  Language Model Head
                           │
                           ▼
                   Vocabulary Logits
                           │
                           ▼
                    Next Token
```

---

# ⚙️ Model Configuration

The default model configuration is:

| Parameter              |   Value |
| ---------------------- | ------: |
| Vocabulary size        |  50,257 |
| Context length         |     128 |
| Transformer layers     |       6 |
| Attention heads        |       6 |
| Embedding dimension    |     384 |
| Feed-forward expansion |      4× |
| Dropout                |     0.1 |
| Bias                   | Enabled |

The resulting model contains approximately:

**29.97 million trainable parameters**

The output language-model head shares its weights with the token embedding matrix through **weight tying**, reducing the number of independent parameters.

---

# 🔍 Multi-Head Causal Self-Attention

The attention mechanism is implemented directly in:

```text
models/attention.py
```

The module performs:

```text
Hidden States
      ↓
QKV Projection
      ↓
Split into Q / K / V
      ↓
Multiple Attention Heads
      ↓
Scaled Dot-Product Attention
      ↓
Causal Masking
      ↓
Merge Attention Heads
      ↓
Output Projection
```

For a sequence, causal attention prevents a token from accessing future tokens.

For example:

```text
Token 1 → Token 1

Token 2 → Token 1, Token 2

Token 3 → Token 1, Token 2, Token 3

Token 4 → Token 1, Token 2, Token 3, Token 4
```

This is essential for autoregressive next-token prediction.

---

# ⚡ Flash / Scaled Dot-Product Attention

When supported by the installed PyTorch version, the project uses:

```python
torch.nn.functional.scaled_dot_product_attention()
```

with:

```python
is_causal=True
```

The implementation also contains a manual attention path as a fallback.

This gives the project both:

* A readable implementation of the attention mechanism
* Compatibility with optimized PyTorch attention primitives

---

# 🧱 Transformer Block

Each Transformer block follows a **pre-normalization** architecture:

```text
Input
  │
  ├───────────────┐
  │               │
  ▼               │
LayerNorm         │
  ↓               │
Self-Attention    │
  │               │
  └─────── + ◄────┘
          │
          ├───────────────┐
          │               │
          ▼               │
      LayerNorm           │
          ↓               │
     Feed Forward         │
          │               │
          └─────── + ◄─────┘
                  │
                  ▼
                Output
```

The feed-forward network expands the embedding dimension by a factor of four before projecting it back to the original dimension.

---

# 📚 Dataset

The project uses the **TinyStories** dataset:

```text
roneneldan/TinyStories
```

The dataset is downloaded through the Hugging Face `datasets` library and converted into token sequences using the GPT-2 tokenizer provided by `tiktoken`.

The processing workflow is:

```text
TinyStories
     ↓
Dataset Loading
     ↓
GPT-2 Tokenization
     ↓
Token Statistics
     ↓
Binary Conversion
     ↓
Memory-Mapped .bin Files
```

---

# 🔤 Tokenization

The project uses:

```python
tiktoken.get_encoding("gpt2")
```

The tokenizer converts text into integer token IDs.

For example:

```text
"Once upon a time..."
        ↓
[7454, 2402, 257, ...]
```

The model operates entirely on these token IDs.

The tokenizer also supports decoding generated token sequences back into human-readable text.

---

# 💾 Memory-Mapped Dataset

Instead of loading the entire processed dataset into RAM, tokenized data is stored as binary files:

```text
data/processed/
├── train.bin
└── validation.bin
```

The `BatchLoader` uses NumPy memory mapping:

```python
np.memmap(...)
```

This allows the training process to access portions of large token arrays without loading the entire dataset into memory.

Random context windows are sampled during training.

---

# 🔄 Training Data Flow

For every training batch:

```text
Binary Token Dataset
        ↓
Random Starting Positions
        ↓
Input Sequence
        ↓
Target Sequence
        ↓
Transformer
        ↓
Vocabulary Logits
        ↓
Cross-Entropy Loss
        ↓
Backpropagation
        ↓
Gradient Clipping
        ↓
AdamW Update
```

The target sequence is shifted by one token relative to the input.

Example:

```text
Input:
The cat sat on

Target:
cat sat on the
```

This teaches the model to predict the next token at every position.

---

# 🎯 Training Configuration

Default training configuration:

| Parameter             |  Value |
| --------------------- | -----: |
| Maximum iterations    | 20,000 |
| Batch size            |     32 |
| Learning rate         |   1e-4 |
| Minimum learning rate |   5e-5 |
| Warmup steps          |  1,000 |
| Gradient clipping     |    0.5 |
| Weight decay          |    0.1 |
| Adam β₁               |    0.9 |
| Adam β₂               |   0.95 |
| Evaluation interval   |    500 |
| Random seed           |     42 |

---

# 📉 Learning Rate Schedule

The project implements a two-stage learning-rate schedule:

```text
Learning Rate
     │
     │       /\
     │      /  \
     │     /    \
     │    /      \____
     │   /             \____
     │  /
     └─────────────────────────
        Warmup      Cosine Decay
```

### Stage 1 — Linear Warmup

The learning rate gradually increases during the first 1,000 iterations.

### Stage 2 — Cosine Decay

After warmup, the learning rate gradually decreases toward the configured minimum learning rate.

This provides a simple and commonly used optimization strategy for neural language-model training.

---

# 💾 Checkpoint Management

Training checkpoints are handled by:

```text
training/checkpoint.py
```

The system supports:

* Latest checkpoint
* Best model checkpoint
* Iteration checkpoints
* Model state
* Optimizer state
* Validation loss
* Scheduler state

Example:

```text
checkpoints/
├── best_model.pt
├── latest_checkpoint.pt
└── checkpoint_500.pt
```

The best model is selected based on validation loss.

---

# ✍️ Text Generation

After training, the model can generate text autoregressively.

Generation follows:

```text
Prompt
  ↓
Tokenize
  ↓
Transformer
  ↓
Next-token probabilities
  ↓
Sampling
  ↓
Append token
  ↓
Repeat
```

The generation process supports:

* Temperature
* Top-k sampling
* Maximum generated tokens
* Context-window truncation

Default generation configuration:

| Parameter          | Value |
| ------------------ | ----: |
| Maximum new tokens |   200 |
| Temperature        |   0.8 |
| Top-k              |    40 |

---

# 💬 Interactive Chat

The project includes a command-line chat interface.

Start it with:

```bash
python inference/inference.py
```

The interface provides:

```text
============================================================
        TinyTransformerLM Interactive Chat
============================================================

You : Once upon a time

Model :
...
```

Type:

```text
exit
```

or:

```text
quit
```

to end the session.

---

# 📊 Evaluation

The project evaluates the language model using:

### Cross-Entropy Loss

Measures how well the model predicts the target token distribution.

Lower is better.

### Perplexity

Calculated as:

```text
Perplexity = exp(Loss)
```

Lower perplexity generally indicates better language-model prediction performance.

### Token Accuracy

Measures the proportion of tokens for which the highest-probability prediction matches the target token.

---

# 🧪 Evaluation Workflow

```text
Validation Dataset
        ↓
Random Validation Batches
        ↓
Transformer
        ↓
Predictions
        ↓
Cross-Entropy Loss
        ↓
Perplexity
        ↓
Token Accuracy
        ↓
Evaluation Summary
```

Run evaluation with:

```bash
python evaluation/evaluate.py
```

> Training/evaluation metrics should be reported only after running the current repository configuration and recording the resulting values.

---

# 🧪 Testing

The repository includes unit tests for important model components.

Current tests cover:

```text
tests/
├── test_attention.py
├── test_embeddings.py
├── test_feedforward.py
├── test_model.py
├── test_training.py
├── test_transformer.py
└── test_transformer_block.py
```

Run the test suite with:

```bash
pytest
```

---

# 📁 Project Structure

```text
TinyTransformerLM/
│
├── configs/
│   ├── model_config.py
│   ├── training_config.py
│   └── generation_config.py
│
├── data/
│   ├── batch_loader.py
│   ├── dataset.py
│   ├── preprocessing.py
│   ├── tokenizer.py
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── attention.py
│   ├── embeddings.py
│   ├── feedforward.py
│   ├── normalization.py
│   ├── transformer.py
│   └── transformer_block.py
│
├── training/
│   ├── checkpoint.py
│   ├── losses.py
│   ├── optimizer.py
│   ├── scheduler.py
│   └── trainer.py
│
├── evaluation/
│   ├── evaluate.py
│   └── metrics.py
│
├── inference/
│   ├── generate.py
│   ├── inference.py
│   └── predictor.py
│
├── utils/
│   ├── device.py
│   ├── helpers.py
│   ├── logger.py
│   └── seed.py
│
├── tests/
│
├── notebooks/
│   └── experiments.ipynb
│
├── checkpoints/
├── outputs/
│   ├── figures/
│   ├── generated_text/
│   └── logs/
│
├── train.py
├── requirements.txt
└── README.md
```

---

# 🛠️ Technology Stack

| Category            | Technologies                        |
| ------------------- | ----------------------------------- |
| Language            | Python                              |
| Deep Learning       | PyTorch                             |
| Tokenization        | tiktoken                            |
| Dataset             | Hugging Face Datasets / TinyStories |
| Numerical Computing | NumPy                               |
| Visualization       | Matplotlib                          |
| Progress Tracking   | tqdm                                |
| Testing             | pytest                              |
| Architecture        | Decoder-only Transformer            |

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/TinyTransformerLM.git
cd TinyTransformerLM
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows:

```bash
.venv\Scripts\activate
```

Linux/macOS:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 📥 Prepare the Dataset

The project uses the TinyStories dataset.

The preprocessing pipeline tokenizes the dataset and creates binary training files.

The resulting files should be placed under:

```text
data/processed/
├── train.bin
└── validation.bin
```

The project uses memory-mapped arrays for efficient batch sampling.

---

# 🏋️ Train the Model

Run:

```bash
python train.py
```

The training script:

1. Detects CPU/GPU availability.
2. Loads model configuration.
3. Loads training configuration.
4. Initializes the batch loader.
5. Creates the Transformer.
6. Initializes the optimizer and scheduler.
7. Runs the training loop.
8. Evaluates periodically.
9. Saves checkpoints.
10. Preserves the best-performing model.

---

# 🔮 Generate Text

Using the reusable predictor:

```python
from inference.predictor import Predictor

predictor = Predictor(
    "checkpoints/best_model.pt"
)

result = predictor.predict(
    "Once upon a time"
)

print(result)
```

You can also control generation:

```python
result = predictor.predict(
    "Once upon a time",
    max_new_tokens=100,
    temperature=0.8,
    top_k=40,
)
```

---

# 🧠 What This Project Demonstrates

This project demonstrates understanding of several fundamental concepts behind modern language models:

### Transformer Fundamentals

* Self-attention
* Multi-head attention
* Causal masking
* Positional representations
* Residual connections
* Layer normalization
* Feed-forward networks

### LLM Training

* Next-token prediction
* Cross-entropy language modeling
* AdamW optimization
* Learning-rate scheduling
* Gradient clipping
* Validation
* Checkpointing

### ML Systems Engineering

* Modular architecture
* Config-driven design
* Efficient dataset storage
* Memory-mapped data
* Reusable inference APIs
* Automated tests
* Device management
* Reproducibility

---

# ⚠️ Limitations

TinyTransformerLM is a **small research/educational language model**, not a production-scale LLM.

Important limitations include:

* Approximately 30M parameters.
* Context window limited to 128 tokens.
* Training data is significantly smaller than modern foundation-model datasets.
* Training compute is limited compared with large language models.
* Generation quality depends heavily on training duration and dataset coverage.
* The model has no instruction tuning or RLHF.
* It has no retrieval system or external knowledge access.
* It should not be considered a general-purpose assistant.

The project is primarily intended to demonstrate the mechanics and engineering of decoder-only Transformer language models.

---

# 🚀 Future Improvements

Potential extensions include:

### Architecture

* Rotary positional embeddings (RoPE)
* RMSNorm
* SwiGLU
* Grouped-query attention
* Improved initialization
* Configurable attention implementations

### Training

* Mixed-precision training
* Gradient accumulation
* Distributed Data Parallel
* Automatic checkpoint recovery
* Experiment tracking
* Larger training datasets

### Generation

* Nucleus / top-p sampling
* Beam search
* Repetition penalties
* Stop-token handling
* Streaming generation

### Performance

* Flash Attention
* `torch.compile`
* Optimized data loading
* GPU memory optimization

### Deployment

* REST inference API
* Web interface
* ONNX export
* Model quantization
* Containerized inference

---

# 📌 Portfolio Description

### Short Version

> A decoder-only Transformer language model built from scratch in PyTorch, implementing causal multi-head self-attention, positional embeddings, Transformer blocks, custom training loops, checkpointing, evaluation, and autoregressive text generation.

### Technical Version

> Designed and implemented a ~30M-parameter decoder-only Transformer in PyTorch with custom causal self-attention, pre-LayerNorm blocks, weight tying, GPT-2 tokenization, memory-mapped datasets, AdamW optimization, warmup/cosine learning-rate scheduling, checkpoint management, and autoregressive text generation.

---

# 👨‍💻 Author

**MA Hannan**

Computer Science · Machine Learning · Artificial Intelligence · Data Science

---

# 📄 License

MIT License
