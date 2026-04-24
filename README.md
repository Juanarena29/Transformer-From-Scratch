# Transformer From Scratch — NumPy & BPE

An educational implementation of a Transformer encoder / language model built entirely in **NumPy**, with a custom **BPE tokenizer** for Spanish, full manual backpropagation, training loop, checkpointing, text generation, and visualizations.

No PyTorch. No TensorFlow. Just math.

---

## What this repo includes

- Full Transformer pipeline: `Embedding → Positional Encoding → Encoder Blocks → LM Head`
- Custom BPE tokenizer trained on Spanish (`vocab_size = 6000`)
- Manual backpropagation for every layer (Cross Entropy, Softmax, LayerNorm, MHA, FFN)
- Training loop with train/val split, batch logging, and CSV export
- Checkpoint save/load via `.npz`
- Autoregressive text generation (`greedy`, `top-k`, `top-p`)
- 3D PCA embedding visualization (interactive HTML via Plotly)
- Attention heatmap visualization

---

## Architecture

```
Raw Text
  → BPE Tokenizer
  → Token IDs
  → Token Embedding       (vocab_size × d_model)
  → Positional Encoding   (sinusoidal, fixed)
  → Encoder Block × N
      ├── Multi-Head Attention
      ├── Add & Norm
      ├── Feed Forward (ReLU)
      └── Add & Norm
  → LM Head               (d_model → vocab_size)
  → Logits / Output Distribution
```

### Base config (`Config.py`)

| Parameter     | Value |
|---------------|-------|
| `vocab_size`  | 6000  |
| `d_model`     | 128   |
| `n_heads`     | 8     |
| `n_layers`    | 4     |
| `d_ff`        | 512   |
| `max_seq_len` | 512   |
| `lr`          | 0.001 |

---

## Project structure

```
Transformer/
├── Config.py                    # Model hyperparameters
├── transformer.py               # Full model assembly + save/load weights
├── encoder_block.py             # Encoder block (MHA + FFN + residuals)
├── multi_head_attention.py      # Multi-head attention (forward + backward)
├── feed_forward.py              # Two-layer MLP with ReLU
├── layer_norm.py                # Layer normalization (forward + backward)
├── token_embedding.py           # Token embedding lookup
├── positional_encoding.py       # Sinusoidal positional encoding
├── loss.py                      # Cross Entropy loss + gradient
├── trainer.py                   # Batch builder + training step + train loop
├── colab_training.py            # Full training pipeline (Colab-ready)
├── inference.py                 # Text generation with loaded checkpoint
├── visualize_embeddings.py      # 3D PCA of embedding matrix (Plotly HTML)
├── visualize_attention.py       # Attention heatmap (interactive HTML)
├── main.py                      # End-to-end demo (train + save + load + verify)
├── quick_start.py               # Environment verification script
├── Tokenizer/
│   ├── preprocess.py            # Corpus cleaning and word frequencies
│   ├── tokenizer.py             # BPE training, encode, decode, save, load
│   ├── main.py                  # Streamlit app to inspect BPE merges
│   └── vocab/
│       └── tokenizer.json       # Trained tokenizer artifact (6000 tokens)
├── checkpoints/
│   └── epoch_6_final.npz        # Pre-trained weights (6 epochs)
├── requirements.txt
└── README.md
```

---

## Installation

```bash
pip install -r requirements.txt
```

Optional: run environment check

```bash
python quick_start.py
```

---

## Quick start — inference with pre-trained weights

A pre-trained checkpoint (`epoch_6_final.npz`) is included in `checkpoints/`. You can generate text immediately without training:

```python
from inference import TextGenerator
from Config import TransformerConfig

cfg = TransformerConfig()
gen = TextGenerator(
    model_path="checkpoints/epoch_6_final.npz",
    tokenizer_path="Tokenizer/vocab/tokenizer.json",
    cfg=cfg
)

text = gen.generate(
    prompt="el transformer aprende",
    max_tokens=50,
    method="topp",
    p=0.9
)
print(text)
```

Available generation methods: `greedy`, `topk`, `topp`.

---

## Training from scratch

```bash
python colab_training.py
```

The script:
- downloads a Spanish Wikipedia dataset (or falls back to local corpus)
- tokenizes and chunks text into training sequences
- trains with Cross Entropy loss and next-token prediction objective
- evaluates on validation split every N batches
- saves checkpoints to `checkpoints/`
- logs metrics to `training_log.csv`

See [`COLAB_INSTRUCTIONS.md`](COLAB_INSTRUCTIONS.md) for the full step-by-step guide to train on Google Colab.

---

## Visualizations

### 3D Embedding space (PCA)

```bash
python visualize_embeddings.py \
  --checkpoint "checkpoints/epoch_6_final.npz" \
  --top-k 500 \
  --output "embedding_visualization.html"
```

### Attention heatmap

```bash
python visualize_attention.py \
  --checkpoint "checkpoints/epoch_6_final.npz" \
  --prompt "el transformer aprende patrones" \
  --layer -1 \
  --head -1 \
  --output "attention_heatmap.html"
```

Parameters:
- `--layer -1`: last encoder layer
- `--head -1`: average across all heads
- `--head 0..7`: inspect a specific attention head

---

## Implementation notes

Every component is implemented analytically — no autograd:

- **BPE tokenizer**: trained from scratch on Spanish with incremental merge optimization
- **Multi-Head Attention**: full forward and backward pass including softmax gradient
- **LayerNorm**: analytical gradient over `gamma` and `beta`
- **Cross Entropy**: numerically stable softmax + `ignore_index` for `<PAD>` tokens
- **SGD**: manual parameter updates across all layers

This is intentional. The goal is not to build the fastest model — it is to understand every step.

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `ModuleNotFoundError: plotly` | `pip install plotly` |
| `Permission denied: training_log.csv` | Close the file if open in Excel |
| Shape mismatch in checkpoint | Check that `Config.py` matches the checkpoint's architecture |
| Checkpoint not found | Verify path and contents of `checkpoints/` |
| Loss explodes (NaN) | Reduce `LR` to `0.0001` |
| Loss not decreasing | Increase `LR` to `0.01` |

---

## Project status

Educational project. Built to understand the internals of the Transformer architecture — not optimized for production use. Ideal for:

- learning how attention, embeddings, and backprop actually work
- experimenting with BPE tokenization in Spanish
- building a technical portfolio with full explainability

---

## License

MIT
