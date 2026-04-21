# Transformer From Scratch (NumPy)

Educational, from-scratch implementation of an encoder-style Transformer in
pure NumPy, including a BPE tokenizer for Spanish, training utilities, and
inference scripts. The project is designed to be readable and modular, so each
component of the architecture can be inspected independently.

## Architecture Diagram (ASCII)

```text
Raw text
  |
  v
BPE Tokenizer
  |
  v
Token IDs (batch, seq)
  |
  v
TokenEmbedding ------------------------------+
  |                                          |
  v                                          |
PositionalEncoding                           |
  |                                          |
  v                                          |
EncoderBlock x N                             |
  |   [MHA -> Add&Norm -> FFN -> Add&Norm]   |
  +------------------------------------------+
  |
  v
LM Head (d_model -> vocab_size)
  |
  v
Logits (batch, seq, vocab_size)
```

## File Structure

- `Config.py`: Global hyperparameter container.
- `transformer.py`: Full Transformer assembly and checkpoint IO.
- `encoder_block.py`: Encoder block with residual attention and FFN.
- `multi_head_attention.py`: Scaled dot-product multi-head self-attention.
- `feed_forward.py`: Position-wise feed-forward network.
- `layer_norm.py`: Layer normalization module.
- `token_embedding.py`: Token embedding lookup table.
- `positional_encoding.py`: Fixed sinusoidal positional encodings.
- `loss.py`: Cross-entropy loss and gradient computation.
- `trainer.py`: Batch building and training loops.
- `main.py`: End-to-end local demo.
- `colab_training.py`: Full training pipeline for Colab.
- `inference.py`: Text generation from trained checkpoints.
- `Tokenizer/tokenizer.py`: BPE tokenizer train/encode/decode implementation.
- `Tokenizer/preprocess.py`: Corpus preprocessing pipeline for tokenizer data.
- `Tokenizer/main.py`: Streamlit tokenizer explorer.

## Install Dependencies

```bash
pip install -r requirements.txt
```

## Train (Placeholder Command)

```bash
python colab_training.py
```

## Parameter Count

| Component | Formula | Parameters |
|---|---|---:|
| Token embedding | `vocab_size * d_model` | 768,000 |
| Encoder block (x1) | `MHA + FFN + Norms` | 197,120 |
| Encoder blocks (x4) | `4 * 197,120` | 788,480 |
| LM head | `d_model * vocab_size + vocab_size` | 774,000 |
| **Total** | `embedding + blocks + head` | **2,330,480** |
