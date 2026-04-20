"""
trainer.py
----------
Utilidades de entrenamiento para el mini transformer.
"""

import numpy as np
from loss import CrossEntropyLoss


def _pad_sequences(sequences: list[list[int]], pad_id: int) -> np.ndarray:
    max_len = max(len(seq) for seq in sequences)
    padded = [seq + [pad_id] * (max_len - len(seq)) for seq in sequences]
    return np.array(padded, dtype=int)


def build_lm_batch(
    tokenizer,
    texts: list[str],
    pad_id: int = 0,
    bos_id: int = 2,
    eos_id: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construye un batch para objetivo next-token prediction.

    Para cada texto:
      tokens = [BOS] + encode(text) + [EOS]
      x = tokens[:-1]
      y = tokens[1:]
    """
    tokenized = []
    for text in texts:
        core = tokenizer.encode(text)
        # Si el texto queda vacío tras pretokenizar, al menos BOS->EOS entrena.
        full = [bos_id] + core + [eos_id]
        tokenized.append(full)

    x_seqs = [seq[:-1] for seq in tokenized]
    y_seqs = [seq[1:] for seq in tokenized]

    x_batch = _pad_sequences(x_seqs, pad_id=pad_id)
    y_batch = _pad_sequences(y_seqs, pad_id=pad_id)
    return x_batch, y_batch


def train_step(
    model,
    loss_fn: CrossEntropyLoss,
    x_batch: np.ndarray,
    y_batch: np.ndarray,
    lr: float,
) -> float:
    """Un paso completo: forward -> loss -> backward -> update."""
    logits = model.forward(x_batch)
    loss, dlogits = loss_fn.forward(logits, y_batch)
    model.backward(dlogits)
    model.update(lr)
    return loss


def train_loop(
    model,
    tokenizer,
    texts: list[str],
    epochs: int,
    lr: float,
    pad_id: int = 0,
    bos_id: int = 2,
    eos_id: int = 3,
    verbose: bool = True,
) -> dict[str, list[float]]:
    """
    Entrena en full-batch sobre `texts` por varias épocas.

    Retorna historial:
      {"loss": [...], "perplexity": [...]}
    """
    x_batch, y_batch = build_lm_batch(
        tokenizer, texts=texts, pad_id=pad_id, bos_id=bos_id, eos_id=eos_id
    )
    loss_fn = CrossEntropyLoss(ignore_index=pad_id)

    history = {"loss": [], "perplexity": []}

    for epoch in range(1, epochs + 1):
        loss = train_step(model, loss_fn, x_batch, y_batch, lr=lr)
        ppl = float(np.exp(np.clip(loss, 0, 20)))

        history["loss"].append(loss)
        history["perplexity"].append(ppl)

        if verbose:
            print(f"[epoch {epoch:03d}] loss={loss:.4f} | ppl={ppl:.4f}")

    return history
