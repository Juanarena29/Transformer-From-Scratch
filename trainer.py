"""trainer.py
Training utilities for language-model batches and optimization loops.
Architecture position: orchestrates data batching around model forward,
backward, and parameter update passes.
"""

import numpy as np
from loss import CrossEntropyLoss


def _pad_sequences(sequences: list[list[int]], pad_id: int) -> np.ndarray:
    """Pad variable-length token ID sequences to a rectangular array.

    Parameters
    ----------
    sequences : list[list[int]]
        Token ID sequences.
    pad_id : int
        Padding token ID used to fill shorter sequences.

    Returns
    -------
    np.ndarray
        Padded matrix with shape ``(batch, max_seq_len)``.
    """
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
    """Build a next-token prediction batch from text samples.

    Parameters
    ----------
    tokenizer : Any
        Tokenizer object exposing ``encode``.
    texts : list[str]
        Input text samples.
    pad_id : int, optional
        Padding token ID.
    bos_id : int, optional
        Begin-of-sequence token ID.
    eos_id : int, optional
        End-of-sequence token ID.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``x_batch`` and ``y_batch`` with aligned LM supervision.
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
    """Run one full optimization step.

    Parameters
    ----------
    model : Any
        Transformer-like model with ``forward``, ``backward``, and ``update``.
    loss_fn : CrossEntropyLoss
        Loss object.
    x_batch : np.ndarray
        Input token IDs with shape ``(batch, seq)``.
    y_batch : np.ndarray
        Target token IDs with shape ``(batch, seq)``.
    lr : float
        Learning rate.

    Returns
    -------
    float
        Scalar training loss for the step.
    """
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
    """Train a model for multiple epochs on in-memory text samples.

    Parameters
    ----------
    model : Any
        Transformer-like model.
    tokenizer : Any
        Tokenizer object exposing ``encode``.
    texts : list[str]
        Training corpus samples.
    epochs : int
        Number of epochs.
    lr : float
        Learning rate.
    pad_id : int, optional
        Padding token ID.
    bos_id : int, optional
        Begin-of-sequence token ID.
    eos_id : int, optional
        End-of-sequence token ID.
    verbose : bool, optional
        Whether to print per-epoch metrics.

    Returns
    -------
    dict[str, list[float]]
        History dictionary with ``loss`` and ``perplexity`` lists.
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
