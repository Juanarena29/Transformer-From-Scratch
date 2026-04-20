"""
loss.py
-------
Funciones de pérdida para entrenar el transformer.
"""

import numpy as np


class CrossEntropyLoss:
    """
    Cross-Entropy para logits de language modeling.

    Inputs esperados:
      - logits : (batch, seq, vocab_size)
      - targets: (batch, seq) con IDs de token
    """

    def __init__(self, ignore_index: int | None = None, eps: float = 1e-12) -> None:
        self.ignore_index = ignore_index
        self.eps = eps

    def forward(self, logits: np.ndarray, targets: np.ndarray) -> tuple[float, np.ndarray]:
        """
        Retorna:
          - loss escalar promedio sobre tokens válidos
          - dlogits con mismo shape que logits
        """
        if logits.ndim != 3:
            raise ValueError(f"logits debe tener 3 dims (B,T,V), recibido {logits.shape}")
        if targets.ndim != 2:
            raise ValueError(f"targets debe tener 2 dims (B,T), recibido {targets.shape}")
        if logits.shape[:2] != targets.shape:
            raise ValueError(
                f"logits y targets incompatibles: {logits.shape[:2]} vs {targets.shape}"
            )

        batch, seq, vocab_size = logits.shape
        logits_2d = logits.reshape(-1, vocab_size)
        targets_1d = targets.reshape(-1)

        # Softmax estable numéricamente.
        shifted = logits_2d - logits_2d.max(axis=1, keepdims=True)
        exp_logits = np.exp(shifted)
        probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

        if self.ignore_index is None:
            valid_mask = np.ones_like(targets_1d, dtype=bool)
        else:
            valid_mask = targets_1d != self.ignore_index

        valid_count = int(valid_mask.sum())
        if valid_count == 0:
            raise ValueError("No hay tokens válidos para calcular Cross-Entropy.")

        # Para evitar indexar posiciones ignoradas con ids potencialmente inválidos.
        safe_targets = targets_1d.copy()
        safe_targets[~valid_mask] = 0

        row_idx = np.arange(targets_1d.shape[0])
        target_probs = probs[row_idx, safe_targets]
        target_probs = np.clip(target_probs, self.eps, 1.0)

        # Ignorar tokens de padding en el promedio de la loss.
        losses = -np.log(target_probs)
        loss = losses[valid_mask].sum() / valid_count

        dlogits = probs
        dlogits[row_idx, safe_targets] -= 1.0
        dlogits[~valid_mask] = 0.0
        dlogits /= valid_count

        return float(loss), dlogits.reshape(batch, seq, vocab_size)
