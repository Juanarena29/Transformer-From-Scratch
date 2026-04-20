"""
layer_norm.py
-------------
Layer Normalization para un transformer desde cero.

Posición en la arquitectura:

    x + MultiHeadAttention(x)  →  LayerNorm  →  siguiente capa
    x + FeedForward(x)         →  LayerNorm  →  siguiente capa
"""

import numpy as np


class LayerNorm:
    """
    Layer Normalization sobre el último eje (las d_model dimensiones).

    Parámetros aprendibles
    ----------------------
    gamma : (d_model,) — escala, inicializada en 1
    beta  : (d_model,) — desplazamiento, inicializado en 0

    Por qué solo dos vectores de 128 params cada uno:
    la normalización ocurre por token, así que gamma y beta
    se aplican igual a todos los tokens — no dependen de la posición.
    """

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        self.d_model = d_model
        self.eps = eps

        # Parámetros aprendibles
        self.gamma = np.ones(d_model)   # (128,)
        self.beta = np.zeros(d_model)  # (128,)

        # Gradientes
        self._dgamma = np.zeros_like(self.gamma)
        self._dbeta = np.zeros_like(self.beta)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Parámetros
        ----------
        x : (batch, seq, d_model)

        Retorna
        -------
        (batch, seq, d_model) — normalizado y reescalado
        """
        # Media y varianza por token — sobre el último eje (las 128 dims)
        self._mu = x.mean(axis=-1, keepdims=True)   # (batch, seq, 1)
        self._var = x.var(axis=-1, keepdims=True)    # (batch, seq, 1)

        # Normalizar
        self._x_norm = (x - self._mu) / np.sqrt(self._var +
                                                self.eps)  # (batch, seq, 128)

        # Guardar input para backward
        self._x = x

        # Escalar y desplazar con parámetros aprendibles
        return self.gamma * self._x_norm + self.beta

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Parámetros
        ----------
        dout : (batch, seq, d_model) — gradiente de la capa siguiente

        Retorna
        -------
        dx : (batch, seq, d_model) — gradiente hacia la capa anterior
        """
        N = self.d_model  # número de dimensiones que se normalizaron
        std_inv = 1.0 / np.sqrt(self._var + self.eps)  # (batch, seq, 1)

        # Gradientes de gamma y beta
        # gamma y beta se aplican a todos los tokens de todos los batches
        # por eso sumamos sobre batch y seq
        self._dgamma = (dout * self._x_norm).sum(axis=(0, 1))  # (128,)
        self._dbeta = dout.sum(axis=(0, 1))                   # (128,)

        # Gradiente hacia x_norm
        dx_norm = dout * self.gamma  # (batch, seq, 128)

        # Gradiente hacia x — fórmula completa de LayerNorm backward
        dx = (1.0 / N) * std_inv * (
            N * dx_norm
            - dx_norm.sum(axis=-1, keepdims=True)
            - self._x_norm *
            (dx_norm * self._x_norm).sum(axis=-1, keepdims=True)
        )

        return dx  # (batch, seq, 128)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, lr: float) -> None:
        """SGD sobre gamma y beta."""
        self.gamma -= lr * self._dgamma
        self.beta -= lr * self._dbeta

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f"LayerNorm(d_model={self.d_model}, eps={self.eps})"

    @property
    def num_parameters(self) -> int:
        return 2 * self.d_model  # gamma + beta
