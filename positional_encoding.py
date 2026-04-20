"""
positional_encoding.py
----------------------
Positional Encoding sinusoidal para un transformer desde cero.

Posición en la arquitectura:

    [TokenEmbedding]   →  embeddings: shape (batch_size, seq_len, d_model)
          ↓
    [PositionalEncoding] →  x: shape (batch_size, seq_len, d_model)
          ↓
    [Transformer blocks]
"""

import numpy as np


class PositionalEncoding:
    """
    Suma un vector posicional fijo (no aprendible) a cada embedding.

    Fórmula (Vaswani et al., 2017):

        PE(pos, 2i)   = sin(pos / 10000^(2i / d_model))
        PE(pos, 2i+1) = cos(pos / 10000^(2i / d_model))

    La matriz PE se construye una sola vez y se reutiliza en cada forward.
    No tiene parámetros ni backward propio: el gradiente fluye directo
    hacia el TokenEmbedding sin modificarse.
    """

    def __init__(self, d_model: int, max_seq_len: int = 512) -> None:
        """
        Parámetros
        ----------
        d_model     : debe coincidir exactamente con TokenEmbedding.d_model
        max_seq_len : límite superior de secuencia. 512 es seguro para empezar.
        """
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.pe = self._build_matrix()  # (max_seq_len, d_model)

    def _build_matrix(self) -> np.ndarray:
        pe = np.zeros((self.max_seq_len, self.d_model))

        positions = np.arange(self.max_seq_len).reshape(-1, 1)   # (seq, 1)
        dims = np.arange(0, self.d_model, 2)                # índices pares

        # Ángulos: (seq, d_model/2)
        angles = positions / np.power(10000.0, dims / self.d_model)

        pe[:, 0::2] = np.sin(angles)  # dimensiones pares
        pe[:, 1::2] = np.cos(angles)  # dimensiones impares

        return pe  # (max_seq_len, d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Parámetros
        ----------
        x : np.ndarray, shape (batch_size, seq_len, d_model)
            Salida directa del TokenEmbedding.forward()

        Retorna
        -------
        np.ndarray, shape (batch_size, seq_len, d_model)
            x + PE[:seq_len, :]  — el broadcasting aplica PE a todo el batch.
        """
        seq_len = x.shape[1]
        assert seq_len <= self.max_seq_len, (
            f"Secuencia ({seq_len}) excede max_seq_len ({self.max_seq_len})"
        )
        # pe[np.newaxis] → (1, seq_len, d_model) — broadcast sobre el batch
        return x + self.pe[:seq_len, :][np.newaxis, :, :]

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        El gradiente pasa sin cambios: PE es constante, su derivada es 0.

        Parámetros
        ----------
        dout : np.ndarray, shape (batch_size, seq_len, d_model)

        Retorna
        -------
        np.ndarray, shape (batch_size, seq_len, d_model)  — dout sin modificar
        """
        return dout

    def __repr__(self) -> str:
        return (
            f"PositionalEncoding("
            f"d_model={self.d_model}, "
            f"max_seq_len={self.max_seq_len})"
        )
