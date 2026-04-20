"""
multi_head_attention.py
-----------------------
Multi-Head Attention para un transformer desde cero.

Posición en la arquitectura:

    [Positional Encoding]  →  x: shape (batch, seq, d_model)
          ↓
    [MultiHeadAttention]   →  x: shape (batch, seq, d_model)
          ↓
    [Feed Forward]
"""

import numpy as np


def softmax(x: np.ndarray) -> np.ndarray:
    """Softmax numéricamente estable sobre el último eje."""
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=-1, keepdims=True)


class MultiHeadAttention:
    """
    Multi-Head Attention implementado desde cero en NumPy.

    Parámetros
    ----------
    d_model : dimensión del embedding (128)
    n_heads : número de cabezas (8)
    d_k     : dimensión por cabeza = d_model // n_heads (16)

    Pesos
    -----
    W_Q, W_K, W_V : (d_model, d_model)  — proyecciones Q, K, V
    W_O           : (d_model, d_model)  — proyección de salida
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        assert d_model % n_heads == 0, "d_model debe ser divisible por n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.scale = 1.0 / np.sqrt(self.d_k)

        # Inicialización Xavier — apropiada para capas lineales sin ReLU
        def xavier(shape):
            limit = np.sqrt(6 / (shape[0] + shape[1]))
            return np.random.uniform(-limit, limit, shape)

        # (128,128) despues se dividen en 8 (n_heads) quedando (128,16)
        self.W_Q = xavier((d_model, d_model))
        # (128,128) al principio para eficiencia en calculos.
        self.W_K = xavier((d_model, d_model))
        self.W_V = xavier((d_model, d_model))
        self.W_O = xavier((d_model, d_model))

        # Gradientes
        self._dW_Q = np.zeros_like(self.W_Q)
        self._dW_K = np.zeros_like(self.W_K)
        self._dW_V = np.zeros_like(self.W_V)
        self._dW_O = np.zeros_like(self.W_O)

    # ------------------------------------------------------------------
    # Helpers: split y merge de cabezas
    # ------------------------------------------------------------------

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """
        (batch, seq, d_model) → (batch, n_heads, seq, d_k)
        Divide el último eje en n_heads cabezas.
        """
        batch, seq, _ = x.shape  # ejemplo 2 batches de 5 tokens
        x = x.reshape(batch, seq, self.n_heads, self.d_k)
        # 128 dimensiones se cortan en 8 grupos de 16 -> (2, 5, 8, 16)
        return x.transpose(0, 2, 1, 3)
        # x.T -> (2, 8, 5, 16)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        """
        (batch, n_heads, seq, d_k) → (batch, seq, d_model)
        Operación inversa a _split_heads.
        """
        batch, _, seq, _ = x.shape
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch, seq, self.d_model)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        Parámetros
        ----------
        x    : (batch, seq, d_model)  — salida del Positional Encoding
        mask : (batch, 1, 1, seq) opcional — 0 donde atender, -inf donde ignorar

        Retorna
        -------
        (batch, seq, d_model)
        """
        # Guardar input para backward
        self._x = x

        # 1. Proyecciones lineales Q, K, V
        # (batch, seq, d_model) @ (d_model, d_model) → (batch, seq, d_model)
        Q = x @ self.W_Q
        K = x @ self.W_K
        V = x @ self.W_V

        # 2. Dividir en cabezas
        # (batch, n_heads, seq, d_k)
        Q = self._split_heads(Q)
        K = self._split_heads(K)
        V = self._split_heads(V)

        # Guardar para backward
        self._Q = Q
        self._K = K
        self._V = V

        # 3. Scaled dot-product attention
        # (batch, n_heads, seq, seq)
        # self scale es el denominador de softmax
        scores = Q @ K.transpose(0, 1, 3, 2) * self.scale
        #   (2, 8, 5, 16) @ (2, 8, 16, 5) → (2, 8, 5, 5)
        #                                                ↑
        #                              cada token contra cada token

        # Mask opcional (para padding o causal)
        if mask is not None:
            scores = scores + mask

        # (2, 8, 5, 5) — pesos que suman 1
        self._attn_weights = softmax(scores)

        # 4. Combinar con V
        # (batch, n_heads, seq, seq) @ (batch, n_heads, seq, d_k)
        # → (batch, n_heads, seq, d_k)
        attn_out = self._attn_weights @ V
        # (2, 8, 5, 5) @ (2, 8, 5, 16) → (2, 8, 5, 16)

        # 5. Reunir cabezas
        # (batch, seq, d_model)
        attn_out = self._merge_heads(attn_out)
        # aca se juntan las cabezas

        self._attn_out = attn_out
        # se guarda

        # 6. Proyección final
        out = attn_out @ self.W_O
        #     (2, 5, 128) @ (128, 128) → (2, 5, 128)
        return out  # (batch, seq, d_model)

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
        dx : (batch, seq, d_model) — gradiente hacia el PE o capa anterior
        """
        # 6. Gradiente de W_O
        self._dW_O = self._attn_out.reshape(-1,
                                            self.d_model).T @ dout.reshape(-1, self.d_model)
        d_attn_out = dout @ self.W_O.T  # (batch, seq, d_model)

        # Volver a forma de cabezas
        # (batch, n_heads, seq, d_k)
        d_attn_out = self._split_heads(d_attn_out)

        # 4. Gradiente a través de attn_weights @ V
        # (batch, n_heads, seq, seq)
        d_attn_weights = d_attn_out @ self._V.transpose(0, 1, 3, 2)
        dV = self._attn_weights.transpose(
            0, 1, 3, 2) @ d_attn_out   # (batch, n_heads, seq, d_k)

        # 3. Gradiente a través del softmax
        A = self._attn_weights
        # dL/dscores = A * (dA - sum(dA * A, keepdims))
        d_scores = A * (d_attn_weights - (d_attn_weights *
                        A).sum(axis=-1, keepdims=True))
        d_scores *= self.scale  # (batch, n_heads, seq, seq)

        # 2. Gradiente a Q y K
        # (batch, n_heads, seq, d_k)
        dQ = d_scores @ self._K
        # (batch, n_heads, seq, d_k)
        dK = d_scores.transpose(0, 1, 3, 2) @ self._Q

        # Reunir cabezas
        dQ = self._merge_heads(dQ)  # (batch, seq, d_model)
        dK = self._merge_heads(dK)
        dV = self._merge_heads(dV)

        # 1. Gradientes de W_Q, W_K, W_V
        x_flat = self._x.reshape(-1, self.d_model)
        self._dW_Q = x_flat.T @ dQ.reshape(-1, self.d_model)
        self._dW_K = x_flat.T @ dK.reshape(-1, self.d_model)
        self._dW_V = x_flat.T @ dV.reshape(-1, self.d_model)

        # Gradiente hacia el input x
        dx = dQ @ self.W_Q.T + dK @ self.W_K.T + dV @ self.W_V.T
        return dx  # (batch, seq, d_model)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, lr: float) -> None:
        """SGD sobre los 4 matrices de pesos."""
        self.W_Q -= lr * self._dW_Q
        self.W_K -= lr * self._dW_K
        self.W_V -= lr * self._dW_V
        self.W_O -= lr * self._dW_O

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_params = 4 * self.d_model * self.d_model
        return (
            f"MultiHeadAttention("
            f"d_model={self.d_model}, "
            f"n_heads={self.n_heads}, "
            f"d_k={self.d_k}, "
            f"params={n_params:,})"
        )

    @property
    def num_parameters(self) -> int:
        return 4 * self.d_model * self.d_model
