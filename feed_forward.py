"""feed_forward.py
Feed-Forward Network block used inside each encoder block.
Architecture position: runs after first residual+LayerNorm and before second
residual+LayerNorm within an encoder block.
"""

import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """Apply ReLU activation elementwise.

    Parameters
    ----------
    x : np.ndarray
        Input tensor.

    Returns
    -------
    np.ndarray
        ReLU-activated tensor with the same shape as ``x``.
    """
    return np.maximum(0, x)


def relu_deriv(x: np.ndarray) -> np.ndarray:
    """Compute the derivative of ReLU activation.

    Parameters
    ----------
    x : np.ndarray
        Pre-activation tensor used by ReLU.

    Returns
    -------
    np.ndarray
        Tensor with 1.0 where ``x > 0`` and 0.0 elsewhere.
    """
    return (x > 0).astype(float)


class FeedForward:
    """
    Dos capas lineales con ReLU en el medio.

        x → W1 + b1 → ReLU → W2 + b2 → salida

    Parámetros
    ----------
    d_model : dimensión de entrada y salida (128)
    d_ff    : dimensión interna expandida (512 = 4 × d_model)

    Pesos
    -----
    W1 : (d_model, d_ff)   — expande de 128 a 512
    b1 : (d_ff,)
    W2 : (d_ff, d_model)   — comprime de 512 a 128
    b2 : (d_model,)
    """

    def __init__(self, d_model: int, d_ff: int) -> None:
        self.d_model = d_model
        self.d_ff = d_ff

        # He initialization — apropiada para ReLU, igual que en tu net.py
        self.W1 = np.random.randn(d_model, d_ff) * np.sqrt(2 / d_model)
        self.W2 = np.random.randn(d_ff,   d_model) * np.sqrt(2 / d_ff)
        self.b1 = np.zeros(d_ff)
        self.b2 = np.zeros(d_model)

        # Gradientes
        self._dW1 = np.zeros_like(self.W1)
        self._dW2 = np.zeros_like(self.W2)
        self._db1 = np.zeros_like(self.b1)
        self._db2 = np.zeros_like(self.b2)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Parámetros
        ----------
        x : (batch, seq, d_model=128)

        Retorna
        -------
        (batch, seq, d_model=128)
        """
        # Guardar input para backward
        self._x = x

        # Capa 1: expansión
        self._z1 = x @ self.W1 + self.b1
        # (batch, seq, 128) @ (128, 512) → (batch, seq, 512)

        # ReLU
        self._a1 = relu(self._z1)
        # (batch, seq, 512)

        # Capa 2: compresión
        out = self._a1 @ self.W2 + self.b2
        # (batch, seq, 512) @ (512, 128) → (batch, seq, 128)

        return out

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Parámetros
        ----------
        dout : (batch, seq, d_model=128) — gradiente de la capa siguiente

        Retorna
        -------
        dx : (batch, seq, d_model=128) — gradiente hacia la capa anterior
        """
        # Capa 2 al revés
        # dout: (batch, seq, 128)
        self._dW2 = self._a1.reshape(-1,
                                     self.d_ff).T @ dout.reshape(-1, self.d_model)
        # (512, batch*seq) @ (batch*seq, 128) → (512, 128)

        self._db2 = dout.sum(axis=(0, 1))
        # (128,)

        da1 = dout @ self.W2.T
        # (batch, seq, 128) @ (128, 512) → (batch, seq, 512)

        # ReLU al revés
        dz1 = da1 * relu_deriv(self._z1)
        # (batch, seq, 512)

        # Capa 1 al revés
        self._dW1 = self._x.reshape(-1,
                                    self.d_model).T @ dz1.reshape(-1, self.d_ff)
        # (128, batch*seq) @ (batch*seq, 512) → (128, 512)

        self._db1 = dz1.sum(axis=(0, 1))
        # (512,)

        dx = dz1 @ self.W1.T
        # (batch, seq, 512) @ (512, 128) → (batch, seq, 128)

        return dx

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, lr: float) -> None:
        """Apply one SGD step to FFN parameters.

        Parameters
        ----------
        lr : float
            Learning rate.

        Returns
        -------
        None
            Parameters are updated in place.
        """
        self.W1 -= lr * self._dW1
        self.W2 -= lr * self._dW2
        self.b1 -= lr * self._db1
        self.b2 -= lr * self._db2

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        n_params = self.d_model * self.d_ff + self.d_ff * self.d_model
        return (
            f"FeedForward("
            f"d_model={self.d_model}, "
            f"d_ff={self.d_ff}, "
            f"params={n_params:,})"
        )

    @property
    def num_parameters(self) -> int:
        return (self.d_model * self.d_ff) + (self.d_ff * self.d_model)
