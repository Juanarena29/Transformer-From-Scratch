"""token_embedding.py
Token embedding lookup layer backed by a learnable matrix.
Architecture position: receives tokenizer IDs and produces dense vectors before
positional encoding.
"""

import numpy as np


class TokenEmbedding:
    """
    Tabla de embeddings: una matriz W ∈ R^(vocab_size × d_model) donde
    cada fila W[i] es el vector aprendible que representa al token con ID i.

        embed(i) = W[i]   (la i-ésima fila de W)

    W es el único parámetro de esta capa. Durante el entrenamiento, solo
    se actualizan las filas de los tokens que aparecieron en el batch actual.
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        """
        Parámetros
        ----------
        vocab_size : int
            Número de tokens distintos en el vocabulario.
            Debe coincidir con el tamaño del vocabulario del tokenizador.
        d_model : int
            Dimensión del espacio vectorial de salida de cada embedding.
        """
        self.vocab_size = vocab_size
        self.d_model = d_model

        # Inicialización con std = 1/√d_model para mantener norma ~1
        # independientemente del tamaño de d_model.
        self.W: np.ndarray = np.random.randn(vocab_size, d_model) / np.sqrt(d_model)

        self._dW: np.ndarray = np.zeros_like(self.W)
        self._token_ids_cache: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Extrae las filas de W correspondientes a cada token ID.

        Parámetros
        ----------
        token_ids : np.ndarray, shape (batch_size, seq_len), dtype int

        Retorna
        -------
        np.ndarray, shape (batch_size, seq_len, d_model)
            output[b, t, :] == W[token_ids[b, t]]
        """
        self._token_ids_cache = token_ids
        return self.W[token_ids]  # (batch_size, seq_len, d_model)

    # ------------------------------------------------------------------
    # Backward pass
    # ------------------------------------------------------------------

    def backward(self, dout: np.ndarray) -> None:
        """
        Calcula el gradiente de la pérdida respecto a W.

        Parámetros
        ----------
        dout : np.ndarray, shape (batch_size, seq_len, d_model)
            Gradiente proveniente de la capa siguiente.

        No retorna gradiente respecto a token_ids: son índices discretos,
        no tienen derivada.
        """
        self._dW = np.zeros_like(self.W)

        # np.add.at acumula sin buffering, necesario cuando el mismo token
        # aparece más de una vez en el batch.
        np.add.at(self._dW, self._token_ids_cache, dout)

    # ------------------------------------------------------------------
    # Actualización de parámetros (SGD)
    # ------------------------------------------------------------------

    def update(self, lr: float) -> None:
        """Update embedding matrix with one SGD step.

        Parameters
        ----------
        lr : float
            Learning rate.

        Returns
        -------
        None
            Embedding matrix is updated in place.
        """
        self.W -= lr * self._dW

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        """Return a compact string representation.

        Parameters
        ----------
        None
            Uses instance attributes only.

        Returns
        -------
        str
            Debug-friendly representation.
        """
        n_params = self.vocab_size * self.d_model
        return (
            f"TokenEmbedding("
            f"vocab_size={self.vocab_size}, "
            f"d_model={self.d_model}, "
            f"params={n_params:,})"
        )

    @property
    def num_parameters(self) -> int:
        """Número total de parámetros escalares en W."""
        return self.vocab_size * self.d_model
