"""
transformer.py
--------------
Transformer completo ensamblado desde cero.

Arquitectura completa:

    [texto]
       ↓
    [BPE Tokenizer]
       ↓
    [TokenEmbedding]
       ↓
    [PositionalEncoding]
       ↓
    [EncoderBlock x4]
       ↓
    [LM Head]  →  logits: (batch, seq, vocab_size)
       ↓
    [Softmax]  →  probabilidades sobre el vocabulario
"""

import numpy as np
import os
from token_embedding import TokenEmbedding
from positional_encoding import PositionalEncoding
from encoder_block import EncoderBlock


class Transformer:
    """
    Transformer completo con:
        - Token Embedding
        - Positional Encoding
        - N Encoder Blocks apilados
        - LM Head (proyección a vocab_size)

    Parámetros
    ----------
    vocab_size  : tamaño del vocabulario (6000)
    d_model     : dimensión del embedding (128)
    n_heads     : cabezas de atención (8)
    n_layers    : número de bloques encoder (4)
    d_ff        : dimensión interna del FFN (512)
    max_seq_len : longitud máxima de secuencia (512)
    """

    def __init__(
        self,
        vocab_size:  int,
        d_model:     int,
        n_heads:     int,
        n_layers:    int,
        d_ff:        int,
        max_seq_len: int,
    ) -> None:
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers

        # ── Capas ────────────────────────────────────────────────────────
        self.embedding = TokenEmbedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_seq_len)

        # 4 bloques encoder apilados
        self.blocks = [
            EncoderBlock(d_model, n_heads, d_ff)
            for _ in range(n_layers)
        ]

        # LM Head: proyecta d_model → vocab_size
        # Es una capa lineal simple sin activación
        # W_head: (d_model, vocab_size) = (128, 6000)
        limit = np.sqrt(6 / (d_model + vocab_size))
        self.W_head = np.random.uniform(-limit, limit, (d_model, vocab_size))
        self.b_head = np.zeros(vocab_size)

        self._dW_head = np.zeros_like(self.W_head)
        self._db_head = np.zeros_like(self.b_head)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """
        Parámetros
        ----------
        token_ids : (batch, seq)  dtype int

        Retorna
        -------
        logits : (batch, seq, vocab_size)
            Scores sin normalizar sobre el vocabulario.
            Para obtener probabilidades: softmax(logits)
        """
        # 1. Embedding
        x = self.embedding.forward(token_ids)
        # (batch, seq, 128)

        # 2. Positional Encoding
        x = self.pos_enc.forward(x)
        # (batch, seq, 128)

        # 3. Encoder Blocks apilados
        for block in self.blocks:
            x = block.forward(x)
        # (batch, seq, 128)

        # Guardar para backward del LM Head
        self._x_before_head = x

        # 4. LM Head
        logits = x @ self.W_head + self.b_head
        # (batch, seq, 128) @ (128, 6000) → (batch, seq, 6000)

        return logits

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------

    def backward(self, dlogits: np.ndarray) -> None:
        """
        Parámetros
        ----------
        dlogits : (batch, seq, vocab_size)
            Gradiente de la loss respecto a los logits.
            Lo calcula la función de loss, no este módulo.
        """
        # 4. LM Head al revés
        x = self._x_before_head
        self._dW_head = x.reshape(-1,
                                  self.d_model).T @ dlogits.reshape(-1, self.vocab_size)
        # (128, batch*seq) @ (batch*seq, 6000) → (128, 6000)

        self._db_head = dlogits.sum(axis=(0, 1))
        # (6000,)

        dx = dlogits @ self.W_head.T
        # (batch, seq, 6000) @ (6000, 128) → (batch, seq, 128)

        # 3. Encoder Blocks al revés
        for block in reversed(self.blocks):
            dx = block.backward(dx)
        # (batch, seq, 128)

        # 2. PE backward — pasa directo sin cambios
        dx = self.pos_enc.backward(dx)

        # 1. Embedding backward
        self.embedding.backward(dx)

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, lr: float) -> None:
        """SGD sobre todos los parámetros del transformer."""
        self.embedding.update(lr)
        # pos_enc no tiene parámetros

        for block in self.blocks:
            block.update(lr)

        self.W_head -= lr * self._dW_head
        self.b_head -= lr * self._db_head

    # ------------------------------------------------------------------
    # Persistencia de pesos
    # ------------------------------------------------------------------

    def state_dict(self) -> dict[str, np.ndarray]:
        """Retorna todos los pesos aprendibles en un dict plano."""
        state = {
            "embedding.W": self.embedding.W,
            "head.W": self.W_head,
            "head.b": self.b_head,
        }

        for i, block in enumerate(self.blocks):
            prefix = f"blocks.{i}"

            state[f"{prefix}.attn.W_Q"] = block.attention.W_Q
            state[f"{prefix}.attn.W_K"] = block.attention.W_K
            state[f"{prefix}.attn.W_V"] = block.attention.W_V
            state[f"{prefix}.attn.W_O"] = block.attention.W_O

            state[f"{prefix}.norm1.gamma"] = block.norm1.gamma
            state[f"{prefix}.norm1.beta"] = block.norm1.beta

            state[f"{prefix}.ff.W1"] = block.ff.W1
            state[f"{prefix}.ff.b1"] = block.ff.b1
            state[f"{prefix}.ff.W2"] = block.ff.W2
            state[f"{prefix}.ff.b2"] = block.ff.b2

            state[f"{prefix}.norm2.gamma"] = block.norm2.gamma
            state[f"{prefix}.norm2.beta"] = block.norm2.beta

        return state

    def load_state_dict(self, state: dict[str, np.ndarray]) -> None:
        """Carga pesos desde un state_dict compatible."""
        expected_keys = set(self.state_dict().keys())
        provided_keys = set(state.keys())

        missing = expected_keys - provided_keys
        extra = provided_keys - expected_keys

        if missing:
            raise ValueError(f"Faltan claves en state_dict: {sorted(missing)}")
        if extra:
            raise ValueError(f"Claves inesperadas en state_dict: {sorted(extra)}")

        def _assign(attr: np.ndarray, key: str) -> np.ndarray:
            value = state[key]
            if attr.shape != value.shape:
                raise ValueError(
                    f"Shape incompatible para '{key}': esperado {attr.shape}, recibido {value.shape}"
                )
            return value.astype(attr.dtype, copy=True)

        self.embedding.W = _assign(self.embedding.W, "embedding.W")
        self.W_head = _assign(self.W_head, "head.W")
        self.b_head = _assign(self.b_head, "head.b")

        for i, block in enumerate(self.blocks):
            prefix = f"blocks.{i}"

            block.attention.W_Q = _assign(block.attention.W_Q, f"{prefix}.attn.W_Q")
            block.attention.W_K = _assign(block.attention.W_K, f"{prefix}.attn.W_K")
            block.attention.W_V = _assign(block.attention.W_V, f"{prefix}.attn.W_V")
            block.attention.W_O = _assign(block.attention.W_O, f"{prefix}.attn.W_O")

            block.norm1.gamma = _assign(block.norm1.gamma, f"{prefix}.norm1.gamma")
            block.norm1.beta = _assign(block.norm1.beta, f"{prefix}.norm1.beta")

            block.ff.W1 = _assign(block.ff.W1, f"{prefix}.ff.W1")
            block.ff.b1 = _assign(block.ff.b1, f"{prefix}.ff.b1")
            block.ff.W2 = _assign(block.ff.W2, f"{prefix}.ff.W2")
            block.ff.b2 = _assign(block.ff.b2, f"{prefix}.ff.b2")

            block.norm2.gamma = _assign(block.norm2.gamma, f"{prefix}.norm2.gamma")
            block.norm2.beta = _assign(block.norm2.beta, f"{prefix}.norm2.beta")

    def save_weights(self, path: str) -> None:
        """Guarda los pesos del modelo en formato .npz."""
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        np.savez(path, **self.state_dict())

    def load_weights(self, path: str) -> None:
        """Carga pesos guardados con save_weights()."""
        with np.load(path, allow_pickle=False) as data:
            state = {k: data[k] for k in data.files}
        self.load_state_dict(state)

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    @property
    def num_parameters(self) -> int:
        total = 0
        total += self.embedding.num_parameters
        for block in self.blocks:
            total += block.num_parameters
        total += self.d_model * self.vocab_size  # W_head
        total += self.vocab_size                 # b_head
        return total

    def __repr__(self) -> str:
        lines = ["Transformer("]
        lines.append(f"  {self.embedding}")
        lines.append(f"  {self.pos_enc}")
        for i, block in enumerate(self.blocks):
            lines.append(
                f"  EncoderBlock[{i}]: {block.num_parameters:,} params")
        lines.append(f"  LM Head: ({self.d_model}, {self.vocab_size})")
        lines.append(f"  Total params: {self.num_parameters:,}")
        lines.append(")")
        return "\n".join(lines)
