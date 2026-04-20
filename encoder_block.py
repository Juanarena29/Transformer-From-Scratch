"""
encoder_block.py
----------------
Encoder Block completo para un transformer desde cero.

Un bloque encoder es la unidad repetible del transformer.
El transformer completo apila N de estos bloques en secuencia.

Flujo interno:
    x → MHA → Add & Norm → FFN → Add & Norm → salida

Posición en la arquitectura:

    [Positional Encoding]
          ↓
    [Encoder Block 1]
          ↓
    [Encoder Block 2]
          ↓
         ...
    [Encoder Block N]  (N = n_layers = 4)
          ↓
    [Output / LM Head]
"""

import numpy as np
from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward
from layer_norm import LayerNorm


class EncoderBlock:
    """
    Un bloque encoder con:
        - Multi-Head Attention
        - Conexión residual + LayerNorm
        - Feed Forward
        - Conexión residual + LayerNorm

    Parámetros
    ----------
    d_model : dimensión del embedding (128)
    n_heads : número de cabezas de atención (8)
    d_ff    : dimensión interna del FFN (512)
    """

    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff)
        self.norm2 = LayerNorm(d_model)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """
        Parámetros
        ----------
        x    : (batch, seq, d_model)
        mask : opcional, para ignorar tokens de padding

        Retorna
        -------
        (batch, seq, d_model) — mismo shape que entró
        """
        # ── Sub-bloque 1: Multi-Head Attention ──────────────────────────
        attn_out = self.attention.forward(x, mask)
        # (batch, seq, 128)

        # Conexión residual + LayerNorm
        x = self.norm1.forward(x + attn_out)
        # x + attn_out: el input original se suma al output de attention
        # norm1.forward: normaliza el resultado
        # (batch, seq, 128)

        # Guardar para backward
        self._x_after_norm1 = x

        # ── Sub-bloque 2: Feed Forward ───────────────────────────────────
        ff_out = self.ff.forward(x)
        # (batch, seq, 128)

        # Conexión residual + LayerNorm
        x = self.norm2.forward(x + ff_out)
        # (batch, seq, 128)

        return x

    # ------------------------------------------------------------------
    # Backward
    # ------------------------------------------------------------------

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """
        Recorre el forward al revés.

        Parámetros
        ----------
        dout : (batch, seq, d_model)

        Retorna
        -------
        dx : (batch, seq, d_model) — gradiente hacia el bloque anterior
        """
        # ── Sub-bloque 2 al revés ────────────────────────────────────────

        # norm2 backward
        d_residual2 = self.norm2.backward(dout)
        # (batch, seq, 128)

        # La conexión residual se bifurca en dos caminos:
        # uno va hacia ff.backward, otro pasa directo
        #
        #   x ──────────────────────────────┐
        #   │                               │
        #   └──→ ff.forward(x) → ff_out ───┤ + → norm2 → salida
        #
        # En el backward, d_residual2 se distribuye a ambos caminos
        dx_ff = self.ff.backward(d_residual2)
        dx_skip2 = d_residual2  # camino directo de la residual

        # Suma de los dos caminos
        dx = dx_ff + dx_skip2
        # (batch, seq, 128)

        # ── Sub-bloque 1 al revés ────────────────────────────────────────

        # norm1 backward
        d_residual1 = self.norm1.backward(dx)
        # (batch, seq, 128)

        # Mismo patrón: d_residual1 se distribuye a attention y al skip
        dx_attn = self.attention.backward(d_residual1)
        dx_skip1 = d_residual1

        dx = dx_attn + dx_skip1
        # (batch, seq, 128)

        return dx

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    def update(self, lr: float) -> None:
        """Actualiza todos los parámetros del bloque."""
        self.attention.update(lr)
        self.norm1.update(lr)
        self.ff.update(lr)
        self.norm2.update(lr)

    # ------------------------------------------------------------------
    # Utilidades
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"EncoderBlock(\n"
            f"  {self.attention}\n"
            f"  {self.norm1}\n"
            f"  {self.ff}\n"
            f"  {self.norm2}\n"
            f")"
        )

    @property
    def num_parameters(self) -> int:
        return (
            self.attention.num_parameters +
            self.norm1.num_parameters +
            self.ff.num_parameters +
            self.norm2.num_parameters
        )
