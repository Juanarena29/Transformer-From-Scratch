"""main.py
End-to-end smoke test and mini training demo.
Architecture position: executable entrypoint that wires tokenizer, model,
batching, training loop, and checkpoint persistence.
"""

import numpy as np
from Tokenizer.tokenizer import BPETokenizer
from transformer import Transformer
from Config import TransformerConfig
from trainer import build_lm_batch, train_loop


def build_model(cfg: TransformerConfig) -> Transformer:
    """Instantiate a Transformer from configuration values.

    Parameters
    ----------
    cfg : TransformerConfig
        Hyperparameter container.

    Returns
    -------
    Transformer
        Initialized Transformer model.
    """
    return Transformer(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        max_seq_len=cfg.max_seq_len,
    )


def main() -> None:
    """Run the local end-to-end demonstration workflow.

    Parameters
    ----------
    None
        Uses local files and default configuration.

    Returns
    -------
    None
        Prints checks and writes a checkpoint file.
    """
    # ── Config ───────────────────────────────────────────────────────────
    cfg = TransformerConfig()

    # ── Instanciar ───────────────────────────────────────────────────────
    tokenizer = BPETokenizer.load("Tokenizer/vocab/tokenizer.json")
    model = build_model(cfg)

    print(model)

    # ── Dataset mínimo de ejemplo para LM ────────────────────────────────
    texts = [
        "hola mundo",
        "el transformer aprende",
        "aprender desde cero ayuda a entender",
        "la atencion mira todas las posiciones",
    ]

    x_batch, y_batch = build_lm_batch(tokenizer, texts)
    logits = model.forward(x_batch)
    print(f"\nx_batch : {x_batch.shape}")
    print(f"y_batch : {y_batch.shape}")
    print(f"logits  : {logits.shape}")

    # ── Verificaciones iniciales ────────────────────────────────────────
    assert logits.shape == (x_batch.shape[0], x_batch.shape[1], cfg.vocab_size)
    print("[OK] Shapes correctos")

    assert not np.isnan(logits).any()
    assert not np.isinf(logits).any()
    print("[OK] Sin NaN ni Inf")

    # ── Entrenamiento real con Cross Entropy ────────────────────────────
    history = train_loop(
        model=model,
        tokenizer=tokenizer,
        texts=texts,
        epochs=20,
        lr=cfg.lr,
        pad_id=tokenizer.special_tokens["<PAD>"],
        bos_id=tokenizer.special_tokens["<BOS>"],
        eos_id=tokenizer.special_tokens["<EOS>"],
    )

    assert history["loss"][-1] <= history["loss"][0], (
        "La loss no bajó. Considera más épocas o ajustar learning rate."
    )
    print("[OK] Entrenamiento con Cross Entropy ejecutado")

    # ── Guardar/cargar pesos ─────────────────────────────────────────────
    ckpt_path = "checkpoints/mini_transformer_weights.npz"
    model.save_weights(ckpt_path)
    print(f"[OK] Pesos guardados en {ckpt_path}")

    reloaded_model = build_model(cfg)
    reloaded_model.load_weights(ckpt_path)
    print("[OK] Pesos cargados en un nuevo modelo")

    logits_original = model.forward(x_batch)
    logits_reloaded = reloaded_model.forward(x_batch)
    assert np.allclose(logits_original, logits_reloaded), (
        "Modelo recargado no reproduce los mismos logits."
    )
    print("[OK] Save/Load consistente (logits identicos)")

    print("\n[OK] Proyecto listo para entrenar de punta a punta")


if __name__ == "__main__":
    main()
