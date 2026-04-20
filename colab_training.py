"""
colab_training.py
-----------------
Script de entrenamiento completo para Google Colab.

Características:
  - descarga Wikipedia español automáticamente
  - mini-batching real
  - train/val split
  - checkpointing periódico
  - logging a CSV
  - plotting de métricas

Instrucciones:
  1. Abre https://colab.research.google.com
  2. Copia este script en la primera celda
  3. Ejecuta (Cell > Run All)
  4. Monitorea en tiempo real los gráficos y logs
"""

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print("[1/6] Instalando dependencias...")
os.system("pip install -q datasets transformers")

# ── Importar módulos locales ─────────────────────────────────────────────────

print("[2/6] Importando módulos del transformer...")

# Si estás en Colab, descargar repo primero
try:
    from transformer import Transformer
    from Config import TransformerConfig
    from Tokenizer.tokenizer import BPETokenizer
    from trainer import build_lm_batch, train_step
    from loss import CrossEntropyLoss
except ImportError:
    print("    Clonando repositorio desde GitHub...")
    os.system(
        "git clone https://github.com/TU_USUARIO/TU_REPO.git /tmp/transformer && "
        "cp -r /tmp/transformer/* . && "
        "rm -rf /tmp/transformer"
    )
    from transformer import Transformer
    from Config import TransformerConfig
    from Tokenizer.tokenizer import BPETokenizer
    from trainer import build_lm_batch, train_step
    from loss import CrossEntropyLoss

# ── Configuración ────────────────────────────────────────────────────────────

COLAB_MODE = "google.colab" in sys.modules  # Detectar si es Colab
DEVICE = "gpu" if COLAB_MODE else "cpu"
CHECKPOINT_DIR = "checkpoints"
LOG_FILE = "training_log.csv"

# Hiperparámetros
DATASET_SIZE = "small"  # "small" (10K), "medium" (100K), "large" (500K)
EPOCHS = 3
BATCH_SIZE = 16 if DATASET_SIZE == "small" else (32 if DATASET_SIZE == "medium" else 64)
LR = 0.001 if DATASET_SIZE == "small" else (0.0005 if DATASET_SIZE == "medium" else 0.0003)
EVAL_EVERY = 100
SAVE_EVERY = 500

print(f"[CONFIG] dataset={DATASET_SIZE} | epochs={EPOCHS} | batch={BATCH_SIZE} | lr={LR}")
print(f"[CONFIG] device={DEVICE}")

# ── Crear carpetas ───────────────────────────────────────────────────────────

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print(f"[OK] Carpeta {CHECKPOINT_DIR} creada")

# ── Descargar dataset (Wikipedia en español) ─────────────────────────────────

print("\n[3/6] Descargando dataset Wikipedia español...")
from datasets import load_dataset

size_map = {"small": 10000, "medium": 100000, "large": 500000}
split_str = f"train[:{size_map[DATASET_SIZE]}]"

try:
    dataset = load_dataset("wikipedia", "20231101.es", split=split_str, trust_remote_code=True)
    print(f"[OK] Dataset cargado: {len(dataset)} documentos")
except Exception as e:
    print(f"[WARN] No se pudo descargar Wikipedia: {e}")
    print("       Usando corpus de ejemplo local...")
    dataset = [
        {"text": "el transformador es una arquitectura neural revolucionaria"},
        {"text": "la atencion es el mecanismo clave del transformer"},
        {"text": "se puede entrenar desde cero para aprender patrones"},
        {"text": "los tokens se procesan de forma paralela"},
        {"text": "la codificacion posicional preserva la informacion de posicion"},
    ] * 200  # repetir para tener al menos 1000


# ── Train / Val split ────────────────────────────────────────────────────────

print("\n[4/6] Preparando dataset: 80% train / 20% val...")
from sklearn.model_selection import train_test_split

all_texts = [doc["text"] if isinstance(doc, dict) else str(doc) for doc in dataset]
train_texts, val_texts = train_test_split(all_texts, test_size=0.2, random_state=42)
print(f"[OK] Train: {len(train_texts)} | Val: {len(val_texts)}")

# ── Inicializar modelo ───────────────────────────────────────────────────────

print("\n[5/6] Inicializando modelo...")
cfg = TransformerConfig()

# Ajustar config según dataset
if DATASET_SIZE == "small":
    cfg.n_layers = 2
    cfg.d_model = 64
    cfg.n_heads = 4
    cfg.d_ff = 256
elif DATASET_SIZE == "medium":
    cfg.n_layers = 4
    cfg.d_model = 128
    cfg.n_heads = 8
    cfg.d_ff = 512
else:
    cfg.n_layers = 6
    cfg.d_model = 256
    cfg.n_heads = 8
    cfg.d_ff = 1024

tokenizer = BPETokenizer.load("Tokenizer/vocab/tokenizer.json")
model = Transformer(
    vocab_size=cfg.vocab_size,
    d_model=cfg.d_model,
    n_heads=cfg.n_heads,
    n_layers=cfg.n_layers,
    d_ff=cfg.d_ff,
    max_seq_len=cfg.max_seq_len,
)
print(f"[OK] Modelo con {model.num_parameters:,} parámetros")

# ── Data loader por mini-batches ─────────────────────────────────────────────

def batch_iterator(texts, batch_size):
    """Yield batches de texts."""
    for i in range(0, len(texts), batch_size):
        yield texts[i : i + batch_size]


# ── Loss function ────────────────────────────────────────────────────────────

loss_fn = CrossEntropyLoss(ignore_index=tokenizer.special_tokens["<PAD>"])

# ── Logging ──────────────────────────────────────────────────────────────────

def init_logger():
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "batch", "train_loss", "train_ppl", "val_loss", "val_ppl"])


def log_step(epoch, batch, train_loss, val_loss=None):
    train_ppl = float(np.exp(np.clip(train_loss, 0, 20)))
    val_ppl = float(np.exp(np.clip(val_loss, 0, 20))) if val_loss else None
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, batch, train_loss, train_ppl, val_loss, val_ppl])


def read_log():
    """Lee log para plotear."""
    try:
        with open(LOG_FILE, "r") as f:
            reader = csv.DictReader(f)
            data = list(reader)
        return data
    except:
        return []


# ── Evaluación en val set ────────────────────────────────────────────────────

def eval_on_val(model, loss_fn, val_texts, tokenizer, max_batches=10):
    """Calcula loss en validation set (primeros max_batches)."""
    total_loss = 0.0
    count = 0
    for batch_texts in batch_iterator(val_texts, BATCH_SIZE):
        if count >= max_batches:
            break
        try:
            x, y = build_lm_batch(
                tokenizer,
                batch_texts,
                pad_id=tokenizer.special_tokens["<PAD>"],
                bos_id=tokenizer.special_tokens["<BOS>"],
                eos_id=tokenizer.special_tokens["<EOS>"],
            )
            logits = model.forward(x)
            loss, _ = loss_fn.forward(logits, y)
            total_loss += loss
            count += 1
        except Exception as e:
            print(f"[WARN] Error en batch val: {e}")
            continue
    return total_loss / count if count > 0 else float("inf")


# ── Entrenamiento ───────────────────────────────────────────────────────────

print("\n[6/6] Iniciando entrenamiento...")
init_logger()

global_batch = 0

for epoch in range(1, EPOCHS + 1):
    print(f"\n{'='*60}")
    print(f"EPOCH {epoch}/{EPOCHS}")
    print(f"{'='*60}")

    epoch_losses = []

    for batch_idx, batch_texts in enumerate(batch_iterator(train_texts, BATCH_SIZE)):
        try:
            x, y = build_lm_batch(
                tokenizer,
                batch_texts,
                pad_id=tokenizer.special_tokens["<PAD>"],
                bos_id=tokenizer.special_tokens["<BOS>"],
                eos_id=tokenizer.special_tokens["<EOS>"],
            )

            loss = train_step(model, loss_fn, x, y, lr=LR)
            epoch_losses.append(loss)
            global_batch += 1

            # ── Logging ──────────────────────────────────────────────
            if (batch_idx + 1) % EVAL_EVERY == 0:
                val_loss = eval_on_val(
                    model, loss_fn, val_texts, tokenizer, max_batches=5
                )
                ppl = float(np.exp(np.clip(loss, 0, 20)))
                val_ppl = float(np.exp(np.clip(val_loss, 0, 20)))
                log_step(epoch, global_batch, loss, val_loss)
                print(
                    f"[batch {batch_idx+1:04d}] "
                    f"train_loss={loss:.4f} (ppl={ppl:.1f}) | "
                    f"val_loss={val_loss:.4f} (ppl={val_ppl:.1f})"
                )

            # ── Checkpointing ────────────────────────────────────────
            if global_batch % SAVE_EVERY == 0:
                ckpt_path = f"{CHECKPOINT_DIR}/epoch_{epoch}_batch_{global_batch}.npz"
                model.save_weights(ckpt_path)
                print(f"[CKPT] Guardado en {ckpt_path}")

        except Exception as e:
            print(f"[ERROR] En batch {batch_idx}: {e}")
            continue

    # ── Fin de época ─────────────────────────────────────────────────
    avg_loss = float(np.mean(epoch_losses))
    print(f"\n[EPOCH {epoch}] Promedio loss: {avg_loss:.4f}")

    # ── Guardar checkpoint de época ──────────────────────────────────
    epoch_ckpt = f"{CHECKPOINT_DIR}/epoch_{epoch}_final.npz"
    model.save_weights(epoch_ckpt)
    print(f"[CKPT] Guardado en {epoch_ckpt}")

# ── Plotear resultados ───────────────────────────────────────────────────────

print("\n" + "="*60)
print("ENTRENAMIENTO COMPLETADO")
print("="*60)

log_data = read_log()
if log_data:
    epochs_list = [int(d["epoch"]) for d in log_data]
    train_losses = [float(d["train_loss"]) for d in log_data]
    val_losses = [float(d["val_loss"]) if d["val_loss"] else None for d in log_data]
    train_ppls = [float(d["train_ppl"]) for d in log_data]
    val_ppls = [float(d["val_ppl"]) if d["val_ppl"] else None for d in log_data]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss plot
    axes[0].plot(epochs_list, train_losses, "o-", label="Train Loss", linewidth=2)
    if any(v is not None for v in val_losses):
        axes[0].plot(epochs_list, val_losses, "s-", label="Val Loss", linewidth=2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Perplexity plot
    axes[1].plot(epochs_list, train_ppls, "o-", label="Train PPL", linewidth=2)
    if any(v is not None for v in val_ppls):
        axes[1].plot(epochs_list, val_ppls, "s-", label="Val PPL", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Perplexity")
    axes[1].set_title("Training & Validation Perplexity")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("training_metrics.png", dpi=100)
    print(f"[SAVED] Gráficos guardados en training_metrics.png")
    plt.show()

print(f"\n[OK] Logs en {LOG_FILE}")
print(f"[OK] Checkpoints en carpeta {CHECKPOINT_DIR}/")
print("\n¡Entrenamiento completado! Pesos guardados. Listo para inferencia.")
