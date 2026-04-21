"""colab_training.py
Full training pipeline designed for Google Colab execution.
Architecture position: orchestrates dataset loading, batching, model updates,
evaluation, checkpointing, and metric visualization around the Transformer.
"""

import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

print("[1/6] Instalando dependencias...")
os.system("pip install -q datasets transformers scikit-learn matplotlib")

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
EPOCHS = 2
BATCH_SIZE = 16 if DATASET_SIZE == "small" else (32 if DATASET_SIZE == "medium" else 64)
LR = TransformerConfig.lr
EVAL_EVERY = 50  # solo aplica si hay suficientes batches por época
SAVE_EVERY = 250
RESUME_FROM_CHECKPOINT = True
RESUME_PATH = None  # e.g. "checkpoints/epoch_3_final.npz"
DATASET_SHUFFLE_SEED = 42  # None => seed aleatorio en cada rerun

# Wikipedia larga genera MUCHISIMOS chunks. Para demos/Colab, conviene topar.
# Si pones None, no se submuestrea (puede volverse muy lento en NumPy).
MAX_TRAIN_CHUNKS = 20000
MAX_VAL_CHUNKS = 5000
print(f"[CONFIG] dataset={DATASET_SIZE} | epochs={EPOCHS} | batch={BATCH_SIZE} | lr={LR}")
print(f"[CONFIG] device={DEVICE}")
print(f"[CONFIG] max_train_chunks={MAX_TRAIN_CHUNKS} | max_val_chunks={MAX_VAL_CHUNKS}")

if DATASET_SHUFFLE_SEED is None:
    DATASET_SHUFFLE_SEED = int(np.random.default_rng().integers(0, 2**31 - 1))
print(f"[CONFIG] dataset_shuffle_seed={DATASET_SHUFFLE_SEED}")
# Resume training
RESUME_FROM_CHECKPOINT = True
RESUME_PATH = None  # e.g. "checkpoints/epoch_3_final.npz"


# ── Crear carpetas ───────────────────────────────────────────────────────────

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
print(f"[OK] Carpeta {CHECKPOINT_DIR} creada")

# ── Descargar dataset (Wikipedia en español) ─────────────────────────────────

print("\n[3/6] Descargando dataset Wikipedia español...")
from datasets import load_dataset

size_map = {"small": 10000, "medium": 100000, "large": 500000}
target_n = size_map[DATASET_SIZE]

try:
    # Hugging Face `datasets` ya no soporta scripts legacy como `wikipedia.py`.
    # La forma estable es cargar los shards Parquet publicados en el Hub.
    #
    # Referencia: dataset `wikimedia/wikipedia` con subset `20231101.es`.
    # Importante:
    # Usar split="train[:N]" puede terminar construyendo/leyendo casi todo el split
    # antes de aplicar el slice (caro en tiempo/disco).
    # Streaming + take(N) suele ser mucho mas eficiente para prototipos.
    stream = load_dataset(
        "parquet",
        data_files="hf://datasets/wikimedia/wikipedia/20231101.es/*.parquet",
        split="train",
        streaming=True,
    )
    # Barajar el stream hace que cada rerun tome documentos distintos.
    stream = stream.shuffle(seed=DATASET_SHUFFLE_SEED, buffer_size=50_000)
    dataset = list(stream.take(target_n))
    print(f"[OK] Dataset cargado: {len(dataset)} documentos (streaming.shuffle + take)")
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

print("\n[4/6] Extrayendo texto plano del dataset (un registro = un documento)...")
from sklearn.model_selection import train_test_split

all_texts = [doc["text"] if isinstance(doc, dict) else str(doc) for doc in dataset]

# ── Inicializar modelo ───────────────────────────────────────────────────────

print("\n[5/6] Inicializando modelo...")
cfg = TransformerConfig()
print(
    "[CONFIG] arquitectura (desde Config.py) -> "
    f"vocab={cfg.vocab_size}, d_model={cfg.d_model}, n_heads={cfg.n_heads}, "
    f"n_layers={cfg.n_layers}, d_ff={cfg.d_ff}, max_seq_len={cfg.max_seq_len}"
)

tokenizer = BPETokenizer.load("Tokenizer/vocab/tokenizer.json")

# Wikipedia trae articulos muy largos. Debemos partir cada documento en chunks
# de tokens que, luego de agregar <BOS>/<EOS>, no superen cfg.max_seq_len.
#
# IMPORTANTE:
# Guardamos cada chunk como *texto* (`decode(ids)`), pero en training volvemos a
# tokenizar ese texto (`encode`). Ese round-trip puede cambiar levemente el largo.
# Por eso usamos un margen de seguridad (-3) para que casi siempre:
#   2 + len(encode(chunk)) <= max_seq_len
MAX_CONTENT_TOKENS = max(8, cfg.max_seq_len - 3)
STRIDE_CONTENT_TOKENS = max(MAX_CONTENT_TOKENS // 2, 1)


def chunk_wikipedia_documents(
    tokenizer: BPETokenizer,
    texts: list[str],
    max_content_tokens: int,
    stride_tokens: int,
    max_seq_len: int,
    max_chunks: int | None = None,
) -> list[str]:
    """
    Convierte articulos largos en muchos ejemplos cortos de LM.

    Estrategia:
      encode(doc) -> ids largos -> ventanas con stride (solapadas) sobre ids.

    Por que ids y no texto crudo:
      cortar texto UTF-8 al azar puede romper palabras; cortar sobre tokens del
      mismo tokenizador garantiza chunks consistentes.
    """
    chunked_texts: list[str] = []
    skipped_empty = 0

    hard_limit = max_seq_len - 2  # lugar para <BOS> y <EOS>

    def _finalize_chunk(raw_ids: list[int]) -> str | None:
        """
        ids -> texto -> ids, garantizando len(ids) <= hard_limit.
        Si el round-trip crece, recorta desde el final (conservativo y simple).
        """
        if not raw_ids:
            return None

        ids_tail = raw_ids[-hard_limit:] if len(raw_ids) > hard_limit else raw_ids

        text_chunk = tokenizer.decode(ids_tail).strip()
        if not text_chunk:
            return None

        chk = tokenizer.encode(text_chunk)
        if len(chk) > hard_limit:
            chk = chk[:hard_limit]
            text_chunk = tokenizer.decode(chk).strip()

        chk2 = tokenizer.encode(text_chunk)
        if len(chk2) > hard_limit:
            chk2 = chk2[:hard_limit]
            text_chunk = tokenizer.decode(chk2).strip()

        return text_chunk or None

    for doc_idx, doc in enumerate(texts):
        if not isinstance(doc, str):
            doc = str(doc)

        ids = tokenizer.encode(doc)
        if len(ids) == 0:
            skipped_empty += 1
            continue

        # Si entra completo y es corto, no fragmentamos.
        if len(ids) <= max_content_tokens:
            text_chunk = _finalize_chunk(ids)
            if text_chunk:
                chunked_texts.append(text_chunk)
                if max_chunks is not None and len(chunked_texts) >= max_chunks:
                    break
            continue

        step = stride_tokens if stride_tokens > 0 else max_content_tokens
        for start in range(0, len(ids), step):
            window = ids[start : start + max_content_tokens]
            if len(window) < max(16, max_content_tokens // 4):
                continue

            text_chunk = _finalize_chunk(window)
            if text_chunk:
                chunked_texts.append(text_chunk)

            if max_chunks is not None and len(chunked_texts) >= max_chunks:
                break

        if max_chunks is not None and len(chunked_texts) >= max_chunks:
            break

    print(
        f"[DATA] Docs originales: {len(texts)} | "
        f"Ejemplos chunkados: {len(chunked_texts)} | "
        f"Vacios omitidos: {skipped_empty}"
    )
    return chunked_texts


train_docs, val_docs = train_test_split(
    all_texts,
    test_size=0.2,
    random_state=DATASET_SHUFFLE_SEED,
)
print(f"[OK] Docs train: {len(train_docs)} | Docs val: {len(val_docs)}")

train_texts = chunk_wikipedia_documents(
    tokenizer,
    train_docs,
    MAX_CONTENT_TOKENS,
    STRIDE_CONTENT_TOKENS,
    cfg.max_seq_len,
    max_chunks=MAX_TRAIN_CHUNKS,
)

val_texts = chunk_wikipedia_documents(
    tokenizer,
    val_docs,
    MAX_CONTENT_TOKENS,
    STRIDE_CONTENT_TOKENS,
    cfg.max_seq_len,
    max_chunks=MAX_VAL_CHUNKS,
)

print(f"[OK] Ejemplos LM train (cap early-stop): {len(train_texts)} | "
      f"Ejemplos LM val (cap early-stop): {len(val_texts)}")


model = Transformer(
    vocab_size=cfg.vocab_size,
    d_model=cfg.d_model,
    n_heads=cfg.n_heads,
    n_layers=cfg.n_layers,
    d_ff=cfg.d_ff,
    max_seq_len=cfg.max_seq_len,
)
print(f"[OK] Modelo con {model.num_parameters:,} parámetros")

# ── Resume opcional de checkpoint ───────────────────────────────────────────
start_epoch = 1

if RESUME_FROM_CHECKPOINT:
    ckpt_to_load = RESUME_PATH

    if ckpt_to_load is None:
        candidates = []
        for fn in os.listdir(CHECKPOINT_DIR):
            if fn.startswith("epoch_") and fn.endswith("_final.npz"):
                try:
                    ep = int(fn.split("_")[1])
                    candidates.append((ep, fn))
                except Exception:
                    pass

        if candidates:
            candidates.sort(key=lambda x: x[0])
            last_ep, last_fn = candidates[-1]
            ckpt_to_load = os.path.join(CHECKPOINT_DIR, last_fn)
            start_epoch = last_ep + 1

    if ckpt_to_load and os.path.exists(ckpt_to_load):
        model.load_weights(ckpt_to_load)
        print(f"[OK] Resume desde: {ckpt_to_load}")
        print(f"[OK] Continuando desde epoch {start_epoch}")
    else:
        print("[INFO] No se encontro checkpoint para resume; entrenamiento desde cero.")


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

for epoch in range(start_epoch, start_epoch + EPOCHS):
    print(f"\n{'='*60}")
    print(f"EPOCH {epoch}/{EPOCHS}")
    print(f"{'='*60}")

    epoch_losses = []
    num_batches = max(1, int(np.ceil(len(train_texts) / BATCH_SIZE)))
    eval_every = min(EVAL_EVERY, max(1, num_batches))

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
            if (batch_idx + 1) % eval_every == 0:
                val_loss = eval_on_val(
                    model, loss_fn, val_texts, tokenizer, max_batches=5
                )
                ppl = float(np.exp(np.clip(loss, 0, 20)))
                val_ppl = float(np.exp(np.clip(val_loss, 0, 20)))
                log_step(epoch, global_batch, loss, val_loss)
                print(
                    f"[batch {batch_idx+1:04d}/{num_batches:04d}] "
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

    # Log final de época (para que SIEMPRE haya filas en training_log.csv)
    try:
        train_tail_loss = float(epoch_losses[-1])
        val_loss_epoch = eval_on_val(
            model, loss_fn, val_texts, tokenizer, max_batches=5
        )
        val_ppl_epoch = float(np.exp(np.clip(val_loss_epoch, 0, 20)))
        train_ppl_epoch = float(np.exp(np.clip(train_tail_loss, 0, 20)))
        log_step(epoch, global_batch, train_tail_loss, val_loss_epoch)
        print(
            f"[epoch_end] train_tail_loss={train_tail_loss:.4f} (ppl={train_ppl_epoch:.1f}) | "
            f"val_loss={val_loss_epoch:.4f} (ppl={val_ppl_epoch:.1f})"
        )
    except Exception as e:
        print(f"[WARN] No se pudo loguear fin de época: {e}")

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
