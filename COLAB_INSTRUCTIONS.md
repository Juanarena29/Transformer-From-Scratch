# Training on Google Colab — Step-by-Step Guide

A pre-trained checkpoint (`epoch_6_final.npz`) is already available in the repo — you can run inference immediately without training anything. This guide is for those who want to train the model from scratch or continue training from the provided checkpoint.

---

## Overview

| Step | Description | Time |
|------|-------------|------|
| 1 | Create notebook and enable GPU | 2 min |
| 2 | Clone repo | 1 min |
| 3 | Verify setup | 1 min |
| 4 | Run training | 15–60 min |
| 5 | Save checkpoints to Drive | 2 min |

---

## Step 1 — Create notebook and enable GPU

Open [colab.research.google.com](https://colab.research.google.com) and create a new notebook.

Enable GPU: **Runtime → Change runtime type → T4 GPU → Save**

Verify GPU is available:

```python
import torch
print("GPU available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("Device:", torch.cuda.get_device_name(0))
```

> **Note:** this project trains in pure NumPy (CPU). GPU does not accelerate it unless the backend is migrated to PyTorch/JAX. For small datasets it runs fine on CPU.

---

## Step 2 — Clone the repository

```python
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO transformer
%cd transformer
```

Or mount from Google Drive if you have the code there:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/My\ Drive/Transformer
```

---

## Step 3 — Install dependencies and verify setup

```python
!pip install -r requirements.txt -q
%run quick_start.py
```

Expected output:

```
[OK] VERIFICATION COMPLETE — READY TO TRAIN
```

If you see errors, check:
- Are you in the right directory? (`%pwd`)
- Are all files present? (`%ls`)
- Does the tokenizer exist? (`%ls Tokenizer/vocab/`)

---

## Step 4 — Configure training

Open `colab_training.py` and adjust these parameters if needed:

```python
DATASET_SIZE = "small"      # "small" (10K docs), "medium" (100K), "large" (500K)
EPOCHS = 3                  # number of training epochs
BATCH_SIZE = 16             # reduce to 8 if you hit memory errors
LR = 0.001                  # learning rate
EVAL_EVERY = 100            # validate every N batches
SAVE_EVERY = 500            # save checkpoint every N batches
```

For a first run, start with:

```python
DATASET_SIZE = "small"
EPOCHS = 2
```

### Resume from the provided checkpoint

If you want to continue training from `epoch_6_final.npz` instead of starting from scratch:

```python
RESUME_FROM_CHECKPOINT = True
RESUME_PATH = "checkpoints/epoch_6_final.npz"
```

---

## Step 5 — Run training

```python
%run colab_training.py
```

You will see progress like:

```
======================================================================
EPOCH 1/3
======================================================================
[batch 0100] train_loss=8.6233 (ppl=5505.4) | val_loss=8.6234 (ppl=5505.5)
[CKPT] Saved to checkpoints/epoch_1_batch_100.npz
[batch 0200] train_loss=8.5987 (ppl=5397.2) | val_loss=8.5989 (ppl=5397.3)
...
```

Checkpoints are saved automatically to `checkpoints/`. Plots update at the end of each epoch.

---

## Step 6 — Monitor training

While training runs, open another cell to inspect logs:

```python
import pandas as pd
df = pd.read_csv("training_log.csv")
print(df.tail(10))
```

Or plot:

```python
import matplotlib.pyplot as plt
df = pd.read_csv("training_log.csv")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(df["train_loss"], label="train"); ax1.plot(df["val_loss"], label="val")
ax1.set_title("Loss"); ax1.legend(); ax1.grid(True)
ax2.plot(df["train_ppl"], label="train"); ax2.plot(df["val_ppl"], label="val")
ax2.set_title("Perplexity"); ax2.legend(); ax2.grid(True)
plt.tight_layout(); plt.show()
```

---

## Step 7 — Save checkpoints to Google Drive

Colab sessions are temporary. Save your checkpoints before the session ends:

```python
from shutil import copy2
import os

drive_path = "/content/drive/My Drive/Transformer_Checkpoints"
os.makedirs(drive_path, exist_ok=True)

for file in os.listdir("checkpoints"):
    copy2(f"checkpoints/{file}", f"{drive_path}/{file}")
    print(f"Saved: {file}")

copy2("training_log.csv", f"{drive_path}/training_log.csv")
print("Logs saved.")
```

---

## Step 8 — Generate text

Once training is complete (or using the provided `epoch_6_final.npz`):

```python
from inference import TextGenerator
from Config import TransformerConfig

cfg = TransformerConfig()
gen = TextGenerator(
    model_path="checkpoints/epoch_6_final.npz",
    tokenizer_path="Tokenizer/vocab/tokenizer.json",
    cfg=cfg
)

prompts = [
    "el transformer es",
    "la inteligencia artificial",
    "aprender desde cero",
]

for prompt in prompts:
    text = gen.generate(prompt, max_tokens=40, method="topp", p=0.9)
    print(f"[PROMPT] {prompt}")
    print(f"[OUTPUT] {text}\n")
```

---

## Troubleshooting

| Problem | Fix |
|---------|-----|
| `ModuleNotFoundError` | `!pip install datasets transformers scikit-learn -q` |
| GPU not available | Runtime → Change runtime type → T4 GPU |
| Out of memory | Reduce `BATCH_SIZE` to 8 or 4 |
| Session disconnected | Checkpoints are saved — resume from Drive in a new session |
| Loss explodes (NaN) | Reduce `LR` to `0.0001` |
| Loss not decreasing | Increase `LR` to `0.01` |
| Wrong checkpoint shape | Ensure `Config.py` matches the checkpoint architecture |

---

## Suggested training roadmap

**Session 1 — verification (20–40 min)**
- [ ] Enable GPU
- [ ] Clone repo and verify setup
- [ ] Train with `small` dataset, 2–3 epochs
- [ ] Save checkpoints to Drive

**Session 2 — scaling up (2–4 h)**
- [ ] Resume from checkpoint or train fresh with `medium` dataset
- [ ] Monitor train/val loss for divergence
- [ ] Run inference and inspect generated text

**Session 3+ — full training (8–24 h)**
- [ ] `large` dataset, 20+ epochs
- [ ] Tune learning rate and batch size
- [ ] Save final checkpoint
