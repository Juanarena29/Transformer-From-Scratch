# Entrenar en Google Colab: Instrucciones Paso-a-Paso

## Resumen rápido

1. Abre Colab
2. Clona repo
3. Ejecuta verificación
4. Ejecuta entrenamiento
5. Monitorea y guarda

Tiempo total: 5 min de setup + 15-60 min de entrenamiento (depende de dataset)

---

## Paso 1: Crear/abrir notebook en Colab

Abre [colab.research.google.com](https://colab.research.google.com) en tu navegador.

Click en "New notebook" (o abre uno existente si lo compartiste).

Renombra el notebook: "Transformer_Training_YYYY-MM-DD"

---

## Paso 2: Activar GPU (CRÍTICO)

En el menu: **Runtime > Change runtime type**

Asegúrate de que está seleccionado:
- **Runtime type**: Python 3
- **GPU**: T4 (gratis) o A100 (si tienes cuota)
- **TPU**: NO (para este proyecto no lo necesitas)

Click "Save"

---

## Paso 3: Verificar GPU (opcional pero recomendado)

En una celda nueva, ejecuta:

```python
import torch
print("GPU disponible:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM:", torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")
```

Deberías ver algo como:
```
GPU disponible: True
GPU: Tesla T4
VRAM: 15.9 GB
```

Si dice `False`, repite Paso 2.

---

## Paso 4: Descargar código del repositorio

En una celda nueva, ejecuta:

```python
# Opción A: Desde GitHub
!git clone https://github.com/TU_USUARIO/TU_REPO transformer_repo
%cd transformer_repo

# Si el repo no es tuyo, usa mi template:
# !git clone https://github.com/ignacio-transformer/mini-transformer transformer_repo
# %cd transformer_repo
```

Si tienes el código en Google Drive en su lugar:

```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/My\ Drive/Transformer  # Ajusta la ruta
```

---

## Paso 5: Verificación rápida del setup

En una celda nueva, ejecuta:

```python
%run quick_start.py
```

Deberías ver:
```
[OK] VERIFICACION COMPLETADA - LISTO PARA ENTRENAR
```

Si hay error, revisa:
- ¿Estás en el directorio correcto? (%pwd para confirmar)
- ¿Todos los archivos están? (%ls para listar)
- ¿Tokenizador existe? (`ls Tokenizer/vocab/`)

---

## Paso 6: Configurar entrenamiento

En una celda nueva, editaa el archivo `colab_training.py` si necesitas ajustar:

```python
# Lee el archivo para ver opciones
with open("colab_training.py", "r") as f:
    lines = f.readlines()
    for i, line in enumerate(lines[20:60], start=20):  # primeras líneas de config
        print(f"{i}: {line}", end="")
```

Las opciones principales en `colab_training.py`:

```python
DATASET_SIZE = "small"   # "small" (10K), "medium" (100K), "large" (500K)
EPOCHS = 3               # cuántas épocas entrenar
BATCH_SIZE = 16          # tamaño del batch (auto-ajustable por DATASET_SIZE)
LR = 0.001               # learning rate
EVAL_EVERY = 100         # cada cuántos batches evaluar en validation
SAVE_EVERY = 500         # cada cuántos batches guardar checkpoint
```

Para **primera vez** recomiendo:
```python
DATASET_SIZE = "small"
EPOCHS = 2
```

---

## Paso 7: Ejecutar entrenamiento

En una celda nueva, ejecuta:

```python
%run colab_training.py
```

Verás progreso como:

```
======================================================================
EPOCH 1/2
======================================================================
[batch 0100] train_loss=8.6233 (ppl=5505.4) | val_loss=8.6234 (ppl=5505.5)
[CKPT] Guardado en checkpoints/epoch_1_batch_100.npz
[batch 0200] train_loss=8.5987 (ppl=5397.2) | val_loss=8.5989 (ppl=5397.3)
...
```

**Esto puede tomar 15-60 minutos** dependiendo del dataset.

Durante el entrenamiento:
- Los gráficos se actualizan cada época
- Los checkpoints se guardan automáticamente
- Si Colab se desconecta, puedes continuar en una nueva sesión (los checkpoints se guardan)

---

## Paso 8: Monitorear el entrenamiento

Mientras se ejecuta, puedes abrir otra celda para ver los logs en tiempo real:

```python
import pandas as pd
df = pd.read_csv("training_log.csv")
print(df.tail(10))  # últimas 10 líneas
```

O plotear:

```python
import matplotlib.pyplot as plt
df = pd.read_csv("training_log.csv")
plt.plot(df["epoch"], df["train_loss"], label="Train Loss")
plt.plot(df["epoch"], df["val_loss"], label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
```

---

## Paso 9: Guardar checkpoints a Google Drive (MUY IMPORTANTE)

Los checkpoints se guardan en `checkpoints/` pero **solo en la sesión de Colab temporal**.

Cuando termine el entrenamiento (o durante), **guarda a Drive**:

```python
from shutil import copytree, copy2
import os

# Crear carpeta en Drive
drive_path = "/content/drive/My Drive/Transformer_Checkpoints"
os.makedirs(drive_path, exist_ok=True)

# Copiar todos los checkpoints
for file in os.listdir("checkpoints"):
    src = f"checkpoints/{file}"
    dst = f"{drive_path}/{file}"
    if os.path.isfile(src):
        copy2(src, dst)
        print(f"Copiado: {file}")

# Copiar también logs
copy2("training_log.csv", f"{drive_path}/training_log.csv")
print("Logs copiados!")
```

---

## Paso 10: Generar texto con modelo entrenado

Una vez que tengas un checkpoint guardado:

```python
%run inference.py  # Usa el último checkpoint automáticamente
```

O especificar checkpoint manualmente:

```python
from inference import TextGenerator
from Config import TransformerConfig

cfg = TransformerConfig()
gen = TextGenerator(
    model_path="checkpoints/epoch_2_final.npz",
    tokenizer_path="Tokenizer/vocab/tokenizer.json",
    cfg=cfg
)

prompts = [
    "el transformer es",
    "la inteligencia artificial",
    "aprender desde cero",
]

for prompt in prompts:
    text = gen.generate(prompt, max_tokens=30, method="topp", p=0.9)
    print(f"[PROMPT] {prompt}")
    print(f"[OUTPUT] {text}\n")
```

---

## Troubleshooting

### "Module not found" error
```
ModuleNotFoundError: No module named 'datasets'
```
Solución: Instala en la primera celda:
```python
!pip install datasets transformers scikit-learn -q
```

### "GPU no disponible"
Solución: Runtime > Change runtime type > GPU T4

### "Memory out" (OOM)
Solución: Reduce `BATCH_SIZE` en `colab_training.py`:
```python
BATCH_SIZE = 8  # en lugar de 16, 32, etc.
```

### La sesión se desconectó
No te preocupes: los checkpoints están guardados en `checkpoints/`.

En una nueva sesión:
1. Clona repo nuevamente
2. Copia checkpoints de Drive a `checkpoints/`
3. Modifica `colab_training.py` para cargar checkpoint inicial (opcional)
4. Vuelve a ejecutar entrenamiento

### Loss explota (NaN)
Solución: Reduce learning rate:
```python
LR = 0.0001  # en lugar de 0.001
```

### Loss no baja
Solución: Aumenta learning rate:
```python
LR = 0.01  # en lugar de 0.001
```

---

## Recomendaciones finales

1. **Comienza con dataset pequeño**: prueba `DATASET_SIZE = "small"`, `EPOCHS = 2`
2. **Monitorea loss**: debe bajar consistentemente
3. **Guarda a Drive**: no confíes solo en Colab temporal
4. **Itera**: si loss baja bien, escala a dataset más grande
5. **Experimenta**: ajusta hiperparámetros según resulados

---

## Próxima sesión después de entrenar

Si quieres continuar entrenamiento en otra sesión (con más datos, más épocas):

```python
# En la nueva sesión

# 1. Setup igual que antes
!git clone ...
%cd ...
%run quick_start.py

# 2. Cargar checkpoint anterior
model.load_weights("checkpoints/epoch_2_final.npz")

# 3. Ejecutar train_loop nuevamente con más épocas/datos
from trainer import train_loop
history = train_loop(model, tokenizer, new_texts, epochs=5, lr=0.0001, ...)
```

---

## Hoja de ruta

**Primera sesión (20-40 min):**
- [ ] Setup GPU
- [ ] Descargar código
- [ ] Verificación rápida
- [ ] Entrenar con dataset pequeño (3 épocas)
- [ ] Guardar a Drive

**Segunda sesión (2-4h):**
- [ ] Cargar código y checkpoints
- [ ] Entrenar con dataset mediano (5-10 épocas)
- [ ] Monitorear loss/val_loss
- [ ] Generar texto de prueba

**Tercera sesión+ (8-24h):**
- [ ] Dataset grande
- [ ] 20+ épocas
- [ ] Fine-tuning de hiperparámetros
- [ ] Evaluación final

---

**¿Listo? Abre Colab y ¡comienza!**

Cualquier duda, consulta `ENTRENAMIENTO_GUIDE.md` o `PROYECTO_TRANSFORMER.md`.
