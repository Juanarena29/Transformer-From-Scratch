# Guía de Entrenamiento para Mini Transformer

## 1) Estado actual: ¿Qué tienes?

- Transformer funcional end-to-end ✅
- Cross Entropy Loss ✅
- Training loop simple (full-batch) ✅
- Save/load de pesos ✅
- Tokenizador BPE entrenado ✅

**Limitación actual**: `main.py` entrena en full-batch sobre 4 textos de ejemplo. No es realista.

---

## 2) Decisión 1: Datasets en español

### Opciones recomendadas (ordenadas por facilidad):

#### A) **Wikipedia en español (RECOMENDADO para empezar)**
- **Fuente**: [Hugging Face Datasets](https://huggingface.co/datasets/wikipedia)
- **Ventaja**: datos limpios, bien estructurados, acceso sencillo
- **Tamaño**: ~4 GB (idioma completo), puedes usar un subset
- **Cómo obtenerlo**:
  ```python
  from datasets import load_dataset
  
  dataset = load_dataset("wikipedia", "20231101.es", split="train[:1000000]")  # 1M docs
  ```

#### B) **Oscar/Common Crawl**
- **Fuente**: [OSCAR Corpus](https://huggingface.co/datasets/oscar-corpus/OSCAR-2301)
- **Ventaja**: muy grande, diverso
- **Desventaja**: menos limpio que Wikipedia
- **Tamaño**: ~50+ GB para español

#### C) **Custom: tu propio corpus**
- Si tienes texto español local (libros, artículos, etc.)
- Mejor control sobre datos
- Más trabajo de limpieza

### Mi recomendación para empezar:
**Wikipedia español + subset pequeño (100K-500K documentos)** = balance perfecto entrenamiento/tiempo

---

## 3) Decisión 2: Estrategia de entrenamiento

### Opción A: Local (tu PC)
**Ventajas**:
- Control total
- sin límites de tiempo/GPU

**Desventajas**:
- lento (solo CPU, a menos que tengas GPU local)
- largo (días/semanas)

**Recomendación**: SOLO si tienes GPU (NVIDIA + CUDA)

---

### Opción B: Google Colab (RECOMENDADO para ti)
**Ventajas**:
- GPU T4/A100 gratis (~40GB VRAM)
- suficiente para entrenar nuestro modelo
- ambiente listo (numpy, pandas, HF datasets)
- puedes pausar/reanudar

**Desventajas**:
- sesión máx 12h (pero colab guarda checkpoints automáticos)
- cuota diaria limitada (pero generosa)
- no local

**Recomendación**: **PERFECTO para ti ahora**

---

### Opción C: Kaggle Notebooks (alternativa a Colab)
- Similar a Colab
- 30h de GPU por semana
- interfaz similar

---

## 4) Plan de entrenamiento recomendado

### **Fase 1: Prototipo en Colab (1-2 sesiones)**
1. descargar dataset pequeño (10K docs Wikipedia)
2. entrenar 1-2 épocas
3. ver que loss baje
4. guardar checkpoint
5. medir tiempo/memoria

### **Fase 2: Escala intermedia (5-10 sesiones)**
1. aumentar a 100K docs
2. entrenar 5-10 épocas
3. implementar mini-batching (no full-batch)
4. agregar validación set
5. guardar checkpoints periódicamente

### **Fase 3: Escala real (múltiples sesiones)**
1. 500K+ docs o dataset completo
2. 20+ épocas
3. ajuste fino de hiperparámetros
4. inferencia con peso final

---

## 5) Cambios técnicos necesarios para escalar

### Cambio 1: Mini-batching (CRÍTICO)
**Problema actual**: full-batch = todos los textos al mismo tiempo

**Solución**: procesar en lotes de 32-64

```python
# Pseudo-código
for epoch in range(epochs):
    for batch_idx, batch_texts in enumerate(dataloader):
        x, y = build_lm_batch(tokenizer, batch_texts)
        loss = train_step(model, loss_fn, x, y, lr)
        
        if batch_idx % 100 == 0:
            print(f"batch {batch_idx} | loss {loss:.4f}")
```

**Implementación**: agregar `data_loader.py` (ver sección 7 abajo)

---

### Cambio 2: Train/Val split
**Problema actual**: no sabes si el modelo generaliza

**Solución**: 80% train, 20% val

```python
train_texts, val_texts = train_test_split(all_texts, test_size=0.2)
train_loss = train_loop(model, tokenizer, train_texts, epochs=5, ...)
val_loss = eval_loop(model, loss_fn, val_texts, ...)
```

---

### Cambio 3: Checkpointing cada N batches
**Problema actual**: si cae Colab a las 5h, pierdes todo

**Solución**: guardar cada 1000 batches

```python
if batch_idx % 1000 == 0:
    model.save_weights(f"checkpoints/epoch_{epoch}_batch_{batch_idx}.npz")
```

---

### Cambio 4: Logging a archivo + métricas en tiempo real
**Problema actual**: solo ves consola (incómodo en Colab)

**Solución**: registrar en CSV, plotear

```python
import csv

logger = csv.writer(open("train_log.csv", "w"))
logger.writerow(["epoch", "batch", "loss", "val_loss", "ppl"])

for epoch, batch, loss in train_loop(...):
    logger.writerow([epoch, batch, loss, ...])
    # después plotear con matplotlib
```

---

## 6) Configuraciones recomendadas por dataset size

### Dataset pequeño (10K docs, ~50MB)
```python
cfg.n_layers = 2        # más ligero
cfg.d_model = 64
cfg.n_heads = 4
cfg.d_ff = 256
batch_size = 16
epochs = 3
lr = 0.001
```
**Tiempo esperado**: 10-30 min en GPU T4

---

### Dataset medio (100K docs, ~500MB)
```python
cfg.n_layers = 4
cfg.d_model = 128
cfg.n_heads = 8
cfg.d_ff = 512
batch_size = 32
epochs = 5
lr = 0.0005
```
**Tiempo esperado**: 2-4 horas en GPU T4

---

### Dataset grande (500K+ docs, ~2GB+)
```python
cfg.n_layers = 6
cfg.d_model = 256
cfg.n_heads = 8
cfg.d_ff = 1024
batch_size = 64
epochs = 10
lr = 0.0003
```
**Tiempo esperado**: 8-24 horas en GPU T4 (multipart en sesiones Colab)

---

## 7) Pasos concretos para Google Colab

### Paso 1: Crear notebook Colab
1. Ir a [colab.research.google.com](https://colab.research.google.com)
2. "New notebook"
3. Renombrar a `"Transformer_Training"`

### Paso 2: Configurar GPU
```python
# En primera celda
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
print("GPU disponible:", torch.cuda.is_available())  # (si verificas con torch)
```

### Paso 3: Instalar dependencias
```python
!pip install datasets transformers numpy -q
```

### Paso 4: Subir archivos del repo
Opciones:
- **A) Desde GitHub** (más fácil):
  ```python
  !git clone https://github.com/TU_USUARIO/TU_REPO.git
  %cd TU_REPO
  ```
- **B) Subirlos directamente**: menu "Archivos" en Colab
- **C) Desde Google Drive**:
  ```python
  from google.colab import drive
  drive.mount('/content/drive')
  %cd /content/drive/My\ Drive/Transformer
  ```

### Paso 5: Entrenar
Ve a sección 8 abajo (`colab_training.py`)

---

## 8) Script listo para Colab (colab_training.py)

**Te lo proporcionaré en archivo separado** con:
- descarga automática de Wikipedia
- mini-batching
- checkpointing
- logging a CSV
- plotting de loss/val

---

## 9) Checklist antes de entrenar

- [ ] Tienes acceso a GPU (Colab o local)
- [ ] Dataset descargado/accesible
- [ ] Tokenizador guardado (`Tokenizer/vocab/tokenizer.json`)
- [ ] `Config.py` con hiperparámetros ajustados
- [ ] Carpeta `checkpoints/` creada
- [ ] `requirements.txt` actualizado si añades dependencias
- [ ] Versión del código en Git/backup (por si necesitas revertir)

---

## 10) Monitoreo durante entrenamiento

### Métrica principal: Loss
- **Debe bajar** linealmente o logarítmicamente
- Si no baja → learning rate muy bajo o datos problemáticos
- Si sube → learning rate muy alto (divergencia)

### Métrica secundaria: Perplexity (PPL)
- `PPL = exp(loss)`
- Más interpretable que loss (# de tokens equiprobables)
- Meta: PPL < 100-200 es bueno para LM básico

### Validación
- Si val_loss > train_loss + 0.5 → **overfitting** (más datos/regularización)
- Si val_loss ≈ train_loss → generalizando bien

---

## 11) Próximas optimizaciones (después del entrenamiento básico)

1. **Cambiar a Adam optimizer** (vs SGD actual)
   - Convergencia más rápida
   - Requiere cambio en `trainer.py`

2. **Learning rate scheduler**
   - Bajar LR conforme el training avanza
   - Mejora convergencia

3. **Gradient clipping**
   - Evita explosión de gradientes
   - Importante si usas modelo más profundo

4. **Regularización (dropout)**
   - No implementado aún en nuestro transformer
   - Reduce overfitting

5. **Distributed training** (si necesitas)
   - PyTorch + TPU en Colab
   - Bastante más complejo

---

## 12) Troubleshooting común

| Problema | Causa | Solución |
|----------|-------|----------|
| Loss no baja | LR muy bajo | Aumentar `lr` a 0.001-0.01 |
| Loss explota (NaN) | LR muy alto | Disminuir `lr` a 0.0001 |
| Memory out (OOM) | Batch muy grande | Reducir `batch_size` |
| Entrenamiento muy lento | CPU vs GPU | Verificar GPU activa en Colab |
| Val loss sube | Overfitting | Más datos, menos épocas, regularización |

---

## 13) Resumen: qué haces ahora

1. **Elige dataset**: Wikipedia español (recomendado)
2. **Elige plataforma**: Google Colab (recomendado)
3. **Descarga script Colab**: lo agrego en sección siguiente
4. **Configura hiperparámetros**: empieza con config "Dataset pequeño"
5. **Ejecuta en Colab**: 1-2 sesiones para prototipo
6. **Monitorea**: loss, perplexity, validación
7. **Guarda checkpoints**: después de cada sesión
8. **Itera**: ajusta config, más datos, más épocas

---

## Siguiente: Necesitas `colab_training.py` ready-to-run?

Si dices que sí, te lo escribo ahora con:
- descarga Wikipedia automática
- data loader por mini-batches
- train/val split
- checkpointing
- logging a CSV
- ploteo de métricas
