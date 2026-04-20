# Mini Transformer From Scratch en NumPy

Implementación educativa completa de un Transformer encoder-only en NumPy puro, con tokenizador BPE para español.

## ⚡ Quick Start

### 1) Ver el proyecto ya funcionando
```bash
python main.py
```
Entrena 20 épocas en un mini-dataset de ejemplo y guarda pesos.

### 2) Entrenar en serio en Google Colab
1. Abre [colab.research.google.com](https://colab.research.google.com)
2. Sube/clona este repositorio
3. Ejecuta:
   ```python
   %run colab_training.py
   ```
4. Monitorea loss, perplexity, validación en tiempo real

### 3) Generar texto con modelo entrenado
```python
python inference.py
```
Genera texto usando métodos: greedy, top-K, nucleus sampling.

---

## 📚 Documentación

| Archivo | Propósito |
|---------|-----------|
| `PROYECTO_TRANSFORMER.md` | Explicación técnica completa del proyecto (para tu video) |
| `ENTRENAMIENTO_GUIDE.md` | Guía paso-a-paso para elegir dataset y entrenar |
| `colab_training.py` | Script ready-to-run para Google Colab |
| `inference.py` | Generación de texto post-entrenamiento |

---

## 🏗️ Arquitectura

**Transformer encoder de 4 capas:**
- Token Embedding (vocab=6000)
- Positional Encoding sinusoidal
- 4× Encoder Blocks, cada uno con:
  - Multi-Head Attention (8 cabezas)
  - Feed Forward (expansión 4x)
  - Layer Norm + Residual connections
- LM Head (proyección a vocab)

**Pérdida:** Cross Entropy con soporte para `<PAD>`

**Optimizador:** SGD (reemplazar por Adam para mejor convergencia)

---

## 📊 Configuración (Config.py)

```python
vocab_size = 6000       # tokens del vocabulario
d_model = 128           # dimensión de embeddings
n_heads = 8             # cabezas de atención
n_layers = 4            # bloques encoder
d_ff = 512              # expansión FFN
max_seq_len = 512       # máxima longitud de secuencia
lr = 0.001              # learning rate
```

Ajustable según datos/recurso disponible.

---

## 🎯 Paso a paso: entrenar tu propio modelo

### Opción A: Local (rápido para prototipar)
```bash
python main.py                    # validar setup
python colab_training.py          # entrenar (ajustar DATASET_SIZE)
```

### Opción B: Google Colab (RECOMENDADO)
1. Subir código
2. Ejecutar `colab_training.py` con GPU gratuita
3. Guardar checkpoints a Google Drive
4. Continuar en otra sesión si es necesario

### Opción C: Kaggle Notebooks
Similar a Colab, 30h GPU/semana.

---

## 🔍 Monitoreo

Durante entrenamiento, `colab_training.py` registra:
- **Training loss** y **perplexity** por batch
- **Validation loss** cada N batches
- Checkpoints cada M batches
- Gráficos al final de cada sesión

Archivo: `training_log.csv`

---

## 💾 Guardar / Cargar pesos

Automático durante entrenamiento:
```python
model.save_weights("checkpoints/my_model.npz")
model.load_weights("checkpoints/my_model.npz")
```

---

## 🤖 Generar texto

```python
from inference import TextGenerator
from Config import TransformerConfig

cfg = TransformerConfig()
gen = TextGenerator("checkpoints/epoch_3_final.npz", "Tokenizer/vocab/tokenizer.json", cfg)

# Greedy (determinístico)
text = gen.generate("el transformer", max_tokens=30, method="greedy")

# Top-K (más variedad)
text = gen.generate("el transformer", max_tokens=30, method="topk", k=5)

# Nucleus (mejor balance)
text = gen.generate("el transformer", max_tokens=30, method="topp", p=0.9)

print(text)
```

---

## 📈 Rendimiento esperado

| Dataset | Batch Size | Epochs | GPU | Tiempo | Loss final |
|---------|-----------|--------|-----|--------|-----------|
| 10K docs | 16 | 3 | T4 | 15 min | ~8.5 |
| 100K docs | 32 | 5 | T4 | 2-4h | ~7.5 |
| 500K+ docs | 64 | 10 | T4 | 8-24h | ~6.5 |

*(valores aproximados; depende del contenido/limpieza de datos)*

---

## 🧠 Tokenizador BPE

Entrenado en Wikipedia español con 6000 tokens.

### Usar para nuevas secuencias:
```python
from Tokenizer.tokenizer import BPETokenizer

tok = BPETokenizer.load("Tokenizer/vocab/tokenizer.json")
ids = tok.encode("hola mundo")          # → [IDs]
text = tok.decode(ids)                  # → "hola mundo"
```

### Ver análisis de merges:
```bash
streamlit run Tokenizer/main.py
```

---

## 🐛 Troubleshooting

| Problema | Solución |
|----------|----------|
| Loss no baja | ↑ learning rate, más datos |
| Loss explota (NaN) | ↓ learning rate (0.0001), gradient clipping |
| Memory out | ↓ batch size |
| Entrenamiento lento | GPU en Colab, no CPU local |
| Val loss >> train loss | Overfitting: más regularización/datos |

---

## 🎓 Estructura de archivos

```
.
├── Config.py                    # hiperparámetros
├── main.py                      # demo básico
├── colab_training.py            # entrenamiento Colab
├── inference.py                 # generación de texto
│
├── transformer.py               # modelo principal
├── encoder_block.py             # bloque encoder
├── multi_head_attention.py      # atención multi-cabeza
├── feed_forward.py              # MLP
├── layer_norm.py                # normalización
├── token_embedding.py           # embeddings
├── positional_encoding.py       # PE sinusoidal
├── loss.py                      # Cross Entropy
├── trainer.py                   # utilidades train
│
├── Tokenizer/
│   ├── tokenizer.py             # BPE
│   ├── preprocess.py            # limpieza corpus
│   ├── main.py                  # UI Streamlit
│   └── vocab/
│       └── tokenizer.json       # vocab (6000 tokens)
│
├── checkpoints/                 # (se crea en entrenamiento)
├── training_log.csv             # (se crea en entrenamiento)
│
├── PROYECTO_TRANSFORMER.md      # documentación técnica
├── ENTRENAMIENTO_GUIDE.md       # guía de entrenamiento
└── README.md                    # este archivo
```

---

## 🚀 Próximos pasos

### Ya implementado ✅
- Token Embedding + Positional Encoding
- Multi-Head Attention (8 cabezas)
- Encoder Blocks apilados (4 capas)
- Layer Norm + Residual connections
- Cross Entropy Loss
- Training loop con mini-batches
- Save/Load de pesos
- Generación de texto (3 métodos)

### Para mejorar (después de entrenar)
- [ ] Cambiar a Adam optimizer
- [ ] Learning rate scheduler
- [ ] Gradient clipping
- [ ] Dropout para regularización
- [ ] Distributed training (PyTorch + TPU)
- [ ] Fine-tune en tareas downstream

---

## 📝 Para tu video

Los archivos `PROYECTO_TRANSFORMER.md` y `ENTRENAMIENTO_GUIDE.md` tienen todo explicado paso-a-paso:
- Motivación y decisiones de diseño
- Cómo fluye el tensor por cada capa
- Matemática de atención, layer norm, cross entropy
- Backward propagation
- Entrenamiento real en Colab

---

## 📄 Licencia

Libre para uso educativo y personal.

---

**¿Preguntas? Revisar `PROYECTO_TRANSFORMER.md` o `ENTRENAMIENTO_GUIDE.md`**
