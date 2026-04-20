# Mini Transformer From Scratch (NumPy + BPE)

## 1) Objetivo del proyecto

Construir un **Transformer encoder-only** desde cero usando NumPy, entendiendo cada componente matemático y de ingeniería sin depender de frameworks de alto nivel.

El foco del proyecto es:

- implementar toda la arquitectura base (embedding, PE, MHA, FFN, residuals, layer norm, head de salida);
- construir un **tokenizador BPE propio** para español con vocabulario de `6000` tokens;
- validar que el pipeline corre **end-to-end** con `forward`, `backward` y `update`;
- entrenar con **Cross Entropy Loss** real para language modeling;
- agregar persistencia de pesos (`save/load`) para continuar entrenamiento o hacer inferencia luego.

---

## 2) Estructura general del repo

### Núcleo del modelo

- `transformer.py`: ensambla el modelo completo.
- `loss.py`: implementación de Cross Entropy con gradiente.
- `trainer.py`: utilidades para construir batches LM y loop de entrenamiento.
- `encoder_block.py`: bloque repetible del encoder.
- `multi_head_attention.py`: atención multi-cabeza desde cero.
- `feed_forward.py`: MLP interno del bloque.
- `layer_norm.py`: normalización por token.
- `token_embedding.py`: lookup de embeddings.
- `positional_encoding.py`: codificación posicional sinusoidal.
- `Config.py`: hiperparámetros globales.
- `main.py`: demo integral: entrenamiento + guardado/carga de pesos.

### Tokenizador

- `Tokenizer/preprocess.py`: limpieza de corpus y frecuencias de palabras.
- `Tokenizer/tokenizer.py`: entrenamiento BPE + encode/decode.
- `Tokenizer/vocab/tokenizer.json`: artefacto entrenado del tokenizador.
- `Tokenizer/main.py`: app Streamlit para inspeccionar merges de BPE.

---

## 3) Configuración actual y decisiones clave

Configuración en `Config.py`:

- `vocab_size = 6000`
- `d_model = 128`
- `n_heads = 8`
- `n_layers = 4`
- `max_seq_len = 512`
- `d_ff = 512`
- `lr = 0.001`

Decisiones de diseño:

- **`d_model=128`**: tamaño manejable para experimentar rápido y depurar.
- **`n_heads=8`**: cada cabeza trabaja con `d_k=16` (`128/8`), equilibrio entre granularidad y costo.
- **`d_ff=512`**: expansión típica `4x d_model`.
- **`n_layers=4`**: profundidad suficiente para demostrar stacking sin volver lento el prototipo.
- **`max_seq_len=512`**: margen razonable para textos cortos/medios en pruebas.

---

## 4) Pipeline de datos y tokenización (BPE)

### 4.1 Preprocesamiento

`Tokenizer/preprocess.py`:

1. Normaliza texto (`lowercase` + NFC para caracteres acentuados).
2. Pretokeniza con regex para español (`[a-záéíóúüñ]+`).
3. Convierte cada palabra a símbolos con marcador de fin de palabra `</w>`.
4. Construye frecuencias de palabras en streaming (línea por línea).
5. Filtra ruido por frecuencia mínima.

### 4.2 Entrenamiento BPE

`Tokenizer/tokenizer.py`:

- Arranca con tokens especiales:
  - `<PAD>=0`, `<UNK>=1`, `<BOS>=2`, `<EOS>=3`
- Inicializa vocabulario base de símbolos.
- Aprende merges hasta llegar a `vocab_size=6000`.
- Usa una estrategia **incremental optimizada** con `pair_counts` y `pair_locations` (evita recomputar todo en cada iteración).

### 4.3 Inferencia de tokenización

Para `encode(text)`:

1. limpieza + pretokenización;
2. aplicación de merges en orden de prioridad aprendida (`merges_index`);
3. mapeo símbolo -> id (`<UNK>` para fuera de vocabulario).

Para `decode(ids)`:

1. id -> símbolo;
2. remoción de tokens especiales;
3. reemplazo de `</w>` por espacio.

**Motivo de BPE**: reducir OOV y representar morfología del español (prefijos/sufijos) mejor que tokenización por palabra completa.

---

## 5) Arquitectura del Transformer implementado

Flujo de alto nivel:

`texto -> BPE -> token_ids -> Embedding -> Positional Encoding -> Encoder x N -> LM Head -> logits`

### 5.1 Token Embedding

`token_embedding.py`:

- Matriz `W` de forma `(vocab_size, d_model)`.
- `forward`: lookup por índices.
- `backward`: acumulación de gradientes por índice con `np.add.at`.
- `update`: SGD.

### 5.2 Positional Encoding sinusoidal

`positional_encoding.py`:

- implementación clásica de Vaswani et al. (sin/cos);
- matriz fija precomputada de tamaño `(max_seq_len, d_model)`;
- sin parámetros aprendibles;
- en backward, el gradiente pasa directo.

### 5.3 Encoder Block (repetido 4 veces)

`encoder_block.py`:

1. **Multi-Head Attention**
2. **Residual + LayerNorm**
3. **FeedForward**
4. **Residual + LayerNorm**

Mismo shape de entrada/salida: `(batch, seq, d_model)`.

### 5.4 Multi-Head Attention

`multi_head_attention.py`:

- proyecciones lineales `Q, K, V` con pesos `(d_model, d_model)`;
- split/merge de cabezas;
- `scaled dot-product attention` con factor `1/sqrt(d_k)`;
- softmax estable numéricamente;
- proyección de salida `W_O`;
- backward completo implementado (incluyendo gradiente de softmax).

### 5.5 Feed Forward

`feed_forward.py`:

- dos capas lineales con ReLU intermedia:
  - `128 -> 512 -> 128`
- inicialización tipo He para ramas con ReLU;
- backward y actualización SGD implementados.

### 5.6 LayerNorm

`layer_norm.py`:

- normaliza por token sobre el eje de `d_model`;
- parámetros aprendibles `gamma` y `beta`;
- backward analítico completo.

### 5.7 LM Head

`transformer.py`:

- proyección final `d_model -> vocab_size`;
- produce `logits` de forma `(batch, seq, vocab_size)`;
- inicialización Xavier para estabilidad.

---

## 6) Flujo de entrenamiento actual (con Cross Entropy)

En `main.py` se valida el circuito completo:

1. carga del tokenizador entrenado;
2. construcción de batch para objetivo autoregresivo (next-token):
   - `x = [BOS, t1, t2, ...]`
   - `y = [t1, t2, ..., EOS]`
3. `forward` del Transformer;
4. cálculo de `loss` + `dlogits` con Cross Entropy;
5. `backward` y `update` por SGD;
6. repetición por épocas (`train_loop`) con registro de `loss` y `perplexity`;
7. verificación de que la loss disminuye.

Resultado: el sistema ahora funciona **de punta a punta también para entrenar**.

---

## 7) Cross Entropy Loss implementada

Archivo: `loss.py`

Características:

- entrada `logits: (B, T, V)` y `targets: (B, T)`;
- softmax estable numéricamente (resta del máximo por fila);
- `ignore_index` para ignorar `<PAD>` al promediar la pérdida;
- retorna:
  - `loss` escalar promedio sobre tokens válidos;
  - `dlogits` con shape `(B, T, V)` para backprop.

Detalle clave:

- cuando hay padding, esos tokens no contribuyen ni a la `loss` ni al gradiente.

---

## 8) Training loop implementado

Archivo: `trainer.py`

Módulos principales:

1. `build_lm_batch(...)`
   - agrega `BOS` y `EOS`;
   - construye pares de entrenamiento `x/y` desplazados en 1 token;
   - hace padding al largo máximo del batch.
2. `train_step(...)`
   - `forward -> loss -> backward -> update`.
3. `train_loop(...)`
   - ejecuta múltiples épocas;
   - devuelve historial de `loss` y `perplexity`;
   - imprime progreso por época.

---

## 9) Guardar / cargar pesos implementado

Archivo: `transformer.py`

Se agregaron:

- `state_dict()`: serializa todos los pesos aprendibles en un dict plano.
- `load_state_dict(state)`: valida claves/shapes y restaura pesos.
- `save_weights(path)`: guarda en `.npz`.
- `load_weights(path)`: carga desde `.npz`.

En `main.py` se prueba la consistencia:

1. entrenar modelo;
2. guardar checkpoint;
3. crear un modelo nuevo y cargar pesos;
4. verificar que ambos producen logits idénticos para el mismo input.

---

## 10) Guion sugerido para tu video (paso a paso)

1. **Contexto y objetivo**: por qué construir un Transformer desde cero.
2. **Tokenizador BPE**: problema de vocabulario, entrenamiento de merges, ejemplo real.
3. **Configuración**: justificar `6000`, `128`, `8`, `4`, `512`.
4. **Arquitectura**: recorrido del tensor desde texto hasta logits.
5. **Bloque encoder**: atención + residual + norm + FFN.
6. **Backward**: cómo fluye el gradiente en cada submódulo.
7. **Cross Entropy**: cómo se obtiene `dlogits` real desde targets.
8. **Training loop**: batching LM, épocas y métrica de perplexity.
9. **Persistencia de pesos**: guardar/cargar y reproducibilidad.
10. **Próximos pasos**: dataset más grande, batching por mini-lotes, scheduler/optimizador.

---

## 11) Resumen técnico

Proyecto actual:

- tokenizador BPE entrenado para español (`vocab=6000`);
- Transformer encoder implementado completo en NumPy;
- Cross Entropy Loss integrada con soporte de `ignore_index` para `<PAD>`;
- training loop implementado para objetivo next-token prediction;
- guardado y carga de pesos funcional con validación de consistencia.

Pendientes implementados en `colab_training.py`:

- ✅ dataloader por mini-batches reales;
- ✅ separación train/val + métricas de generalización;
- ✅ descarga automática de Wikipedia español;
- ✅ checkpointing periódico;
- ✅ logging a CSV + plotting de métricas.

---

## 12) Cómo entrenar en Google Colab (paso a paso)

### Preparación (5 minutos)
1. Abre [colab.research.google.com](https://colab.research.google.com)
2. Click en "New notebook"
3. Renombra a `Transformer_Training`

### Configurar GPU
En la primera celda, asegúrate de tener GPU:
```python
import torch
print("GPU disponible:", torch.cuda.is_available())
```

### Subir archivos
Opción A (recomendado): desde GitHub
```python
!git clone https://github.com/TU_USUARIO/TU_REPO.git transformer
%cd transformer
```

Opción B: desde Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
%cd /content/drive/My\ Drive/Transformer
```

### Ejecutar entrenamiento
```python
%run colab_training.py
```

### Monitorear
- Los gráficos se mostrarán automáticamente
- Log guardado en `training_log.csv`
- Checkpoints en carpeta `checkpoints/`

---

## 13) Generación de texto con modelo entrenado

Archivo: `inference.py`

Métodos soportados:
- **Greedy**: token más probable (determinístico)
- **Top-K**: muestrea entre K tokens más probables (más variedad)
- **Top-P (Nucleus)**: muestrea entre tokens que sumen p% de probabilidad (mejor calidad)

Ejemplo:
```python
from inference import TextGenerator
from Config import TransformerConfig

cfg = TransformerConfig()
generator = TextGenerator(
    model_path="checkpoints/epoch_3_final.npz",
    tokenizer_path="Tokenizer/vocab/tokenizer.json",
    cfg=cfg
)

text = generator.generate(
    prompt="el transformador",
    max_tokens=50,
    method="topp",
    p=0.9
)
print(text)
```

---

## 14) Estructura final del proyecto

```
Transformer/
├── PROYECTO_TRANSFORMER.md          # este documento
├── ENTRENAMIENTO_GUIDE.md           # guía detallada de entrenamiento
├── Config.py                        # hiperparámetros
├── main.py                          # demo básico end-to-end
├── colab_training.py                # script listo para Colab
├── inference.py                     # generación de texto
├── transformer.py                   # modelo principal
├── encoder_block.py                 # bloque encoder
├── multi_head_attention.py          # atención multi-cabeza
├── feed_forward.py                  # MLP
├── layer_norm.py                    # normalización
├── token_embedding.py               # embeddings
├── positional_encoding.py           # PE sinusoidal
├── loss.py                          # Cross Entropy
├── trainer.py                       # utilidades de entrenamiento
├── Tokenizer/
│   ├── preprocess.py                # limpieza de corpus
│   ├── tokenizer.py                 # BPE implementation
│   ├── main.py                      # app Streamlit para explorar
│   └── vocab/
│       └── tokenizer.json           # vocab entrenado (6000 tokens)
├── checkpoints/
│   └── (se crean durante entrenamiento)
└── README.md                        # instrucciones generales
```

---

## 15) Próximos pasos después de entrenar

### Corto plazo (inmediato)
1. Ejecutar `colab_training.py` en Colab con dataset pequeño
2. Monitorear loss y perplexity
3. Guardar checkpoints

### Mediano plazo (1-2 semanas)
1. Escalar a dataset más grande (100K+ docs)
2. Implementar Adam optimizer (vs SGD actual)
3. Agregar learning rate scheduler
4. Hacer inferencia con modelo entrenado

### Largo plazo (futuro)
1. Agregar dropout para regularización
2. Implementar gradient clipping
3. Pasar a PyTorch para distribuido training
4. Fine-tune en tareas específicas (clasificación, Q&A, etc.)

