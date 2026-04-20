# Roadmap: Mini Transformer From Scratch

## Estado Actual (26 de Abril 2026)

### ✅ Completado

#### Arquitectura
- [x] Token Embedding (vocab_size=6000, d_model=128)
- [x] Positional Encoding sinusoidal
- [x] Multi-Head Attention (8 cabezas, d_k=16)
- [x] Feed Forward Network (expansión 4x)
- [x] Layer Normalization con parámetros aprendibles
- [x] 4 Encoder Blocks apilados con residual connections
- [x] LM Head (proyección a vocab_size)

#### Training
- [x] Cross Entropy Loss (con ignore_index para padding)
- [x] Forward pass completo (tokens -> logits)
- [x] Backward pass completo (gradientes)
- [x] SGD update
- [x] Mini-batching real
- [x] Train/Val split
- [x] Checkpointing periódico
- [x] Logging a CSV
- [x] Plotting de métricas

#### Tokenizador
- [x] BPE desde cero (6000 tokens español)
- [x] Encode/decode funcionando
- [x] Soporte de tokens especiales (<PAD>, <BOS>, <EOS>, <UNK>)
- [x] UI Streamlit para inspeccionar

#### Persistencia
- [x] state_dict() / load_state_dict()
- [x] save_weights() / load_weights()
- [x] Validación de consistencia (logits idénticos tras recargar)

#### Inferencia
- [x] Generación greedy (más probable)
- [x] Generación Top-K (K más probables)
- [x] Generación Top-P / Nucleus (masa de probabilidad)

#### Documentación
- [x] README.md (quick start)
- [x] PROYECTO_TRANSFORMER.md (arquitectura técnica)
- [x] ENTRENAMIENTO_GUIDE.md (paso-a-paso datasets/entrenamiento)
- [x] COLAB_INSTRUCTIONS.md (Colab específico)
- [x] Este roadmap

#### Utilidades
- [x] Config.py (hiperparámetros centralizados)
- [x] quick_start.py (verificación de setup)
- [x] colab_training.py (script ready-to-run para Colab)
- [x] inference.py (generación de texto)
- [x] requirements.txt (dependencias)

---

## Verificado Funcionando

✅ **End-to-end pipeline**
```
main.py: 20 épocas completas, loss baja de 8.74 -> 8.62
```

✅ **Save/Load**
```
Pesos guardados y recargados: logits idénticos
```

✅ **Quick Start Verification**
```
[OK] VERIFICACION COMPLETADA - LISTO PARA ENTRENAR
```

---

## Lo que falta / Roadmap futuro

### Fase 2: Optimizaciones de entrenamiento (1-2 semanas)

- [ ] **Adam optimizer** (reemplazar SGD)
  - Mejor convergencia
  - Momentum + adaptive learning rates
  
- [ ] **Learning Rate Scheduler**
  - Warm-up inicial
  - Decay a medida que avanza
  - Ejemplo: cosine annealing
  
- [ ] **Gradient clipping**
  - Evitar explosión de gradientes (importante en modelos profundos)
  
- [ ] **Dropout regularización**
  - Reducir overfitting
  - Aplicar en atención + FFN

### Fase 3: Infraestructura de datos (1 semana)

- [ ] **DataLoader más robusto**
  - Actualmente: mini-batching manual
  - Mejorar: shuffling, prefetching, multiprocessing
  
- [ ] **Validación dinámica**
  - Actualmente: eval cada 100 batches en subset
  - Mejorar: eval completo en val set cada época
  
- [ ] **Early stopping**
  - Parar si val_loss no mejora en N épocas
  
- [ ] **Diferentes datasets**
  - Wikipedia (grande)
  - Common Crawl / OSCAR (muy grande)
  - Custom corpus (más control)

### Fase 4: Escalabilidad (2-4 semanas)

- [ ] **Aumentar modelo**
  - d_model=256, n_layers=8, n_heads=16
  - d_ff=2048
  - Evaluar que siga entrenable en T4
  
- [ ] **Distributed training** (si necesitas)
  - PyTorch + DataParallel (simple)
  - O migrara PyTorch completamente
  
- [ ] **Mixed precision training**
  - FP32 (actual) -> FP16 para más velocidad/memoria
  - Importante si escala significativamente

### Fase 5: Evaluación y análisis (1-2 semanas)

- [ ] **Métricas más ricas**
  - Perplexity (ya implementada)
  - Token accuracy (% de predicciones correctas)
  - BPE-token accuracy vs char-level accuracy
  
- [ ] **Análisis de atención**
  - Visualizar attention heads
  - Entender qué atiende a qué
  
- [ ] **Benchmarks**
  - Comparar con LLMs de referencia
  - Fine-tune en downstream tasks
  
- [ ] **Error analysis**
  - Qué tipos de tokens predice mal
  - Dónde falla la generación

### Fase 6: Fine-tuning / Downstream tasks (2-4 semanas)

- [ ] **Clasificación de texto**
  - Agregar head de clasificación
  - Fine-tune pre-trained transformer
  
- [ ] **Generación mejorada**
  - Beam search (vs greedy)
  - Length penalties
  - Repeat penalties
  
- [ ] **Question Answering**
  - Span extraction
  - Reader comprehension
  
- [ ] **Named Entity Recognition (NER)**
  - Token-level classification
  - Español específico

### Fase 7: Producción (1-2 semanas)

- [ ] **API REST**
  - Servir modelo con FastAPI/Flask
  - Endpoints: encode, decode, generate, inference
  
- [ ] **Containerización**
  - Docker para reproducibilidad
  - Desplegar en GPU cloud (AWS, GCP)
  
- [ ] **Benchmarking latency**
  - Cuánto tarda generar N tokens
  - Optimizaciones de velocidad
  
- [ ] **Documentación de deployment**
  - Cómo usar en producción
  - Cómo escalar

---

## Decisiones de diseño (justificación)

### ¿Por qué NumPy?
- **Educativo**: entender cada operación matemática
- **Transparencia**: no hay "magia" en frameworks
- **Control total**: gradiantes, forward, backward bajo tu mando

### ¿Por qué no PyTorch directamente?
- PyTorch es genial pero "esconde" detalles
- Objetivo: **entender desde cero**
- Buena preparación antes de usar frameworks

### ¿Por qué BPE?
- Reduce OOV (out-of-vocabulary)
- Representa subpalabras (morphology aware)
- Estándar en LLMs modernos

### ¿Por qué 6000 tokens?
- Balance: suficiente variedad + entrenable en T4
- Alternativas: 10K (más preciso), 1K (más eficiente)

### ¿Por qué d_model=128?
- Pequeño: prototipado rápido (15min en T4)
- Suficiente para demostrar arquitectura
- Si escala: aumentar a 256-512

### ¿Por qué 4 capas?
- Menos: poco expresivo
- Más: overfitting con datasets pequeños
- Trade-off en prototipado

---

## Métricas esperadas (benchmarks)

### Tiempo de entrenamiento

| Config | Dataset | Épocas | Batch | Device | Tiempo | Loss final |
|--------|---------|--------|-------|--------|--------|-----------|
| Small | 10K docs | 3 | 16 | T4 GPU | 15 min | 8.5 |
| Medium | 100K docs | 5 | 32 | T4 GPU | 2-3h | 7.5 |
| Large | 500K+ docs | 10 | 64 | T4 GPU | 8-12h | 6.5 |

### Parámetros del modelo

| Componente | Parámetros |
|-----------|-----------|
| Token Embedding | 768K |
| 4x Encoder Block | 789K |
| LM Head | 768K |
| **TOTAL** | **2.3M** |

### Requerimientos de memoria

- **Pesos**: ~9 MB (float32)
- **Activations (batch=32)**: ~150 MB
- **Gradientes**: ~9 MB
- **Optimizer state (si Adam)**: ~18 MB
- **Total**: ~200 MB (cómodo en T4 15GB)

---

## Próximas instrucciones

1. **Hoy**: Leer README.md, ENTRENAMIENTO_GUIDE.md
2. **Mañana**: Entrenar en Colab con dataset pequeño
3. **Semana 1**: Escalar a dataset mediano, ver resultados
4. **Semana 2-3**: Optimizaciones (Adam, scheduler, etc.)
5. **Semana 4+**: Fases 4, 5, 6, 7 según motivación

---

## Archivos clave por fase

### Fase 1 (Actual - Arquitectura)
- `transformer.py`: modelo
- `loss.py`: cross entropy
- `trainer.py`: training loop
- `main.py`: demo básico

### Fase 2 (Optimizaciones)
- `optimizer.py` (nuevo): Adam, SGD
- `scheduler.py` (nuevo): learning rate schedules
- `trainer.py`: modificar train_step para gradient clipping

### Fase 3 (Datos)
- `data_loader.py` (nuevo): DataLoader más robusto
- `datasets.py` (nuevo): soporte para múltiples datasets
- `colab_training.py`: mejorar

### Fase 4 (Escalabilidad)
- Aumentar Config.py
- Profiling de memoria/velocidad
- `distributed.py` (si necesario)

### Fase 5 (Evaluación)
- `evaluate.py` (nuevo): métricas detalladas
- `analyze_attention.py` (nuevo): visualizar atención
- Jupyter notebooks para experimentos

### Fase 6 (Fine-tuning)
- `downstream_tasks.py` (nuevo)
- Heads específicas por tarea

### Fase 7 (Producción)
- `api.py` (nuevo): FastAPI
- `requirements_prod.txt`
- Docker + deployment configs

---

## Video script (basado en PROYECTO_TRANSFORMER.md)

1. Intro (2 min)
2. Tokenizador BPE (3 min)
3. Arquitectura Transformer (5 min)
4. Forward pass (4 min)
5. Backward pass (3 min)
6. Entrenamiento real (2 min)
7. Resultados y demo (2 min)
8. Próximos pasos (1 min)

Total: ~22 minutos

---

## Conclusión

**Hoy terminaste:**
- ✅ Transformer funcional end-to-end
- ✅ Training loop con Cross Entropy real
- ✅ Save/load de pesos
- ✅ Generación de texto
- ✅ Script listo para Colab
- ✅ Documentación completa

**Siguiente:** Entrenar en Colab y ver loss bajar en datos reales.

¡A entrenar! 🚀
