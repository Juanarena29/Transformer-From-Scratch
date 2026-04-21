"""quick_start.py
Quick environment and integration verification script.
Architecture position: executed before training to validate runtime
dependencies, tokenizer loading, and model forward/backward/update paths.
"""

import os
import sys
import numpy as np

print("\n" + "="*70)
print("TRANSFORMER: QUICK START VERIFICATION")
print("="*70 + "\n")

# -- 1. Verificar Python y NumPy -------------------------------------------
print("[1/6] Verificando Python y NumPy...")
print(f"      Python: {sys.version}")
print(f"      NumPy:  {np.__version__}")
assert np.__version__.split(".")[0] in ["1", "2"], "NumPy 1.x o 2.x requerido"
print("      [OK]\n")

# -- 2. Verificar ambiente (Colab vs Local) --------------------------------
print("[2/6] Detectando ambiente...")
try:
    import google.colab
    print("      Ejecutando en: GOOGLE COLAB [OK]")
    IN_COLAB = True
except:
    print("      Ejecutando en: LOCAL (maquina)")
    IN_COLAB = False
print()

# -- 3. Verificar GPU disponible -------------------------------------------
print("[3/6] Comprobando GPU...")
try:
    import torch
    if torch.cuda.is_available():
        print(f"      GPU: {torch.cuda.get_device_name(0)} [OK]")
        print(f"      CUDA: {torch.version.cuda}")
    else:
        print("      [WARN] GPU NO disponible. Colab > Entorno > GPU T4/A100")
except ImportError:
    print("      (PyTorch no instalado, no critico para NumPy training)")
print()

# -- 4. Verificar modulos del transformer ----------------------------------
print("[4/6] Importando modulos del transformer...")
try:
    from Config import TransformerConfig
    print("      [OK] Config")
    from transformer import Transformer
    print("      [OK] Transformer")
    from loss import CrossEntropyLoss
    print("      [OK] Loss")
    from trainer import build_lm_batch, train_loop
    print("      [OK] Trainer")
    from Tokenizer.tokenizer import BPETokenizer
    print("      [OK] BPETokenizer")
except ImportError as e:
    print(f"      [ERROR] {e}")
    print("      > Asegurate de estar en la carpeta raiz del proyecto")
    sys.exit(1)
print()

# -- 5. Verificar tokenizador entrenado ------------------------------------
print("[5/6] Cargando tokenizador entrenado...")
try:
    tokenizer = BPETokenizer.load("Tokenizer/vocab/tokenizer.json")
    print(f"      Vocab size: {len(tokenizer.vocab)} tokens [OK]")
    
    test_encode = tokenizer.encode("hola mundo")
    test_decode = tokenizer.decode(test_encode)
    print(f"      Encode/decode: '{test_decode}' [OK]")
except Exception as e:
    print(f"      [ERROR] {e}")
    sys.exit(1)
print()

# -- 6. Verificar modelo forward/backward ----------------------------------
print("[6/6] Probando modelo forward/backward...")
try:
    cfg = TransformerConfig()
    model = Transformer(
        vocab_size=cfg.vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_layers=cfg.n_layers,
        d_ff=cfg.d_ff,
        max_seq_len=cfg.max_seq_len,
    )
    
    x_batch = np.array([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=int)
    logits = model.forward(x_batch)
    assert logits.shape == (2, 4, cfg.vocab_size)
    print(f"      Forward: [OK] logits shape {logits.shape}")
    
    y_batch = np.array([[2, 3, 4, 5], [6, 7, 8, 9]], dtype=int)
    loss_fn = CrossEntropyLoss(ignore_index=0)
    loss, dlogits = loss_fn.forward(logits, y_batch)
    print(f"      Loss: [OK] loss = {loss:.4f}")
    
    model.backward(dlogits)
    print(f"      Backward: [OK]")
    
    model.update(cfg.lr)
    print(f"      Update: [OK]")
    
except Exception as e:
    print(f"      [ERROR] {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("="*70)
print("[OK] VERIFICACION COMPLETADA - LISTO PARA ENTRENAR")
print("="*70)
print()
print("Proximos pasos:")
print("  1. Ajusta hiperparametros en colab_training.py (DATASET_SIZE, EPOCHS, etc.)")
print("  2. Ejecuta: %run colab_training.py")
print("  3. Monitorea loss, perplexity y validacion")
print("  4. Checkpoints guardados en 'checkpoints/' automaticamente")
print()
print("Para generar texto:")
print("  %run inference.py")
print()
print("Documentacion: lee ENTRENAMIENTO_GUIDE.md")
print("="*70 + "\n")
