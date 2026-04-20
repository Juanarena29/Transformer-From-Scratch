"""
inference.py
-------------
Generación de texto con el transformer entrenado.

Métodos soportados:
  - Greedy: selecciona el token más probable
  - Top-K: muestrea entre los K tokens más probables
  - Top-p (nucleus): muestrea entre tokens cuya prob acumulada < p
"""

import numpy as np
from transformer import Transformer
from Config import TransformerConfig
from Tokenizer.tokenizer import BPETokenizer


class TextGenerator:
    """Generador de texto usando el transformer entrenado."""

    def __init__(self, model_path: str, tokenizer_path: str, cfg: TransformerConfig) -> None:
        """
        Parámetros
        ----------
        model_path : ruta al archivo .npz con pesos guardados
        tokenizer_path : ruta al tokenizador
        cfg : configuración del modelo
        """
        self.cfg = cfg
        self.tokenizer = BPETokenizer.load(tokenizer_path)

        self.model = Transformer(
            vocab_size=cfg.vocab_size,
            d_model=cfg.d_model,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            d_ff=cfg.d_ff,
            max_seq_len=cfg.max_seq_len,
        )
        self.model.load_weights(model_path)

        self.pad_id = self.tokenizer.special_tokens["<PAD>"]
        self.bos_id = self.tokenizer.special_tokens["<BOS>"]
        self.eos_id = self.tokenizer.special_tokens["<EOS>"]
        self.unk_id = self.tokenizer.special_tokens["<UNK>"]

    def _softmax(self, logits: np.ndarray) -> np.ndarray:
        """Softmax estable."""
        shifted = logits - logits.max()
        exp_logits = np.exp(shifted)
        return exp_logits / exp_logits.sum()

    def _sample_greedy(self, logits: np.ndarray) -> int:
        """Selecciona el token con probabilidad máxima."""
        return int(np.argmax(logits))

    def _sample_topk(self, logits: np.ndarray, k: int = 10, temperature: float = 1.0) -> int:
        """Muestrea entre los K tokens más probables."""
        logits = logits / temperature
        probs = self._softmax(logits)

        top_k_idx = np.argsort(probs)[-k:]
        top_k_probs = probs[top_k_idx]
        top_k_probs /= top_k_probs.sum()

        return int(np.random.choice(top_k_idx, p=top_k_probs))

    def _sample_topp(self, logits: np.ndarray, p: float = 0.9, temperature: float = 1.0) -> int:
        """Nucleus sampling: muestrea hasta acumular probabilidad p."""
        logits = logits / temperature
        probs = self._softmax(logits)

        sorted_idx = np.argsort(probs)[::-1]
        sorted_probs = probs[sorted_idx]
        cumsum_probs = np.cumsum(sorted_probs)

        mask = cumsum_probs <= p
        mask[0] = True  # siempre incluir el más probable

        nucleus_idx = sorted_idx[mask]
        nucleus_probs = probs[nucleus_idx]
        nucleus_probs /= nucleus_probs.sum()

        return int(np.random.choice(nucleus_idx, p=nucleus_probs))

    def generate(
        self,
        prompt: str,
        max_tokens: int = 50,
        method: str = "greedy",
        temperature: float = 1.0,
        k: int = 10,
        p: float = 0.9,
    ) -> str:
        """
        Genera texto a partir de un prompt.

        Parámetros
        ----------
        prompt : texto de entrada
        max_tokens : cantidad de tokens a generar
        method : "greedy", "topk", o "topp"
        temperature : (para topk/topp) controla randomness
        k : (para topk) número de top tokens
        p : (para topp) umbral de probabilidad acumulada

        Retorna
        -------
        texto generado
        """
        # Tokenizar prompt
        prompt_ids = self.tokenizer.encode(prompt)
        if not prompt_ids:
            prompt_ids = [self.bos_id]
        else:
            prompt_ids = [self.bos_id] + prompt_ids

        # Generar token a token
        generated_ids = prompt_ids.copy()
        for _ in range(max_tokens):
            # Limitar a max_seq_len
            x = np.array(generated_ids[-self.cfg.max_seq_len :], dtype=int)
            x = x[np.newaxis, :]

            # Forward
            logits = self.model.forward(x)  # (1, seq, vocab_size)

            # Tomar logits del último token
            next_logits = logits[0, -1, :]

            # Samplear según método
            if method == "greedy":
                next_token = self._sample_greedy(next_logits)
            elif method == "topk":
                next_token = self._sample_topk(next_logits, k=k, temperature=temperature)
            elif method == "topp":
                next_token = self._sample_topp(next_logits, p=p, temperature=temperature)
            else:
                raise ValueError(f"método desconocido: {method}")

            generated_ids.append(next_token)

            # Stop si generamos EOS
            if next_token == self.eos_id:
                break

        # Decodificar
        output_ids = generated_ids[1:]  # quitar BOS
        text = self.tokenizer.decode(output_ids)
        return text


# ── Ejemplo de uso ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    cfg = TransformerConfig()

    # Cargar último checkpoint
    model_path = "checkpoints/epoch_1_final.npz"  # Cambiar según tu checkpoint
    tokenizer_path = "Tokenizer/vocab/tokenizer.json"

    generator = TextGenerator(model_path, tokenizer_path, cfg)

    # Ejemplos de generación
    prompts = [
        "el transformer",
        "la inteligencia artificial",
        "en español",
        "el aprendizaje",
    ]

    print("=" * 60)
    print("GENERACIÓN DE TEXTO CON TRANSFORMER ENTRENADO")
    print("=" * 60)

    for prompt in prompts:
        print(f"\n[PROMPT] {prompt}")
        print("-" * 60)

        # Greedy
        text_greedy = generator.generate(prompt, max_tokens=30, method="greedy")
        print(f"[GREEDY]  {text_greedy}")

        # Top-K
        text_topk = generator.generate(prompt, max_tokens=30, method="topk", k=5)
        print(f"[TOP-K]   {text_topk}")

        # Top-P
        text_topp = generator.generate(prompt, max_tokens=30, method="topp", p=0.9)
        print(f"[TOP-P]   {text_topp}")

    print("\n" + "=" * 60)
