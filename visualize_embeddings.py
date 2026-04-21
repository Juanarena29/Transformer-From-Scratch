"""
Visualiza embeddings de checkpoints del Transformer en 3D usando PCA + Plotly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.io import templates
from sklearn.decomposition import PCA

from Tokenizer.tokenizer import BPETokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera un HTML interactivo con la proyeccion 3D de "
            "los embeddings guardados en checkpoints .npz."
        )
    )
    parser.add_argument(
        "--checkpoints-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directorio donde buscar archivos .npz (default: checkpoints).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help=(
            "Ruta exacta al checkpoint .npz. "
            "Si no se especifica, se usa el mas reciente en checkpoints-dir."
        ),
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=Path("Tokenizer/vocab/tokenizer.json"),
        help="Ruta al tokenizer.json de BPETokenizer.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("embedding_visualization.html"),
        help="Archivo HTML de salida.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=500,
        help=(
            "Cantidad maxima de tokens a mostrar. "
            "Usa 0 o negativo para mostrar todos."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Semilla para PCA reproducible.",
    )
    return parser.parse_args()


def resolve_checkpoint(checkpoint: Path | None, checkpoints_dir: Path) -> Path:
    if checkpoint is not None:
        if not checkpoint.exists():
            raise FileNotFoundError(f"No existe checkpoint: {checkpoint}")
        return checkpoint

    if not checkpoints_dir.exists():
        raise FileNotFoundError(
            f"No existe el directorio de checkpoints: {checkpoints_dir}"
        )

    candidates = sorted(checkpoints_dir.rglob("*.npz"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(
            f"No se encontraron archivos .npz dentro de {checkpoints_dir}"
        )
    return candidates[-1]


def load_embeddings(checkpoint_path: Path) -> np.ndarray:
    with np.load(checkpoint_path, allow_pickle=False) as data:
        if "embedding.W" not in data.files:
            raise KeyError(
                f"El checkpoint {checkpoint_path} no contiene la clave 'embedding.W'. "
                f"Claves disponibles: {data.files}"
            )
        embeddings = data["embedding.W"]

    if embeddings.ndim != 2:
        raise ValueError(f"Se esperaba matriz 2D en 'embedding.W', recibido shape={embeddings.shape}")
    return embeddings


def load_tokenizer_tokens(tokenizer_path: Path) -> list[str]:
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"No existe tokenizer.json: {tokenizer_path}")

    tokenizer = BPETokenizer.load(str(tokenizer_path))
    max_id = max(tokenizer.inverse_vocab.keys())
    tokens = [tokenizer.inverse_vocab.get(i, f"<ID_{i}>") for i in range(max_id + 1)]
    return tokens


def select_top_tokens(embeddings: np.ndarray, tokens: list[str], top_k: int) -> tuple[np.ndarray, list[str]]:
    vocab_size = min(len(tokens), embeddings.shape[0])
    embeddings = embeddings[:vocab_size]
    tokens = tokens[:vocab_size]

    if top_k <= 0 or top_k >= vocab_size:
        return embeddings, tokens

    # Sin estadisticas de frecuencia en checkpoint/tokenizer, usamos norma L2
    # como proxy de "importancia" para evitar saturacion visual.
    norms = np.linalg.norm(embeddings, axis=1)
    top_indices = np.argsort(norms)[-top_k:]
    top_indices = np.sort(top_indices)
    return embeddings[top_indices], [tokens[i] for i in top_indices]


def build_figure(points_3d: np.ndarray, labels: list[str], title: str) -> go.Figure:
    distances = np.linalg.norm(points_3d, axis=1)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points_3d[:, 0],
                y=points_3d[:, 1],
                z=points_3d[:, 2],
                mode="markers",
                text=labels,
                hovertemplate="<b>%{text}</b><br>x=%{x:.3f}<br>y=%{y:.3f}<br>z=%{z:.3f}<extra></extra>",
                marker=dict(
                    size=4,
                    color=distances,
                    colorscale="Cividis",
                    opacity=0.85,
                    colorbar=dict(title="Dist. al origen"),
                ),
            )
        ]
    )

    fig.update_layout(
        template=templates["plotly_dark"],
        title=title,
        scene=dict(
            xaxis_title="PC1",
            yaxis_title="PC2",
            zaxis_title="PC3",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    return fig


def main() -> None:
    args = parse_args()

    checkpoint_path = resolve_checkpoint(args.checkpoint, args.checkpoints_dir)
    embeddings = load_embeddings(checkpoint_path)
    tokens = load_tokenizer_tokens(args.tokenizer_path)

    selected_embeddings, selected_tokens = select_top_tokens(embeddings, tokens, args.top_k)
    if selected_embeddings.shape[0] < 3:
        raise ValueError(
            f"Se necesitan al menos 3 tokens para PCA 3D; recibido {selected_embeddings.shape[0]}"
        )

    pca = PCA(n_components=3, random_state=args.seed)
    reduced = pca.fit_transform(selected_embeddings)

    explained = pca.explained_variance_ratio_ * 100.0
    title = (
        f"Embeddings 3D (PCA) - {checkpoint_path.name}<br>"
        f"Varianza explicada: PC1={explained[0]:.2f}%, "
        f"PC2={explained[1]:.2f}%, PC3={explained[2]:.2f}%"
    )

    fig = build_figure(reduced, selected_tokens, title)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(args.output), include_plotlyjs=True, full_html=True)

    print(f"Checkpoint usado: {checkpoint_path}")
    print(f"Tokens mostrados: {len(selected_tokens)}")
    print(f"HTML generado en: {args.output.resolve()}")


if __name__ == "__main__":
    main()
