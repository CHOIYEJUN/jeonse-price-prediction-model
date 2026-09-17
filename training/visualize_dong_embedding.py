"""학습된 dong 임베딩을 2차원으로 축소해 시각화한다."""

import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch

from embedding_mlp_model import JeonseEmbeddingMLP

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "embedding_mlp_model.pt")
MAPPING_PATH = os.path.join(MODEL_DIR, "embedding_category_mapping.pkl")
OUTPUT_PATH = os.path.join(MODEL_DIR, "dong_embedding_visualization.png")


def _korean_font() -> None:
    plt.rcParams["axes.unicode_minus"] = False
    for name in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
        try:
            plt.rcParams["font.family"] = name
            return
        except Exception:
            continue


def reduce_2d(vectors: np.ndarray) -> np.ndarray:
    n = len(vectors)
    if n < 3:
        pad = np.zeros((n, 2), dtype=float)
        pad[:, 0] = vectors[:, 0] if vectors.shape[1] else 0
        return pad
    try:
        from sklearn.manifold import TSNE

        perplexity = max(2, min(30, n - 1))
        return TSNE(
            n_components=2,
            random_state=42,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
        ).fit_transform(vectors)
    except Exception as exc:
        print(f"t-SNE 실패, PCA로 대체: {exc}")
        from sklearn.decomposition import PCA

        return PCA(n_components=2, random_state=42).fit_transform(vectors)


def main() -> None:
    if not os.path.exists(MODEL_PATH) or not os.path.exists(MAPPING_PATH):
        print("먼저 training/train_embedding_mlp.py 를 실행하세요.")
        sys.exit(1)

    mapping = joblib.load(MAPPING_PATH)
    try:
        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    except TypeError:
        ckpt = torch.load(MODEL_PATH, map_location="cpu")
    model = JeonseEmbeddingMLP(
        n_apt=ckpt["n_apt"],
        n_dong=ckpt["n_dong"],
        n_numeric=ckpt["n_numeric"],
        embedding_dim=ckpt["embedding_dim"],
        dropout=ckpt.get("dropout", 0.25),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    dong_to_idx = mapping["dong_to_idx"]
    idx_to_dong = {idx: name for name, idx in dong_to_idx.items()}
    weights = model.dong_emb.weight.detach().cpu().numpy()

    indices = sorted(idx_to_dong.keys())
    labels = [idx_to_dong[i] for i in indices]
    vectors = weights[indices]
    xy = reduce_2d(vectors)

    _korean_font()
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(xy[:, 0], xy[:, 1], s=40, alpha=0.85)
    for (x, y), label in zip(xy, labels):
        ax.annotate(label, (x, y), fontsize=8, ha="left", va="bottom")
    ax.set_title("Dong entity embeddings (2D)")
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    fig.tight_layout()
    os.makedirs(MODEL_DIR, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150)
    plt.close(fig)
    print(f"저장: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
