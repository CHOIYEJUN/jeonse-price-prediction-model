"""학습이 끝난 동 임베딩을 2차원으로 줄여 그림으로 저장한다.

8차원 벡터는 그대로 그리기 어려워서, t-SNE(또는 PCA)로 2칸으로 축소한다.
비슷한 전세 패턴의 동은 가까이 모이는 경향이 있다.
"""

import os
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch

from embedding_mlp_model import JeonseEmbeddingMLP

# __file__: 이 스크립트 경로. 상위 두 칸이 프로젝트 루트다.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(PROJECT_ROOT, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "embedding_mlp_model.pt")
MAPPING_PATH = os.path.join(MODEL_DIR, "embedding_category_mapping.pkl")
OUTPUT_PATH = os.path.join(MODEL_DIR, "dong_embedding_visualization.png")


def _korean_font() -> None:
    """그래프에 한글 동 이름이 깨지지 않도록 글꼴을 고른다.

    설치된 글꼴이 없으면 다음 후보로 넘어간다.
    """
    # 마이너스 기호가 네모로 깨지는 것을 막는다
    plt.rcParams["axes.unicode_minus"] = False
    for name in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
        try:
            plt.rcParams["font.family"] = name
            return
        except Exception:
            continue


def reduce_2d(vectors: np.ndarray) -> np.ndarray:
    """고차원 임베딩을 2열짜리 좌표로 줄인다.

    vectors: (동 개수, embedding_dim) 배열
    반환: (동 개수가, 2) — 가로·세로 좌표

    동이 2개 이하면 t-SNE를 쓸 수 없어 첫 번째 축만 사용한다.
    t-SNE가 실패하면 PCA(주성분 분석)로 대체한다.
    """
    n = len(vectors)  # 점(동)의 개수
    if n < 3:
        pad = np.zeros((n, 2), dtype=float)
        pad[:, 0] = vectors[:, 0] if vectors.shape[1] else 0
        return pad
    try:
        from sklearn.manifold import TSNE

        # perplexity: 한 점이 몇 개 이웃을 볼지. 점 수보다 작아야 한다.
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
    """저장해 둔 MLP에서 동 임베딩을 꺼내 산점도를 그린다."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(MAPPING_PATH):
        print("먼저 training/train_embedding_mlp.py 를 실행하세요.")
        sys.exit(1)

    # mapping: 동 이름 ↔ 인덱스, 숫자 피처 목록, 스케일러 등이 들어 있는 사전
    mapping = joblib.load(MAPPING_PATH)
    # ckpt: 학습된 가중치(state_dict)와 모형 크기를 담은 사전
    try:
        ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    except TypeError:
        # 예전 PyTorch는 weights_only 인자가 없다
        ckpt = torch.load(MODEL_PATH, map_location="cpu")
    model = JeonseEmbeddingMLP(
        n_apt=ckpt["n_apt"],
        n_dong=ckpt["n_dong"],
        n_numeric=ckpt["n_numeric"],
        embedding_dim=ckpt["embedding_dim"],
        dropout=ckpt.get("dropout", 0.25),
    )
    model.load_state_dict(ckpt["state_dict"])
    model.eval()  # Dropout 등을 끄는 평가 모드

    dong_to_idx = mapping["dong_to_idx"]  # {"정자동": 3, ...}
    idx_to_dong = {idx: name for name, idx in dong_to_idx.items()}  # 반대 방향 사전
    # dong_emb.weight: (n_dong, embedding_dim) 학습된 좌표표
    weights = model.dong_emb.weight.detach().cpu().numpy()

    indices = sorted(idx_to_dong.keys())  # 그릴 동 인덱스 목록
    labels = [idx_to_dong[i] for i in indices]  # 같은 순서의 동 이름
    vectors = weights[indices]  # 해당 행만 뽑은 임베딩
    xy = reduce_2d(vectors)  # 2차원 좌표

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
    # 이 파일을 직접 실행했을 때만 main()을 부른다. import 하면 실행되지 않는다.
    main()
