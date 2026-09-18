"""단지명·동을 임베딩으로 넣고 전세가율을 예측하는 작은 신경망.

nn.Module
    PyTorch에서 모형(레이어 묶음)을 만들 때 상속하는 기본 클래스.
nn.Embedding
    정수 인덱스(단지 번호 등)를 실수 벡터로 바꾸는 표.
    예: 동 인덱스 3 → [0.12, -0.05, ...] 처럼 embedding_dim칸.
nn.Linear
    y = xW + b 형태의 완전연결층. 입력 칸 수 × 출력 칸 수를 곱한다.
"""

import torch
from torch import nn


class JeonseEmbeddingMLP(nn.Module):
    """전세가율(0~1 근처)을 출력하는 다층퍼셉트론.

    입력은 세 갈래다.
    - apt_idx: 단지 이름 → 정수 인덱스
    - dong_idx: 법정동 → 정수 인덱스
    - numeric: 면적·매매가 등 숫자 피처 (이미 표준화된 값)
    """

    def __init__(
        self,
        n_apt: int,
        n_dong: int,
        n_numeric: int,
        embedding_dim: int = 8,
        dropout: float = 0.25,
        hidden1: int = 128,
        hidden2: int = 64,
    ):
        """레이어를 만들어 둔다. 아직 학습은 하지 않는다.

        n_apt: 단지 종류의 개수 (인덱스 0=기타 포함)
        n_dong: 동의 종류 개수
        n_numeric: 숫자 피처 개수
        embedding_dim: 단지·동을 몇 칸 실수 벡터로 표현할지
        dropout: 학습 때 일부를 무작위로 끄는 비율. 과적합을 줄인다.
        hidden1, hidden2: 은닉층 뉴런 수
        """
        super().__init__()
        # 단지 인덱스 → embedding_dim차원 벡터
        self.apt_emb = nn.Embedding(n_apt, embedding_dim)
        # 동 인덱스 → embedding_dim차원 벡터
        self.dong_emb = nn.Embedding(n_dong, embedding_dim)
        # 두 임베딩을 이어 붙인 뒤 숫자 피처를 옆에 붙인다
        in_dim = embedding_dim * 2 + n_numeric
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.BatchNorm1d(hidden1),  # 번치 안 값의 평균·분산을 맞춰 학습을 안정화
            nn.ReLU(),  # 음수는 0, 양수는 그대로. 비선형을 넣는다
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),  # 전세가율 하나
        )

    def forward(self, apt_idx, dong_idx, numeric):
        """한 배치를 앞에서 뒤로 통과시켜 전세가율 예측을 낸다.

        apt_idx: (배치 크기,) 정수 텐서
        dong_idx: (배치 크기,) 정수 텐서
        numeric: (배치 크기, n_numeric) 실수 텐서
        반환: (배치 크기,) 예측 전세가율
        """
        # dim=1: 행은 그대로 두고, 열 방향으로 이어 붙인다
        x = torch.cat(
            [self.apt_emb(apt_idx), self.dong_emb(dong_idx), numeric],
            dim=1,
        )
        # squeeze(-1): (n, 1) → (n,) 마지막 크기 1인 축을 없앤다
        return self.net(x).squeeze(-1)
