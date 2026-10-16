"""전세 거래만으로 다음 보증금의 로그를 예측하는 두 신경망.

LagEmbeddingMLP
    직전 보증금·직전 3건 평균처럼 사람이 자른 자기회귀 피처를 임베딩과 함께 넣는다.
JeonseHistoryLSTM
    최근 거래를 가변 길이 시퀀스로 넣고, LSTM이 순서를 요약한다.
둘 다 출력은 log(이번 보증금) 하나다. 매매가는 받지 않는다.
"""

import torch
from torch import nn


class LagEmbeddingMLP(nn.Module):
    """단지·동 임베딩과 자기회귀 숫자 피처로 로그 보증금을 낸다.

    nn.Embedding
        단지명·동 같은 범주를 정수 인덱스에서 실수 벡터로 바꾼다.
    nn.BatchNorm1d
        배치 안 은닉값의 평균과 분산을 맞춰 학습을 안정시킨다.
    Dropout
        학습 때 은닉 유닛 일부를 꺼서 특정 단지 암기에 덜 의존하게 한다.
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
        """레이어만 만든다. 학습은 호출 쪽에서 한다.

        n_apt, n_dong: 인덱스 0(기타)을 포함한 범주 개수
        n_numeric: 표준화된 숫자 피처 칸 수
        embedding_dim: 단지·동 벡터의 칸 수
        """
        super().__init__()
        self.apt_emb = nn.Embedding(n_apt, embedding_dim)
        self.dong_emb = nn.Embedding(n_dong, embedding_dim)
        in_dim = embedding_dim * 2 + n_numeric
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.BatchNorm1d(hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, 1),
        )

    def forward(self, apt_idx, dong_idx, numeric):
        """한 배치의 로그 보증금을 낸다.

        apt_idx, dong_idx: (배치,) 정수
        numeric: (배치, n_numeric) 실수. 학습 구간의 평균·표준편차로 이미 맞춰져 있다.
        반환: (배치,) 로그 보증금
        """
        x = torch.cat(
            [self.apt_emb(apt_idx), self.dong_emb(dong_idx), numeric],
            dim=1,
        )
        return self.net(x).squeeze(-1)


class JeonseHistoryLSTM(nn.Module):
    """과거 전세 거래 시퀀스를 LSTM으로 요약해 로그 보증금을 낸다.

    시퀀스가 단지마다 길이가 다르므로 짧은 쪽은 0으로 패딩한다.
    pack_padded_sequence는 그 0이 은닉 상태에 더해지지 않게
    실제 길이까지만 LSTM 칸을 진행시킨다.

    한 층 LSTM이라 층 사이 Dropout은 걸리지 않는다.
    Dropout은 은닉 상태·임베딩·정적 피처를 붙인 뒤의 머리에서 건다.
    """

    def __init__(
        self,
        n_apt: int,
        n_dong: int,
        n_seq: int,
        n_static: int,
        embedding_dim: int = 8,
        hidden_size: int = 64,
        dropout: float = 0.25,
    ):
        """레이어만 만든다.

        n_seq: 거래 한 스텝의 피처 수 (로그 보증금, 면적, 층, 다음 거래까지 일수)
        n_static: 이번 계약에서 아는 숫자 (면적, 층, 연식)
        hidden_size: LSTM 은닉 상태 칸 수. 마지막 시점의 상태가 시퀀스 요약이다.
        """
        super().__init__()
        self.apt_emb = nn.Embedding(n_apt, embedding_dim)
        self.dong_emb = nn.Embedding(n_dong, embedding_dim)
        self.lstm = nn.LSTM(
            input_size=n_seq,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        fused = hidden_size + embedding_dim * 2 + n_static
        self.head = nn.Sequential(
            nn.Linear(fused, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, apt_idx, dong_idx, static, seq, lengths):
        """패딩을 무시하고 마지막 실제 거래까지 읽은 뒤 로그 보증금을 낸다.

        seq: (배치, 최대길이, n_seq). 오른쪽이 패딩이다.
        lengths: (배치,) 실제 과거 거래 수. 1 이상이어야 한다.
        static: (배치, n_static) 이번 거래의 면적·층·연식
        반환: (배치,) 로그 보증금
        """
        # pack은 길이 텐서가 CPU에 있어야 한다.
        packed = nn.utils.rnn.pack_padded_sequence(
            seq,
            lengths.detach().cpu().long(),
            batch_first=True,
            enforce_sorted=False,
        )
        _output, (h_n, _c_n) = self.lstm(packed)
        # h_n: (층 수, 배치, hidden). 마지막 층의, 실제 마지막 시점 상태.
        history = h_n[-1]
        x = torch.cat(
            [history, self.apt_emb(apt_idx), self.dong_emb(dong_idx), static],
            dim=1,
        )
        return self.head(x).squeeze(-1)
