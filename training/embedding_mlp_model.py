"""Jeonse ratio MLP with entity embeddings for apartmentName and dong."""

import torch
from torch import nn


class JeonseEmbeddingMLP(nn.Module):
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
        x = torch.cat(
            [self.apt_emb(apt_idx), self.dong_emb(dong_idx), numeric],
            dim=1,
        )
        return self.net(x).squeeze(-1)
