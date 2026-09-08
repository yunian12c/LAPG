"""APA char head for v102b."""

from __future__ import annotations

import torch
import torch.nn as nn


class HierarchicalApaCharHead(nn.Module):
    """Fuse ini/fin/tone/global (+ interaction) for whole-character APA regression."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        fuse_in = hidden_dim * 6
        mid = hidden_dim * 2
        self.net = nn.Sequential(
            nn.Linear(fuse_in, mid),
            nn.LayerNorm(mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mid, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        h_ini: torch.Tensor,
        h_fin: torch.Tensor,
        h_tone: torch.Tensor,
        h_graph: torch.Tensor,
    ) -> torch.Tensor:
        diff = torch.abs(h_ini - h_fin)
        prod = h_ini * h_fin
        x = torch.cat([h_ini, h_fin, h_tone, h_graph, diff, prod], dim=-1)
        return self.net(x).squeeze(-1)
