"""Edge-aware TransformerConv GNN block (v102b stem/MDD/APA backbone)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torch_geometric.nn import TransformerConv
except Exception:
    TransformerConv = None  # type: ignore


def _pick_heads(dim: int, heads: int = 4) -> tuple[int, int]:
    nh = heads
    while dim % nh != 0 and nh > 1:
        nh -= 1
    return nh, dim // nh


class EdgeTransformerBlock(nn.Module):
    def __init__(self, dim: int, num_edge_types: int, heads: int = 4):
        super().__init__()
        if TransformerConv is None:
            raise SystemExit("TransformerConv unavailable")
        nh, out_ch = _pick_heads(dim, heads)
        self.edge_emb = nn.Embedding(int(num_edge_types), out_ch)
        self.conv = TransformerConv(
            in_channels=dim,
            out_channels=out_ch,
            heads=nh,
            concat=True,
            dropout=0.1,
            edge_dim=out_ch,
        )
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(0.15), nn.Linear(dim * 2, dim))
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor | None = None,
        mismatch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del mismatch  # plain blocks ignore mismatch (APA / stem)
        edge_attr = None if edge_type is None else self.edge_emb(edge_type.reshape(-1).long())
        h = F.gelu(self.conv(x, edge_index, edge_attr=edge_attr))
        x = self.norm(x + F.dropout(h, p=0.15, training=self.training))
        h2 = self.ffn(x)
        return self.ffn_norm(x + h2)


class MismatchEdgeTransformerBlock(nn.Module):
    """EdgeTransformer whose edge_attr is typed emb + acoustic–canonical mismatch pair.

    Used on the MDD branch so within-word / sequential messages carry mismatch context
    (canonical↔acoustic disagreement), without changing the APA path.
    """

    def __init__(self, dim: int, num_edge_types: int, mismatch_dim: int, heads: int = 4):
        super().__init__()
        if TransformerConv is None:
            raise SystemExit("TransformerConv unavailable")
        nh, out_ch = _pick_heads(dim, heads)
        self.edge_emb = nn.Embedding(int(num_edge_types), out_ch)
        self.mm_edge = nn.Sequential(
            nn.Linear(2 * int(mismatch_dim), out_ch),
            nn.GELU(),
            nn.Linear(out_ch, out_ch),
        )
        # Start near 0 so training begins ≈ type-only edges, then learns mismatch gating.
        self.mm_scale = nn.Parameter(torch.tensor(0.0))
        self.conv = TransformerConv(
            in_channels=dim,
            out_channels=out_ch,
            heads=nh,
            concat=True,
            dropout=0.1,
            edge_dim=out_ch,
        )
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(0.15), nn.Linear(dim * 2, dim))
        self.ffn_norm = nn.LayerNorm(dim)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor | None = None,
        mismatch: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if edge_type is None:
            edge_attr = None
        else:
            edge_attr = self.edge_emb(edge_type.reshape(-1).long())
            if mismatch is not None and edge_index.numel() > 0:
                src, dst = edge_index[0], edge_index[1]
                mm = self.mm_edge(torch.cat([mismatch[src], mismatch[dst]], dim=-1))
                edge_attr = edge_attr + torch.tanh(self.mm_scale) * mm
        h = F.gelu(self.conv(x, edge_index, edge_attr=edge_attr))
        x = self.norm(x + F.dropout(h, p=0.15, training=self.training))
        h2 = self.ffn(x)
        return self.ffn_norm(x + h2)
