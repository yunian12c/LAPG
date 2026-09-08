"""APA frame cross-attention readout (fixed seg bias, v102b defaults)."""

from __future__ import annotations

import torch
import torch.nn as nn

SEG_INITIAL = 1
SEG_FINAL = 2


class FrameSegmentCrossAttnReadout(nn.Module):
    def __init__(self, hidden_dim: int, seg_bias: float = 2.0, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.seg_bias = float(seg_bias)
        nh = num_heads
        while hidden_dim % nh != 0 and nh > 1:
            nh -= 1
        self.num_heads = nh
        self.head_dim = hidden_dim // nh
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.graph_char_fuse = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )

    def _seg_bias_from_labels(self, seg: torch.Tensor, target: int, n: int, device: torch.device) -> torch.Tensor:
        mag = torch.tensor(self.seg_bias, device=device, dtype=torch.float32)
        if seg.numel() == 0:
            return torch.full((n,), -1.0, device=device, dtype=torch.float32) * mag
        in_seg = seg == target
        sign = torch.where(
            in_seg,
            torch.ones_like(seg, dtype=torch.float32),
            -torch.ones_like(seg, dtype=torch.float32),
        )
        return sign * mag

    def _cross_attn_one(
        self,
        rows: torch.Tensor,
        query: torch.Tensor,
        seg: torch.Tensor,
        target_seg: int,
    ) -> torch.Tensor:
        if rows.size(0) == 0:
            return query.new_zeros(query.shape)
        n = rows.size(0)
        hd = rows.size(-1)
        nh = self.num_heads
        dh = self.head_dim
        q = self.q_proj(query).view(nh, dh)
        k = self.k_proj(rows).view(n, nh, dh)
        v = self.v_proj(rows).view(n, nh, dh)
        bias = self._seg_bias_from_labels(seg, target_seg, n, rows.device)
        logits = torch.einsum("hd,nhd->hn", q, k) * self.scale + bias.unsqueeze(0)
        attn = torch.softmax(logits, dim=-1)
        attn = self.dropout(attn)
        ctx = torch.einsum("hn,nhd->hd", attn, v)
        out = ctx.reshape(hd)
        return self.out_proj(out) + query
