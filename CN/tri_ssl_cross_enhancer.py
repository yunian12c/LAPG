"""SSL modality blocks used by PyramidTriSSLDualEnhancerFA."""

from __future__ import annotations

import torch
import torch.nn as nn


def _ssl_proj(in_dim: int, hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim),
        nn.LayerNorm(hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
    )


def _resolve_n_heads(hidden_dim: int, n_heads: int) -> int:
    nh = int(n_heads)
    while hidden_dim % nh != 0 and nh > 1:
        nh -= 1
    return nh


class _ModalitySelfAttnBlock(nn.Module):
    def __init__(self, hidden_dim: int, n_heads: int, dropout: float):
        super().__init__()
        nh = _resolve_n_heads(hidden_dim, n_heads)
        self.attn = nn.MultiheadAttention(hidden_dim, nh, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(tokens, tokens, tokens, need_weights=False)
        return self.norm(tokens + attn_out)


class TriSSLCrossEnhancerV66B(nn.Module):
    """v66b enhancer (init-only legacy path; replaced by PyramidTriSSLDualEnhancerFA in v84c)."""

    def __init__(
        self,
        feat_dims: tuple[int, ...],
        hidden_dim: int = 256,
        n_heads: int = 4,
        dropout: float = 0.1,
        mode: str = "self_attn2",
        n_self_attn_layers: int = 2,
    ):
        super().__init__()
        if len(feat_dims) not in (3, 4) or any(d <= 0 for d in feat_dims):
            raise ValueError(f"feat_dims must be 3 or 4 positive ints, got {feat_dims}")
        if mode not in ("self_attn2", "hubert_q"):
            raise ValueError(f"mode must be self_attn2 or hubert_q, got {mode}")
        self.feat_dims = feat_dims
        self.hidden_dim = int(hidden_dim)
        self.mode = str(mode)
        self.encoders = nn.ModuleList(_ssl_proj(d, hidden_dim, dropout) for d in feat_dims)
        nh = _resolve_n_heads(hidden_dim, n_heads)
        self.n_heads = nh
        if mode == "self_attn2":
            n_layers = max(int(n_self_attn_layers), 1)
            self.modality_blocks = nn.ModuleList(
                [_ModalitySelfAttnBlock(hidden_dim, nh, dropout) for _ in range(n_layers)]
            )
            self.hubert_cross_attn = None
            self.hubert_cross_norm = None
        else:
            self.modality_blocks = nn.ModuleList()
            self.hubert_cross_attn = nn.MultiheadAttention(hidden_dim, nh, dropout=dropout, batch_first=True)
            self.hubert_cross_norm = nn.LayerNorm(hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate_logit = nn.Linear(hidden_dim, 1, bias=False)

    def _encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        parts: list[torch.Tensor] = []
        off = 0
        for enc, d in zip(self.encoders, self.feat_dims):
            parts.append(enc(x[:, off : off + d]))
            off += d
        return torch.stack(parts, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self._encode_tokens(x)
        if self.mode == "self_attn2":
            for block in self.modality_blocks:
                tokens = block(tokens)
        else:
            h = tokens[:, 0]
            kv = tokens[:, 1:]
            cross_out, _ = self.hubert_cross_attn(h.unsqueeze(1), kv, kv, need_weights=False)
            h_enh = self.hubert_cross_norm(h + cross_out.squeeze(1))
            tokens = torch.cat([h_enh.unsqueeze(1), tokens[:, 1:]], dim=1)
        z = torch.tanh(self.gate_proj(tokens))
        beta = torch.softmax(self.gate_logit(z).squeeze(-1), dim=-1)
        return (beta.unsqueeze(-1) * tokens).sum(dim=1)
