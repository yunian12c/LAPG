"""Pyramid SSL enhancer with FA-mask node↔frame bridges (v102b trunk)."""

from __future__ import annotations

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from phoneme_ctc_align import CLS_FINAL, CLS_INITIAL
from tri_ssl_cross_enhancer import _ModalitySelfAttnBlock, _resolve_n_heads, _ssl_proj


def _gated_pool_tokens(tokens: torch.Tensor, gate_proj: nn.Linear, gate_logit: nn.Linear) -> torch.Tensor:
    z = torch.tanh(gate_proj(tokens))
    beta = torch.softmax(gate_logit(z).squeeze(-1), dim=-1)
    return (beta.unsqueeze(-1) * tokens).sum(dim=1)


def _segment_node_compat(frame_seg: torch.Tensor, node_type: torch.Tensor) -> torch.Tensor:
    fs = frame_seg.long().view(-1, 1)
    nt = node_type.long().view(1, -1)
    compat = torch.zeros(fs.size(0), nt.size(1), device=fs.device, dtype=torch.float32)
    # Chinese: ini=1 / fin=2 / tone=3 / global=4  (seg CLS_INITIAL/FINAL)
    compat = compat + ((fs == CLS_INITIAL) & (nt == 1)).float() * 1.0
    compat = compat + ((fs == CLS_FINAL) & (nt == 2)).float() * 1.0
    compat = compat + (nt == 4).float() * 0.5
    compat = compat + (nt == 3).float() * 0.25
    # Speechocean: phone=1 / word=2 / utt=3 with seg phone=1 / word=2
    compat = compat + ((fs == 1) & (nt == 1)).float() * 1.0
    compat = compat + ((fs == 2) & (nt == 2)).float() * 1.0
    return compat


class CrossLayerSyncBridgeFA(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.1, seg_bias: float = 2.0):
        super().__init__()
        hd = int(hidden_dim)
        nh = _resolve_n_heads(hd, num_heads)
        self.seg_bias = float(seg_bias)
        self.frame_attn = nn.MultiheadAttention(hd, nh, dropout=dropout, batch_first=True)
        self.frame_norm = nn.LayerNorm(hd)
        self.node_attn = nn.MultiheadAttention(hd, nh, dropout=dropout, batch_first=True)
        self.node_norm = nn.LayerNorm(hd)
        self.frame_gate = nn.Sequential(nn.Linear(hd, hd), nn.GELU(), nn.Linear(hd, 1))
        self.node_gate = nn.Sequential(nn.Linear(hd, hd), nn.GELU(), nn.Linear(hd, 1))

    def forward(
        self,
        h_node: torch.Tensor,
        h_frame: torch.Tensor,
        fa_mask: torch.Tensor | None,
        node_type: torch.Tensor | None = None,
        frame_seg: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if h_node.size(0) == 0 or h_frame.size(0) == 0:
            return h_node, h_frame

        attn_fn = None
        if node_type is not None and frame_seg is not None and frame_seg.numel() > 0:
            compat_fn = _segment_node_compat(frame_seg.reshape(-1), node_type.reshape(-1))
            attn_fn = compat_fn * self.seg_bias

        h_frame_tuned, _ = self.frame_attn(h_frame, h_node, h_node, attn_mask=attn_fn, need_weights=False)
        g_f = torch.sigmoid(self.frame_gate(h_frame_tuned))
        h_frame_out = self.frame_norm(h_frame + g_f * h_frame_tuned)

        attn_bias = None
        if fa_mask is not None and fa_mask.numel() > 0:
            attn_bias = (~fa_mask.bool()).to(dtype=h_node.dtype) * (-10000.0)
            if attn_bias.dim() == 3:
                attn_bias = attn_bias.repeat_interleave(self.node_attn.num_heads, dim=0)

        h_node_tuned, _ = self.node_attn(h_node, h_frame, h_frame, attn_mask=attn_bias, need_weights=False)
        g_n = torch.sigmoid(self.node_gate(h_node_tuned))
        h_node_out = self.node_norm(h_node + g_n * h_node_tuned)
        return h_node_out, h_frame_out


def build_fa_mask_for_graph(
    pool_mask_ini: torch.Tensor,
    pool_mask_fin: torch.Tensor,
    pool_mask_tone: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    t_len = int(pool_mask_ini.numel())
    m_glob = torch.ones(t_len, device=device, dtype=dtype)
    return torch.stack(
        [
            pool_mask_ini.to(device=device, dtype=dtype),
            pool_mask_fin.to(device=device, dtype=dtype),
            pool_mask_tone.to(device=device, dtype=dtype),
            m_glob,
        ],
        dim=0,
    )


def _fa_mask_for_graph(
    batch: Batch,
    gid: int,
    frame_off: int,
    t_len: int,
    n_nodes: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    # Speechocean: per-graph [N_nodes, T] masks attached as a Python list
    node_pool_masks = getattr(batch, "node_pool_masks", None)
    if node_pool_masks is not None:
        m = node_pool_masks[gid]
        if not torch.is_tensor(m):
            m = torch.as_tensor(m)
        m = m.to(device=device, dtype=dtype)
        if m.dim() == 2 and m.size(0) == n_nodes:
            if m.size(1) != t_len:
                # nearest resize along time
                idx = torch.linspace(0, m.size(1) - 1, steps=t_len, device=m.device).round().long()
                m = m[:, idx]
            return m
    sl = slice(frame_off, frame_off + t_len)
    return build_fa_mask_for_graph(
        batch.pool_mask_ini[sl],
        batch.pool_mask_fin[sl],
        batch.pool_mask_tone[sl],
        device,
        dtype,
    )


def _apply_fa_bridge_per_graph(
    bridge: CrossLayerSyncBridgeFA,
    node_h: torch.Tensor,
    frame_h: torch.Tensor,
    node_graph: torch.Tensor,
    frame_graph: torch.Tensor,
    batch: Batch,
    node_type: torch.Tensor,
    frame_seg: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    out_node = node_h.clone()
    out_frame = frame_h.clone()
    frame_off = 0
    for gid in range(int(batch.num_graphs)):
        t_len = int(batch.n_time_frames[gid].item())
        nm = node_graph == gid
        fm = frame_graph == gid
        if bool(nm.any()) and bool(fm.any()):
            n_nodes = int(nm.sum().item())
            fa_mask = _fa_mask_for_graph(
                batch, gid, frame_off, t_len, n_nodes, node_h.device, node_h.dtype
            ).unsqueeze(0)
            nh, fh = bridge(
                node_h[nm].unsqueeze(0),
                frame_h[fm].unsqueeze(0),
                fa_mask,
                node_type=node_type[nm],
                frame_seg=frame_seg[fm],
            )
            out_node[nm] = nh.squeeze(0)
            out_frame[fm] = fh.squeeze(0)
        frame_off += t_len
    return out_node, out_frame


class PyramidTriSSLDualEnhancerFA(nn.Module):
    def __init__(
        self,
        feat_dims: tuple[int, ...],
        hidden_dim: int = 256,
        n_heads: int = 4,
        dropout: float = 0.1,
        n_ssl_layers: int = 4,
        sync_layers: tuple[int, ...] = (2, 4),
        sync_attn_heads: int = 4,
        seg_sync_bias: float = 2.0,
    ):
        super().__init__()
        if len(feat_dims) not in (1, 2, 3, 4) or any(d <= 0 for d in feat_dims):
            raise ValueError(f"feat_dims must be 1–4 positive ints, got {feat_dims}")
        self.feat_dims = feat_dims
        self.n_modalities = len(feat_dims)
        self.hidden_dim = int(hidden_dim)
        self.n_ssl_layers = max(int(n_ssl_layers), 1)
        sync_set = {int(x) for x in sync_layers if 1 <= int(x) <= self.n_ssl_layers}
        self.sync_layers = tuple(sorted(sync_set))
        self.encoders = nn.ModuleList(_ssl_proj(d, hidden_dim, dropout) for d in feat_dims)
        nh = _resolve_n_heads(hidden_dim, n_heads)
        self.node_layers = nn.ModuleList(
            [_ModalitySelfAttnBlock(hidden_dim, nh, dropout) for _ in range(self.n_ssl_layers)]
        )
        self.frame_layers = nn.ModuleList(
            [_ModalitySelfAttnBlock(hidden_dim, nh, dropout) for _ in range(self.n_ssl_layers)]
        )
        self.bridges = nn.ModuleDict(
            {
                str(li): CrossLayerSyncBridgeFA(hidden_dim, sync_attn_heads, dropout, seg_bias=seg_sync_bias)
                for li in self.sync_layers
            }
        )
        self.h_to_tokens = nn.Linear(hidden_dim, hidden_dim * self.n_modalities, bias=False)
        self.inject_gate = nn.Linear(hidden_dim, hidden_dim)
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate_logit = nn.Linear(hidden_dim, 1, bias=False)

    def _split_modalities(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        parts: list[torch.Tensor] = []
        off = 0
        for enc, d in zip(self.encoders, self.feat_dims):
            parts.append(enc(x[:, off : off + d]))
            off += d
        return tuple(parts)

    def _to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return torch.stack(self._split_modalities(x), dim=1)

    def _inject_hidden(self, tokens: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        delta = self.h_to_tokens(h).view(h.size(0), self.n_modalities, self.hidden_dim)
        g = torch.sigmoid(self.inject_gate(h)).unsqueeze(1)
        return tokens + g * delta

    @staticmethod
    def _batch_frame_graph_ids(batch: Batch) -> torch.Tensor:
        ids: list[int] = []
        for gid in range(int(batch.num_graphs)):
            t_len = int(batch.n_time_frames[gid].item())
            ids.extend([gid] * t_len)
        return torch.tensor(ids, dtype=torch.long, device=batch.frame_feat.device)

    def forward(
        self,
        node_raw: torch.Tensor,
        frame_raw: torch.Tensor,
        batch: Batch,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        node_tokens = self._to_tokens(node_raw)
        frame_tokens = self._to_tokens(frame_raw)
        node_graph = batch.batch
        frame_graph = self._batch_frame_graph_ids(batch)
        node_type = batch.node_type
        frame_seg = batch.seg_label_frames.long()

        for layer_idx in range(1, self.n_ssl_layers + 1):
            node_tokens = self.node_layers[layer_idx - 1](node_tokens)
            frame_tokens = self.frame_layers[layer_idx - 1](frame_tokens)
            node_h = _gated_pool_tokens(node_tokens, self.gate_proj, self.gate_logit)
            frame_h = _gated_pool_tokens(frame_tokens, self.gate_proj, self.gate_logit)
            if layer_idx in self.sync_layers:
                bridge = self.bridges[str(layer_idx)]
                node_h, frame_h = _apply_fa_bridge_per_graph(
                    bridge, node_h, frame_h, node_graph, frame_graph, batch, node_type, frame_seg
                )
                node_tokens = self._inject_hidden(node_tokens, node_h)
                frame_tokens = self._inject_hidden(frame_tokens, frame_h)
        return node_h, frame_h
