"""GOP phoneme graph model v102b (v56→v66b→v68→v84→v92→v98 merged)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch

from cross_layer_sync import PyramidTriSSLDualEnhancerFA
from phoneme_graph_edges import NUM_PHONEME_EDGE_TYPES
from tri_ssl_cross_enhancer import TriSSLCrossEnhancerV66B

from apa_heads import HierarchicalApaCharHead
from apa_segment_readout import SEG_FINAL, SEG_INITIAL, FrameSegmentCrossAttnReadout
from typed_gnn_blocks import EdgeTransformerBlock

PROSODY_F0_ENERGY_DIM = 8

SLOT_INI = 0
SLOT_FIN = 1
SLOT_TONE = 2
SLOT_GLOBAL = 3


def _pick_transformer_nhead(hidden_dim: int) -> int:
    for nh in (8, 4, 2, 1):
        if hidden_dim % nh == 0:
            return nh
    return 1


def make_phoneme_gnn_block(dim: int, backbone: str) -> nn.Module:
    if str(backbone) != "edge_transformer":
        raise ValueError(f"v102b only supports edge_transformer, got {backbone!r}")
    return EdgeTransformerBlock(dim, num_edge_types=NUM_PHONEME_EDGE_TYPES)


class SegmentTriGatedFusion(nn.Module):
    """Legacy v56 fusion (instantiated then discarded for init-order parity with inheritance chain)."""

    def __init__(self, feat_dims: tuple[int, ...], hidden_dim: int):
        super().__init__()
        self.feat_dims = feat_dims
        self.projs = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(d, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.1))
                for d in feat_dims
            ]
        )
        self.gate_proj = nn.Linear(hidden_dim, hidden_dim)
        self.gate_logit = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        parts = []
        off = 0
        for m, d in enumerate(self.feat_dims):
            parts.append(self.projs[m](x[:, off : off + d]))
            off += d
        h = torch.stack(parts, dim=1)
        z = torch.tanh(self.gate_proj(h))
        beta = torch.softmax(self.gate_logit(z).squeeze(-1), dim=-1)
        return (beta.unsqueeze(-1) * h).sum(dim=1)


def _pool_one_graph_fin_sidepath(
    cross_attn: FrameSegmentCrossAttnReadout,
    rows: torch.Tensor,
    rows_fin: torch.Tensor,
    seg: torch.Tensor,
    sem: torch.Tensor,
    *,
    want_init: bool,
    want_final: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = rows.device
    dtype = rows.dtype
    hd = rows.size(-1) if rows.numel() else sem.size(-1)
    z = torch.zeros(hd, device=device, dtype=dtype)

    q_ini = sem[0] if sem.size(0) > 0 else z
    q_fin = sem[1] if sem.size(0) > 1 else z
    h_ini = cross_attn._cross_attn_one(rows, q_ini, seg, SEG_INITIAL) if want_init else z
    h_fin = cross_attn._cross_attn_one(rows_fin, q_fin, seg, SEG_FINAL) if want_final else z
    h_tone = sem[2] if sem.size(0) > 2 else z
    row_mean = rows.mean(dim=0) if rows.size(0) > 0 else z
    h_graph = sem[3] + row_mean if sem.size(0) > 3 else z
    if rows.size(0) > 0:
        h_graph = cross_attn.graph_char_fuse(torch.cat([h_graph.unsqueeze(0), row_mean.unsqueeze(0)], dim=-1)).squeeze(0)
    return h_ini, h_fin, h_tone, h_graph


class TaskDecoder(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, in_dim), nn.GELU(), nn.Dropout(0.15), nn.Linear(in_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ProsodyFiLM8(nn.Module):
    """8-dim prosody (f0||energy) -> gamma, beta; h' = (1+gamma)*h + beta (identity init)."""

    def __init__(self, prosody_dim: int = PROSODY_F0_ENERGY_DIM, hidden_dim: int = 256):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(prosody_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 2 * hidden_dim),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, h: torch.Tensor, prosody: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.mlp(prosody).chunk(2, dim=-1)
        return (1.0 + gamma) * h + beta


class QwenSslCrossAttnFusion(nn.Module):
    """SSL hidden as Q, Qwen frames as K/V; residual gated cross-attention (identity-biased init)."""

    def __init__(self, qwen_dim: int, hidden_dim: int, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.qwen_proj = nn.Sequential(
            nn.Linear(int(qwen_dim), hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        nh = _pick_transformer_nhead(hidden_dim) if hidden_dim % int(n_heads) != 0 else int(n_heads)
        self.cross_attn = nn.MultiheadAttention(hidden_dim, nh, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        self.gate = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.constant_(self.gate[-1].bias, -2.0)

    def forward(self, h_ssl: torch.Tensor, qwen: torch.Tensor) -> torch.Tensor:
        if h_ssl.size(0) == 0 or qwen.size(0) == 0:
            return h_ssl
        kv = self.qwen_proj(qwen).unsqueeze(0)
        q = h_ssl.unsqueeze(0)
        attn_out, _ = self.cross_attn(q, kv, kv, need_weights=False)
        delta = attn_out.squeeze(0)
        g = torch.sigmoid(self.gate(delta))
        return self.norm(h_ssl + g * delta)


class SegmentAttentionPoolFeat(nn.Module):
    """Masked attention pool on concatenated SSL frame features [T, feat_dim]."""

    def __init__(self, feat_dim: int, num_slots: int = 4, dropout: float = 0.1):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.num_slots = int(num_slots)
        self.queries = nn.Parameter(torch.randn(num_slots, self.feat_dim) * 0.02)
        self.k_proj = nn.Linear(self.feat_dim, self.feat_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(self.feat_dim)
        self.scale = self.feat_dim**-0.5

    def pool(self, feat_seq: torch.Tensor, mask: torch.Tensor | None, slot: int) -> torch.Tensor:
        fd = feat_seq.size(-1)
        device = feat_seq.device
        dtype = feat_seq.dtype
        if feat_seq.size(0) == 0:
            return torch.zeros(fd, device=device, dtype=dtype)

        if mask is not None and mask.numel() == feat_seq.size(0) and bool(mask.any()):
            h = feat_seq[mask]
        else:
            h = feat_seq

        if h.size(0) == 1:
            return h.squeeze(0)

        q = self.queries[int(slot)]
        k = self.k_proj(h)
        logits = (k @ q) * self.scale
        attn = F.softmax(logits, dim=0)
        attn = self.dropout(attn)
        out = (attn.unsqueeze(-1) * h).sum(dim=0)
        return self.out_norm(out)


class PhonemeGraphModelV102b(nn.Module):
    """v102b snapshot: SSL base → Prosody FiLM → Qwen FiLM (nodes: SSL→Qwen; frames: SSL→Prosody→Qwen)."""

    def __init__(
        self,
        feat_dims: tuple[int, ...],
        vocab_size: int,
        qwen_dim: int | None = None,
        qwen_fusion: str = "concat",
        hidden_dim: int = 256,
        stem_layers: int = 4,
        stem_backbone: str = "edge_transformer",
        mdd_layers: int = 4,
        mdd_backbone: str = "edge_transformer",
        apa_layers: int = 4,
        apa_backbone: str = "edge_transformer",
        use_phoneme_triplet_graph: bool = True,
        phoneme_slot_causal: bool = True,
        apa_seg_bias: float = 2.0,
        prosody_dim: int = PROSODY_F0_ENERGY_DIM,
        ssl_attn_heads: int = 4,
        ssl_enhancer_mode: str = "self_attn2",
        ssl_self_attn_layers: int = 2,
        n_ssl_layers: int = 4,
        sync_layers: tuple[int, ...] = (2, 4),
        sync_attn_heads: int = 4,
        seg_sync_bias: float = 2.0,
        graph_pool_dropout: float = 0.1,
        contra_dim: int | None = None,
    ):
        super().__init__()
        self.feat_dims = feat_dims
        self.hidden_dim = hidden_dim
        self.qwen_fusion = str(qwen_fusion)
        self.qwen_dim = int(qwen_dim) if qwen_dim else 0
        self.phoneme_slot_causal = bool(phoneme_slot_causal)

        # v56 segment_fusion → v66b delete: preserve torch RNG draw order.
        _legacy_segment_fusion = SegmentTriGatedFusion(feat_dims, hidden_dim)
        del _legacy_segment_fusion

        token_dim = min(64, max(16, hidden_dim // 4))
        type_dim = 16
        self.ref_token_emb = nn.Embedding(vocab_size, token_dim)
        self.type_emb = nn.Embedding(5, type_dim)
        self.ref_proj = nn.Sequential(nn.Linear(token_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.meta_proj = nn.Sequential(nn.Linear(type_dim + 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU())
        self.in_proj = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
        )

        if use_phoneme_triplet_graph:
            nh = _pick_transformer_nhead(hidden_dim)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=nh,
                dim_feedforward=max(hidden_dim * 2, 128),
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            self.phoneme_slot_encoder = nn.TransformerEncoder(enc_layer, num_layers=1)
            self.phoneme_slot_logit = nn.Parameter(torch.tensor(-0.307))
        else:
            self.phoneme_slot_encoder = None
            self.phoneme_slot_logit = None

        self.stem_blocks = nn.ModuleList(
            [make_phoneme_gnn_block(hidden_dim, stem_backbone) for _ in range(max(stem_layers, 1))]
        )
        self.mdd_blocks = nn.ModuleList(
            [make_phoneme_gnn_block(hidden_dim, mdd_backbone) for _ in range(max(mdd_layers, 1))]
        )
        self.apa_blocks = nn.ModuleList(
            [make_phoneme_gnn_block(hidden_dim, apa_backbone) for _ in range(max(apa_layers, 1))]
        )

        self.asr_initial = TaskDecoder(hidden_dim, vocab_size)
        self.asr_final = TaskDecoder(hidden_dim, vocab_size)
        self.asr_tone = TaskDecoder(hidden_dim, vocab_size)
        self.mdd_initial = TaskDecoder(hidden_dim, 3)
        self.mdd_final = TaskDecoder(hidden_dim, 3)
        self.mdd_tone = TaskDecoder(hidden_dim, 3)
        self.mdd_char = TaskDecoder(hidden_dim, 3)
        self.apa_initial = TaskDecoder(hidden_dim, 1)
        self.apa_final = TaskDecoder(hidden_dim, 1)
        self.apa_char_hier = HierarchicalApaCharHead(hidden_dim)

        # v66b ssl_enhancer → v84c replace: preserve torch RNG draw order.
        _legacy_ssl_enhancer = TriSSLCrossEnhancerV66B(
            feat_dims,
            hidden_dim,
            n_heads=int(ssl_attn_heads),
            mode=str(ssl_enhancer_mode),
            n_self_attn_layers=int(ssl_self_attn_layers),
        )
        del _legacy_ssl_enhancer

        self.frame_prosody_film = ProsodyFiLM8(prosody_dim=prosody_dim, hidden_dim=hidden_dim)
        self.frame_prosody_film_fin = ProsodyFiLM8(prosody_dim=prosody_dim, hidden_dim=hidden_dim)
        self.apa_cross_attn = FrameSegmentCrossAttnReadout(hidden_dim, seg_bias=apa_seg_bias)

        use_qwen_side = self.qwen_dim > 0 and self.qwen_fusion in ("film", "xattn")
        if use_qwen_side:
            self.qwen_attn_pool = SegmentAttentionPoolFeat(self.qwen_dim, dropout=graph_pool_dropout)
        else:
            self.qwen_attn_pool = None

        if self.qwen_fusion == "film" and self.qwen_dim > 0:
            self.node_qwen_film = ProsodyFiLM8(prosody_dim=self.qwen_dim, hidden_dim=hidden_dim)
            self.frame_qwen_film = ProsodyFiLM8(prosody_dim=self.qwen_dim, hidden_dim=hidden_dim)
            self.frame_qwen_film_fin = ProsodyFiLM8(prosody_dim=self.qwen_dim, hidden_dim=hidden_dim)
            self.node_qwen_xattn = None
            self.frame_qwen_xattn = None
            self.frame_qwen_xattn_fin = None
        elif self.qwen_fusion == "xattn" and self.qwen_dim > 0:
            self.node_qwen_film = None
            self.frame_qwen_film = None
            self.frame_qwen_film_fin = None
            self.node_qwen_xattn = QwenSslCrossAttnFusion(self.qwen_dim, hidden_dim, n_heads=int(ssl_attn_heads))
            self.frame_qwen_xattn = QwenSslCrossAttnFusion(self.qwen_dim, hidden_dim, n_heads=int(ssl_attn_heads))
            self.frame_qwen_xattn_fin = QwenSslCrossAttnFusion(self.qwen_dim, hidden_dim, n_heads=int(ssl_attn_heads))
        else:
            self.node_qwen_film = None
            self.frame_qwen_film = None
            self.frame_qwen_film_fin = None
            self.node_qwen_xattn = None
            self.frame_qwen_xattn = None
            self.frame_qwen_xattn_fin = None

        feat_dim = sum(int(d) for d in feat_dims)
        self.graph_attn_pool = SegmentAttentionPoolFeat(feat_dim, dropout=graph_pool_dropout)
        self.ssl_enhancer = PyramidTriSSLDualEnhancerFA(
            feat_dims,
            hidden_dim=hidden_dim,
            n_heads=int(ssl_attn_heads),
            n_ssl_layers=int(n_ssl_layers),
            sync_layers=tuple(int(x) for x in sync_layers),
            sync_attn_heads=int(sync_attn_heads),
            seg_sync_bias=float(seg_sync_bias),
        )

        self.match_dim_reducer = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        self.match_gate = nn.Parameter(torch.tensor(-1.5))

        cd = int(contra_dim or hidden_dim)
        self.contra_dim = cd
        self.contra_proj = nn.Sequential(
            nn.Linear(hidden_dim, cd),
            nn.LayerNorm(cd),
            nn.GELU(),
        )

        self._dual_ssl_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self._dual_ssl_forward_batch: Batch | None = None

    @staticmethod
    def _run_blocks(
        x: torch.Tensor,
        edge_index: torch.Tensor,
        blocks: nn.ModuleList,
        edge_type: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for block in blocks:
            if edge_type is not None:
                x = block(x, edge_index, edge_type)
            else:
                x = block(x, edge_index)
        return x

    def _apply_phoneme_slot_mix(self, x: torch.Tensor, batch: Batch) -> torch.Tensor:
        enc = self.phoneme_slot_encoder
        logit = self.phoneme_slot_logit
        if enc is None or logit is None:
            return x
        ptr = batch.ptr
        device = x.device
        slot_idx = ptr[:-1].unsqueeze(1) + torch.arange(3, device=device, dtype=torch.long).unsqueeze(0)
        slot_h = x[slot_idx]
        if self.phoneme_slot_causal:
            causal_mask = torch.triu(torch.ones(3, 3, device=device, dtype=torch.bool), diagonal=1)
            h_out = enc(slot_h, mask=causal_mask)
        else:
            h_out = enc(slot_h)
        delta = h_out - slot_h
        gate = torch.sigmoid(logit)
        x = x.clone()
        x[slot_idx] = slot_h + gate * delta
        return x

    def _nodes_by_type(self, x: torch.Tensor, batch: Batch, node_type: int) -> torch.Tensor:
        return x[batch.node_type == node_type]

    def _pool_graph_node_feats(self, batch: Batch) -> torch.Tensor:
        pool = self.graph_attn_pool
        num_graphs = int(batch.num_graphs)
        feat_dim = pool.feat_dim
        device = batch.frame_feat.device
        dtype = batch.frame_feat.dtype
        pooled: list[torch.Tensor] = []
        off = 0
        for gid in range(num_graphs):
            t_len = int(batch.n_time_frames[gid].item())
            feat = batch.frame_feat[off : off + t_len]
            m_ini = batch.pool_mask_ini[off : off + t_len]
            m_fin = batch.pool_mask_fin[off : off + t_len]
            m_tone = batch.pool_mask_tone[off : off + t_len]
            pooled.append(pool.pool(feat, m_ini, SLOT_INI))
            pooled.append(pool.pool(feat, m_fin, SLOT_FIN))
            pooled.append(pool.pool(feat, m_tone, SLOT_TONE))
            if feat.size(0) > 0:
                pooled.append(pool.pool(feat, None, SLOT_GLOBAL))
            else:
                pooled.append(torch.zeros(feat_dim, device=device, dtype=dtype))
            off += t_len
        return torch.stack(pooled, dim=0)

    def _pool_graph_qwen(self, batch: Batch) -> torch.Tensor:
        pool = self.qwen_attn_pool
        assert pool is not None
        num_graphs = int(batch.num_graphs)
        device = batch.frame_qwen.device
        dtype = batch.frame_qwen.dtype
        qwen_dim = pool.feat_dim
        pooled: list[torch.Tensor] = []
        off = 0
        for gid in range(num_graphs):
            t_len = int(batch.n_time_frames[gid].item())
            feat = batch.frame_qwen[off : off + t_len]
            m_ini = batch.pool_mask_ini[off : off + t_len]
            m_fin = batch.pool_mask_fin[off : off + t_len]
            m_tone = batch.pool_mask_tone[off : off + t_len]
            pooled.append(pool.pool(feat, m_ini, SLOT_INI))
            pooled.append(pool.pool(feat, m_fin, SLOT_FIN))
            pooled.append(pool.pool(feat, m_tone, SLOT_TONE))
            if feat.size(0) > 0:
                pooled.append(pool.pool(feat, None, SLOT_GLOBAL))
            else:
                pooled.append(torch.zeros(qwen_dim, device=device, dtype=dtype))
            off += t_len
        return torch.stack(pooled, dim=0)

    def _has_frame_qwen(self, batch: Batch) -> bool:
        fq = getattr(batch, "frame_qwen", None)
        return fq is not None and fq.numel() > 0

    def _apply_node_qwen_xattn(self, batch: Batch, node_h: torch.Tensor) -> torch.Tensor:
        mod = self.node_qwen_xattn
        assert mod is not None
        out = node_h.clone()
        frame_off = 0
        for gid in range(int(batch.num_graphs)):
            t_len = int(batch.n_time_frames[gid].item())
            nm = batch.batch == gid
            if bool(nm.any()) and t_len > 0:
                qwen = batch.frame_qwen[frame_off : frame_off + t_len]
                out[nm] = mod(node_h[nm], qwen)
            frame_off += t_len
        return out

    def _apply_frame_qwen_xattn(self, batch: Batch, frame_h: torch.Tensor, fin: bool) -> torch.Tensor:
        mod = self.frame_qwen_xattn_fin if fin else self.frame_qwen_xattn
        assert mod is not None
        out = frame_h.clone()
        off = 0
        for gid in range(int(batch.num_graphs)):
            t_len = int(batch.n_time_frames[gid].item())
            if t_len > 0:
                out[off : off + t_len] = mod(frame_h[off : off + t_len], batch.frame_qwen[off : off + t_len])
            off += t_len
        return out

    def _get_dual_ssl(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        if self._dual_ssl_forward_batch is not batch:
            pooled_x = self._pool_graph_node_feats(batch)
            node_h, frame_h = self.ssl_enhancer(pooled_x, batch.frame_feat, batch)
            if self.qwen_fusion == "film" and self.node_qwen_film is not None and self._has_frame_qwen(batch):
                node_h = self.node_qwen_film(node_h, self._pool_graph_qwen(batch))
            elif self.qwen_fusion == "xattn" and self.node_qwen_xattn is not None and self._has_frame_qwen(batch):
                node_h = self._apply_node_qwen_xattn(batch, node_h)
            self._dual_ssl_cache = (node_h, frame_h)
            self._dual_ssl_forward_batch = batch
        assert self._dual_ssl_cache is not None
        return self._dual_ssl_cache

    def _encode_nodes_with_match(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        h0, _ = self._get_dual_ssl(batch)
        ref_h = self.ref_proj(self.ref_token_emb(batch.ref_token_id))
        meta_h = self.meta_proj(
            torch.cat([self.type_emb(batch.node_type), batch.position, batch.duration_ratio], dim=-1)
        )
        match_h = torch.abs(h0 - ref_h) + (h0 * ref_h)
        x = self.in_proj(torch.cat([h0, ref_h, meta_h, match_h], dim=-1))
        return x, match_h

    def _encode_frames(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        _, base = self._get_dual_ssl(batch)
        # _get_dual_ssl returns SSL frame features (Qwen is applied on nodes only there).
        prosody = batch.frame_prosody
        base_p = self.frame_prosody_film(base, prosody)
        base_p_fin = self.frame_prosody_film_fin(base, prosody)
        if self.qwen_fusion == "film" and self.frame_qwen_film is not None and self._has_frame_qwen(batch):
            # SSL base → Prosody FiLM → Qwen FiLM
            frame_h = self.frame_qwen_film(base_p, batch.frame_qwen)
            frame_h_fin = self.frame_qwen_film_fin(base_p_fin, batch.frame_qwen)
        elif self.qwen_fusion == "xattn" and self.frame_qwen_xattn is not None and self._has_frame_qwen(batch):
            frame_h = self._apply_frame_qwen_xattn(batch, base_p, fin=False)
            frame_h_fin = self._apply_frame_qwen_xattn(batch, base_p_fin, fin=True)
        else:
            frame_h, frame_h_fin = base_p, base_p_fin
        return frame_h, frame_h_fin, batch.seg_label_frames.long()

    def _apa_cross_readout(
        self,
        batch: Batch,
        frame_h: torch.Tensor,
        frame_h_fin: torch.Tensor,
        frame_seg: torch.Tensor,
        x_apa: torch.Tensor,
        ref_h: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_graphs = int(batch.num_graphs)
        device = x_apa.device
        dtype = x_apa.dtype
        hd = self.hidden_dim
        outs_ini: list[torch.Tensor] = []
        outs_fin: list[torch.Tensor] = []
        outs_tone: list[torch.Tensor] = []
        outs_graph: list[torch.Tensor] = []

        sem_mask = batch.node_type >= 1
        sem_h = x_apa[sem_mask]
        sem_ref = ref_h[sem_mask]
        sem_batch = batch.batch[sem_mask]

        frame_off = 0
        for gid in range(num_graphs):
            t_len = int(batch.n_time_frames[gid].item())
            rows = frame_h[frame_off : frame_off + t_len]
            rows_fin = frame_h_fin[frame_off : frame_off + t_len]
            seg = frame_seg[frame_off : frame_off + t_len]
            frame_off += t_len
            sm = sem_batch == gid
            if not bool(sm.any()) or sem_h[sm].size(0) < 4:
                z = torch.zeros(hd, device=device, dtype=dtype)
                outs_ini.append(z)
                outs_fin.append(z)
                outs_tone.append(z)
                outs_graph.append(z)
                continue
            sem = sem_h[sm] + sem_ref[sm]
            nt = batch.node_type[sem_mask][sm]
            want_ini = bool((nt == 1).any())
            want_fin = bool((nt == 2).any())
            hi, hf, ht, hg = _pool_one_graph_fin_sidepath(
                self.apa_cross_attn,
                rows,
                rows_fin,
                seg,
                sem,
                want_init=want_ini,
                want_final=want_fin,
            )
            outs_ini.append(hi)
            outs_fin.append(hf)
            outs_tone.append(ht)
            outs_graph.append(hg)

        return (
            torch.stack(outs_ini, dim=0),
            torch.stack(outs_fin, dim=0),
            torch.stack(outs_tone, dim=0),
            torch.stack(outs_graph, dim=0),
        )

    def _run_stem_trajectory(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_type: torch.Tensor | None,
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        traj: list[torch.Tensor] = []
        for block in self.stem_blocks:
            if edge_type is not None:
                x = block(x, edge_index, edge_type)
            else:
                x = block(x, edge_index)
            traj.append(x)
        return x, traj

    def forward(self, batch: Batch) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        ref_h_nodes = self.ref_proj(self.ref_token_emb(batch.ref_token_id))
        x, match_h = self._encode_nodes_with_match(batch)
        x = self._apply_phoneme_slot_mix(x, batch)
        edge_type = getattr(batch, "edge_type", None)
        x_stem, stem_traj = self._run_stem_trajectory(x, batch.edge_index, edge_type)

        match_bias = torch.sigmoid(self.match_gate) * self.match_dim_reducer(match_h)
        x_task_in = x_stem.detach() + match_bias
        x_mdd = self._run_blocks(x_task_in, batch.edge_index, self.mdd_blocks, edge_type)
        x_apa = self._run_blocks(x_task_in, batch.edge_index, self.apa_blocks, edge_type)

        hi_a = self._nodes_by_type(x_stem, batch, 1)
        hf_a = self._nodes_by_type(x_stem, batch, 2)
        ht_a = self._nodes_by_type(x_stem, batch, 3)
        hi_m = self._nodes_by_type(x_mdd, batch, 1)
        hf_m = self._nodes_by_type(x_mdd, batch, 2)
        ht_m = self._nodes_by_type(x_mdd, batch, 3)
        hg_m = self._nodes_by_type(x_mdd, batch, 4)

        frame_h, frame_h_fin, frame_seg = self._encode_frames(batch)
        hi_p, hf_p, ht_p, hg_p = self._apa_cross_readout(
            batch, frame_h, frame_h_fin, frame_seg, x_apa, ref_h_nodes
        )

        return {
            "asr_initial_logits": self.asr_initial(hi_a),
            "asr_final_logits": self.asr_final(hf_a),
            "asr_tone_logits": self.asr_tone(ht_a),
            "mdd_initial_logits": self.mdd_initial(hi_m),
            "mdd_final_logits": self.mdd_final(hf_m),
            "mdd_tone_logits": self.mdd_tone(ht_m),
            "mdd_char_logits": self.mdd_char(hg_m),
            "apa_initial_score": self.apa_initial(hi_p).squeeze(-1),
            "apa_final_score": self.apa_final(hf_p).squeeze(-1),
            "apa_char_score": self.apa_char_hier(hi_p, hf_p, ht_p, hg_p),
            "stem_traj": stem_traj,
        }
