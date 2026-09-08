"""Frozen snapshot: Qwen base → GOP FiLM → SSL FiLM (film_order=qwen_gop_ssl).

Copied from src_qwen3/english_gnn2.py for seed/seed_qwen_gop_ssl multi-seed runs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch
from torch_geometric.nn import global_mean_pool

from cross_layer_sync import PyramidTriSSLDualEnhancerFA
from english_graph_edges import NUM_ENGLISH_EDGE_TYPES, NODE_PHONE, NODE_WORD
from phoneme_gnn import PROSODY_F0_ENERGY_DIM, ProsodyFiLM8, QwenSslCrossAttnFusion, SegmentAttentionPoolFeat, TaskDecoder
from typed_gnn_blocks import EdgeTransformerBlock, MismatchEdgeTransformerBlock


def _pick_transformer_nhead(hidden_dim: int) -> int:
    for nh in (8, 4, 2, 1):
        if hidden_dim % nh == 0:
            return nh
    return 1


def make_english_gnn_block(dim: int) -> nn.Module:
    return EdgeTransformerBlock(dim, num_edge_types=NUM_ENGLISH_EDGE_TYPES)


def make_mdd_gnn_block(dim: int, mismatch_dim: int, mismatch_prop: bool) -> nn.Module:
    if mismatch_prop:
        return MismatchEdgeTransformerBlock(
            dim, num_edge_types=NUM_ENGLISH_EDGE_TYPES, mismatch_dim=mismatch_dim
        )
    return EdgeTransformerBlock(dim, num_edge_types=NUM_ENGLISH_EDGE_TYPES)


class LearnableFaSoftPool(nn.Module):
    """FA-centered soft attention pool with learnable shift/width (GOP-conditioned).

    Replaces hard FA masking: prior ∝ N(c+δ, σ²) with FA-region boost; content
    attention (same as SegmentAttentionPoolFeat) is added on top. At zero-init,
    δ≈0 and σ≈0.7·w_FA so behavior starts near the hard FA window.
    """

    def __init__(
        self,
        feat_dim: int,
        cond_dim: int,
        num_slots: int = 2,
        dropout: float = 0.1,
        max_extra: int = 12,
    ):
        super().__init__()
        self.feat_dim = int(feat_dim)
        self.cond_dim = int(cond_dim)
        self.num_slots = int(num_slots)
        self.max_extra = int(max_extra)
        hid = max(32, min(128, self.cond_dim))
        self.width_mlp = nn.Sequential(
            nn.Linear(self.cond_dim, hid),
            nn.GELU(),
            nn.Linear(hid, 2),
        )
        nn.init.zeros_(self.width_mlp[-1].weight)
        nn.init.zeros_(self.width_mlp[-1].bias)
        self.fa_bias = nn.Parameter(torch.tensor(3.0))
        self.queries = nn.Parameter(torch.randn(num_slots, self.feat_dim) * 0.02)
        self.k_proj = nn.Linear(self.feat_dim, self.feat_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.out_norm = nn.LayerNorm(self.feat_dim)
        self.scale = self.feat_dim**-0.5

    def pool(
        self,
        feat_seq: torch.Tensor,
        mask: torch.Tensor | None,
        slot: int,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t_len, fd = feat_seq.size(0), feat_seq.size(-1)
        device, dtype = feat_seq.device, feat_seq.dtype
        if t_len == 0:
            return torch.zeros(fd, device=device, dtype=dtype)

        if mask is not None and mask.numel() == t_len and bool(mask.any()):
            idx = torch.nonzero(mask.bool(), as_tuple=False).squeeze(-1)
            lo = int(idx.min().item())
            hi = int(idx.max().item())
            c = 0.5 * float(lo + hi)
            w0 = max(0.5 * float(hi - lo + 1), 1.0)
            fa_mask = mask.bool()
        else:
            # no FA: attend whole utterance with mild Gaussian at center
            lo, hi = 0, t_len - 1
            c = 0.5 * float(t_len - 1)
            w0 = max(0.25 * float(t_len), 1.0)
            fa_mask = torch.ones(t_len, device=device, dtype=torch.bool)

        if cond is None:
            cond_v = torch.zeros(self.cond_dim, device=device, dtype=dtype)
        else:
            cond_v = cond.to(device=device, dtype=dtype).reshape(-1)
            if cond_v.numel() != self.cond_dim:
                # pad / truncate
                z = torch.zeros(self.cond_dim, device=device, dtype=dtype)
                n = min(self.cond_dim, cond_v.numel())
                z[:n] = cond_v[:n]
                cond_v = z

        delta_raw, sig_raw = self.width_mlp(cond_v).unbind(-1)
        delta = torch.tanh(delta_raw) * float(self.max_extra)
        sigma = F.softplus(sig_raw) * w0 + 0.5

        # Local support: FA span ± max_extra (covers learned shift/width).
        t0 = max(0, lo - self.max_extra)
        t1 = min(t_len, hi + self.max_extra + 1)
        if t1 <= t0:
            t0, t1 = 0, t_len
        h = feat_seq[t0:t1]
        t_idx = torch.arange(t0, t1, device=device, dtype=dtype)
        prior = -0.5 * ((t_idx - (c + delta)) / sigma) ** 2
        prior = prior + fa_mask[t0:t1].to(dtype) * self.fa_bias

        if h.size(0) == 1:
            return self.out_norm(h.squeeze(0))

        q = self.queries[int(slot)]
        k = self.k_proj(h)
        logits = (k @ q) * self.scale + prior
        attn = F.softmax(logits, dim=0)
        attn = self.dropout(attn)
        out = (attn.unsqueeze(-1) * h).sum(dim=0)
        return self.out_norm(out)


class EnglishPhoneWordGraphModel(nn.Module):
    """Variable phone/word nodes; SSL pyramid + optional Qwen FiLM + multitask heads."""

    def __init__(
        self,
        feat_dims: tuple[int, ...],
        vocab_size: int,
        qwen_dim: int | None = None,
        qwen_fusion: str = "film",
        hidden_dim: int = 256,
        stem_layers: int = 4,
        mdd_layers: int = 4,
        apa_layers: int = 4,
        prosody_dim: int = PROSODY_F0_ENERGY_DIM,
        ssl_attn_heads: int = 4,
        n_ssl_layers: int = 4,
        sync_layers: tuple[int, ...] = (2, 4),
        sync_attn_heads: int = 4,
        seg_sync_bias: float = 2.0,
        graph_pool_dropout: float = 0.1,
        phone_gop_dim: int = 0,
        ssl_fuse: str = "overwrite",
        mismatch_prop: bool = False,
        mismatch_dim: int = 64,
        fa_soft_pool: bool = False,
        energy_dur_film: bool = False,
        energy_dur_dim: int = 7,
        energy_film_pos: str = "after",
        film_order: str = "qwen_gop_ssl",
        apa_energy_dur: str = "none",
        apa_energy_dur_dim: int = 8,
        utt_apa: bool = False,
    ):
        super().__init__()
        self.feat_dims = feat_dims
        self.hidden_dim = hidden_dim
        self.qwen_fusion = str(qwen_fusion)
        self.qwen_dim = int(qwen_dim) if qwen_dim else 0
        self.phone_gop_dim = int(phone_gop_dim)
        self.mismatch_prop = bool(mismatch_prop)
        self.mismatch_dim = int(mismatch_dim)
        self.vocab_size = int(vocab_size)
        self.fa_soft_pool = bool(fa_soft_pool)
        self.energy_dur_film = bool(energy_dur_film)
        self.energy_dur_dim = int(energy_dur_dim)
        # after: SSL→Qwen→energy; before_ssl: GOP→energy→SSL→Qwen
        self.energy_film_pos = str(energy_film_pos)
        # ssl_qwen: GOP→SSL→Qwen; qwen_ssl: GOP→Qwen→SSL (best);
        # ssl_gop_qwen: SSL base → GOP FiLM → Qwen FiLM
        # qwen_gop_ssl: Qwen base → GOP FiLM → SSL FiLM
        # ssl_qwen_gop_cat: SSL base → FiLM(concat(Qwen, GOP))
        self.film_order = str(film_order)
        _ok = ("ssl_qwen", "qwen_ssl", "ssl_gop_qwen", "qwen_gop_ssl", "ssl_qwen_gop_cat")
        if self.film_order not in _ok:
            raise ValueError(f"film_order must be {'|'.join(_ok)}, got {self.film_order!r}")
        apa_energy_dur = str(apa_energy_dur).lower().strip()
        if apa_energy_dur not in ("none", "stress", "apa_film"):
            raise ValueError(f"apa_energy_dur must be none|stress|apa_film, got {apa_energy_dur!r}")
        self.apa_energy_dur = apa_energy_dur
        self.apa_energy_dur_dim = int(apa_energy_dur_dim)
        self.utt_apa = bool(utt_apa)
        # overwrite: FA-pool SSL then replace with GOP (old).
        # film: GOP as base, FA-pooled SSL (enhanced) FiLM-modulates nodes (identity init).
        # xattn: GOP first, then cross-attn to SSL frames (+ Qwen), learned alignment.
        # none: GOP as base; optional Qwen FiLM only (no SSL FiLM / xattn).
        self.ssl_fuse = str(ssl_fuse)

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
        # Official GOPT Kaldi GOP (84-d) projected to hidden for phone and/or word nodes.
        if self.phone_gop_dim > 0:
            self.gop_proj = nn.Sequential(
                nn.Linear(self.phone_gop_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            )
        else:
            self.gop_proj = None

        # Chinese-style: stem → fork MDD / APA (each 4 EdgeTransformer layers).
        # MDD may use mismatch-conditioned edges (canonical↔acoustic disagreement on messages).
        self.stem_blocks = nn.ModuleList([make_english_gnn_block(hidden_dim) for _ in range(max(stem_layers, 1))])
        self.mdd_blocks = nn.ModuleList(
            [
                make_mdd_gnn_block(hidden_dim, self.mismatch_dim, self.mismatch_prop)
                for _ in range(max(mdd_layers, 1))
            ]
        )
        self.apa_blocks = nn.ModuleList([make_english_gnn_block(hidden_dim) for _ in range(max(apa_layers, 1))])
        if self.mismatch_prop:
            self.mismatch_proj = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, self.mismatch_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            )
        else:
            self.mismatch_proj = None

        # MDD: diagnosis (realized phone) + detection (CP/MP); APA scoring.
        self.mdd_phone = TaskDecoder(hidden_dim, vocab_size)
        self.mdd_detect = TaskDecoder(hidden_dim, 2)  # 0=correct, 1=mispronounced
        stress_in = hidden_dim + self.apa_energy_dur_dim if self.apa_energy_dur == "stress" else hidden_dim
        self.apa_phone = TaskDecoder(hidden_dim, 1)
        self.apa_word_acc = TaskDecoder(hidden_dim, 1)
        self.apa_word_stress = TaskDecoder(stress_in, 1)
        self.apa_word_total = TaskDecoder(hidden_dim, 1)
        # Utterance APA only when explicitly enabled (e.g. w_utt>0).
        if self.utt_apa:
            self.apa_utt_acc = TaskDecoder(hidden_dim, 1)
            self.apa_utt_comp = TaskDecoder(hidden_dim, 1)
            self.apa_utt_flu = TaskDecoder(hidden_dim, 1)
            self.apa_utt_pros = TaskDecoder(hidden_dim, 1)
            self.apa_utt_total = TaskDecoder(hidden_dim, 1)
        else:
            self.apa_utt_acc = None
            self.apa_utt_comp = None
            self.apa_utt_flu = None
            self.apa_utt_pros = None
            self.apa_utt_total = None

        self.frame_prosody_film = ProsodyFiLM8(prosody_dim=prosody_dim, hidden_dim=hidden_dim)
        # Extra FiLM after SSL/Qwen: GOPT energy(7) conditioner on phone/word nodes.
        self.node_energy_dur_film = (
            ProsodyFiLM8(prosody_dim=self.energy_dur_dim, hidden_dim=hidden_dim)
            if self.energy_dur_film
            else None
        )
        # APA-branch-only FiLM with energy(7)+dur(1); does not touch MDD.
        self.apa_ed_film = (
            ProsodyFiLM8(prosody_dim=self.apa_energy_dur_dim, hidden_dim=hidden_dim)
            if self.apa_energy_dur == "apa_film"
            else None
        )

        use_qwen_side = self.qwen_dim > 0 and self.qwen_fusion in ("film", "xattn")
        # Qwen FiLM needs mask pool unless SSL path already uses full-frame xattn for Qwen.
        soft_cond = max(self.phone_gop_dim, 1) if self.fa_soft_pool else 0
        if use_qwen_side and self.ssl_fuse != "xattn":
            if self.fa_soft_pool:
                self.qwen_attn_pool = LearnableFaSoftPool(
                    self.qwen_dim, soft_cond, num_slots=2, dropout=graph_pool_dropout
                )
            else:
                self.qwen_attn_pool = SegmentAttentionPoolFeat(
                    self.qwen_dim, num_slots=2, dropout=graph_pool_dropout
                )
        else:
            self.qwen_attn_pool = None
        if self.qwen_fusion == "film" and self.qwen_dim > 0 and self.ssl_fuse != "xattn":
            self.node_qwen_film = ProsodyFiLM8(prosody_dim=self.qwen_dim, hidden_dim=hidden_dim)
            self.frame_qwen_film = ProsodyFiLM8(prosody_dim=self.qwen_dim, hidden_dim=hidden_dim)
            self.node_qwen_xattn = None
            self.frame_qwen_xattn = None
        elif self.qwen_dim > 0 and (self.qwen_fusion == "xattn" or self.ssl_fuse == "xattn"):
            # Learned alignment: Qwen frames as K/V, node (GOP) as Q.
            self.node_qwen_film = None
            self.frame_qwen_film = ProsodyFiLM8(prosody_dim=self.qwen_dim, hidden_dim=hidden_dim) if self.qwen_fusion == "film" else None
            self.node_qwen_xattn = QwenSslCrossAttnFusion(self.qwen_dim, hidden_dim, n_heads=int(ssl_attn_heads))
            self.frame_qwen_xattn = (
                QwenSslCrossAttnFusion(self.qwen_dim, hidden_dim, n_heads=int(ssl_attn_heads))
                if self.qwen_fusion == "xattn"
                else None
            )
        else:
            self.node_qwen_film = None
            self.frame_qwen_film = None
            self.node_qwen_xattn = None
            self.frame_qwen_xattn = None

        feat_dim = sum(int(d) for d in feat_dims)
        if self.fa_soft_pool:
            self.graph_attn_pool = LearnableFaSoftPool(
                feat_dim, soft_cond, num_slots=2, dropout=graph_pool_dropout
            )
        else:
            self.graph_attn_pool = SegmentAttentionPoolFeat(feat_dim, num_slots=2, dropout=graph_pool_dropout)
        # GOP-query SSL frames (same module shape as Qwen xattn; KV dim = concat SSL).
        self.node_ssl_xattn = (
            QwenSslCrossAttnFusion(feat_dim, hidden_dim, n_heads=int(ssl_attn_heads))
            if self.ssl_fuse == "xattn"
            else None
        )
        # SSL as FiLM conditioner on GOP base (hidden SSL node feats after pyramid).
        self.node_ssl_film = (
            ProsodyFiLM8(prosody_dim=hidden_dim, hidden_dim=hidden_dim) if self.ssl_fuse == "film" else None
        )
        # GOP as FiLM conditioner on SSL/Qwen base (ssl_gop_qwen / qwen_gop_ssl).
        self.node_gop_film = (
            ProsodyFiLM8(prosody_dim=hidden_dim, hidden_dim=hidden_dim)
            if self.ssl_fuse == "film"
            and self.film_order in ("ssl_gop_qwen", "qwen_gop_ssl")
            and self.gop_proj is not None
            else None
        )
        # Project FA-pooled Qwen → hidden when Qwen is the acoustic base.
        self.qwen_base_proj = (
            nn.Sequential(
                nn.Linear(self.qwen_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
            )
            if self.ssl_fuse == "film" and self.film_order == "qwen_gop_ssl" and self.qwen_dim > 0
            else None
        )
        # SSL base + single FiLM from concat(FA-pooled Qwen, projected GOP).
        self.node_qwen_gop_cat_film = (
            ProsodyFiLM8(prosody_dim=self.qwen_dim + hidden_dim, hidden_dim=hidden_dim)
            if self.ssl_fuse == "film"
            and self.film_order == "ssl_qwen_gop_cat"
            and self.qwen_dim > 0
            and self.gop_proj is not None
            else None
        )
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
        self._dual_ssl_cache: tuple[torch.Tensor, torch.Tensor] | None = None
        self._dual_ssl_forward_batch: Batch | None = None

    @staticmethod
    def _run_blocks(
        x: torch.Tensor,
        edge_index: torch.Tensor,
        blocks: nn.ModuleList,
        edge_type: torch.Tensor | None,
        mismatch: torch.Tensor | None = None,
    ):
        for block in blocks:
            x = block(x, edge_index, edge_type, mismatch=mismatch)
        return x

    def _pool_nodes_from_masks(self, batch: Batch, feat_key: str, pool: nn.Module) -> torch.Tensor:
        """Pool each node with its FA mask; slot chosen by node_type.

        When pool is LearnableFaSoftPool, pass GOP rows as conditioner (δ, σ).
        """
        device = getattr(batch, feat_key).device
        dtype = getattr(batch, feat_key).dtype
        fd = int(getattr(pool, "feat_dim"))
        soft = isinstance(pool, LearnableFaSoftPool)
        out: list[torch.Tensor] = []
        frame_off = 0
        phone_ptr = 0
        word_ptr = 0
        masks = getattr(batch, "node_pool_masks", None)
        phone_gop = getattr(batch, "phone_gop", None) if soft else None
        word_gop = getattr(batch, "word_gop", None) if soft else None
        for gid in range(int(batch.num_graphs)):
            t_len = int(batch.n_time_frames[gid].item())
            feat = getattr(batch, feat_key)[frame_off : frame_off + t_len]
            nm = batch.batch == gid
            n_nodes = int(nm.sum().item())
            types = batch.node_type[nm]
            m = None if masks is None else masks[gid]
            if m is not None and not torch.is_tensor(m):
                m = torch.as_tensor(m, device=device)
            if m is not None:
                m = m.to(device=device)
                if m.size(1) != t_len and m.size(1) > 0:
                    idx = torch.linspace(0, m.size(1) - 1, steps=t_len, device=m.device).round().long()
                    m = m[:, idx]
            for i in range(n_nodes):
                nt = int(types[i].item())
                slot = 0 if nt == NODE_PHONE else 1
                mask_i = None
                if m is not None and i < m.size(0):
                    mask_i = m[i].bool()
                    if not bool(mask_i.any()):
                        mask_i = None
                if soft:
                    cond = None
                    if nt == NODE_PHONE and phone_gop is not None and phone_ptr < phone_gop.size(0):
                        cond = phone_gop[phone_ptr]
                        phone_ptr += 1
                    elif nt == NODE_WORD and word_gop is not None and word_ptr < word_gop.size(0):
                        cond = word_gop[word_ptr]
                        word_ptr += 1
                    elif nt == NODE_PHONE:
                        phone_ptr += 1
                    elif nt == NODE_WORD:
                        word_ptr += 1
                    out.append(pool.pool(feat, mask_i, slot, cond=cond))
                else:
                    out.append(pool.pool(feat, mask_i, slot))
            frame_off += t_len
        if not out:
            return torch.zeros(0, fd, device=device, dtype=dtype)
        return torch.stack(out, dim=0)

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
                out[nm] = mod(node_h[nm], batch.frame_qwen[frame_off : frame_off + t_len])
            frame_off += t_len
        return out

    def _apply_frame_kv_xattn(
        self,
        batch: Batch,
        node_h: torch.Tensor,
        frame_key: str,
        mod: nn.Module,
    ) -> torch.Tensor:
        """Per-graph cross-attn: node queries attend over frame K/V (learned alignment)."""
        out = node_h.clone()
        frame_off = 0
        for gid in range(int(batch.num_graphs)):
            t_len = int(batch.n_time_frames[gid].item())
            nm = batch.batch == gid
            frames = getattr(batch, frame_key)
            if bool(nm.any()) and t_len > 0 and frames is not None:
                out[nm] = mod(node_h[nm], frames[frame_off : frame_off + t_len])
            frame_off += t_len
        return out

    def _apply_gop_nodes(self, batch: Batch, node_h: torch.Tensor) -> torch.Tensor:
        """Replace phone/word node acoustics with projected official GOP when provided."""
        if self.gop_proj is None:
            return node_h
        out = node_h
        phone_gop = getattr(batch, "phone_gop", None)
        word_gop = getattr(batch, "word_gop", None)
        if phone_gop is not None and phone_gop.numel() > 0:
            phone_m = batch.node_type == NODE_PHONE
            n_phone = int(phone_m.sum().item())
            if n_phone > 0:
                if int(phone_gop.size(0)) != n_phone:
                    raise RuntimeError(f"phone_gop rows={phone_gop.size(0)} != n_phone={n_phone}")
                if out is node_h:
                    out = node_h.clone()
                out[phone_m] = self.gop_proj(phone_gop.to(device=node_h.device, dtype=node_h.dtype))
        if word_gop is not None and word_gop.numel() > 0:
            word_m = batch.node_type == NODE_WORD
            n_word = int(word_m.sum().item())
            if n_word > 0:
                if int(word_gop.size(0)) != n_word:
                    raise RuntimeError(f"word_gop rows={word_gop.size(0)} != n_word={n_word}")
                if out is node_h:
                    out = node_h.clone()
                out[word_m] = self.gop_proj(word_gop.to(device=node_h.device, dtype=node_h.dtype))
        return out

    def _apply_gop_film(self, batch: Batch, node_h: torch.Tensor) -> torch.Tensor:
        """FiLM-modulate SSL phone/word nodes with projected GOP (identity init)."""
        if self.node_gop_film is None or self.gop_proj is None:
            return node_h
        out = node_h
        phone_gop = getattr(batch, "phone_gop", None)
        word_gop = getattr(batch, "word_gop", None)
        if phone_gop is not None and phone_gop.numel() > 0:
            phone_m = batch.node_type == NODE_PHONE
            n_phone = int(phone_m.sum().item())
            if n_phone > 0:
                if int(phone_gop.size(0)) != n_phone:
                    raise RuntimeError(f"phone_gop rows={phone_gop.size(0)} != n_phone={n_phone}")
                cond = self.gop_proj(phone_gop.to(device=node_h.device, dtype=node_h.dtype))
                if out is node_h:
                    out = node_h.clone()
                out[phone_m] = self.node_gop_film(out[phone_m], cond)
        if word_gop is not None and word_gop.numel() > 0:
            word_m = batch.node_type == NODE_WORD
            n_word = int(word_m.sum().item())
            if n_word > 0:
                if int(word_gop.size(0)) != n_word:
                    raise RuntimeError(f"word_gop rows={word_gop.size(0)} != n_word={n_word}")
                cond = self.gop_proj(word_gop.to(device=node_h.device, dtype=node_h.dtype))
                if out is node_h:
                    out = node_h.clone()
                out[word_m] = self.node_gop_film(out[word_m], cond)
        return out

    def _apply_qwen_gop_cat_film(self, batch: Batch, node_h: torch.Tensor) -> torch.Tensor:
        """FiLM SSL nodes with concat(FA-pooled Qwen, projected GOP)."""
        if self.node_qwen_gop_cat_film is None or self.gop_proj is None or self.qwen_attn_pool is None:
            return node_h
        if not self._has_frame_qwen(batch):
            raise RuntimeError("ssl_qwen_gop_cat requires batch.frame_qwen")
        qwen_pool = self._pool_nodes_from_masks(batch, "frame_qwen", self.qwen_attn_pool)
        out = node_h
        phone_gop = getattr(batch, "phone_gop", None)
        word_gop = getattr(batch, "word_gop", None)
        if phone_gop is not None and phone_gop.numel() > 0:
            phone_m = batch.node_type == NODE_PHONE
            n_phone = int(phone_m.sum().item())
            if n_phone > 0:
                if int(phone_gop.size(0)) != n_phone:
                    raise RuntimeError(f"phone_gop rows={phone_gop.size(0)} != n_phone={n_phone}")
                gop_h = self.gop_proj(phone_gop.to(device=node_h.device, dtype=node_h.dtype))
                cond = torch.cat([qwen_pool[phone_m], gop_h], dim=-1)
                if out is node_h:
                    out = node_h.clone()
                out[phone_m] = self.node_qwen_gop_cat_film(out[phone_m], cond)
        if word_gop is not None and word_gop.numel() > 0:
            word_m = batch.node_type == NODE_WORD
            n_word = int(word_m.sum().item())
            if n_word > 0:
                if int(word_gop.size(0)) != n_word:
                    raise RuntimeError(f"word_gop rows={word_gop.size(0)} != n_word={n_word}")
                gop_h = self.gop_proj(word_gop.to(device=node_h.device, dtype=node_h.dtype))
                cond = torch.cat([qwen_pool[word_m], gop_h], dim=-1)
                if out is node_h:
                    out = node_h.clone()
                out[word_m] = self.node_qwen_gop_cat_film(out[word_m], cond)
        return out

    def _apply_energy_dur_film(self, batch: Batch, node_h: torch.Tensor) -> torch.Tensor:
        """FiLM-modulate phone/word nodes with GOPT energy(7) (identity init)."""
        if self.node_energy_dur_film is None:
            return node_h
        return self._film_nodes_with_ed(
            batch, node_h, self.node_energy_dur_film, self.energy_dur_dim, "main"
        )

    def _apply_apa_energy_dur_film(self, batch: Batch, node_h: torch.Tensor) -> torch.Tensor:
        """APA-branch FiLM with energy(7)+dur(1); MDD path untouched."""
        if self.apa_ed_film is None:
            return node_h
        return self._film_nodes_with_ed(
            batch, node_h, self.apa_ed_film, self.apa_energy_dur_dim, "apa"
        )

    def _film_nodes_with_ed(
        self,
        batch: Batch,
        node_h: torch.Tensor,
        film: nn.Module,
        ed_dim: int,
        tag: str,
    ) -> torch.Tensor:
        out = node_h
        phone_ed = getattr(batch, "phone_energy_dur", None)
        word_ed = getattr(batch, "word_energy_dur", None)
        if phone_ed is not None and phone_ed.numel() > 0:
            phone_m = batch.node_type == NODE_PHONE
            n_phone = int(phone_m.sum().item())
            if n_phone > 0:
                if int(phone_ed.size(0)) != n_phone:
                    raise RuntimeError(f"{tag} phone_energy_dur rows={phone_ed.size(0)} != n_phone={n_phone}")
                if int(phone_ed.size(-1)) < ed_dim:
                    raise RuntimeError(f"{tag} phone_energy_dur dim={phone_ed.size(-1)} < {ed_dim}")
                if out is node_h:
                    out = node_h.clone()
                cond = phone_ed[..., :ed_dim].to(device=node_h.device, dtype=node_h.dtype)
                out[phone_m] = film(out[phone_m], cond)
        if word_ed is not None and word_ed.numel() > 0:
            word_m = batch.node_type == NODE_WORD
            n_word = int(word_m.sum().item())
            if n_word > 0:
                if int(word_ed.size(0)) != n_word:
                    raise RuntimeError(f"{tag} word_energy_dur rows={word_ed.size(0)} != n_word={n_word}")
                if int(word_ed.size(-1)) < ed_dim:
                    raise RuntimeError(f"{tag} word_energy_dur dim={word_ed.size(-1)} < {ed_dim}")
                if out is node_h:
                    out = node_h.clone()
                cond = word_ed[..., :ed_dim].to(device=node_h.device, dtype=node_h.dtype)
                out[word_m] = film(out[word_m], cond)
        return out

    def _get_dual_ssl(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        if self._dual_ssl_forward_batch is not batch:
            pooled_x = self._pool_nodes_from_masks(batch, "frame_feat", self.graph_attn_pool)
            ssl_node, frame_h = self.ssl_enhancer(pooled_x, batch.frame_feat, batch)

            if self.ssl_fuse == "xattn" and self.gop_proj is not None:
                # Scheme-2: GOP defines phone/word acoustics, then attend SSL (+Qwen) frames.
                node_h = self._apply_gop_nodes(batch, ssl_node)
                if self.node_ssl_xattn is not None:
                    node_h = self._apply_frame_kv_xattn(batch, node_h, "frame_feat", self.node_ssl_xattn)
                if self.node_qwen_xattn is not None and self._has_frame_qwen(batch):
                    node_h = self._apply_frame_kv_xattn(batch, node_h, "frame_qwen", self.node_qwen_xattn)
            elif self.ssl_fuse == "none" and self.gop_proj is not None:
                # Ablation: GOP base + optional Qwen FiLM only (no SSL FiLM).
                node_h = self._apply_gop_nodes(batch, ssl_node)
                if self.qwen_fusion == "film" and self.node_qwen_film is not None and self._has_frame_qwen(batch):
                    assert self.qwen_attn_pool is not None
                    node_h = self.node_qwen_film(
                        node_h, self._pool_nodes_from_masks(batch, "frame_qwen", self.qwen_attn_pool)
                    )
                elif self.qwen_fusion == "xattn" and self.node_qwen_xattn is not None and self._has_frame_qwen(batch):
                    node_h = self._apply_node_qwen_xattn(batch, node_h)
            elif self.ssl_fuse == "film" and self.gop_proj is not None and self.node_ssl_film is not None:
                # Scheme-3: FiLM stack (order via film_order).
                def _apply_ssl_film(h: torch.Tensor) -> torch.Tensor:
                    return self.node_ssl_film(h, ssl_node)  # type: ignore[misc]

                def _apply_qwen_film(h: torch.Tensor) -> torch.Tensor:
                    if self.qwen_fusion == "film" and self.node_qwen_film is not None and self._has_frame_qwen(batch):
                        assert self.qwen_attn_pool is not None
                        return self.node_qwen_film(
                            h, self._pool_nodes_from_masks(batch, "frame_qwen", self.qwen_attn_pool)
                        )
                    if self.qwen_fusion == "xattn" and self.node_qwen_xattn is not None and self._has_frame_qwen(batch):
                        return self._apply_node_qwen_xattn(batch, h)
                    return h

                if self.film_order == "ssl_gop_qwen":
                    # SSL base → GOP FiLM → Qwen FiLM
                    node_h = ssl_node
                    if self.energy_film_pos == "before_ssl":
                        node_h = self._apply_energy_dur_film(batch, node_h)
                    node_h = self._apply_gop_film(batch, node_h)
                    node_h = _apply_qwen_film(node_h)
                elif self.film_order == "ssl_qwen_gop_cat":
                    # SSL base → FiLM(concat(Qwen, GOP))
                    node_h = ssl_node
                    if self.energy_film_pos == "before_ssl":
                        node_h = self._apply_energy_dur_film(batch, node_h)
                    node_h = self._apply_qwen_gop_cat_film(batch, node_h)
                elif self.film_order == "qwen_gop_ssl":
                    # Qwen base → GOP FiLM → SSL FiLM
                    if self.qwen_base_proj is None or self.qwen_attn_pool is None:
                        raise RuntimeError("qwen_gop_ssl requires qwen_base_proj + qwen_attn_pool")
                    if not self._has_frame_qwen(batch):
                        raise RuntimeError("qwen_gop_ssl requires batch.frame_qwen")
                    node_h = self.qwen_base_proj(
                        self._pool_nodes_from_masks(batch, "frame_qwen", self.qwen_attn_pool)
                    )
                    if self.energy_film_pos == "before_ssl":
                        node_h = self._apply_energy_dur_film(batch, node_h)
                    node_h = self._apply_gop_film(batch, node_h)
                    node_h = _apply_ssl_film(node_h)
                else:
                    # GOP base + FiLM stack
                    node_h = self._apply_gop_nodes(batch, ssl_node)
                    # Optional: energy FiLM before SSL/Qwen (GOP → energy → …).
                    if self.energy_film_pos == "before_ssl":
                        node_h = self._apply_energy_dur_film(batch, node_h)

                    if self.film_order == "qwen_ssl":
                        # GOP → Qwen FiLM → SSL FiLM
                        node_h = _apply_qwen_film(node_h)
                        node_h = _apply_ssl_film(node_h)
                    else:
                        # Default: GOP → SSL FiLM → Qwen FiLM
                        node_h = _apply_ssl_film(node_h)
                        node_h = _apply_qwen_film(node_h)
            else:
                node_h = ssl_node
                if self.qwen_fusion == "film" and self.node_qwen_film is not None and self._has_frame_qwen(batch):
                    assert self.qwen_attn_pool is not None
                    node_h = self.node_qwen_film(
                        node_h, self._pool_nodes_from_masks(batch, "frame_qwen", self.qwen_attn_pool)
                    )
                elif self.qwen_fusion == "xattn" and self.node_qwen_xattn is not None and self._has_frame_qwen(batch):
                    node_h = self._apply_node_qwen_xattn(batch, node_h)
                # Old path: FA-pool SSL/Qwen then overwrite with GOP.
                node_h = self._apply_gop_nodes(batch, node_h)

            # Default energy FiLM after SSL/Qwen (skipped if already applied before SSL).
            if self.energy_film_pos != "before_ssl":
                node_h = self._apply_energy_dur_film(batch, node_h)

            self._dual_ssl_cache = (node_h, frame_h)
            self._dual_ssl_forward_batch = batch
        assert self._dual_ssl_cache is not None
        return self._dual_ssl_cache

    def forward(self, batch: Batch) -> dict[str, torch.Tensor]:
        h0, _ = self._get_dual_ssl(batch)
        ref_h = self.ref_proj(self.ref_token_emb(batch.ref_token_id))
        meta_h = self.meta_proj(
            torch.cat([self.type_emb(batch.node_type), batch.position, batch.duration_ratio], dim=-1)
        )
        match_h = torch.abs(h0 - ref_h) + (h0 * ref_h)
        x = self.in_proj(torch.cat([h0, ref_h, meta_h, match_h], dim=-1))

        edge_type = getattr(batch, "edge_type", None)
        x_stem = self._run_blocks(x, batch.edge_index, self.stem_blocks, edge_type)
        match_bias = torch.sigmoid(self.match_gate) * self.match_dim_reducer(match_h)
        # Shared stem → independent MDD / APA branches.
        x_task = x_stem + match_bias
        mismatch = self.mismatch_proj(match_h) if self.mismatch_proj is not None else None
        x_mdd = self._run_blocks(
            x_task, batch.edge_index, self.mdd_blocks, edge_type, mismatch=mismatch
        )
        x_apa = self._run_blocks(x_task, batch.edge_index, self.apa_blocks, edge_type)
        # APA-only energy/dur FiLM (after APA GNN; does not affect MDD).
        if self.apa_energy_dur == "apa_film":
            x_apa = self._apply_apa_energy_dur_film(batch, x_apa)

        phone_m = batch.node_type == NODE_PHONE
        word_m = batch.node_type == NODE_WORD
        w = x_apa[word_m]
        x_mdd_phone = x_mdd[phone_m]

        if self.apa_energy_dur == "stress":
            word_ed = getattr(batch, "word_energy_dur", None)
            if word_ed is None or word_ed.numel() == 0:
                raise RuntimeError("apa_energy_dur=stress requires batch.word_energy_dur")
            if int(word_ed.size(0)) != int(w.size(0)):
                raise RuntimeError(
                    f"word_energy_dur rows={word_ed.size(0)} != n_word={w.size(0)}"
                )
            if int(word_ed.size(-1)) < self.apa_energy_dur_dim:
                raise RuntimeError(
                    f"word_energy_dur dim={word_ed.size(-1)} < {self.apa_energy_dur_dim}"
                )
            w_stress = torch.cat(
                [w, word_ed[..., : self.apa_energy_dur_dim].to(device=w.device, dtype=w.dtype)],
                dim=-1,
            )
            stress_score = self.apa_word_stress(w_stress).squeeze(-1)
        else:
            stress_score = self.apa_word_stress(w).squeeze(-1)

        out = {
            "mdd_phone_logits": self.mdd_phone(x_mdd_phone),
            "mdd_detect_logits": self.mdd_detect(x_mdd_phone),
            "apa_phone_score": self.apa_phone(x_apa[phone_m]).squeeze(-1),
            "apa_word_acc": self.apa_word_acc(w).squeeze(-1),
            "apa_word_stress": stress_score,
            "apa_word_total": self.apa_word_total(w).squeeze(-1),
        }
        if self.utt_apa:
            utt_h = global_mean_pool(x_apa, batch.batch)
            out["apa_utt_acc"] = self.apa_utt_acc(utt_h).squeeze(-1)  # type: ignore[misc]
            out["apa_utt_comp"] = self.apa_utt_comp(utt_h).squeeze(-1)  # type: ignore[misc]
            out["apa_utt_flu"] = self.apa_utt_flu(utt_h).squeeze(-1)  # type: ignore[misc]
            out["apa_utt_pros"] = self.apa_utt_pros(utt_h).squeeze(-1)  # type: ignore[misc]
            out["apa_utt_total"] = self.apa_utt_total(utt_h).squeeze(-1)  # type: ignore[misc]
        return out
