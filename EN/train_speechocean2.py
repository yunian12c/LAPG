#!/usr/bin/env python3
"""Frozen snapshot: Qwen→GOP FiLM→SSL FiLM multi-seed trainer.

Copied from src_qwen3/train_speechocean2.py for seed/seed_qwen_gop_ssl runs.
Local english_gnn2.py is used; other helpers come from src_qwen3.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

_HERE = Path(__file__).resolve().parent
# Self-contained package: all helpers live next to this script.
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

try:
    from torch_geometric.data import Batch, Data
except Exception:
    Batch = None  # type: ignore
    Data = None  # type: ignore

from classic_mdd_eval import ids_to_phone_str, run_classic_mdd
from english_gnn2 import EnglishPhoneWordGraphModel
from english_graph_edges import (
    NODE_PHONE,
    NODE_WORD,
    build_english_phone_word_edges,
    edges_to_tensors,
)
from fa_cache_speechocean import load_speechocean_fa_masks
from hmamba_phone_vocab import aligned_realized_ids, canonical_ids, load_hmamba_vocab, phone_to_id
from hubert_side_cache import (
    HUBERT_ENERGY_CACHE_KEY,
    HUBERT_ENERGY_DIM,
    HUBERT_F0_CACHE_KEY,
    HUBERT_F0_DIM,
    _load_hubert_side_cache,
    _normalize_side,
    compute_hubert_side_stats,
)
from phoneme_gnn import PROSODY_F0_ENERGY_DIM
from train_gop_graph import align_wav2vec_to_target_frames, compute_acoustic_stats_native, read_jsonl

IGNORE_INDEX = -100
BASE = _HERE
GOP_RAW = BASE / "gop_feature/raw_kaldi_gop/librispeech"
GOP_SEQ = BASE / "gop_feature/seq_data_librispeech"


def load_gop_utt_map(split: str) -> dict[str, np.ndarray]:
    """Load official GOPT Librispeech GOP: utt_id -> (n_phones, 84) float32.

    CSV feat row = [phone_id, gop_0..gop_83]; keys = utt.phone_idx.
    """
    prefix = "tr" if split == "train" else "te"
    keys_path = GOP_RAW / f"{prefix}_keys_phn.csv"
    feats_path = GOP_RAW / f"{prefix}_feats.csv"
    if not keys_path.is_file() or not feats_path.is_file():
        raise FileNotFoundError(f"missing GOP raw files under {GOP_RAW}")
    keys = [ln.strip() for ln in keys_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    feats = np.loadtxt(feats_path, delimiter=",", dtype=np.float32)
    if feats.ndim == 1:
        feats = feats.reshape(1, -1)
    if feats.shape[0] != len(keys):
        raise RuntimeError(f"GOP keys/feats size mismatch: {len(keys)} vs {feats.shape[0]}")
    if feats.shape[1] < 2:
        raise RuntimeError(f"GOP feat dim too small: {feats.shape}")
    buckets: dict[str, list[tuple[int, np.ndarray]]] = {}
    for key, row in zip(keys, feats):
        utt, idx_s = key.rsplit(".", 1)
        buckets.setdefault(utt, []).append((int(float(idx_s)), row[1:].astype(np.float32)))
    out: dict[str, np.ndarray] = {}
    for utt, items in buckets.items():
        items.sort(key=lambda x: x[0])
        out[utt] = np.stack([v for _, v in items], axis=0)
    return out


def load_seq_energy_dur_map(split: str, *, padded: bool = False) -> dict[str, np.ndarray]:
    """GOPT seq packs: utt_id -> (T, 8) = energy(7) + dur(1).

    padded=False: truncate to nonzero phone count (legacy GOP-concat).
    padded=True: keep full pad length; dataset slices to label n_phones.
    """
    prefix = "tr" if split == "train" else "te"
    utt_path = BASE / "data" / split / "utt_ids.txt"
    energy_path = GOP_SEQ / f"{prefix}_energy_feat.npy"
    dur_path = GOP_SEQ / f"{prefix}_dur_feat.npy"
    if not utt_path.is_file():
        raise FileNotFoundError(utt_path)
    if not energy_path.is_file() or not dur_path.is_file():
        raise FileNotFoundError(f"missing energy/dur under {GOP_SEQ}")
    utt_ids = [ln.strip() for ln in utt_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    energy = np.asarray(np.load(energy_path), dtype=np.float32)
    dur = np.asarray(np.load(dur_path), dtype=np.float32)
    if energy.ndim != 3 or dur.ndim != 3:
        raise RuntimeError(f"bad energy/dur shapes: {energy.shape} {dur.shape}")
    if energy.shape[0] != len(utt_ids) or dur.shape[0] != len(utt_ids):
        raise RuntimeError(
            f"utt/energy/dur size mismatch: {len(utt_ids)} vs {energy.shape[0]} vs {dur.shape[0]}"
        )
    out: dict[str, np.ndarray] = {}
    for i, sid in enumerate(utt_ids):
        e = energy[i]
        d = dur[i]
        if padded:
            out[sid] = np.concatenate([e, d], axis=-1).astype(np.float32)
            continue
        n = int((np.abs(e).sum(axis=-1) > 0).sum())
        if n <= 0:
            n = int((np.abs(d).sum(axis=-1) > 0).sum())
        if n <= 0:
            raise RuntimeError(f"empty energy/dur for {sid}")
        out[sid] = np.concatenate([e[:n], d[:n]], axis=-1).astype(np.float32)
    return out


def load_seq_dur_map(split: str) -> dict[str, np.ndarray]:
    """GOPT seq duration only: utt_id -> (max_phones, 1), padded; concat uses GOP length."""
    prefix = "tr" if split == "train" else "te"
    utt_path = BASE / "data" / split / "utt_ids.txt"
    dur_path = GOP_SEQ / f"{prefix}_dur_feat.npy"
    if not utt_path.is_file():
        raise FileNotFoundError(utt_path)
    if not dur_path.is_file():
        raise FileNotFoundError(dur_path)
    utt_ids = [ln.strip() for ln in utt_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    dur = np.asarray(np.load(dur_path), dtype=np.float32)
    if dur.ndim != 3 or int(dur.shape[-1]) != 1:
        raise RuntimeError(f"bad dur shape: {dur.shape}")
    if dur.shape[0] != len(utt_ids):
        raise RuntimeError(f"utt/dur size mismatch: {len(utt_ids)} vs {dur.shape[0]}")
    return {sid: dur[i].astype(np.float32) for i, sid in enumerate(utt_ids)}


def load_seq_energy_map(split: str) -> dict[str, np.ndarray]:
    """GOPT seq energy only: utt_id -> (max_phones, 7), padded; slice to n_phones in dataset."""
    prefix = "tr" if split == "train" else "te"
    utt_path = BASE / "data" / split / "utt_ids.txt"
    energy_path = GOP_SEQ / f"{prefix}_energy_feat.npy"
    if not utt_path.is_file():
        raise FileNotFoundError(utt_path)
    if not energy_path.is_file():
        raise FileNotFoundError(energy_path)
    utt_ids = [ln.strip() for ln in utt_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    energy = np.asarray(np.load(energy_path), dtype=np.float32)
    if energy.ndim != 3 or int(energy.shape[-1]) != 7:
        raise RuntimeError(f"bad energy shape: {energy.shape}")
    if energy.shape[0] != len(utt_ids):
        raise RuntimeError(f"utt/energy size mismatch: {len(utt_ids)} vs {energy.shape[0]}")
    return {sid: energy[i].astype(np.float32) for i, sid in enumerate(utt_ids)}


def compute_energy_film_stats(
    ed_map: dict[str, np.ndarray],
    phone_len_map: dict[str, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Train-set mean/std for energy(7) FiLM, sliced to label/GOP phone lengths."""
    xs: list[np.ndarray] = []
    for sid, n in phone_len_map.items():
        if sid not in ed_map:
            continue
        e = np.asarray(ed_map[sid], dtype=np.float32)
        if int(e.shape[0]) < int(n) or int(n) <= 0:
            continue
        xs.append(e[: int(n)])
    if not xs:
        raise RuntimeError("empty energy map for stats")
    cat = np.concatenate(xs, axis=0)
    mean = cat.mean(axis=0).astype(np.float32)
    std = np.maximum(cat.std(axis=0).astype(np.float32), 1e-5)
    return mean, std


def compute_energy_dur_stats(ed_map: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Train-set mean/std for energy(7)+dur(1) FiLM conditioner."""
    xs = [np.asarray(v, dtype=np.float32) for v in ed_map.values() if v.size > 0]
    if not xs:
        raise RuntimeError("empty energy/dur map for stats")
    cat = np.concatenate(xs, axis=0)
    mean = cat.mean(axis=0).astype(np.float32)
    std = cat.std(axis=0).astype(np.float32)
    std = np.maximum(std, 1e-5)
    return mean, std


def concat_gop_with_energy_dur(
    gop_map: dict[str, np.ndarray],
    extra_map: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Phone node feat = [GOP(84), energy(7), dur(1)] = 92-d."""
    out: dict[str, np.ndarray] = {}
    for sid, g in gop_map.items():
        if sid not in extra_map:
            raise KeyError(f"missing energy/dur for {sid}")
        e = extra_map[sid]
        n = int(g.shape[0])
        if int(e.shape[0]) != n:
            if int(e.shape[0]) > n:
                e = e[:n]
            else:
                pad = np.zeros((n - int(e.shape[0]), int(e.shape[1])), dtype=np.float32)
                e = np.concatenate([e, pad], axis=0)
        out[sid] = np.concatenate([g.astype(np.float32), e.astype(np.float32)], axis=-1)
    return out


def compute_gop_stats(gop_map: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    if not gop_map:
        raise RuntimeError("empty gop_map")
    dim = int(next(iter(gop_map.values())).shape[1])
    total = np.zeros(dim, dtype=np.float64)
    total_sq = np.zeros(dim, dtype=np.float64)
    n = 0
    for arr in gop_map.values():
        a = arr.astype(np.float64)
        total += a.sum(axis=0)
        total_sq += (a * a).sum(axis=0)
        n += a.shape[0]
    mean = (total / max(n, 1)).astype(np.float32)
    var = total_sq / max(n, 1) - mean.astype(np.float64) ** 2
    std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
    return mean, std


# Phone inventory: HMamba vocab_merge.json (see hmamba_phone_vocab.py).


def mse(pred: list[float], ref: list[float]) -> float:
    if not ref:
        return 0.0
    a = np.asarray(pred, dtype=np.float64)
    b = np.asarray(ref, dtype=np.float64)
    return float(np.mean((a - b) ** 2))


def pcc(pred: list[float], ref: list[float]) -> float:
    if len(pred) < 2:
        return 0.0
    a = np.asarray(pred, dtype=np.float64)
    b = np.asarray(ref, dtype=np.float64)
    if a.std() < 1e-8 or b.std() < 1e-8:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def expand_binary_masks(masks: np.ndarray, expand_frames: int) -> np.ndarray:
    """Dilate each row's contiguous True span by ±expand_frames. masks: [N, T] bool/uint8."""
    if expand_frames <= 0:
        return masks
    out = np.asarray(masks, dtype=bool).copy()
    n, t = out.shape
    for i in range(n):
        row = out[i]
        if not row.any():
            continue
        idx = np.flatnonzero(row)
        lo = max(0, int(idx.min()) - expand_frames)
        hi = min(t, int(idx.max()) + expand_frames + 1)
        out[i, :] = False
        out[i, lo:hi] = True
    return out


def expand_binary_masks_at(masks: np.ndarray, indices: list[int], expand_frames: int) -> np.ndarray:
    """Dilate only selected rows by ±expand_frames."""
    if expand_frames <= 0 or not indices:
        return masks
    out = np.asarray(masks, dtype=bool).copy()
    t = out.shape[1]
    for i in indices:
        if i < 0 or i >= out.shape[0]:
            continue
        row = out[i]
        if not row.any():
            continue
        idx = np.flatnonzero(row)
        lo = max(0, int(idx.min()) - expand_frames)
        hi = min(t, int(idx.max()) + expand_frames + 1)
        out[i, :] = False
        out[i, lo:hi] = True
    return out


def edit_distance(a: list[int], b: list[int]) -> int:
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev, dp[0] = dp[0], i
        for j in range(1, m + 1):
            cur = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = cur
    return dp[m]


class SpeechoceanGraphDataset(Dataset):
    def __init__(
        self,
        split_dir: Path,
        vocab: dict[str, int],
        hubert_dir: Path,
        hubert_mean: np.ndarray,
        hubert_std: np.ndarray,
        w2v_dir: Path,
        w2v_mean: np.ndarray,
        w2v_std: np.ndarray,
        wavlm_dir: Path,
        wavlm_mean: np.ndarray,
        wavlm_std: np.ndarray,
        fa_cache_dir: Path,
        f0_cache_dir: Path,
        energy_cache_dir: Path,
        f0_mean: np.ndarray,
        f0_std: np.ndarray,
        energy_mean: np.ndarray,
        energy_std: np.ndarray,
        qwen_dir: Path | None = None,
        qwen_mean: np.ndarray | None = None,
        qwen_std: np.ndarray | None = None,
        qwen_fusion: str = "film",
        phone_feat: str = "ssl",
        word_feat: str = "ssl",
        gop_map: dict[str, np.ndarray] | None = None,
        gop_mean: np.ndarray | None = None,
        gop_std: np.ndarray | None = None,
        energy_dur_map: dict[str, np.ndarray] | None = None,
        energy_dur_mean: np.ndarray | None = None,
        energy_dur_std: np.ndarray | None = None,
        coword_edges: bool = False,
        fa_expand_frames: int = 0,
        fa_expand_err_train: int = 0,
        detect_label: str = "accuracy",
    ):
        self.records = read_jsonl(split_dir / "labels.jsonl")
        self.vocab = vocab
        self.split_name = split_dir.name
        self.hubert_dir = hubert_dir
        self.hubert_mean = hubert_mean
        self.hubert_std = hubert_std
        self.w2v_dir = w2v_dir
        self.w2v_mean = w2v_mean
        self.w2v_std = w2v_std
        self.wavlm_dir = wavlm_dir
        self.wavlm_mean = wavlm_mean
        self.wavlm_std = wavlm_std
        self.fa_cache_dir = fa_cache_dir
        self.f0_cache_dir = f0_cache_dir
        self.energy_cache_dir = energy_cache_dir
        self.f0_mean = f0_mean
        self.f0_std = f0_std
        self.energy_mean = energy_mean
        self.energy_std = energy_std
        self.qwen_dir = qwen_dir
        self.qwen_mean = qwen_mean
        self.qwen_std = qwen_std
        self.qwen_fusion = qwen_fusion
        self.phone_feat = str(phone_feat)
        self.word_feat = str(word_feat)
        self.gop_map = gop_map or {}
        self.gop_mean = gop_mean
        self.gop_std = gop_std
        self.energy_dur_map = energy_dur_map or {}
        self.energy_dur_mean = energy_dur_mean
        self.energy_dur_std = energy_dur_std
        self.need_gop = self.phone_feat == "gop" or self.word_feat == "gop"
        self.need_energy_dur = bool(self.energy_dur_map)
        self.coword_edges = bool(coword_edges)
        self.fa_expand_frames = int(fa_expand_frames)
        self.fa_expand_err_train = int(fa_expand_err_train)
        self.detect_label = str(detect_label)

        qwen_dim = int(qwen_mean.shape[0]) if qwen_mean is not None else 0
        if qwen_dir is not None and qwen_fusion == "concat":
            self.feat_dims = (
                int(hubert_mean.shape[0]),
                int(w2v_mean.shape[0]),
                int(wavlm_mean.shape[0]),
                qwen_dim,
            )
        else:
            self.feat_dims = (
                int(hubert_mean.shape[0]),
                int(w2v_mean.shape[0]),
                int(wavlm_mean.shape[0]),
            )

        kept = []
        miss = {
            "hubert": 0,
            "w2v": 0,
            "wavlm": 0,
            "qwen": 0,
            "fa": 0,
            "f0": 0,
            "energy": 0,
            "gop": 0,
            "energy_dur": 0,
        }
        for r in self.records:
            sid = str(r.get("id", ""))
            if not sid:
                continue
            ok = True
            for key, d in (("hubert", hubert_dir), ("w2v", w2v_dir), ("wavlm", wavlm_dir)):
                if not (d / self.split_name / f"{sid}.npy").is_file():
                    miss[key] += 1
                    ok = False
            if qwen_dir is not None and not (qwen_dir / self.split_name / f"{sid}.npy").is_file():
                miss["qwen"] += 1
                ok = False
            if not (fa_cache_dir / self.split_name / f"{sid}.npz").is_file():
                miss["fa"] += 1
                ok = False
            if not (f0_cache_dir / self.split_name / f"{sid}.npz").is_file():
                miss["f0"] += 1
                ok = False
            if not (energy_cache_dir / self.split_name / f"{sid}.npz").is_file():
                miss["energy"] += 1
                ok = False
            if self.need_gop:
                g = self.gop_map.get(sid)
                n_ph = len(r.get("phones", []) or [])
                if g is None or int(g.shape[0]) != n_ph:
                    miss["gop"] += 1
                    ok = False
            if self.need_energy_dur:
                ed = self.energy_dur_map.get(sid)
                n_ph = len(r.get("phones", []) or [])
                # padded energy/dur (T,7|8); need T >= n_phones
                if (
                    ed is None
                    or int(ed.shape[0]) < n_ph
                    or int(ed.shape[-1]) not in (7, 8)
                ):
                    miss["energy_dur"] += 1
                    ok = False
            if ok:
                kept.append(r)
        self.records = kept
        print(
            f"[dataset] {self.split_name}: kept={len(kept)} miss_hubert={miss['hubert']} "
            f"miss_w2v={miss['w2v']} miss_wavlm={miss['wavlm']} miss_qwen={miss['qwen']} "
            f"miss_fa={miss['fa']} miss_f0={miss['f0']} miss_energy={miss['energy']} "
            f"miss_gop={miss['gop']} miss_energy_dur={miss['energy_dur']} "
            f"phone_feat={self.phone_feat} word_feat={self.word_feat}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.records)

    def phone_error_flags(self) -> list[bool]:
        """True if utterance has ≥1 phone with realized≠canonical (HMamba ids)."""
        flags: list[bool] = []
        for r in self.records:
            phones = r.get("phones", []) or []
            canon = canonical_ids(phones, self.vocab)
            real = aligned_realized_ids(phones, self.vocab)
            flags.append(any(int(a) != int(b) for a, b in zip(canon, real)))
        return flags

    def _native_t(self, sid: str) -> int:
        feat = np.asarray(np.load(self.hubert_dir / self.split_name / f"{sid}.npy", mmap_mode="r"), dtype=np.float32)
        if feat.ndim != 2:
            feat = feat.reshape(feat.shape[0], -1)
        return max(int(feat.shape[0]), 1)

    def _load_norm(self, path: Path, mean: np.ndarray, std: np.ndarray, target_t: int) -> np.ndarray:
        feat = np.asarray(np.load(path), dtype=np.float32)
        if feat.ndim != 2:
            feat = feat.reshape(feat.shape[0], -1)
        feat = align_wav2vec_to_target_frames(feat, target_t)
        feat = (feat - mean) / std
        return np.clip(feat, -8.0, 8.0).astype(np.float32)

    def _load_frame_prosody(self, sid: str, t: int) -> np.ndarray:
        f0 = _load_hubert_side_cache(self.f0_cache_dir / self.split_name / f"{sid}.npz", HUBERT_F0_CACHE_KEY, t, HUBERT_F0_DIM)
        energy = _load_hubert_side_cache(
            self.energy_cache_dir / self.split_name / f"{sid}.npz", HUBERT_ENERGY_CACHE_KEY, t, HUBERT_ENERGY_DIM
        )
        if f0 is None or energy is None:
            raise RuntimeError(f"missing prosody for {sid}")
        f0 = _normalize_side(f0, self.f0_mean, self.f0_std)
        energy = _normalize_side(energy, self.energy_mean, self.energy_std)
        return np.concatenate([f0, energy], axis=1).astype(np.float32)

    def __getitem__(self, idx: int) -> Data:
        if Data is None:
            raise SystemExit("torch_geometric required")
        r = self.records[idx]
        sid = str(r["id"])
        t = self._native_t(sid)
        hub = self._load_norm(self.hubert_dir / self.split_name / f"{sid}.npy", self.hubert_mean, self.hubert_std, t)
        w2v = self._load_norm(self.w2v_dir / self.split_name / f"{sid}.npy", self.w2v_mean, self.w2v_std, t)
        wlm = self._load_norm(self.wavlm_dir / self.split_name / f"{sid}.npy", self.wavlm_mean, self.wavlm_std, t)
        model_feats = [hub, w2v, wlm]
        qwen_feat = None
        if self.qwen_dir is not None:
            qwen_feat = self._load_norm(
                self.qwen_dir / self.split_name / f"{sid}.npy",
                self.qwen_mean,  # type: ignore[arg-type]
                self.qwen_std,  # type: ignore[arg-type]
                t,
            )
            if self.qwen_fusion == "concat":
                model_feats.append(qwen_feat)

        fa = load_speechocean_fa_masks(self.fa_cache_dir / self.split_name / f"{sid}.npz", t)
        if fa is None:
            raise RuntimeError(f"missing FA for {sid}")
        mask_phones, mask_words, pwi, _labels = fa
        phones = r.get("phones", []) or []
        words = r.get("words", []) or []
        n_phones = len(phones)
        n_words = len(words)
        if mask_phones.shape[0] != n_phones or mask_words.shape[0] != n_words:
            # trust FA masks; trim labels if needed
            n_phones = int(mask_phones.shape[0])
            n_words = int(mask_words.shape[0])
            phones = phones[:n_phones]
            words = words[:n_words]
            pwi = pwi[:n_phones]

        # Wider FA windows: catch boundary shift / sub acoustics under canonical slots.
        # fa_expand_frames: all phones/words, train+test (no label needed → no leakage).
        if self.fa_expand_frames > 0:
            mask_phones = expand_binary_masks(mask_phones, self.fa_expand_frames)
            mask_words = expand_binary_masks(mask_words, self.fa_expand_frames)
        # Train-only: further dilate GT-error phone slots (test keeps unexpanded-or-uniform).
        if self.fa_expand_err_train > 0 and self.split_name == "train":
            canon_tmp = canonical_ids(phones, self.vocab)
            real_tmp = aligned_realized_ids(phones, self.vocab)
            err_idx = [i for i in range(n_phones) if canon_tmp[i] != real_tmp[i]]
            mask_phones = expand_binary_masks_at(mask_phones, err_idx, self.fa_expand_err_train)

        edges, etypes, node_types = build_english_phone_word_edges(
            n_phones, n_words, pwi, bidirectional=True, coword_edges=self.coword_edges
        )
        edge_index_np, edge_type_np = edges_to_tensors(edges, etypes)
        n_nodes = n_phones + n_words

        # node FA masks: phones + words only
        node_pool = np.concatenate([mask_phones, mask_words], axis=0)

        pad = self.vocab["<pad>"]
        ref_ids = []
        phone_canon = []
        phone_real = []
        phone_apa = []
        phone_mdd = []
        canon = canonical_ids(phones, self.vocab)
        real = aligned_realized_ids(phones, self.vocab)
        for i, p in enumerate(phones):
            pid = canon[i]
            ref_ids.append(pid)
            phone_canon.append(pid)  # text-prompt / canonical (classic MDD ref)
            phone_real.append(real[i])  # MDD target = aligned realized (keeps <del>)
            # GOPT scale: phone score stays in [0, 2]
            acc = float(p.get("accuracy", 0.0))
            phone_apa.append(acc)
            if self.detect_label == "mismatch":
                phone_mdd.append(1 if int(real[i]) != int(canon[i]) else 0)
            else:
                # default: accuracy<2 (soft error from human scores)
                phone_mdd.append(1 if acc < 2.0 else 0)
        word_acc = []
        word_stress = []
        word_total = []
        for w in words:
            ref_ids.append(pad)
            # GOPT scale: word scores /5 → [0, 2]
            word_acc.append(float(w.get("accuracy", 0.0)) / 5.0)
            word_stress.append(float(w.get("stress", 0.0)) / 5.0)
            word_total.append(float(w.get("total", 0.0)) / 5.0)

        # position / duration
        position = np.zeros((n_nodes, 1), dtype=np.float32)
        duration_ratio = np.zeros((n_nodes, 1), dtype=np.float32)
        for i in range(n_phones):
            position[i, 0] = (i + 0.5) / max(n_phones, 1)
            duration_ratio[i, 0] = float(mask_phones[i].sum()) / max(t, 1)
        for j in range(n_words):
            position[n_phones + j, 0] = (j + 0.5) / max(n_words, 1)
            duration_ratio[n_phones + j, 0] = float(mask_words[j].sum()) / max(t, 1)

        # frame segment labels: phone-covered → 1, else word-covered → 2, else 0
        seg = np.zeros(t, dtype=np.int64)
        any_phone = mask_phones.any(axis=0) if n_phones > 0 else np.zeros(t, dtype=bool)
        any_word = mask_words.any(axis=0) if n_words > 0 else np.zeros(t, dtype=bool)
        seg[any_word] = 2
        seg[any_phone] = 1

        concat_feat = np.concatenate(model_feats, axis=1)
        frame_prosody = self._load_frame_prosody(sid, t)

        data = Data(
            x=torch.zeros(n_nodes, 1, dtype=torch.float32),
            edge_index=torch.tensor(edge_index_np, dtype=torch.long),
            edge_type=torch.tensor(edge_type_np, dtype=torch.long),
        )
        data.frame_feat = torch.tensor(concat_feat, dtype=torch.float32)
        if qwen_feat is not None and self.qwen_fusion in ("film", "xattn"):
            data.frame_qwen = torch.tensor(qwen_feat, dtype=torch.float32)
        data.frame_prosody = torch.tensor(frame_prosody, dtype=torch.float32)
        data.seg_label_frames = torch.tensor(seg, dtype=torch.long)
        data.n_time_frames = torch.tensor(t, dtype=torch.long)
        data.node_type = torch.tensor(node_types, dtype=torch.long)
        data.ref_token_id = torch.tensor(ref_ids, dtype=torch.long)
        data.position = torch.tensor(position, dtype=torch.float32)
        data.duration_ratio = torch.tensor(duration_ratio, dtype=torch.float32)
        data.node_pool_masks = torch.tensor(node_pool, dtype=torch.bool)  # custom collate

        data.phone_canon_id = torch.tensor(phone_canon, dtype=torch.long)
        data.phone_real_id = torch.tensor(phone_real, dtype=torch.long)
        data.phone_mdd = torch.tensor(phone_mdd, dtype=torch.long)
        data.phone_apa = torch.tensor(phone_apa, dtype=torch.float32)
        data.word_acc = torch.tensor(word_acc, dtype=torch.float32)
        data.word_stress = torch.tensor(word_stress, dtype=torch.float32)
        data.word_total = torch.tensor(word_total, dtype=torch.float32)
        # Sentence / utterance APA (SO762 0–10) → /5 like word scores → [0, 2]
        sent = r.get("sentence") or {}
        data.utt_acc = torch.tensor(float(sent.get("accuracy", 0.0)) / 5.0, dtype=torch.float32)
        data.utt_comp = torch.tensor(float(sent.get("completeness", 0.0)) / 5.0, dtype=torch.float32)
        data.utt_flu = torch.tensor(float(sent.get("fluency", 0.0)) / 5.0, dtype=torch.float32)
        data.utt_pros = torch.tensor(float(sent.get("prosodic", 0.0)) / 5.0, dtype=torch.float32)
        data.utt_total = torch.tensor(float(sent.get("total", 0.0)) / 5.0, dtype=torch.float32)
        data.n_phones = torch.tensor([n_phones], dtype=torch.long)
        data.n_words = torch.tensor([n_words], dtype=torch.long)
        data._utt_id = sid  # collate only (string)
        if self.need_gop:
            g = np.asarray(self.gop_map[sid], dtype=np.float32)
            if self.gop_mean is not None and self.gop_std is not None:
                g = (g - self.gop_mean) / self.gop_std
                g = np.clip(g, -8.0, 8.0).astype(np.float32)
            if g.shape[0] != n_phones:
                raise RuntimeError(f"GOP/phone mismatch {sid}: {g.shape[0]} vs {n_phones}")
            if self.phone_feat == "gop":
                data.phone_gop = torch.tensor(g, dtype=torch.float32)
            if self.word_feat == "gop":
                # Mean-pool phone GOP within each word (inherits Kaldi phone FA).
                wdim = int(g.shape[1])
                wg = np.zeros((n_words, wdim), dtype=np.float32)
                cnt = np.zeros((n_words,), dtype=np.float32)
                for i in range(n_phones):
                    wi = int(pwi[i])
                    if 0 <= wi < n_words:
                        wg[wi] += g[i]
                        cnt[wi] += 1.0
                cnt = np.maximum(cnt, 1.0)
                wg = wg / cnt[:, None]
                data.word_gop = torch.tensor(wg, dtype=torch.float32)
        if self.need_energy_dur:
            ed_full = np.asarray(self.energy_dur_map[sid], dtype=np.float32)
            if ed_full.shape[0] < n_phones:
                raise RuntimeError(
                    f"energy/phone mismatch {sid}: padded={ed_full.shape[0]} vs n_phones={n_phones}"
                )
            ed_dim = int(ed_full.shape[-1])
            if ed_dim not in (7, 8):
                raise RuntimeError(f"energy/dur dim must be 7 or 8 for {sid}: {ed_full.shape}")
            ed = ed_full[:n_phones].copy()
            if self.energy_dur_mean is not None and self.energy_dur_std is not None:
                mean = np.asarray(self.energy_dur_mean, dtype=np.float32)
                std = np.asarray(self.energy_dur_std, dtype=np.float32)
                if mean.shape[0] != ed_dim or std.shape[0] != ed_dim:
                    raise RuntimeError(
                        f"energy/dur stats dim mismatch: feat={ed_dim} mean={mean.shape} std={std.shape}"
                    )
                ed = (ed - mean) / std
                ed = np.clip(ed, -8.0, 8.0).astype(np.float32)
            data.phone_energy_dur = torch.tensor(ed, dtype=torch.float32)
            wed = np.zeros((n_words, ed_dim), dtype=np.float32)
            cnt = np.zeros((n_words,), dtype=np.float32)
            for i in range(n_phones):
                wi = int(pwi[i])
                if 0 <= wi < n_words:
                    wed[wi] += ed[i]
                    cnt[wi] += 1.0
            cnt = np.maximum(cnt, 1.0)
            wed = wed / cnt[:, None]
            data.word_energy_dur = torch.tensor(wed, dtype=torch.float32)
        return data


def collate_speechocean(data_list: list[Data]) -> Batch:
    masks = [d.node_pool_masks for d in data_list]
    utt_ids = [str(getattr(d, "_utt_id", f"utt{i}")) for i, d in enumerate(data_list)]
    for d in data_list:
        if hasattr(d, "node_pool_masks"):
            delattr(d, "node_pool_masks")
        if hasattr(d, "_utt_id"):
            delattr(d, "_utt_id")
    batch = Batch.from_data_list(data_list)
    batch.node_pool_masks = masks
    batch.utt_ids = utt_ids
    return batch


def soft_apa_loss(pred: torch.Tensor, target: torch.Tensor, beta: float = 0.05) -> torch.Tensor:
    if pred.numel() == 0:
        return pred.new_zeros(())
    return F.smooth_l1_loss(pred, target, beta=beta)


def mse_apa_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.numel() == 0:
        return pred.new_zeros(())
    return F.mse_loss(pred, target)


def batch_pcc_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if pred.numel() < 2:
        return pred.new_zeros(())
    p = pred - pred.mean()
    t = target - target.mean()
    denom = p.norm() * t.norm() + 1e-8
    return 1.0 - (p * t).sum() / denom


def mdd_realized_ce(
    logits: torch.Tensor,
    real_id: torch.Tensor,
    canon_id: torch.Tensor,
    error_weight: float = 1.0,
) -> torch.Tensor:
    """Phone CE vs realized; upweight positions where realized ≠ canonical (true errors)."""
    ce = F.cross_entropy(logits, real_id, reduction="none")
    if float(error_weight) <= 1.0:
        return ce.mean()
    w = torch.ones_like(ce)
    w = w + (float(error_weight) - 1.0) * (real_id != canon_id).to(dtype=ce.dtype)
    return (ce * w).mean()


def mdd_dexent(
    logits: torch.Tensor,
    real_id: torch.Tensor,
    canon_id: torch.Tensor,
    *,
    alpha: float = 0.7,
) -> torch.Tensor:
    """HMamba-style decoupled CE: L = L_cor + (N_cor/N_mis)^α * L_mis (group means)."""
    ce = F.cross_entropy(logits, real_id, reduction="none")
    mis = real_id != canon_id
    cor = ~mis
    if bool(cor.any()):
        loss_cor = ce[cor].mean()
    else:
        loss_cor = ce.new_zeros(())
    if bool(mis.any()):
        loss_mis = ce[mis].mean()
    else:
        loss_mis = ce.new_zeros(())
    n_cor = cor.sum().to(dtype=ce.dtype).clamp(min=1.0)
    n_mis = mis.sum().to(dtype=ce.dtype).clamp(min=1.0)
    w_mis = (n_cor / n_mis) ** float(alpha)
    return loss_cor + w_mis * loss_mis


def mdd_group_ce(
    logits: torch.Tensor,
    real_id: torch.Tensor,
    canon_id: torch.Tensor,
    *,
    mis_weight: float = 1.0,
) -> torch.Tensor:
    """Decoupled CE without HMamba α: L = L_cor + λ * L_mis (group means, fixed λ)."""
    ce = F.cross_entropy(logits, real_id, reduction="none")
    mis = real_id != canon_id
    cor = ~mis
    if bool(cor.any()):
        loss_cor = ce[cor].mean()
    else:
        loss_cor = ce.new_zeros(())
    if bool(mis.any()):
        loss_mis = ce[mis].mean()
    else:
        loss_mis = ce.new_zeros(())
    return loss_cor + float(mis_weight) * loss_mis


def decode_phone_vs_canon(
    phone_logits: torch.Tensor,
    canon_id: torch.Tensor,
    *,
    decision_margin: float = 0.0,
    apa_score: torch.Tensor | None = None,
    apa_max_for_error: float | None = None,
) -> torch.Tensor:
    """Single-head MDD decode: leave canonical only if best alt beats it by margin.

    margin=0 ≈ argmax. margin>0 more conservative (↑P/↓R). margin<0 more aggressive.
    Optional APA gate: only allow non-canon if apa_score < apa_max_for_error (phone acc scale 0–2).
    """
    alt = phone_logits.clone()
    alt.scatter_(1, canon_id.unsqueeze(1), float("-inf"))
    best_alt_score, best_alt_id = alt.max(dim=-1)
    canon_score = phone_logits.gather(1, canon_id.unsqueeze(1)).squeeze(1)
    predict_error = best_alt_score > (canon_score + float(decision_margin))
    if apa_score is not None and apa_max_for_error is not None:
        predict_error = predict_error & (apa_score < float(apa_max_for_error))
    return torch.where(predict_error, best_alt_id, canon_id)



def binary_prf(pred: list[int], ref: list[int]) -> tuple[float, float, float]:
    """Positive class = 1 (mispronounced)."""
    tp = fp = fn = 0
    for p, r in zip(pred, ref):
        if p == 1 and r == 1:
            tp += 1
        elif p == 1 and r == 0:
            fp += 1
        elif p == 0 and r == 1:
            fn += 1
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-8)
    return float(prec), float(rec), float(f1)


def decode_joint_mdd(
    detect_logits: torch.Tensor,
    phone_logits: torch.Tensor,
    canon_id: torch.Tensor,
    *,
    phone_weight: float = 1.0,
    decision_margin: float = 0.2,
) -> torch.Tensor:
    """Joint detect+diagnose: flip from canonical only if score_error > score_correct + margin."""
    eps = 1e-8
    p_error = detect_logits.softmax(dim=-1)[:, 1]
    phone_log_prob = F.log_softmax(phone_logits, dim=-1)
    canon_phone_score = phone_log_prob.gather(1, canon_id.unsqueeze(1)).squeeze(1)
    score_correct = torch.log(1.0 - p_error + eps) + float(phone_weight) * canon_phone_score

    alt_log_prob = phone_log_prob.clone()
    alt_log_prob.scatter_(1, canon_id.unsqueeze(1), float("-inf"))
    best_alt_score, best_alt_id = alt_log_prob.max(dim=-1)
    score_error = torch.log(p_error + eps) + float(phone_weight) * best_alt_score

    predict_error = score_error > (score_correct + float(decision_margin))
    return torch.where(predict_error, best_alt_id, canon_id)


def compute_batch_loss(
    out: dict[str, torch.Tensor],
    batch: Batch,
    *,
    w_mdd: float,
    w_detect: float,
    w_apa: float,
    w_utt: float = 0.0,
    apa_pcc_weight: float,
    apa_phone_loss: str,
    apa_word_loss: str,
    apa_beta: float,
    mdd_error_weight: float = 1.0,
    mdd_loss: str = "ce",
    dexent_alpha: float = 0.7,
    group_mis_weight: float = 1.0,
    scer_base_weight: float = 0.3,
    stress_loss: str = "mse",
    stress_pos_weight: float = 50.0,
    loss_log_vars: torch.Tensor | None = None,
) -> torch.Tensor:
    """Diagnosis CE + detect CE + APA (GOPT-style) + optional utterance APA."""
    loss_name = str(mdd_loss).lower()
    if loss_name == "dexent":
        loss_mdd = mdd_dexent(
            out["mdd_phone_logits"],
            batch.phone_real_id,
            batch.phone_canon_id,
            alpha=dexent_alpha,
        )
    elif loss_name == "group_ce":
        loss_mdd = mdd_group_ce(
            out["mdd_phone_logits"],
            batch.phone_real_id,
            batch.phone_canon_id,
            mis_weight=group_mis_weight,
        )
    else:
        loss_mdd = mdd_realized_ce(
            out["mdd_phone_logits"],
            batch.phone_real_id,
            batch.phone_canon_id,
            error_weight=mdd_error_weight,
        )
    loss_detect = F.cross_entropy(out["mdd_detect_logits"], batch.phone_mdd)
    phone_reg = mse_apa_loss if apa_phone_loss == "mse" else (
        lambda p, t: soft_apa_loss(p, t, apa_beta)
    )
    word_reg = mse_apa_loss if apa_word_loss == "mse" else (
        lambda p, t: soft_apa_loss(p, t, apa_beta)
    )
    loss_phn = phone_reg(out["apa_phone_score"], batch.phone_apa)

    stress_mode = str(stress_loss).lower().strip()
    if stress_mode == "bce":
        # Labels are /5 → {1.0, 2.0}; treat <1.5 as stress-error (minority, positive class).
        y_err = (batch.word_stress < 1.5).to(dtype=out["apa_word_stress"].dtype)
        pos_w = torch.tensor(
            [float(stress_pos_weight)],
            device=out["apa_word_stress"].device,
            dtype=out["apa_word_stress"].dtype,
        )
        loss_stress = F.binary_cross_entropy_with_logits(
            out["apa_word_stress"], y_err, pos_weight=pos_w
        )
        # Map P(error) → score on same scale as labels for PCC aux loss.
        stress_score = 2.0 - torch.sigmoid(out["apa_word_stress"])
    else:
        loss_stress = word_reg(out["apa_word_stress"], batch.word_stress)
        stress_score = out["apa_word_stress"]

    loss_word = (
        word_reg(out["apa_word_acc"], batch.word_acc)
        + loss_stress
        + word_reg(out["apa_word_total"], batch.word_total)
    ) / 3.0
    loss_apa = loss_phn + loss_word
    loss_pcc = (
        batch_pcc_loss(out["apa_phone_score"], batch.phone_apa)
        + (
            batch_pcc_loss(out["apa_word_acc"], batch.word_acc)
            + batch_pcc_loss(stress_score, batch.word_stress)
            + batch_pcc_loss(out["apa_word_total"], batch.word_total)
        )
        / 3.0
    )
    loss_apa_full = loss_apa + float(apa_pcc_weight) * loss_pcc
    log_vars = loss_log_vars
    if log_vars is not None:
        # Kendall uncertainty weighting: exp(-s)*L + s ; init s=-log(w0) ≈ baseline 3/1/2
        s = log_vars
        total = (
            torch.exp(-s[0]) * loss_mdd
            + s[0]
            + torch.exp(-s[1]) * loss_detect
            + s[1]
            + torch.exp(-s[2]) * loss_apa_full
            + s[2]
        )
    else:
        total = (
            float(w_mdd) * loss_mdd
            + float(w_detect) * loss_detect
            + float(w_apa) * loss_apa_full
        )
    if float(w_utt) > 0.0 and "apa_utt_acc" in out:
        loss_utt = (
            phone_reg(out["apa_utt_acc"], batch.utt_acc)
            + phone_reg(out["apa_utt_comp"], batch.utt_comp)
            + phone_reg(out["apa_utt_flu"], batch.utt_flu)
            + phone_reg(out["apa_utt_pros"], batch.utt_pros)
            + phone_reg(out["apa_utt_total"], batch.utt_total)
        ) / 5.0
        loss_utt_pcc = (
            batch_pcc_loss(out["apa_utt_acc"], batch.utt_acc)
            + batch_pcc_loss(out["apa_utt_comp"], batch.utt_comp)
            + batch_pcc_loss(out["apa_utt_flu"], batch.utt_flu)
            + batch_pcc_loss(out["apa_utt_pros"], batch.utt_pros)
            + batch_pcc_loss(out["apa_utt_total"], batch.utt_total)
        ) / 5.0
        total = total + float(w_utt) * (loss_utt + float(apa_pcc_weight) * loss_utt_pcc)
    return total


class UncertaintyLossWeights(nn.Module):
    """Learnable multi-task weights via log-variance (Kendall et al.)."""

    def __init__(self, w_mdd: float = 3.0, w_detect: float = 1.0, w_apa: float = 2.0):
        super().__init__()
        init = torch.tensor(
            [
                -math.log(max(float(w_mdd), 1e-6)),
                -math.log(max(float(w_detect), 1e-6)),
                -math.log(max(float(w_apa), 1e-6)),
            ],
            dtype=torch.float32,
        )
        self.log_vars = nn.Parameter(init)

    def effective_weights(self) -> tuple[float, float, float]:
        w = torch.exp(-self.log_vars.detach())
        return float(w[0].item()), float(w[1].item()), float(w[2].item())


def stress_pred_for_metric(pred: torch.Tensor, stress_loss: str) -> torch.Tensor:
    """Convert stress head output to label-scale scores for PCC reporting."""
    if str(stress_loss).lower().strip() == "bce":
        return 2.0 - torch.sigmoid(pred)
    return pred


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    vocab: dict[str, int] | None = None,
    out_dir: Path | None = None,
    *,
    w_mdd: float = 1.0,
    w_detect: float = 1.0,
    w_apa: float = 1.0,
    w_utt: float = 0.0,
    apa_pcc_weight: float = 0.35,
    apa_phone_loss: str = "mse",
    apa_word_loss: str = "mse",
    apa_beta: float = 0.05,
    stress_loss: str = "mse",
    stress_pos_weight: float = 50.0,
    mdd_error_weight: float = 1.0,
    mdd_loss: str = "ce",
    dexent_alpha: float = 0.7,
    group_mis_weight: float = 1.0,
    scer_base_weight: float = 0.3,
    joint_phone_weight: float = 1.0,
    joint_margin: float = 0.2,
    decode_margin: float = 0.0,
    apa_max_for_error: float | None = None,
    loss_log_vars: torch.Tensor | None = None,
) -> dict[str, float]:
    model.eval()
    phone_p: list[float] = []
    phone_r: list[float] = []
    w_acc_p: list[float] = []
    w_acc_r: list[float] = []
    w_str_p: list[float] = []
    w_str_r: list[float] = []
    w_tot_p: list[float] = []
    w_tot_r: list[float] = []
    u_acc_p: list[float] = []
    u_acc_r: list[float] = []
    u_comp_p: list[float] = []
    u_comp_r: list[float] = []
    u_flu_p: list[float] = []
    u_flu_r: list[float] = []
    u_pros_p: list[float] = []
    u_pros_r: list[float] = []
    u_tot_p: list[float] = []
    u_tot_r: list[float] = []
    det_pred: list[int] = []
    det_ref: list[int] = []
    diag_top1_hit = 0
    diag_top3_hit = 0
    diag_n = 0
    id2tok = {int(v): str(k) for k, v in (vocab or {}).items()}
    skip_ids: set[int] = set()
    if vocab:
        for name in ("<pad>", "<blank>", "sil"):
            if name in vocab:
                skip_ids.add(int(vocab[name]))

    hyp_map: dict[str, str] = {}
    ref_map: dict[str, str] = {}
    human_map: dict[str, str] = {}
    val_loss_sum = 0.0
    n_batches = 0

    for batch in loader:
        batch = batch.to(device, non_blocking=device.type == "cuda")
        out = model(batch)
        val_loss_sum += float(
            compute_batch_loss(
                out,
                batch,
                w_mdd=w_mdd,
                w_detect=w_detect,
                w_apa=w_apa,
                w_utt=w_utt,
                apa_pcc_weight=apa_pcc_weight,
                apa_phone_loss=apa_phone_loss,
                apa_word_loss=apa_word_loss,
                apa_beta=apa_beta,
                mdd_error_weight=mdd_error_weight,
                mdd_loss=mdd_loss,
                dexent_alpha=dexent_alpha,
                group_mis_weight=group_mis_weight,
                scer_base_weight=scer_base_weight,
                stress_loss=stress_loss,
                stress_pos_weight=stress_pos_weight,
                loss_log_vars=loss_log_vars,
            ).item()
        )
        n_batches += 1
        phone_p.extend(out["apa_phone_score"].detach().cpu().tolist())
        phone_r.extend(batch.phone_apa.detach().cpu().tolist())
        w_acc_p.extend(out["apa_word_acc"].detach().cpu().tolist())
        w_acc_r.extend(batch.word_acc.detach().cpu().tolist())
        w_str_p.extend(
            stress_pred_for_metric(out["apa_word_stress"], stress_loss).detach().cpu().tolist()
        )
        w_str_r.extend(batch.word_stress.detach().cpu().tolist())
        w_tot_p.extend(out["apa_word_total"].detach().cpu().tolist())
        w_tot_r.extend(batch.word_total.detach().cpu().tolist())
        if "apa_utt_acc" in out:
            u_acc_p.extend(out["apa_utt_acc"].detach().cpu().tolist())
            u_acc_r.extend(batch.utt_acc.detach().cpu().tolist())
            u_comp_p.extend(out["apa_utt_comp"].detach().cpu().tolist())
            u_comp_r.extend(batch.utt_comp.detach().cpu().tolist())
            u_flu_p.extend(out["apa_utt_flu"].detach().cpu().tolist())
            u_flu_r.extend(batch.utt_flu.detach().cpu().tolist())
            u_pros_p.extend(out["apa_utt_pros"].detach().cpu().tolist())
            u_pros_r.extend(batch.utt_pros.detach().cpu().tolist())
            u_tot_p.extend(out["apa_utt_total"].detach().cpu().tolist())
            u_tot_r.extend(batch.utt_total.detach().cpu().tolist())

        det_pred.extend(out["mdd_detect_logits"].argmax(-1).detach().cpu().tolist())
        det_ref.extend(batch.phone_mdd.detach().cpu().tolist())

        err = batch.phone_real_id != batch.phone_canon_id
        if bool(err.any()):
            logits = out["mdd_phone_logits"][err]
            real = batch.phone_real_id[err]
            top1 = logits.argmax(-1)
            diag_top1_hit += int((top1 == real).sum().item())
            top3 = logits.topk(k=min(3, logits.size(-1)), dim=-1).indices
            diag_top3_hit += int((top3 == real.unsqueeze(1)).any(dim=1).sum().item())
            diag_n += int(err.sum().item())

        hyp_ids = decode_joint_mdd(
            out["mdd_detect_logits"],
            out["mdd_phone_logits"],
            batch.phone_canon_id,
            phone_weight=joint_phone_weight,
            decision_margin=joint_margin,
        ).cpu().tolist()
        canon_list = batch.phone_canon_id.cpu().tolist()
        real_all = batch.phone_real_id.cpu().tolist()
        utt_ids = list(getattr(batch, "utt_ids", []))
        phone_off = 0
        for gid in range(int(batch.num_graphs)):
            n_p = int(((batch.batch == gid) & (batch.node_type == NODE_PHONE)).sum().item())
            hyp = hyp_ids[phone_off : phone_off + n_p]
            canon = canon_list[phone_off : phone_off + n_p]
            realized = real_all[phone_off : phone_off + n_p]
            utt = utt_ids[gid] if gid < len(utt_ids) else f"utt{gid}"
            hyp_map[utt] = ids_to_phone_str(hyp, id2tok, skip_ids)
            ref_map[utt] = ids_to_phone_str(canon, id2tok, skip_ids)
            human_map[utt] = ids_to_phone_str(realized, id2tok, skip_ids)
            phone_off += n_p

    det_p, det_r, det_f = binary_prf(det_pred, det_ref)
    metrics: dict[str, Any] = {
        "val_loss": val_loss_sum / max(n_batches, 1),
        "phone_mse": mse(phone_p, phone_r),
        "phone_pcc": pcc(phone_p, phone_r),
        "word_acc_pcc": pcc(w_acc_p, w_acc_r),
        "word_stress_pcc": pcc(w_str_p, w_str_r),
        "word_total_pcc": pcc(w_tot_p, w_tot_r),
        "utt_acc_pcc": pcc(u_acc_p, u_acc_r) if u_acc_p else 0.0,
        "utt_comp_pcc": pcc(u_comp_p, u_comp_r) if u_comp_p else 0.0,
        "utt_flu_pcc": pcc(u_flu_p, u_flu_r) if u_flu_p else 0.0,
        "utt_pros_pcc": pcc(u_pros_p, u_pros_r) if u_pros_p else 0.0,
        "utt_total_pcc": pcc(u_tot_p, u_tot_r) if u_tot_p else 0.0,
        "detect_precision": det_p,
        "detect_recall": det_r,
        "detect_f1": det_f,
        "diag_top1": diag_top1_hit / max(diag_n, 1),
        "diag_top3": diag_top3_hit / max(diag_n, 1),
        "mdd_precision": 0.0,
        "mdd_recall": 0.0,
        "mdd_f1": 0.0,
        "per": 1.0,
    }

    if out_dir is not None and hyp_map and vocab:
        work = Path(out_dir).resolve() / "mdd_eval"
        try:
            classic = run_classic_mdd(
                work_dir=work,
                hyp=hyp_map,
                ref=ref_map,
                human=human_map,
            )
            for k in ("mdd_precision", "mdd_recall", "mdd_f1", "per", "mdd_correct_diag", "mdd_der"):
                if k in classic:
                    metrics[k] = classic[k]
            metrics["mdd_eval_ok"] = classic.get("mdd_eval_ok", 0.0)
        except Exception as e:
            metrics["mdd_eval_ok"] = 0.0
            metrics["mdd_eval_err"] = str(e)
            print(f"[warn] classic MDD eval failed: {e}", flush=True)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=BASE / "data")
    parser.add_argument("--hubert-dir", type=Path, default=BASE / "features/hubert_feature")
    parser.add_argument("--wav2vec-dir", type=Path, default=BASE / "features/tencent_wav2vec2_feature")
    parser.add_argument("--wavlm-dir", type=Path, default=BASE / "features/wavlm_feature")
    parser.add_argument("--qwen-dir", type=Path, default=BASE / "features/qwen_feature_14layer")
    parser.add_argument("--qwen-fusion", type=str, default="film", choices=("concat", "film", "xattn"))
    parser.add_argument(
        "--no-qwen",
        action="store_true",
        help="ablation: disable Qwen fusion (SSL/GOP acoustics only)",
    )
    parser.add_argument("--fa-cache-dir", type=Path, default=BASE / "caches/fa_cache")
    parser.add_argument("--f0-cache-dir", type=Path, default=BASE / "caches/hubert_f0")
    parser.add_argument("--energy-cache-dir", type=Path, default=BASE / "caches/hubert_energy")
    parser.add_argument("--out-dir", type=Path, default=BASE / "exp/seed42_qwen_gop_ssl")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--decode-margin",
        type=float,
        default=0.0,
        help="single-head decode: require best_alt > canon + margin (0≈argmax)",
    )
    parser.add_argument(
        "--apa-max-for-error",
        type=float,
        default=-1.0,
        help="if >=0, only allow non-canon when apa_phone < this (0–2 scale); -1=off",
    )
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument(
        "--lr-schedule",
        type=str,
        default="none",
        choices=("none", "cosine", "warmup_cosine"),
        help="none=fixed lr; cosine=CosineAnnealing over epochs; warmup_cosine=LinearLR warmup then cosine",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=2,
        help="warmup epochs for --lr-schedule=warmup_cosine",
    )
    parser.add_argument(
        "--min-lr-ratio",
        type=float,
        default=0.1,
        help="eta_min = lr * min_lr_ratio for cosine / warmup_cosine",
    )
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cuda-device", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--w-mdd",
        type=float,
        default=3.0,
        help="weight for MDD-branch realized phone CE",
    )
    parser.add_argument(
        "--mdd-error-weight",
        type=float,
        default=1.0,
        help="per-phone CE multiplier when realized≠canonical (1=uniform; only for --mdd-loss ce)",
    )
    parser.add_argument(
        "--mdd-loss",
        type=str,
        default="ce",
        choices=("ce", "dexent", "group_ce"),
        help="ce=mean realized CE; dexent=HMamba α; group_ce=L_cor+λ*L_mis (fixed λ)",
    )
    parser.add_argument(
        "--dexent-alpha",
        type=float,
        default=0.7,
        help="deXent α: w_mis=(N_cor/N_mis)^α (HMamba best≈0.7)",
    )
    parser.add_argument(
        "--group-mis-weight",
        type=float,
        default=1.0,
        help="group_ce λ: L = L_cor + λ*L_mis (no count-ratio reweight)",
    )
    parser.add_argument(
        "--w-detect",
        type=float,
        default=1.0,
        help="weight for binary detect CE",
    )
    parser.add_argument(
        "--detect-label",
        type=str,
        default="accuracy",
        choices=("accuracy", "mismatch"),
        help="accuracy: phone_mdd=(acc<2); mismatch: phone_mdd=(real≠canon)",
    )
    parser.add_argument(
        "--joint-phone-weight",
        type=float,
        default=1.0,
        help="λ for phone logprob in joint detect+diagnose decode (start value if schedule)",
    )
    parser.add_argument(
        "--joint-margin",
        type=float,
        default=0.2,
        help="decision margin (start value if schedule)",
    )
    parser.add_argument(
        "--joint-schedule",
        type=str,
        default="none",
        choices=("none", "linear"),
        help="none=fixed λ/m; linear=epoch-wise interpolate start→end for eval/decode",
    )
    parser.add_argument(
        "--joint-phone-weight-end",
        type=float,
        default=None,
        help="λ at last epoch when --joint-schedule=linear (default=start)",
    )
    parser.add_argument(
        "--joint-margin-end",
        type=float,
        default=None,
        help="margin at last epoch when --joint-schedule=linear (default=start)",
    )
    parser.add_argument(
        "--loss-schedule",
        type=str,
        default="none",
        choices=("none", "linear"),
        help="none=fixed w_mdd/w_detect/w_apa; linear=epoch-wise interpolate start→end",
    )
    parser.add_argument(
        "--w-mdd-end",
        type=float,
        default=None,
        help="w_mdd at last epoch when --loss-schedule=linear (default=start)",
    )
    parser.add_argument(
        "--w-detect-end",
        type=float,
        default=None,
        help="w_detect at last epoch when --loss-schedule=linear (default=start)",
    )
    parser.add_argument(
        "--w-apa-end",
        type=float,
        default=None,
        help="w_apa at last epoch when --loss-schedule=linear (default=start)",
    )
    parser.add_argument(
        "--error-oversample",
        type=float,
        default=1.0,
        help="train sampler weight for utts with realized≠canonical (≥1; 1=uniform, 3≈3× error utts)",
    )
    parser.add_argument(
        "--fa-expand-frames",
        type=int,
        default=0,
        help="dilate all phone/word FA masks by ±N frames (train+test; no GT needed)",
    )
    parser.add_argument(
        "--fa-expand-err-train",
        type=int,
        default=0,
        help="train-only: extra ±N dilation on GT-error phone masks (test unchanged by this flag)",
    )
    parser.add_argument(
        "--fa-soft-pool",
        action="store_true",
        help="learnable FA soft window (GOP-conditioned δ/σ) instead of hard mask pool",
    )
    parser.add_argument(
        "--scer",
        action="store_true",
        help="SCER: APA-score (+entropy) residual refine on single-head MDD logits",
    )
    parser.add_argument(
        "--scer-base-weight",
        type=float,
        default=0.3,
        help="aux CE weight on unrefined base MDD logits when --scer",
    )
    parser.add_argument(
        "--mismatch-prop",
        action="store_true",
        help="MDD-branch mismatch-conditioned edge messages + within-word COWORD edges",
    )
    parser.add_argument(
        "--anti-copy-gate",
        action="store_true",
        help="model-side gate: suppress canonical logit from match features (no extra loss)",
    )
    parser.add_argument("--w-apa", type=float, default=2.0)
    parser.add_argument(
        "--loss-weight-mode",
        type=str,
        default="fixed",
        choices=("fixed", "uncertainty"),
        help="fixed=use --w-mdd/--w-detect/--w-apa; uncertainty=learnable Kendall log-vars "
        "initialized from those weights",
    )
    parser.add_argument(
        "--w-utt",
        type=float,
        default=0.0,
        help="weight for utterance-level APA (5 sentence scores); 0=off",
    )
    parser.add_argument("--apa-beta", type=float, default=0.05)
    parser.add_argument("--apa-pcc-weight", type=float, default=0.35)
    parser.add_argument(
        "--stress-loss",
        type=str,
        default="mse",
        choices=("mse", "bce"),
        help="word stress: mse=regression; bce=binary stress-error with pos_weight (minority=raw5)",
    )
    parser.add_argument(
        "--stress-pos-weight",
        type=float,
        default=50.0,
        help="BCE pos_weight for stress-error class (raw stress=5); ~n_ok/n_err≈110 on SO762",
    )
    parser.add_argument(
        "--apa-phone-loss",
        type=str,
        default="mse",
        choices=("smooth_l1", "mse"),
        help="phone APA regression loss; use mse to match GOPT",
    )
    parser.add_argument(
        "--apa-word-loss",
        type=str,
        default="mse",
        choices=("smooth_l1", "mse"),
        help="word APA regression loss",
    )
    parser.add_argument("--early-stop-patience", type=int, default=10)
    parser.add_argument("--early-stop-min-epochs", type=int, default=12)
    parser.add_argument(
        "--best-metric",
        type=str,
        default="mdd_f1",
        choices=("val_loss", "mdd_f1", "phone_mse", "per", "min_pr"),
        help="checkpoint / early-stop: val_loss↓, mdd_f1↑, phone_mse↓, per↓, or min(P,R)↑",
    )
    parser.add_argument(
        "--phone-vocab",
        type=Path,
        default=BASE / "resource/vocab_merge.json",
        help="HMamba vocab_merge.json (48 phones + <unk>/<pad>)",
    )
    parser.add_argument(
        "--phone-feat",
        type=str,
        default="gop",
        choices=("ssl", "gop"),
        help="ssl=FA-pooled SSL for phones; gop=official Kaldi GOP 84-d for phones",
    )
    parser.add_argument(
        "--word-feat",
        type=str,
        default="gop",
        choices=("ssl", "gop"),
        help="ssl=FA-pooled SSL for words; gop=mean of phone GOP within each word",
    )
    parser.add_argument(
        "--ssl-fuse",
        type=str,
        default="film",
        choices=("overwrite", "film", "xattn", "none"),
        help="overwrite=FA-pool SSL then GOP; film=GOP+SSL FiLM; "
        "xattn=GOP then cross-attn SSL/Qwen; none=GOP+optional Qwen FiLM only",
    )
    parser.add_argument(
        "--film-order",
        type=str,
        default="qwen_gop_ssl",
        choices=("ssl_qwen", "qwen_ssl", "ssl_gop_qwen", "qwen_gop_ssl", "ssl_qwen_gop_cat"),
        help="when ssl-fuse=film: ssl_qwen=GOP→SSL→Qwen; qwen_ssl=GOP→Qwen→SSL (best); "
        "ssl_gop_qwen=SSL→GOP→Qwen; qwen_gop_ssl=Qwen→GOP→SSL; "
        "ssl_qwen_gop_cat=SSL→FiLM(concat(Qwen,GOP))",
    )
    parser.add_argument(
        "--gop-concat-energy-dur",
        action="store_true",
        help="concat seq_data energy(7)+dur(1) onto phone/word GOP → 92-d node feat",
    )
    parser.add_argument(
        "--gop-concat-dur",
        action="store_true",
        help="concat seq_data dur(1) onto phone/word GOP → 85-d node feat",
    )
    parser.add_argument(
        "--energy-dur-film",
        action="store_true",
        help="FiLM-modulate phone/word nodes with GOPT energy(7); position via --energy-film-pos",
    )
    parser.add_argument(
        "--apa-energy-dur",
        type=str,
        default="none",
        choices=("none", "stress", "apa_film"),
        help="APA-side energy(7)+dur(1): none=off; stress=concat into stress head only; "
        "apa_film=FiLM APA branch only (MDD untouched)",
    )
    parser.add_argument(
        "--energy-film-pos",
        type=str,
        default="after",
        choices=("after", "before_ssl"),
        help="after=SSL→Qwen→energy; before_ssl=GOP→energy→SSL→Qwen",
    )
    parser.add_argument("--smoke", action="store_true", help="one-batch forward only")
    args = parser.parse_args()

    if args.no_qwen:
        args.qwen_dir = None  # type: ignore[assignment]

    if Batch is None:
        raise SystemExit("torch_geometric required")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    train_records = read_jsonl(args.data_root / "train" / "labels.jsonl")
    test_records = read_jsonl(args.data_root / "test" / "labels.jsonl")
    vocab = load_hmamba_vocab(Path(args.phone_vocab))
    print(
        f"[vocab] hmamba phones={len(vocab)} path={args.phone_vocab} best_metric={args.best_metric}",
        flush=True,
    )

    hubert_mean, hubert_std = compute_acoustic_stats_native(train_records, args.hubert_dir / "train")
    w2v_mean, w2v_std = compute_acoustic_stats_native(train_records, args.wav2vec_dir / "train")
    wavlm_mean, wavlm_std = compute_acoustic_stats_native(train_records, args.wavlm_dir / "train")
    if args.qwen_dir is not None:
        qwen_mean, qwen_std = compute_acoustic_stats_native(train_records, args.qwen_dir / "train")
        qwen_dim = int(qwen_mean.shape[0])
    else:
        qwen_mean = qwen_std = None
        qwen_dim = 0

    if args.qwen_dir is not None and args.qwen_fusion == "concat":
        feat_dims = (int(hubert_mean.shape[0]), int(w2v_mean.shape[0]), int(wavlm_mean.shape[0]), qwen_dim)
    else:
        feat_dims = (int(hubert_mean.shape[0]), int(w2v_mean.shape[0]), int(wavlm_mean.shape[0]))
    print(
        f"[feat] fusion={args.qwen_fusion} dims={feat_dims} qwen={qwen_dim} no_qwen={int(args.no_qwen)}",
        flush=True,
    )

    f0_mean, f0_std = compute_hubert_side_stats(
        train_records, args.f0_cache_dir, HUBERT_F0_CACHE_KEY, HUBERT_F0_DIM, args.hubert_dir, "train"
    )
    energy_mean, energy_std = compute_hubert_side_stats(
        train_records, args.energy_cache_dir, HUBERT_ENERGY_CACHE_KEY, HUBERT_ENERGY_DIM, args.hubert_dir, "train"
    )

    gop_train = gop_test = {}
    gop_mean = gop_std = None
    phone_gop_dim = 0
    if args.phone_feat == "gop" or args.word_feat == "gop":
        print("[gop] loading official librispeech GOP maps...", flush=True)
        gop_train = load_gop_utt_map("train")
        gop_test = load_gop_utt_map("test")
        if args.gop_concat_energy_dur and args.gop_concat_dur:
            raise SystemExit("use only one of --gop-concat-energy-dur / --gop-concat-dur")
        if args.gop_concat_energy_dur:
            print("[gop] concat seq energy(7)+dur(1) onto GOP...", flush=True)
            gop_train = concat_gop_with_energy_dur(gop_train, load_seq_energy_dur_map("train"))
            gop_test = concat_gop_with_energy_dur(gop_test, load_seq_energy_dur_map("test"))
        elif args.gop_concat_dur:
            print("[gop] concat seq dur(1) onto GOP → 85-d...", flush=True)
            gop_train = concat_gop_with_energy_dur(gop_train, load_seq_dur_map("train"))
            gop_test = concat_gop_with_energy_dur(gop_test, load_seq_dur_map("test"))
        gop_mean, gop_std = compute_gop_stats(gop_train)
        phone_gop_dim = int(next(iter(gop_train.values())).shape[1])
        print(
            f"[gop] train_utts={len(gop_train)} test_utts={len(gop_test)} dim={phone_gop_dim} "
            f"phone_feat={args.phone_feat} word_feat={args.word_feat} "
            f"gop_concat_energy_dur={int(args.gop_concat_energy_dur)} "
            f"gop_concat_dur={int(args.gop_concat_dur)}",
            flush=True,
        )

    ed_train = ed_test = {}
    ed_mean = ed_std = None
    apa_ed = str(args.apa_energy_dur).lower().strip()
    if apa_ed != "none":
        # Prefer full energy(7)+dur(1) for APA-side use.
        print(f"[apa_ed] loading GOPT energy(7)+dur(1) for apa_energy_dur={apa_ed}...", flush=True)
        ed_train = load_seq_energy_dur_map("train", padded=True)
        ed_test = load_seq_energy_dur_map("test", padded=True)
        phone_len_train = {
            str(r["id"]): len(r.get("phones", []) or [])
            for r in train_records
            if r.get("id")
        }
        ed_mean, ed_std = compute_energy_film_stats(ed_train, phone_len_train)
        print(
            f"[apa_ed] train_utts={len(ed_train)} test_utts={len(ed_test)} "
            f"dim={int(ed_mean.shape[0])} mode={apa_ed}",
            flush=True,
        )
    elif args.energy_dur_film:
        print("[energy_film] loading GOPT energy(7) for FiLM (no dur)...", flush=True)
        ed_train = load_seq_energy_map("train")
        ed_test = load_seq_energy_map("test")
        # phone lengths from labels (same as GOP alignment)
        phone_len_train = {
            str(r["id"]): len(r.get("phones", []) or [])
            for r in train_records
            if r.get("id")
        }
        ed_mean, ed_std = compute_energy_film_stats(ed_train, phone_len_train)
        print(
            f"[energy_film] train_utts={len(ed_train)} test_utts={len(ed_test)} dim=7 film=1",
            flush=True,
        )

    common_kw = dict(
        vocab=vocab,
        hubert_dir=args.hubert_dir,
        hubert_mean=hubert_mean,
        hubert_std=hubert_std,
        w2v_dir=args.wav2vec_dir,
        w2v_mean=w2v_mean,
        w2v_std=w2v_std,
        wavlm_dir=args.wavlm_dir,
        wavlm_mean=wavlm_mean,
        wavlm_std=wavlm_std,
        fa_cache_dir=args.fa_cache_dir,
        f0_cache_dir=args.f0_cache_dir,
        energy_cache_dir=args.energy_cache_dir,
        f0_mean=f0_mean,
        f0_std=f0_std,
        energy_mean=energy_mean,
        energy_std=energy_std,
        qwen_dir=args.qwen_dir,
        qwen_mean=qwen_mean,
        qwen_std=qwen_std,
        qwen_fusion=args.qwen_fusion,
        phone_feat=args.phone_feat,
        word_feat=args.word_feat,
        gop_mean=gop_mean,
        gop_std=gop_std,
        energy_dur_mean=ed_mean,
        energy_dur_std=ed_std,
        coword_edges=bool(args.mismatch_prop),
        fa_expand_frames=int(args.fa_expand_frames),
        fa_expand_err_train=int(args.fa_expand_err_train),
        detect_label=str(args.detect_label),
    )
    train_ds = SpeechoceanGraphDataset(
        args.data_root / "train", gop_map=gop_train, energy_dur_map=ed_train, **common_kw
    )
    test_ds = SpeechoceanGraphDataset(
        args.data_root / "test", gop_map=gop_test, energy_dur_map=ed_test, **common_kw
    )
    if len(train_ds) == 0 or len(test_ds) == 0:
        raise SystemExit("empty dataset — wait for prosody / check feature paths")

    train_sampler = None
    train_shuffle = True
    if float(args.error_oversample) > 1.0:
        err_flags = train_ds.phone_error_flags()
        n_err = sum(1 for x in err_flags if x)
        w_err = float(args.error_oversample)
        weights = [w_err if f else 1.0 for f in err_flags]
        train_sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(train_ds),
            replacement=True,
        )
        train_shuffle = False
        print(
            f"[sample] error_oversample={w_err} err_utts={n_err}/{len(train_ds)} "
            f"({100.0 * n_err / max(len(train_ds), 1):.1f}%)",
            flush=True,
        )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=train_shuffle,
        sampler=train_sampler,
        num_workers=args.num_workers,
        collate_fn=collate_speechocean,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_speechocean,
        pin_memory=device.type == "cuda",
    )

    model = EnglishPhoneWordGraphModel(
        feat_dims=feat_dims,
        vocab_size=len(vocab),
        qwen_dim=(qwen_dim if (args.qwen_dir is not None and args.qwen_fusion != "concat") else None),
        qwen_fusion=args.qwen_fusion,
        hidden_dim=args.hidden_dim,
        prosody_dim=PROSODY_F0_ENERGY_DIM,
        phone_gop_dim=phone_gop_dim,
        ssl_fuse=args.ssl_fuse,
        mismatch_prop=bool(args.mismatch_prop),
        fa_soft_pool=bool(args.fa_soft_pool),
        energy_dur_film=bool(args.energy_dur_film),
        energy_dur_dim=7,
        energy_film_pos=str(args.energy_film_pos),
        film_order=str(args.film_order),
        apa_energy_dur=str(args.apa_energy_dur),
        apa_energy_dur_dim=(int(ed_mean.shape[0]) if ed_mean is not None and apa_ed != "none" else 8),
        utt_apa=float(args.w_utt) > 0.0,
    ).to(device)

    if args.smoke:
        batch = next(iter(train_loader)).to(device)
        out = model(batch)
        print({k: tuple(v.shape) for k, v in out.items()}, flush=True)
        loss_mdd = F.cross_entropy(out["mdd_phone_logits"], batch.phone_real_id)
        print(f"[smoke] mdd_loss={float(loss_mdd.item()):.4f}", flush=True)
        metrics = evaluate(
            model,
            test_loader,
            device,
            vocab=vocab,
            out_dir=args.out_dir,
            w_mdd=args.w_mdd,
            w_detect=args.w_detect,
            w_apa=args.w_apa,
            w_utt=args.w_utt,
            apa_pcc_weight=args.apa_pcc_weight,
            apa_phone_loss=args.apa_phone_loss,
            apa_word_loss=args.apa_word_loss,
            apa_beta=args.apa_beta,
            stress_loss=args.stress_loss,
            stress_pos_weight=args.stress_pos_weight,
            mdd_error_weight=args.mdd_error_weight,
            mdd_loss=args.mdd_loss,
            dexent_alpha=args.dexent_alpha,
            group_mis_weight=args.group_mis_weight,
            scer_base_weight=args.scer_base_weight,
            joint_phone_weight=args.joint_phone_weight,
            joint_margin=args.joint_margin,
            decode_margin=args.decode_margin,
            apa_max_for_error=(
                None if float(args.apa_max_for_error) < 0 else float(args.apa_max_for_error)
            ),
        )
        print(
            f"[smoke] val_loss={metrics.get('val_loss', 0):.4f} "
            f"classic MDD P={metrics.get('mdd_precision', 0):.3f} "
            f"R={metrics.get('mdd_recall', 0):.3f} F1={metrics.get('mdd_f1', 0):.3f} "
            f"PER={metrics['per']:.4f} phone_mse={metrics['phone_mse']:.4f} "
            f"ok={metrics.get('mdd_eval_ok', 0)}",
            flush=True,
        )
        print("[smoke] OK", flush=True)
        return

    print(
        f"[train] arch=stem4→mdd4|apa4 phone_feat={args.phone_feat} word_feat={args.word_feat} "
        f"ssl_fuse={args.ssl_fuse} film_order={args.film_order} "
        f"apa_phone_loss={args.apa_phone_loss} apa_word_loss={args.apa_word_loss} "
        f"stress_loss={args.stress_loss} stress_pos_weight={args.stress_pos_weight} "
        f"w_mdd={args.w_mdd} w_detect={args.w_detect} detect_label={args.detect_label} "
        f"mdd_loss={args.mdd_loss} dexent_alpha={args.dexent_alpha} "
        f"group_mis_weight={args.group_mis_weight} "
        f"mdd_error_weight={args.mdd_error_weight} "
        f"joint_phone_weight={args.joint_phone_weight} joint_margin={args.joint_margin} "
        f"joint_schedule={args.joint_schedule} "
        f"joint_phone_weight_end={args.joint_phone_weight_end} "
        f"joint_margin_end={args.joint_margin_end} "
        f"loss_schedule={args.loss_schedule} "
        f"w_mdd_end={args.w_mdd_end} w_detect_end={args.w_detect_end} w_apa_end={args.w_apa_end} "
        f"error_oversample={args.error_oversample} "
        f"fa_expand_frames={args.fa_expand_frames} fa_expand_err_train={args.fa_expand_err_train} "
        f"fa_soft_pool={int(args.fa_soft_pool)} energy_dur_film={int(args.energy_dur_film)} "
        f"apa_energy_dur={args.apa_energy_dur} "
        f"energy_film_pos={args.energy_film_pos} "
        f"scer={int(args.scer)} scer_base_weight={args.scer_base_weight} "
        f"decode_margin={args.decode_margin} apa_max_for_error={args.apa_max_for_error} "
        f"anti_copy_gate={int(args.anti_copy_gate)} "
        f"w_apa={args.w_apa} w_utt={args.w_utt} utt_apa={int(float(args.w_utt) > 0)} "
        f"loss_weight_mode={args.loss_weight_mode} "
        f"apa_pcc_weight={args.apa_pcc_weight} "
        f"lr={args.lr} lr_schedule={args.lr_schedule} warmup_epochs={args.warmup_epochs} "
        f"min_lr_ratio={args.min_lr_ratio} "
        f"best_metric={args.best_metric}",
        flush=True,
    )
    unc_weights: UncertaintyLossWeights | None = None
    if str(args.loss_weight_mode) == "uncertainty":
        unc_weights = UncertaintyLossWeights(
            w_mdd=float(args.w_mdd),
            w_detect=float(args.w_detect),
            w_apa=float(args.w_apa),
        ).to(device)
        wm0, wd0, wa0 = unc_weights.effective_weights()
        print(
            f"[loss_weight] mode=uncertainty init_w=({wm0:.3f},{wd0:.3f},{wa0:.3f}) "
            f"from fixed ({args.w_mdd},{args.w_detect},{args.w_apa})",
            flush=True,
        )
        opt = torch.optim.AdamW(
            list(model.parameters()) + list(unc_weights.parameters()),
            lr=args.lr,
            weight_decay=1e-4,
        )
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    eta_min = float(args.lr) * float(args.min_lr_ratio)
    warm_ep = max(0, int(args.warmup_epochs))
    if str(args.lr_schedule) == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(1, int(args.epochs)), eta_min=eta_min
        )
    elif str(args.lr_schedule) == "warmup_cosine":
        warm_ep = min(warm_ep, max(0, int(args.epochs) - 1))
        if warm_ep <= 0:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, int(args.epochs)), eta_min=eta_min
            )
        else:
            warm = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=0.1, end_factor=1.0, total_iters=warm_ep
            )
            cos = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, int(args.epochs) - warm_ep), eta_min=eta_min
            )
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                opt, schedulers=[warm, cos], milestones=[warm_ep]
            )
    else:
        scheduler = None
    # mdd_f1 / min_pr: higher better; val_loss / phone_mse / per: lower better
    best = -1e9 if args.best_metric in ("mdd_f1", "min_pr") else 1e9

    def _epoch_frac(epoch: int) -> float:
        if int(args.epochs) <= 1:
            return 0.0
        t = (epoch - 1) / float(args.epochs - 1)
        return min(max(t, 0.0), 1.0)

    def joint_params_for_epoch(epoch: int) -> tuple[float, float]:
        lam0 = float(args.joint_phone_weight)
        m0 = float(args.joint_margin)
        if str(args.joint_schedule) != "linear":
            return lam0, m0
        lam1 = float(args.joint_phone_weight_end) if args.joint_phone_weight_end is not None else lam0
        m1 = float(args.joint_margin_end) if args.joint_margin_end is not None else m0
        t = _epoch_frac(epoch)
        return lam0 + t * (lam1 - lam0), m0 + t * (m1 - m0)

    def loss_weights_for_epoch(epoch: int) -> tuple[float, float, float]:
        w_m0 = float(args.w_mdd)
        w_d0 = float(args.w_detect)
        w_a0 = float(args.w_apa)
        if str(args.loss_schedule) != "linear":
            return w_m0, w_d0, w_a0
        w_m1 = float(args.w_mdd_end) if args.w_mdd_end is not None else w_m0
        w_d1 = float(args.w_detect_end) if args.w_detect_end is not None else w_d0
        w_a1 = float(args.w_apa_end) if args.w_apa_end is not None else w_a0
        t = _epoch_frac(epoch)
        return (
            w_m0 + t * (w_m1 - w_m0),
            w_d0 + t * (w_d1 - w_d0),
            w_a0 + t * (w_a1 - w_a0),
        )

    best_f1 = -1.0
    best_vl = 1e9
    bad = 0
    log_path = args.out_dir / "train_log.jsonl"
    log_path.write_text("", encoding="utf-8")  # fresh run: overwrite old jsonl
    (args.out_dir / "config.json").write_text(json.dumps(vars(args), default=str, indent=2), encoding="utf-8")
    (args.out_dir / "vocab.json").write_text(json.dumps(vocab, indent=2), encoding="utf-8")

    for epoch in range(1, args.epochs + 1):
        w_m_e, w_d_e, w_a_e = loss_weights_for_epoch(epoch)
        log_vars_t = unc_weights.log_vars if unc_weights is not None else None
        if unc_weights is not None:
            w_m_e, w_d_e, w_a_e = unc_weights.effective_weights()
        model.train()
        if unc_weights is not None:
            unc_weights.train()
        total_loss = 0.0
        n_batches = 0
        for batch in train_loader:
            batch = batch.to(device, non_blocking=device.type == "cuda")
            out = model(batch)
            loss = compute_batch_loss(
                out,
                batch,
                w_mdd=w_m_e,
                w_detect=w_d_e,
                w_apa=w_a_e,
                w_utt=args.w_utt,
                apa_pcc_weight=args.apa_pcc_weight,
                apa_phone_loss=args.apa_phone_loss,
                apa_word_loss=args.apa_word_loss,
                apa_beta=args.apa_beta,
                mdd_error_weight=args.mdd_error_weight,
                mdd_loss=args.mdd_loss,
                dexent_alpha=args.dexent_alpha,
                group_mis_weight=args.group_mis_weight,
                scer_base_weight=args.scer_base_weight,
                stress_loss=args.stress_loss,
                stress_pos_weight=args.stress_pos_weight,
                loss_log_vars=log_vars_t,
            )
            opt.zero_grad(set_to_none=True)
            loss.backward()
            params_to_clip = list(model.parameters())
            if unc_weights is not None:
                params_to_clip += list(unc_weights.parameters())
            nn.utils.clip_grad_norm_(params_to_clip, 5.0)
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1

        if unc_weights is not None:
            w_m_e, w_d_e, w_a_e = unc_weights.effective_weights()
            log_vars_t = unc_weights.log_vars
        lam_e, mar_e = joint_params_for_epoch(epoch)
        metrics = evaluate(
            model,
            test_loader,
            device,
            vocab=vocab,
            out_dir=args.out_dir,
            w_mdd=w_m_e,
            w_detect=w_d_e,
            w_apa=w_a_e,
            w_utt=args.w_utt,
            apa_pcc_weight=args.apa_pcc_weight,
            apa_phone_loss=args.apa_phone_loss,
            apa_word_loss=args.apa_word_loss,
            apa_beta=args.apa_beta,
            stress_loss=args.stress_loss,
            stress_pos_weight=args.stress_pos_weight,
            mdd_error_weight=args.mdd_error_weight,
            mdd_loss=args.mdd_loss,
            dexent_alpha=args.dexent_alpha,
            group_mis_weight=args.group_mis_weight,
            scer_base_weight=args.scer_base_weight,
            joint_phone_weight=lam_e,
            joint_margin=mar_e,
            decode_margin=args.decode_margin,
            apa_max_for_error=(
                None if float(args.apa_max_for_error) < 0 else float(args.apa_max_for_error)
            ),
            loss_log_vars=log_vars_t,
        )
        metrics["epoch"] = epoch
        metrics["train_loss"] = total_loss / max(n_batches, 1)
        metrics["joint_phone_weight"] = float(lam_e)
        metrics["joint_margin"] = float(mar_e)
        metrics["w_mdd"] = float(w_m_e)
        metrics["w_detect"] = float(w_d_e)
        metrics["w_apa"] = float(w_a_e)
        cur_lr = float(opt.param_groups[0]["lr"])
        metrics["lr"] = cur_lr
        # json-safe
        log_row = {k: v for k, v in metrics.items() if isinstance(v, (int, float, str, bool))}
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(log_row) + "\n")
        utt_part = ""
        if float(args.w_utt) > 0.0:
            utt_part = (
                f"utt Acc={metrics.get('utt_acc_pcc', 0):.3f} Flu={metrics.get('utt_flu_pcc', 0):.3f} "
                f"Pro={metrics.get('utt_pros_pcc', 0):.3f} Tot={metrics.get('utt_total_pcc', 0):.3f} | "
            )
        print(
            f"[ep{epoch}] train_loss={metrics['train_loss']:.4f} val_loss={metrics['val_loss']:.4f} | "
            f"lr={cur_lr:.2e} w=({w_m_e:.2f},{w_d_e:.2f},{w_a_e:.2f}) λ={lam_e:.2f} m={mar_e:.2f} | "
            f"APA phone MSE={metrics['phone_mse']:.3f} PCC={metrics['phone_pcc']:.3f} | "
            f"word Acc={metrics['word_acc_pcc']:.3f} Stress={metrics['word_stress_pcc']:.3f} "
            f"Total={metrics['word_total_pcc']:.3f} | "
            f"{utt_part}"
            f"Det P={metrics['detect_precision']:.3f} R={metrics['detect_recall']:.3f} "
            f"F1={metrics['detect_f1']:.3f} | "
            f"Diag@err top1={metrics['diag_top1']:.3f} top3={metrics['diag_top3']:.3f} | "
            f"MDD P={metrics['mdd_precision']:.3f} R={metrics['mdd_recall']:.3f} "
            f"F1={metrics['mdd_f1']:.3f} PER={metrics['per']:.3f}"
            f"{'' if metrics.get('mdd_eval_ok', 1) else ' [mdd_eval_fail]'}",
            flush=True,
        )
        ckpt = {"model": model.state_dict(), "epoch": epoch, "metrics": metrics, "vocab": vocab}
        if unc_weights is not None:
            ckpt["uncertainty_loss_weights"] = unc_weights.state_dict()
        f1 = float(metrics.get("mdd_f1", 0.0))
        vl = float(metrics["val_loss"])
        if f1 > best_f1:
            best_f1 = f1
            torch.save(ckpt, args.out_dir / "best_mdd_f1.pt")
        if vl < best_vl:
            best_vl = vl
            torch.save(ckpt, args.out_dir / "best_val_loss.pt")

        if args.best_metric == "mdd_f1":
            score = f1
            improved = score > best
        elif args.best_metric == "min_pr":
            score = min(float(metrics["mdd_precision"]), float(metrics["mdd_recall"]))
            improved = score > best
        elif args.best_metric == "per":
            score = float(metrics.get("per", 1.0))
            improved = score < best
        elif args.best_metric == "phone_mse":
            score = float(metrics["phone_mse"])
            improved = score < best
        else:  # val_loss
            score = vl
            improved = score < best
        if improved:
            best = score
            bad = 0
            torch.save(ckpt, args.out_dir / "best_model.pt")
            print(
                f"  -> new best {args.best_metric}={best:.4f} "
                f"(P={metrics['mdd_precision']:.3f} R={metrics['mdd_recall']:.3f} "
                f"F1={metrics['mdd_f1']:.3f} PER={metrics['per']:.3f} phone_mse={metrics['phone_mse']:.4f}), saved",
                flush=True,
            )
        else:
            bad += 1
        if scheduler is not None:
            scheduler.step()
        if args.early_stop_patience > 0 and epoch >= args.early_stop_min_epochs and bad >= args.early_stop_patience:
            print(f"[early-stop] best {args.best_metric}={best:.4f}", flush=True)
            break
    print(
        f"[done] best {args.best_metric}={best:.4f} "
        f"best_mdd_f1={best_f1:.4f} best_val_loss={best_vl:.4f} out={args.out_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
