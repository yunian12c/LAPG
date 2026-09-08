#!/usr/bin/env python3
"""GOP graph v102b training: stem L4 SupCon + multitask ASR/MDD/APA."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

_SRC = Path(__file__).resolve().parent
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    from torch_geometric.data import Batch, Data
except Exception:
    Batch = None  # type: ignore
    Data = None  # type: ignore

from hubert_fa_cache import load_hubert_fa_masks
from hubert_side_cache import (
    HUBERT_ENERGY_CACHE_KEY,
    HUBERT_ENERGY_DIM,
    HUBERT_F0_CACHE_KEY,
    HUBERT_F0_DIM,
    _load_hubert_side_cache,
    _normalize_side,
    compute_hubert_side_stats,
)
from phoneme_ctc_align import CLS_FINAL, CLS_INITIAL, align_initial_final, make_tone_tail_mask
from phoneme_gnn import PROSODY_F0_ENERGY_DIM, PhonemeGraphModelV102b
from phoneme_graph_edges import build_phoneme_graph_edges, edges_to_tensors
from phonological_contrast import phonological_supervised_contrast

INITIALS = ["zh", "ch", "sh", "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h", "j", "q", "x", "r", "z", "c", "s", "y", "w"]
IGNORE_INDEX = -100


# ---------------------------------------------------------------------------
# Data / vocab utilities
# ---------------------------------------------------------------------------


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                records.append(json.loads(s))
    return records


def split_pinyin(pinyin: str) -> tuple[str, str, str]:
    s = str(pinyin).strip().lower()
    tone = ""
    if s and s[-1].isdigit():
        tone = s[-1]
        s = s[:-1]
    initial = ""
    final = s
    for cand in INITIALS:
        if s.startswith(cand):
            initial = cand
            final = s[len(cand) :]
            break
    return initial, final, tone


def phoneme_tokens(pinyin: str) -> list[str]:
    initial, final, tone = split_pinyin(pinyin)
    tokens: list[str] = []
    if initial:
        tokens.append(initial)
    if final:
        tokens.append(final)
    if tone:
        tokens.append(f"tone{tone}")
    return tokens or ["<unk>"]


def pinyin_triplet_ids(pinyin: str, vocab: dict[str, int]) -> tuple[list[int], list[str], bool, bool, bool]:
    initial, final, tone = split_pinyin(pinyin)
    pad = vocab["<pad>"]
    unk = vocab["<unk>"]
    ids = [
        vocab.get(initial, unk) if initial else pad,
        vocab.get(final, unk) if final else pad,
        vocab.get(f"tone{tone}", unk) if tone else pad,
    ]
    tokens: list[str] = []
    if initial:
        tokens.append(initial)
    if final:
        tokens.append(final)
    if tone:
        tokens.append(f"tone{tone}")
    return ids, tokens, bool(initial), bool(final), bool(tone)


def build_vocab(records: list[dict[str, Any]]) -> dict[str, int]:
    vocab = {"<pad>": 0, "<unk>": 1}
    for r in records:
        for key in ["reference_pronunciation", "real_pronunciation"]:
            for tok in phoneme_tokens(str(r.get(key, ""))):
                if tok not in vocab:
                    vocab[tok] = len(vocab)
    return vocab


def find_segment(record: dict[str, Any], seg_type: str) -> dict[str, Any]:
    for seg in record.get("subword_segments", []) or []:
        if seg.get("type") == seg_type:
            return seg
    return {}


def target_dict(record: dict[str, Any]) -> dict[str, Any]:
    ini = find_segment(record, "initial")
    fin = find_segment(record, "final")
    has_ini = bool(ini)
    has_fin = bool(fin)

    def apa_from_segment(seg: dict[str, Any]) -> float:
        if seg.get("apa_score") is not None:
            return float(seg["apa_score"]) / 10.0
        return 0.0

    def mdd_from_segment(seg: dict[str, Any]) -> int:
        if seg.get("mdd_class") is not None:
            return int(seg["mdd_class"])
        return 0

    tone_mdd = record.get("tone_mdd")
    if tone_mdd is None:
        tone_mdd = record.get("char_mdd_class", 0)
    return {
        "mdd_initial": mdd_from_segment(ini) if has_ini else 0,
        "mdd_final": mdd_from_segment(fin) if has_fin else 0,
        "mdd_tone": int(tone_mdd),
        "mdd_char": int(record.get("char_mdd_class", 0)),
        "apa_initial": apa_from_segment(ini) if has_ini else 0.0,
        "apa_final": apa_from_segment(fin) if has_fin else 0.0,
        "apa_char": float(record.get("overall_score", 0.0)) / 10.0,
    }


def align_wav2vec_to_target_frames(w2v: np.ndarray, target_t: int) -> np.ndarray:
    w2v = np.asarray(w2v, dtype=np.float32)
    if w2v.ndim != 2:
        w2v = w2v.reshape(w2v.shape[0], -1)
    tw, d = w2v.shape
    if target_t <= 0:
        return np.zeros((0, d), dtype=np.float32)
    if tw == target_t:
        return w2v
    if tw <= 0:
        return np.zeros((target_t, d), dtype=np.float32)
    if tw == 1:
        return np.repeat(w2v, target_t, axis=0)
    x_old = np.linspace(0.0, 1.0, num=tw, dtype=np.float64)
    x_new = np.linspace(0.0, 1.0, num=target_t, dtype=np.float64)
    out = np.empty((target_t, d), dtype=np.float32)
    for j in range(d):
        out[:, j] = np.interp(x_new, x_old, w2v[:, j].astype(np.float64)).astype(np.float32)
    return out


def compute_acoustic_stats_native(records: list[dict[str, Any]], feature_split_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    sums = None
    sq_sums = None
    count = 0
    for r in records:
        sid = str(r.get("id", ""))
        fp = feature_split_dir / f"{sid}.npy"
        if not sid or not fp.is_file():
            continue
        feat = np.asarray(np.load(fp), dtype=np.float32)
        if feat.ndim != 2:
            feat = feat.reshape(feat.shape[0], -1)
        if sums is None:
            sums = np.zeros(feat.shape[1], dtype=np.float64)
            sq_sums = np.zeros(feat.shape[1], dtype=np.float64)
        sums += feat.sum(axis=0)
        sq_sums += np.square(feat).sum(axis=0)
        count += feat.shape[0]
    if sums is None or count == 0:
        raise RuntimeError(f"No native frames under {feature_split_dir}")
    mean = sums / count
    var = np.maximum(sq_sums / count - np.square(mean), 1e-6)
    return mean.astype(np.float32), np.sqrt(var).astype(np.float32)


# ---------------------------------------------------------------------------
# Loss / metrics
# ---------------------------------------------------------------------------


def masked_cross_entropy(logits: torch.Tensor, target: torch.Tensor, ignore_index: int = IGNORE_INDEX) -> torch.Tensor:
    mask = target != ignore_index
    if not bool(mask.any()):
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], target[mask])


def masked_criterion(criterion: nn.Module, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask = mask.bool()
    if not bool(mask.any()):
        return pred.sum() * 0.0
    return criterion(pred[mask], target[mask])


def apa_batch_pearson_one_minus_r(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if mask is not None:
        pred = pred[mask]
        target = target[mask]
    if pred.numel() < 2:
        return pred.sum() * 0.0
    p = pred.reshape(-1).float()
    t = target.reshape(-1).float()
    pm = p - p.mean()
    tm = t - t.mean()
    sp = pm.std(unbiased=False) + 1e-5
    st = tm.std(unbiased=False) + 1e-5
    r = (pm * tm).mean() / (sp * st)
    return 1.0 - r.clamp(min=-1.0, max=1.0)


def apa_pcc_auxiliary_loss(
    out: dict[str, torch.Tensor],
    batch: Batch,
    final_weight: float = 1.0,
) -> torch.Tensor:
    li = apa_batch_pearson_one_minus_r(out["apa_initial_score"], batch.apa_initial, batch.has_initial)
    lf = apa_batch_pearson_one_minus_r(out["apa_final_score"], batch.apa_final, batch.has_final)
    lc = apa_batch_pearson_one_minus_r(out["apa_char_score"], batch.apa_char, None)
    fw = max(float(final_weight), 0.0)
    denom = 2.0 + fw
    return (li + fw * lf + lc) / denom


def apa_smooth_l1_bundle(
    crit: nn.Module,
    out: dict[str, torch.Tensor],
    batch: Batch,
    final_weight: float = 1.0,
) -> torch.Tensor:
    li = masked_criterion(crit, out["apa_initial_score"], batch.apa_initial, batch.has_initial)
    lf = masked_criterion(crit, out["apa_final_score"], batch.apa_final, batch.has_final)
    lc = crit(out["apa_char_score"], batch.apa_char)
    fw = max(float(final_weight), 0.0)
    denom = 2.0 + fw
    return (li + fw * lf + lc) / denom


def apa_char_segment_consistency_loss(out: dict[str, torch.Tensor], batch: Batch) -> torch.Tensor:
    ini = out["apa_initial_score"].reshape(-1)
    fin = out["apa_final_score"].reshape(-1)
    char = out["apa_char_score"].reshape(-1)
    w_ini = batch.ini_dur_frac.reshape(-1).to(ini.dtype)
    w_fin = batch.fin_dur_frac.reshape(-1).to(ini.dtype)
    has_ini = batch.has_initial.reshape(-1).to(ini.dtype)
    denom = (has_ini * w_ini + w_fin).clamp(min=1e-6)
    seg_proxy = (has_ini * w_ini * ini + w_fin * fin) / denom
    return F.smooth_l1_loss(char, seg_proxy)


def compute_apa_loss(
    crit: nn.Module,
    out: dict[str, torch.Tensor],
    batch: Batch,
    pcc_weight: float,
    final_weight: float = 1.0,
    consistency_weight: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    smooth = apa_smooth_l1_bundle(crit, out, batch, final_weight=final_weight)
    pcc = apa_pcc_auxiliary_loss(out, batch, final_weight=final_weight)
    total = smooth + float(pcc_weight) * pcc
    cons = out["apa_initial_score"].sum() * 0.0
    cw_cons = float(consistency_weight)
    if cw_cons > 0.0:
        cons = apa_char_segment_consistency_loss(out, batch)
        total = total + cw_cons * cons
    return total, smooth, pcc, cons


def edit_distance(a: list[Any], b: list[Any]) -> int:
    dp = list(range(len(b) + 1))
    for i, x in enumerate(a, start=1):
        prev = dp[0]
        dp[0] = i
        for j, y in enumerate(b, start=1):
            old = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + int(x != y))
            prev = old
    return dp[-1]


def accuracy(preds: list[int], refs: list[int]) -> float:
    return sum(int(p == r) for p, r in zip(preds, refs)) / max(len(refs), 1)


def macro_f1(preds: list[int], refs: list[int]) -> float:
    labels = sorted(set(preds) | set(refs))
    if not labels:
        return 0.0
    vals = []
    for c in labels:
        tp = sum(int(p == c and r == c) for p, r in zip(preds, refs))
        fp = sum(int(p == c and r != c) for p, r in zip(preds, refs))
        fn = sum(int(p != c and r == c) for p, r in zip(preds, refs))
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        vals.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(vals) / len(vals)


def pcc(preds: list[float], refs: list[float]) -> float:
    if len(preds) < 2:
        return 0.0
    x = torch.tensor(preds, dtype=torch.float32)
    y = torch.tensor(refs, dtype=torch.float32)
    vx = x - x.mean()
    vy = y - y.mean()
    denom = torch.sqrt(vx.square().sum() * vy.square().sum())
    if denom.item() == 0:
        return 0.0
    return float((vx * vy).sum() / denom)


def score_metrics(metrics: dict[str, float]) -> float:
    return (
        metrics["mdd_initial_f1"]
        + metrics["mdd_final_f1"]
        + metrics["mdd_tone_f1"]
        + metrics["mdd_char_f1"]
        + metrics["apa_initial_pcc"]
        + metrics["apa_final_pcc"]
        + metrics["apa_char_pcc"]
        - metrics["asr_per"]
        - metrics["asr_cer"]
    )


def metric_for_best(metrics: dict[str, float], name: str) -> float:
    if name == "apa_char_pcc":
        return float(metrics["apa_char_pcc"])
    if name == "apa_mean_pcc":
        return float(metrics.get("apa_mean_pcc", metrics["apa_char_pcc"]))
    if name == "mdd_char_f1":
        return float(metrics["mdd_char_f1"])
    if name == "mdd_mean_f1":
        return float(
            metrics.get(
                "mdd_mean_f1",
                (
                    metrics["mdd_initial_f1"]
                    + metrics["mdd_final_f1"]
                    + metrics["mdd_tone_f1"]
                    + metrics["mdd_char_f1"]
                )
                / 4.0,
            )
        )
    return score_metrics(metrics)


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float):
        self.decay = float(decay)
        self.shadow: dict[str, torch.Tensor] = {k: v.detach().clone() for k, v in model.state_dict().items()}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for k, v in model.state_dict().items():
            v.copy_(self.shadow[k])

    @torch.no_grad()
    def restore(self, model: nn.Module) -> None:
        for k, v in model.state_dict().items():
            v.copy_(self._backup[k])

    def state_dict_for_save(self) -> dict[str, torch.Tensor]:
        return {k: v.clone() for k, v in self.shadow.items()}


# ---------------------------------------------------------------------------
# SupCon
# ---------------------------------------------------------------------------


def contra_weight_for_epoch(epoch: int, warmup_epochs: int, target_w: float) -> float:
    if target_w <= 0.0:
        return 0.0
    if warmup_epochs <= 0:
        return float(target_w)
    if epoch <= warmup_epochs:
        return float(target_w) * (epoch / warmup_epochs)
    return float(target_w)


def compute_stem_scl_loss(
    model: nn.Module,
    out: dict[str, Any],
    batch: Batch,
    pad_id: int,
    temperature: float,
) -> torch.Tensor:
    stem_traj = out["stem_traj"]  # type: ignore[index]
    stem_l4 = stem_traj[-1]
    return phonological_supervised_contrast(
        stem_l4,
        batch,
        pad_id=int(pad_id),
        temperature=float(temperature),
        proj=model.contra_proj,  # type: ignore[attr-defined]
        slot_weights=(1.0, 1.0, 0.0),
    )


# ---------------------------------------------------------------------------
# Dataset / evaluation
# ---------------------------------------------------------------------------


class PhonemeGraphDataset(Dataset):
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
        semantic_graph_mode: str = "phoneme_dag_plus",
        qwen_dir: Path | None = None,
        qwen_mean: np.ndarray | None = None,
        qwen_std: np.ndarray | None = None,
        qwen_fusion: str = "concat",
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
        self.semantic_graph_mode = semantic_graph_mode
        self.qwen_dir = qwen_dir
        self.qwen_mean = qwen_mean
        self.qwen_std = qwen_std
        self.qwen_fusion = str(qwen_fusion)
        qwen_dim = int(qwen_mean.shape[0]) if qwen_mean is not None else 0
        if qwen_dir is not None and self.qwen_fusion == "concat":
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
        miss: dict[str, int] = {"hubert": 0, "w2v": 0, "wavlm": 0, "qwen": 0, "fa": 0, "f0": 0, "energy": 0}
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
            if ok:
                kept.append(r)
        self.records = kept
        print(
            f"[dataset] {self.split_name}: kept={len(kept)} miss_hubert={miss['hubert']} "
            f"miss_w2v={miss['w2v']} miss_wavlm={miss['wavlm']} miss_qwen={miss['qwen']} "
            f"miss_fa={miss['fa']} miss_f0={miss['f0']} miss_energy={miss['energy']}",
            flush=True,
        )

    def __len__(self) -> int:
        return len(self.records)

    def _native_t(self, sid: str) -> int:
        path = self.hubert_dir / self.split_name / f"{sid}.npy"
        feat = np.asarray(np.load(path, mmap_mode="r"), dtype=np.float32)
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
        f0_path = self.f0_cache_dir / self.split_name / f"{sid}.npz"
        en_path = self.energy_cache_dir / self.split_name / f"{sid}.npz"
        f0 = _load_hubert_side_cache(f0_path, HUBERT_F0_CACHE_KEY, t, HUBERT_F0_DIM)
        energy = _load_hubert_side_cache(en_path, HUBERT_ENERGY_CACHE_KEY, t, HUBERT_ENERGY_DIM)
        if f0 is None:
            raise RuntimeError(f"missing or mismatched f0 cache: {f0_path} t={t}")
        if energy is None:
            raise RuntimeError(f"missing or mismatched energy cache: {en_path} t={t}")
        f0 = _normalize_side(f0, self.f0_mean, self.f0_std)
        energy = _normalize_side(energy, self.energy_mean, self.energy_std)
        return np.concatenate([f0, energy], axis=1).astype(np.float32)

    def __getitem__(self, idx: int) -> Data:
        if Data is None:
            raise SystemExit("torch_geometric required")
        r = self.records[idx]
        sid = str(r.get("id"))
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

        ref_ids, _, has_ref_initial, has_ref_final, has_ref_tone = pinyin_triplet_ids(
            str(r.get("reference_pronunciation", "")), self.vocab
        )
        real_ids, real_tokens, has_real_initial, has_real_final, has_real_tone = pinyin_triplet_ids(
            str(r.get("real_pronunciation", "")), self.vocab
        )

        cache_path = self.fa_cache_dir / self.split_name / f"{sid}.npz"
        cached = load_hubert_fa_masks(cache_path, t)
        if cached is None:
            _, mask_ini, mask_fin = align_initial_final(
                np.concatenate(model_feats, axis=1),
                str(r.get("reference_pronunciation", "")),
                mode="duration",
                split_pinyin_fn=split_pinyin,
            )
        else:
            mask_ini, mask_fin = cached

        mask_tone = make_tone_tail_mask(mask_fin) if self.semantic_graph_mode == "phoneme_dag_plus" else None
        full = np.ones(t, dtype=bool)

        concat_feat = np.concatenate(model_feats, axis=1)
        frame_prosody = self._load_frame_prosody(sid, t)
        pool_mask_ini = mask_ini if has_ref_initial else full
        pool_mask_fin = mask_fin if has_ref_final else full
        pool_mask_tone = mask_tone if has_ref_tone and mask_tone is not None else full

        seg_frames = np.zeros(t, dtype=np.int64)
        seg_frames[mask_ini] = CLS_INITIAL
        seg_frames[mask_fin & ~mask_ini] = CLS_FINAL

        edges, etypes = build_phoneme_graph_edges(
            has_ref_initial=has_ref_initial,
            has_ref_final=has_ref_final,
            has_ref_tone=has_ref_tone,
            semantic_graph_mode=self.semantic_graph_mode,
        )
        edge_index_np, edge_type_np = edges_to_tensors(edges, etypes)

        n_ini = max(int(mask_ini.sum()), 1)
        n_fin = max(int(mask_fin.sum()), 1)

        data = Data(
            x=torch.zeros(4, 1, dtype=torch.float32),
            edge_index=torch.tensor(edge_index_np, dtype=torch.long),
            edge_type=torch.tensor(edge_type_np, dtype=torch.long),
        )
        data.frame_feat = torch.tensor(concat_feat, dtype=torch.float32)
        if qwen_feat is not None and self.qwen_fusion in ("film", "xattn"):
            data.frame_qwen = torch.tensor(qwen_feat, dtype=torch.float32)
        data.frame_prosody = torch.tensor(frame_prosody, dtype=torch.float32)
        data.pool_mask_ini = torch.tensor(pool_mask_ini, dtype=torch.bool)
        data.pool_mask_fin = torch.tensor(pool_mask_fin, dtype=torch.bool)
        data.pool_mask_tone = torch.tensor(pool_mask_tone, dtype=torch.bool)
        data.seg_label_frames = torch.tensor(seg_frames, dtype=torch.long)
        data.n_time_frames = torch.tensor(t, dtype=torch.long)
        data.node_type = torch.tensor([1, 2, 3, 4], dtype=torch.long)
        data.ref_token_id = torch.tensor(
            [
                ref_ids[0] if has_ref_initial else self.vocab["<pad>"],
                ref_ids[1] if has_ref_final else self.vocab["<pad>"],
                ref_ids[2] if has_ref_tone else self.vocab["<pad>"],
                self.vocab["<pad>"],
            ],
            dtype=torch.long,
        )
        data.position = torch.tensor([[0.0], [0.33], [0.66], [1.0]], dtype=torch.float32)
        data.duration_ratio = torch.tensor(
            [[n_ini / max(t, 1)], [n_fin / max(t, 1)], [1.0], [1.0]],
            dtype=torch.float32,
        )
        data.real_tokens = real_tokens
        data.real_ids = [pid for pid, keep in zip(real_ids, [has_real_initial, has_real_final, has_real_tone]) if keep]
        data.initial = torch.tensor(real_ids[0] if has_real_initial else IGNORE_INDEX, dtype=torch.long)
        data.final = torch.tensor(real_ids[1] if has_real_final else IGNORE_INDEX, dtype=torch.long)
        data.tone = torch.tensor(real_ids[2] if has_real_tone else IGNORE_INDEX, dtype=torch.long)
        y = target_dict(r)
        ini_seg = find_segment(r, "initial")
        fin_seg = find_segment(r, "final")
        data.has_initial = torch.tensor(bool(ini_seg) and ini_seg.get("apa_score") is not None, dtype=torch.bool)
        data.has_final = torch.tensor(bool(fin_seg) and fin_seg.get("apa_score") is not None, dtype=torch.bool)
        data.ini_dur_frac = torch.tensor(float(n_ini / max(t, 1)), dtype=torch.float32)
        data.fin_dur_frac = torch.tensor(float(n_fin / max(t, 1)), dtype=torch.float32)
        for k, v in y.items():
            if k.startswith("mdd"):
                setattr(data, k, torch.tensor(int(v), dtype=torch.long))
            elif k.startswith("apa"):
                setattr(data, k, torch.tensor(float(v), dtype=torch.float32))
        return data


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, id_to_token: dict[int, str], pad_id: int) -> dict[str, float]:
    model.eval()
    store: dict[str, list[Any]] = {k: [] for k in [
        "mdd_initial_p", "mdd_initial_r", "mdd_final_p", "mdd_final_r", "mdd_tone_p", "mdd_tone_r", "mdd_char_p", "mdd_char_r",
        "apa_initial_p", "apa_initial_r", "apa_final_p", "apa_final_r", "apa_char_p", "apa_char_r",
    ]}
    phone_edits = phone_refs = char_edits = char_refs = 0
    non_blocking = device.type == "cuda"
    for batch in loader:
        batch = batch.to(device, non_blocking=non_blocking)
        out = model(batch)
        pred_initial = out["asr_initial_logits"].argmax(-1).cpu().tolist()
        pred_final = out["asr_final_logits"].argmax(-1).cpu().tolist()
        pred_tone = out["asr_tone_logits"].argmax(-1).cpu().tolist()
        for i in range(len(pred_initial)):
            pred_ids = [pid for pid in [pred_initial[i], pred_final[i], pred_tone[i]] if pid != pad_id]
            ref_ids = [pid for pid in batch.real_ids[i] if pid != pad_id]
            phone_edits += edit_distance(pred_ids, ref_ids)
            phone_refs += max(len(ref_ids), 1)
            pred_tokens = [id_to_token.get(pid, "<unk>") for pid in pred_ids]
            pred_text = "".join(pred_tokens).replace("tone", "")
            ref_text = "".join(batch.real_tokens[i]).replace("tone", "")
            char_edits += edit_distance(list(pred_text), list(ref_text))
            char_refs += max(len(ref_text), 1)
        for name in ["mdd_initial", "mdd_final", "mdd_tone", "mdd_char"]:
            pred = out[f"{name}_logits"].argmax(-1).cpu().tolist()
            ref = getattr(batch, name).cpu().tolist()
            for p, r in zip(pred, ref):
                if r != IGNORE_INDEX:
                    store[f"{name}_p"].append(p)
                    store[f"{name}_r"].append(r)
        for name in ["apa_initial", "apa_final", "apa_char"]:
            pred = out[f"{name}_score"].reshape(-1).cpu().tolist()
            ref = getattr(batch, name).reshape(-1).cpu().tolist()
            mask = batch.has_initial.reshape(-1).cpu().tolist() if name == "apa_initial" else (
                batch.has_final.reshape(-1).cpu().tolist() if name == "apa_final" else None
            )
            if mask is not None:
                for p, rv, keep in zip(pred, ref, mask):
                    if keep:
                        store[f"{name}_p"].append(p)
                        store[f"{name}_r"].append(rv)
            else:
                store[f"{name}_p"].extend(pred)
                store[f"{name}_r"].extend(ref)
    return {
        "asr_per": phone_edits / max(phone_refs, 1),
        "asr_cer": char_edits / max(char_refs, 1),
        "mdd_initial_acc": accuracy(store["mdd_initial_p"], store["mdd_initial_r"]),
        "mdd_final_acc": accuracy(store["mdd_final_p"], store["mdd_final_r"]),
        "mdd_tone_acc": accuracy(store["mdd_tone_p"], store["mdd_tone_r"]),
        "mdd_char_acc": accuracy(store["mdd_char_p"], store["mdd_char_r"]),
        "mdd_initial_f1": macro_f1(store["mdd_initial_p"], store["mdd_initial_r"]),
        "mdd_final_f1": macro_f1(store["mdd_final_p"], store["mdd_final_r"]),
        "mdd_tone_f1": macro_f1(store["mdd_tone_p"], store["mdd_tone_r"]),
        "mdd_char_f1": macro_f1(store["mdd_char_p"], store["mdd_char_r"]),
        "apa_initial_pcc": pcc(store["apa_initial_p"], store["apa_initial_r"]),
        "apa_final_pcc": pcc(store["apa_final_p"], store["apa_final_r"]),
        "apa_char_pcc": pcc(store["apa_char_p"], store["apa_char_r"]),
        "apa_mean_pcc": (
            pcc(store["apa_initial_p"], store["apa_initial_r"])
            + pcc(store["apa_final_p"], store["apa_final_r"])
            + pcc(store["apa_char_p"], store["apa_char_r"])
        ) / 3.0,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="GOP graph v102b: stem L4 SupCon + multitask ASR/MDD/APA")
    parser.add_argument("--data-root", type=Path, default=Path("/home/chenliye/Graph/GNN/data"))
    parser.add_argument("--hubert-dir", type=Path, default=Path("/home/chenliye/Graph/feature_12layer/hubert_feature"))
    parser.add_argument("--wav2vec-dir", type=Path, default=Path("/home/chenliye/Graph/feature_12layer/tencent_wav2vec2_feature"))
    parser.add_argument("--wavlm-dir", type=Path, default=Path("/home/chenliye/Graph/feature_12layer/wavlm_feature"))
    parser.add_argument("--qwen-dir", type=Path, default=Path("/home/chenliye/Graph/qwen3-feature-layernorm/qwen_feature_14layer"), help="Qwen3-ASR frame features (L14 layernorm→1024d)")
    parser.add_argument(
        "--qwen-fusion",
        type=str,
        default="concat",
        choices=("concat", "film", "xattn"),
        help="concat: 4096-d input; film: FiLM modulate hidden; xattn: SSL Q attends Qwen K/V",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--f0-cache-dir", type=Path, default=Path("/home/chenliye/Graph/GNN/prosody_cache/hubert_f0"))
    parser.add_argument("--energy-cache-dir", type=Path, default=Path("/home/chenliye/Graph/GNN/prosody_cache/hubert_energy"))
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--stem-layers", type=int, default=4)
    parser.add_argument("--stem-backbone", type=str, default="edge_transformer")
    parser.add_argument("--mdd-layers", type=int, default=4)
    parser.add_argument("--mdd-backbone", type=str, default="edge_transformer")
    parser.add_argument("--apa-layers", type=int, default=4)
    parser.add_argument("--apa-backbone", type=str, default="edge_transformer")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-epochs", type=int, default=5)
    parser.add_argument("--min-lr-ratio", type=float, default=0.01)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--apa-beta", type=float, default=0.05)
    parser.add_argument("--w-asr", type=float, default=1.0)
    parser.add_argument("--w-mdd", type=float, default=1.0)
    parser.add_argument("--w-apa", type=float, default=4.0)
    parser.add_argument("--apa-pcc-weight", type=float, default=0.35)
    parser.add_argument("--apa-final-weight", type=float, default=1.5)
    parser.add_argument("--best-metric", type=str, default="apa_mean_pcc")
    parser.add_argument("--early-stop-patience", type=int, default=10, help="0 = disabled")
    parser.add_argument("--early-stop-min-epochs", type=int, default=15)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--cuda-device", type=int, default=0)
    parser.add_argument("--ssl-attn-heads", type=int, default=4)
    parser.add_argument("--ssl-enhancer-mode", type=str, default="self_attn2", choices=("self_attn2", "hubert_q"))
    parser.add_argument("--ssl-self-attn-layers", type=int, default=2)
    parser.add_argument("--n-ssl-layers", type=int, default=4)
    parser.add_argument("--sync-layers", type=str, default="2,4")
    parser.add_argument("--sync-attn-heads", type=int, default=4)
    parser.add_argument("--seg-sync-bias", type=float, default=2.0)
    parser.add_argument("--apa-seg-bias", type=float, default=2.0)
    parser.add_argument("--phoneme-triplet-graph", type=int, default=1, choices=(0, 1))
    parser.add_argument("--phoneme-slot-causal", type=int, default=1, choices=(0, 1))
    parser.add_argument("--semantic-graph-mode", type=str, default="phoneme_dag_plus", choices=("phoneme_dag", "phoneme_dag_plus"))
    parser.add_argument("--fa-cache-dir", type=Path, default=Path("/home/chenliye/Graph/GNN/fa_cache/torchaudio_mms_hubert_native"))
    parser.add_argument("--apa-char-consistency-weight", type=float, default=0.15)
    parser.add_argument("--w-contra", type=float, default=0.05)
    parser.add_argument("--contra-warmup-epochs", type=int, default=5)
    parser.add_argument("--contra-temperature", type=float, default=0.1)
    args = parser.parse_args()
    if args.out_dir is None:
        out_map = {
            "concat": Path("/home/chenliye/Graph/GNN/zzzzz/exp_qwen/seed42"),
            "film": Path("/home/chenliye/Graph/GNN/zzzzz/exp_qwen_film/seed42"),
            "xattn": Path("/home/chenliye/Graph/GNN/zzzzz/exp_qwen_xattn/seed42"),
        }
        args.out_dir = out_map.get(args.qwen_fusion, out_map["concat"])

    if Batch is None:
        raise SystemExit("torch_geometric required")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_path = args.out_dir / "best_model.pt"
    log_path = args.out_dir / "train_log.jsonl"
    config_path = args.out_dir / "config.json"

    train_records = read_jsonl(args.data_root / "train" / "labels.jsonl")
    test_records = read_jsonl(args.data_root / "test" / "labels.jsonl")
    vocab = build_vocab(train_records + test_records)
    id_to_token = {v: k for k, v in vocab.items()}

    hubert_mean, hubert_std = compute_acoustic_stats_native(train_records, args.hubert_dir / "train")
    w2v_mean, w2v_std = compute_acoustic_stats_native(train_records, args.wav2vec_dir / "train")
    wavlm_mean, wavlm_std = compute_acoustic_stats_native(train_records, args.wavlm_dir / "train")
    qwen_dir = args.qwen_dir
    qwen_mean = qwen_std = None
    qwen_dim = 0
    if qwen_dir is not None:
        qwen_mean, qwen_std = compute_acoustic_stats_native(train_records, qwen_dir / "train")
        qwen_dim = int(qwen_mean.shape[0])
        if args.qwen_fusion == "concat":
            feat_dims = (
                int(hubert_mean.shape[0]),
                int(w2v_mean.shape[0]),
                int(wavlm_mean.shape[0]),
                qwen_dim,
            )
            print(
                f"[feat] 4-way concat: hubert|w2v|wavlm|qwen = {feat_dims} -> total {sum(feat_dims)}",
                flush=True,
            )
        else:
            feat_dims = (
                int(hubert_mean.shape[0]),
                int(w2v_mean.shape[0]),
                int(wavlm_mean.shape[0]),
            )
            if args.qwen_fusion == "film":
                print(
                    f"[feat] qwen film: SSL {feat_dims} (total {sum(feat_dims)}), "
                    f"qwen {qwen_dim}-d FiLM-modulates hidden after SSL pyramid",
                    flush=True,
                )
            elif args.qwen_fusion == "xattn":
                print(
                    f"[feat] qwen xattn: SSL {feat_dims} (total {sum(feat_dims)}), "
                    f"qwen {qwen_dim}-d as K/V, SSL hidden as Q (cross-attn after SSL pyramid)",
                    flush=True,
                )
            else:
                print(f"[feat] SSL only {feat_dims}", flush=True)
    else:
        feat_dims = (
            int(hubert_mean.shape[0]),
            int(w2v_mean.shape[0]),
            int(wavlm_mean.shape[0]),
        )
    side_stats_path = args.out_dir / "hubert_f0_energy_mean_std.npz"
    if side_stats_path.is_file():
        st = np.load(side_stats_path)
        f0_mean = np.asarray(st["f0_mean"], dtype=np.float32)
        f0_std = np.asarray(st["f0_std"], dtype=np.float32)
        energy_mean = np.asarray(st["energy_mean"], dtype=np.float32)
        energy_std = np.asarray(st["energy_std"], dtype=np.float32)
        print(f"[side stats] loaded from {side_stats_path}", flush=True)
    else:
        f0_mean, f0_std = compute_hubert_side_stats(
            train_records, args.f0_cache_dir, HUBERT_F0_CACHE_KEY, HUBERT_F0_DIM, args.hubert_dir, "train"
        )
        energy_mean, energy_std = compute_hubert_side_stats(
            train_records, args.energy_cache_dir, HUBERT_ENERGY_CACHE_KEY, HUBERT_ENERGY_DIM, args.hubert_dir, "train"
        )

    ds_kw = dict(
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
        semantic_graph_mode=args.semantic_graph_mode,
        qwen_dir=qwen_dir,
        qwen_mean=qwen_mean,
        qwen_std=qwen_std,
        qwen_fusion=args.qwen_fusion,
    )
    train_ds = PhonemeGraphDataset(args.data_root / "train", **ds_kw)
    test_ds = PhonemeGraphDataset(args.data_root / "test", **ds_kw)
    sync_layers = tuple(int(x.strip()) for x in str(args.sync_layers).split(",") if x.strip())
    print(
        f"[v102b] SupCon w={args.w_contra} warmup={args.contra_warmup_epochs}ep, "
        f"SSL L={args.n_ssl_layers} bridges@{sync_layers}, lr={args.lr}, hidden={args.hidden_dim}",
        flush=True,
    )

    with config_path.open("w", encoding="utf-8") as f:
        json.dump(
            {
                **vars(args),
                "vocab_size": len(vocab),
                "feat_dims": feat_dims,
                "nodes_per_graph": 4,
                "prosody_dim": PROSODY_F0_ENERGY_DIM,
                "f0_dim": HUBERT_F0_DIM,
                "energy_dim": HUBERT_ENERGY_DIM,
                "ssl_enhancer": "PyramidTriSSLDualEnhancerFA",
                "n_ssl_layers": int(args.n_ssl_layers),
                "sync_layers": list(sync_layers),
                "stem_contrast": "supervised_phoneme_supcon_l4_ini_fin",
                "qwen_dim": qwen_dim,
                "qwen_fusion": args.qwen_fusion,
                "feat_fusion": f"ssl_qwen_{args.qwen_fusion}" if qwen_dim > 0 else "ssl_only",
                "recipe": f"gop_graph_v102b_qwen_{args.qwen_fusion}",
                "script": "train_gop_graph.py",
            },
            f,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

    device = torch.device(f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        print(f"[device] {device} ({torch.cuda.get_device_name(device)})", flush=True)

    nw = max(int(args.num_workers), 0)
    pin_mem = device.type == "cuda"
    loader_kw: dict[str, Any] = dict(batch_size=args.batch_size, collate_fn=Batch.from_data_list, num_workers=nw, pin_memory=pin_mem)
    if nw > 0:
        loader_kw["persistent_workers"] = True
        loader_kw["prefetch_factor"] = 2
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kw)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kw)

    use_amp = bool(args.use_amp and device.type == "cuda")
    model = PhonemeGraphModelV102b(
        feat_dims=feat_dims,
        vocab_size=len(vocab),
        qwen_dim=qwen_dim if args.qwen_fusion in ("film", "xattn") and qwen_dim > 0 else None,
        qwen_fusion=args.qwen_fusion,
        hidden_dim=args.hidden_dim,
        prosody_dim=PROSODY_F0_ENERGY_DIM,
        stem_layers=args.stem_layers,
        stem_backbone=args.stem_backbone,
        mdd_layers=args.mdd_layers,
        mdd_backbone=args.mdd_backbone,
        apa_layers=args.apa_layers,
        apa_backbone=args.apa_backbone,
        use_phoneme_triplet_graph=bool(args.phoneme_triplet_graph),
        phoneme_slot_causal=bool(args.phoneme_slot_causal),
        apa_seg_bias=float(args.apa_seg_bias),
        ssl_attn_heads=int(args.ssl_attn_heads),
        ssl_enhancer_mode=str(args.ssl_enhancer_mode),
        ssl_self_attn_layers=int(args.ssl_self_attn_layers),
        n_ssl_layers=int(args.n_ssl_layers),
        sync_layers=sync_layers,
        sync_attn_heads=int(args.sync_attn_heads),
        seg_sync_bias=float(args.seg_sync_bias),
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if args.warmup_epochs > 0 and args.epochs > args.warmup_epochs:
        warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=args.warmup_epochs)
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, args.epochs - args.warmup_epochs), eta_min=args.lr * args.min_lr_ratio)
        scheduler = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warm, cos], milestones=[args.warmup_epochs])
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1), eta_min=args.lr * args.min_lr_ratio)

    ls = float(args.label_smoothing)
    mdd_losses = {
        k: nn.CrossEntropyLoss(label_smoothing=ls, ignore_index=IGNORE_INDEX)
        for k in ["mdd_initial", "mdd_final", "mdd_tone", "mdd_char"]
    }
    apa_crit = nn.SmoothL1Loss(beta=float(args.apa_beta))
    ema = ModelEMA(model, decay=args.ema_decay) if args.ema_decay > 0.0 else None
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if use_amp else None
    best_metric_val = float("-inf")
    epochs_no_improve = 0

    np.savez(
        args.out_dir / "acoustic_mean_std.npz",
        hubert_mean=hubert_mean, hubert_std=hubert_std,
        w2v_mean=w2v_mean, w2v_std=w2v_std,
        wavlm_mean=wavlm_mean, wavlm_std=wavlm_std,
    )
    np.savez(side_stats_path, f0_mean=f0_mean, f0_std=f0_std, energy_mean=energy_mean, energy_std=energy_std)

    with log_path.open("w", encoding="utf-8") as log_f:
        for epoch in range(1, args.epochs + 1):
            print(f"[train] epoch {epoch}/{args.epochs}", flush=True)
            model.train()
            w_contra_ep = contra_weight_for_epoch(epoch, args.contra_warmup_epochs, args.w_contra)
            totals = {"loss": 0.0, "loss_asr": 0.0, "loss_mdd": 0.0, "loss_apa": 0.0, "loss_contra": 0.0}
            steps = 0
            for batch in train_loader:
                batch = batch.to(device, non_blocking=pin_mem)
                opt.zero_grad(set_to_none=True)

                def _step_losses() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                    out = model(batch)
                    loss_asr = (
                        masked_cross_entropy(out["asr_initial_logits"], batch.initial)
                        + masked_cross_entropy(out["asr_final_logits"], batch.final)
                        + masked_cross_entropy(out["asr_tone_logits"], batch.tone)
                    ) / 3.0
                    loss_mdd = (
                        mdd_losses["mdd_initial"](out["mdd_initial_logits"], batch.mdd_initial)
                        + mdd_losses["mdd_final"](out["mdd_final_logits"], batch.mdd_final)
                        + mdd_losses["mdd_tone"](out["mdd_tone_logits"], batch.mdd_tone)
                        + mdd_losses["mdd_char"](out["mdd_char_logits"], batch.mdd_char)
                    ) / 4.0
                    loss_apa, _, _, _ = compute_apa_loss(
                        apa_crit, out, batch,
                        pcc_weight=args.apa_pcc_weight,
                        final_weight=args.apa_final_weight,
                        consistency_weight=args.apa_char_consistency_weight,
                    )
                    loss_contra = compute_stem_scl_loss(model, out, batch, vocab["<pad>"], args.contra_temperature)
                    loss = args.w_asr * loss_asr + args.w_mdd * loss_mdd + args.w_apa * loss_apa + w_contra_ep * loss_contra
                    return loss, loss_asr, loss_mdd, loss_apa, loss_contra

                if use_amp and scaler is not None:
                    with torch.cuda.amp.autocast():
                        loss, loss_asr, loss_mdd, loss_apa, loss_contra = _step_losses()
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(opt)
                    scaler.update()
                else:
                    loss, loss_asr, loss_mdd, loss_apa, loss_contra = _step_losses()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    opt.step()

                if ema is not None:
                    ema.update(model)
                totals["loss"] += float(loss.item())
                totals["loss_asr"] += float(loss_asr.item())
                totals["loss_mdd"] += float(loss_mdd.item())
                totals["loss_apa"] += float(loss_apa.item())
                totals["loss_contra"] += float(loss_contra.item())
                steps += 1

            scheduler.step()
            if ema is not None:
                ema.apply_to(model)
                metrics = evaluate(model, test_loader, device, id_to_token, vocab["<pad>"])
                ema.restore(model)
            else:
                metrics = evaluate(model, test_loader, device, id_to_token, vocab["<pad>"])

            rec = {
                "epoch": epoch,
                "lr": scheduler.get_last_lr()[0],
                "w_contra": w_contra_ep,
                "train_loss": totals["loss"] / max(steps, 1),
                "train_loss_contra": totals["loss_contra"] / max(steps, 1),
                **metrics,
            }
            log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            log_f.flush()
            print(json.dumps(rec, ensure_ascii=False), flush=True)

            cur = metric_for_best(metrics, args.best_metric)
            if cur > best_metric_val:
                best_metric_val = cur
                epochs_no_improve = 0
                state = ema.state_dict_for_save() if ema is not None else model.state_dict()
                torch.save({"model": state, "vocab": vocab, "feat_dims": feat_dims, "args": vars(args)}, save_path)
                print(f"[best] {args.best_metric}={cur:.4f}", flush=True)
            else:
                epochs_no_improve += 1
            if args.early_stop_patience > 0 and epoch >= args.early_stop_min_epochs and epochs_no_improve >= args.early_stop_patience:
                print(f"[early stop] epoch {epoch}", flush=True)
                break


if __name__ == "__main__":
    main()
