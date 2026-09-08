"""HuBERT F0 / energy side-cache helpers (extracted from train_gop_graph_v56c)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

HUBERT_F0_DIM = 4
HUBERT_ENERGY_DIM = 4
HUBERT_F0_CACHE_KEY = "f0"
HUBERT_ENERGY_CACHE_KEY = "prosody"


def _normalize_side(feat: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return np.clip((feat - mean) / std, -8.0, 8.0).astype(np.float32)


def _load_hubert_side_cache(path: Path, key: str, t: int, feat_dim: int) -> np.ndarray | None:
    if not path.is_file():
        return None
    raw = np.asarray(np.load(path)[key], dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != feat_dim or raw.shape[0] != t:
        return None
    return raw


def compute_hubert_side_stats(
    records: list[dict[str, Any]],
    cache_dir: Path,
    cache_key: str,
    feat_dim: int,
    hubert_dir: Path,
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    sums = np.zeros(feat_dim, dtype=np.float64)
    sq = np.zeros(feat_dim, dtype=np.float64)
    count = 0
    for r in records:
        sid = str(r.get("id", ""))
        if not sid:
            continue
        hub_path = hubert_dir / split_name / f"{sid}.npy"
        cache_path = cache_dir / split_name / f"{sid}.npz"
        if not hub_path.is_file() or not cache_path.is_file():
            continue
        feat = np.asarray(np.load(hub_path, mmap_mode="r"), dtype=np.float32)
        if feat.ndim != 2:
            feat = feat.reshape(feat.shape[0], -1)
        t = max(int(feat.shape[0]), 1)
        raw = _load_hubert_side_cache(cache_path, cache_key, t, feat_dim)
        if raw is None:
            continue
        sums += raw.sum(axis=0)
        sq += np.square(raw).sum(axis=0)
        count += raw.shape[0]
    if count == 0:
        raise RuntimeError(f"No cache frames under {cache_dir}/{split_name} key={cache_key}")
    mean = (sums / count).astype(np.float32)
    var = np.maximum(sq / count - np.square(mean), 1e-6)
    std = np.maximum(np.sqrt(var).astype(np.float32), 1e-3)
    return mean, std
