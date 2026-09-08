"""Load Speechocean FA caches (mask_phones / mask_words) and resize to HuBERT T."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _resize_mask_rows(mask: np.ndarray, target_t: int) -> np.ndarray:
    """Resize [N, T] bool masks along time by nearest index."""
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"expected [N,T] mask, got {mask.shape}")
    n, t = mask.shape
    if t == target_t:
        return mask
    if target_t <= 0:
        return np.zeros((n, 0), dtype=bool)
    if t <= 0:
        return np.zeros((n, target_t), dtype=bool)
    idx = (np.linspace(0, t - 1, num=target_t)).round().astype(np.int64)
    return mask[:, idx]


def load_speechocean_fa_masks(
    cache_path: Path,
    t_hubert: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Return (mask_phones [Np,T], mask_words [Nw,T], phone_word_index [Np], phone_labels)."""
    if not cache_path.is_file():
        return None
    data = np.load(cache_path, allow_pickle=True)
    if "mask_phones" not in data or "mask_words" not in data:
        return None
    mp = _resize_mask_rows(data["mask_phones"], t_hubert)
    mw = _resize_mask_rows(data["mask_words"], t_hubert)
    pwi = np.asarray(data["phone_word_index"], dtype=np.int32)
    labels = np.asarray(data["phone_labels"], dtype=object)
    if mp.shape[0] != pwi.shape[0]:
        raise ValueError(f"mask_phones N={mp.shape[0]} != phone_word_index {pwi.shape[0]} @ {cache_path}")
    return mp, mw, pwi, labels
