"""Load HuBERT FA mask cache (ini/fin per frame)."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def load_hubert_fa_masks(cache_path: Path, t_hubert: int) -> tuple[np.ndarray, np.ndarray] | None:
    if not cache_path.is_file():
        return None
    data = np.load(cache_path)
    mi = np.asarray(data["mask_ini"], dtype=bool)
    mf = np.asarray(data["mask_fin"], dtype=bool)
    if mi.shape[0] != t_hubert or mf.shape[0] != t_hubert:
        cached_t = int(data["t_hubert"][0]) if "t_hubert" in data else int(mi.shape[0])
        raise ValueError(
            f"HuBERT FA cache T={cached_t} != graph T={t_hubert} for {cache_path}. "
            f"Re-run extract_hubert_fa_masks.py."
        )
    return mi, mf
