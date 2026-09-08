"""Frame ↔ ini/fin alignment (duration mode) for v102b."""

from __future__ import annotations

import numpy as np

CLS_INITIAL = 1
CLS_FINAL = 2


def length_prior_masks(
    t: int,
    has_initial: bool,
    has_final: bool,
    initial: str,
    final: str,
) -> tuple[np.ndarray, np.ndarray]:
    if t <= 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=bool)
    if has_initial and has_final:
        li = max(len(initial), 1)
        lf = max(len(final), 1)
        cut = int(round(t * li / (li + lf)))
        cut = min(max(cut, 1), t - 1) if t > 1 else 1
        mask_ini = np.zeros(t, dtype=bool)
        mask_fin = np.zeros(t, dtype=bool)
        mask_ini[:cut] = True
        mask_fin[cut:] = True
        return mask_ini, mask_fin
    if has_initial:
        m = np.zeros(t, dtype=bool)
        m[: max(1, t // 3)] = True
        return m, np.zeros(t, dtype=bool)
    if has_final:
        m = np.zeros(t, dtype=bool)
        m[:] = True
        return np.zeros(t, dtype=bool), m
    return np.zeros(t, dtype=bool), np.zeros(t, dtype=bool)


def duration_align_initial_final(
    feat: np.ndarray,
    reference_pinyin: str,
    *,
    split_pinyin_fn,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    initial, final, _tone = split_pinyin_fn(reference_pinyin)
    has_initial = bool(initial)
    has_final = bool(final)
    t = int(feat.shape[0])
    if t == 0:
        z = np.zeros(0, dtype=np.int64)
        return z, z.astype(bool), z.astype(bool)

    mask_ini, mask_fin = length_prior_masks(t, has_initial, has_final, initial, final)
    frame_labels = np.zeros(t, dtype=np.int64)
    if has_initial:
        frame_labels[mask_ini] = CLS_INITIAL
    if has_final:
        frame_labels[mask_fin] = CLS_FINAL
    unassigned = frame_labels == 0
    if unassigned.any():
        mid = t // 2
        if has_initial and not has_final:
            frame_labels[unassigned] = CLS_INITIAL
        elif has_final and not has_initial:
            frame_labels[unassigned] = CLS_FINAL
        else:
            frame_labels[unassigned] = np.where(
                np.arange(t)[unassigned] < mid, CLS_INITIAL, CLS_FINAL
            ).astype(np.int64)
        mask_ini = frame_labels == CLS_INITIAL
        mask_fin = frame_labels == CLS_FINAL
    return frame_labels, mask_ini, mask_fin


def align_initial_final(
    feat: np.ndarray,
    reference_pinyin: str,
    *,
    mode: str,
    split_pinyin_fn,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if mode == "duration":
        return duration_align_initial_final(feat, reference_pinyin, split_pinyin_fn=split_pinyin_fn)
    raise ValueError(f"align mode must be duration, got {mode!r}")


def make_tone_tail_mask(mask_fin: np.ndarray) -> np.ndarray:
    mask_fin = np.asarray(mask_fin, dtype=bool)
    tone_mask = np.zeros_like(mask_fin, dtype=bool)
    idx = np.where(mask_fin)[0]
    if len(idx) == 0:
        return tone_mask
    start = idx[int(len(idx) * 0.4)]
    tone_mask[start : idx[-1] + 1] = True
    return tone_mask
