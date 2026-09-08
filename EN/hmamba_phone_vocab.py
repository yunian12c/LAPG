"""HMamba / Speechocean762 phone inventory (vocab_merge.json).

Maps canonical / realized phone strings onto the 48-token merge set used by
HMamba (+ optional <unk>/<pad> for our graph pipeline).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Multi-phone realized replacements → single L2 merge token (HMamba inventory).
_MULTI_TO_MERGE: dict[str, str] = {
    "IH R": "ir",
    "I R": "ir",
    "EH R": "err",
    "AH R": "ar",
    "AA R": "ar",
    "AO R": "ar",
    "ER R": "err",
    "T R": "tr",
    "D R": "dr",
    "D Z": "dz",
    "T S": "ts",
}

_DEFAULT_VOCAB = Path(__file__).resolve().parents[1] / "resource" / "vocab_merge.json"


def strip_stress(tok: str) -> str:
    t = str(tok).strip().upper()
    if len(t) >= 2 and t[-1] in "012" and t[-2].isalpha():
        return t[:-1]
    return t


def load_hmamba_vocab(path: Path | None = None) -> dict[str, int]:
    """Load HMamba vocab_merge.json and add <unk>/<pad> for our trainer."""
    path = Path(path) if path is not None else _DEFAULT_VOCAB
    raw = json.loads(path.read_text(encoding="utf-8"))
    # Keep HMamba ids intact; append extras.
    vocab = {str(k).lower(): int(v) for k, v in raw.items()}
    if "<unk>" not in vocab:
        vocab["<unk>"] = len(vocab)
    if "<pad>" not in vocab:
        vocab["<pad>"] = len(vocab)
    return vocab


def map_phone_token(tok: str, vocab: dict[str, int]) -> str:
    """Map one phone string (canonical or realized) to a vocab key."""
    raw = str(tok).strip()
    if not raw:
        return "<unk>" if "<unk>" in vocab else "ah"

    up = raw.upper()
    if up in ("<DEL>", "DEL"):
        return "<del>" if "<del>" in vocab else "<unk>"
    if up in ("<UNK>", "UNK"):
        return "<unk>" if "<unk>" in vocab else "ah"
    if up in ("SIL", "SPN", "NSN", "<EPS>"):
        return "sil" if "sil" in vocab else "<unk>"

    # Multi-phone replacement at one canonical slot (aligned CE / HMamba).
    parts = [strip_stress(p) for p in up.split() if p.strip()]
    if len(parts) >= 2:
        key = " ".join(parts)
        if key in _MULTI_TO_MERGE and _MULTI_TO_MERGE[key] in vocab:
            return _MULTI_TO_MERGE[key]
        joined = "".join(p.lower() for p in parts)
        if joined in vocab:
            return joined

    # Single token: strip stress, then optional L2 '*' → base.
    t = strip_stress(parts[0] if parts else up)
    if t.endswith("*"):
        base = t[:-1].lower()
        if base in vocab:
            return base
        return "<unk>" if "<unk>" in vocab else "ah"

    low = t.lower()
    if low in vocab:
        return low
    # Rare lone 'I' in Speechocean → ih
    if low == "i" and "ih" in vocab:
        return "ih"
    return "<unk>" if "<unk>" in vocab else "ah"


def phone_to_id(tok: str, vocab: dict[str, int]) -> int:
    key = map_phone_token(tok, vocab)
    if key in vocab:
        return int(vocab[key])
    return int(vocab.get("<unk>", 0))


def aligned_realized_ids(phones: list[dict[str, Any]], vocab: dict[str, int]) -> list[int]:
    """One realized id per canonical phone position (HMamba-style, keeps <del>)."""
    ids: list[int] = []
    for p in phones:
        raw = str(p.get("real_phone", p.get("phone", "")))
        ids.append(phone_to_id(raw, vocab))
    return ids


def canonical_ids(phones: list[dict[str, Any]], vocab: dict[str, int]) -> list[int]:
    return [phone_to_id(str(p.get("phone", "")), vocab) for p in phones]
