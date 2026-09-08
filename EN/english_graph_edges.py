"""English phone + word graph edges for Speechocean762 (no sentence/utt node)."""

from __future__ import annotations

import numpy as np

NODE_PHONE = 1
NODE_WORD = 2

EDGE_SELF = 0
EDGE_PHONE_NEXT = 1
EDGE_PHONE_TO_WORD = 2
EDGE_WORD_NEXT = 3
EDGE_SEMANTIC = 4
# Phones that share a word (within-word mismatch propagation for MDD).
EDGE_COWORD = 5
NUM_ENGLISH_EDGE_TYPES = 6


def build_english_phone_word_edges(
    n_phones: int,
    n_words: int,
    phone_word_index: list[int] | np.ndarray,
    *,
    bidirectional: bool = True,
    coword_edges: bool = False,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Build edges for nodes ordered as [phones..., words...]."""
    pwi = [int(x) for x in list(phone_word_index)]
    if len(pwi) != n_phones:
        raise ValueError(f"phone_word_index len {len(pwi)} != n_phones {n_phones}")

    n_nodes = n_phones + n_words
    node_types = [NODE_PHONE] * n_phones + [NODE_WORD] * n_words

    edges: list[tuple[int, int]] = []
    etypes: list[int] = []

    def add(u: int, v: int, et: int) -> None:
        edges.append((u, v))
        etypes.append(et)

    for i in range(n_nodes):
        add(i, i, EDGE_SELF)

    for i in range(n_phones - 1):
        add(i, i + 1, EDGE_PHONE_NEXT)
        if bidirectional:
            add(i + 1, i, EDGE_SEMANTIC)

    for pi, wi in enumerate(pwi):
        if 0 <= wi < n_words:
            w_idx = n_phones + wi
            add(pi, w_idx, EDGE_PHONE_TO_WORD)
            if bidirectional:
                add(w_idx, pi, EDGE_SEMANTIC)

    for wi in range(n_words - 1):
        a = n_phones + wi
        b = n_phones + wi + 1
        add(a, b, EDGE_WORD_NEXT)
        if bidirectional:
            add(b, a, EDGE_SEMANTIC)

    # Within-word phone clique (chain of consecutive phones in the same word).
    # Enables local mismatch / error signal to propagate without going through the word hub.
    if coword_edges and n_phones > 1:
        by_word: dict[int, list[int]] = {}
        for pi, wi in enumerate(pwi):
            if 0 <= wi < n_words:
                by_word.setdefault(wi, []).append(pi)
        for phones_in_w in by_word.values():
            for a, b in zip(phones_in_w, phones_in_w[1:]):
                add(a, b, EDGE_COWORD)
                add(b, a, EDGE_COWORD)

    return edges, etypes, node_types


def edges_to_tensors(edges: list[tuple[int, int]], etypes: list[int]) -> tuple[np.ndarray, np.ndarray]:
    if not edges:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    ei = np.array(edges, dtype=np.int64).T
    et = np.asarray(etypes, dtype=np.int64)
    return ei, et
