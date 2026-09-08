"""Phoneme-only graph (ini / fin / tone / global), no frame nodes."""

from __future__ import annotations

import numpy as np

NODE_INITIAL = 0
NODE_FINAL = 1
NODE_TONE = 2
NODE_GLOBAL = 3
NUM_PHONEME_NODES = 4

EDGE_SELF = 0
EDGE_SEM_INI_FIN = 1
EDGE_SEM_FIN_TONE = 2
EDGE_SEM_TONE_GLOBAL = 3
EDGE_SEMANTIC = 4
NUM_PHONEME_EDGE_TYPES = 5


def build_phoneme_graph_edges(
    *,
    has_ref_initial: bool,
    has_ref_final: bool,
    has_ref_tone: bool,
    semantic_graph_mode: str = "phoneme_dag_plus",
) -> tuple[list[tuple[int, int]], list[int]]:
    """4-node phoneme DAG: ini(0) → fin(1) → tone(2) → global(3)."""
    edges: list[tuple[int, int]] = []
    etypes: list[int] = []

    def add(u: int, v: int, et: int) -> None:
        edges.append((u, v))
        etypes.append(et)

    for i in range(NUM_PHONEME_NODES):
        add(i, i, EDGE_SELF)

    mode = str(semantic_graph_mode).lower()
    if has_ref_initial and has_ref_final:
        add(NODE_INITIAL, NODE_FINAL, EDGE_SEM_INI_FIN)
    if has_ref_final and has_ref_tone:
        add(NODE_FINAL, NODE_TONE, EDGE_SEM_FIN_TONE)
    if has_ref_tone:
        add(NODE_TONE, NODE_GLOBAL, EDGE_SEM_TONE_GLOBAL)
    elif has_ref_final:
        add(NODE_FINAL, NODE_GLOBAL, EDGE_SEM_TONE_GLOBAL)

    if mode == "phoneme_dag_plus":
        if has_ref_initial and has_ref_final:
            add(NODE_FINAL, NODE_INITIAL, EDGE_SEMANTIC)
        if has_ref_final and has_ref_tone:
            add(NODE_TONE, NODE_FINAL, EDGE_SEMANTIC)
        if has_ref_tone:
            add(NODE_GLOBAL, NODE_TONE, EDGE_SEMANTIC)
        if has_ref_initial:
            add(NODE_INITIAL, NODE_GLOBAL, EDGE_SEMANTIC)
        if has_ref_final:
            add(NODE_FINAL, NODE_GLOBAL, EDGE_SEMANTIC)

    return edges, etypes


def edges_to_tensors(edges: list[tuple[int, int]], etypes: list[int]) -> tuple[np.ndarray, np.ndarray]:
    if not edges:
        return np.zeros((2, 0), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    ei = np.array(edges, dtype=np.int64).T
    et = np.asarray(etypes, dtype=np.int64)
    return ei, et
