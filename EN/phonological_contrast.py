"""Stem L4 supervised contrastive loss (ini/fin by ref token id)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch


def supervised_contrastive_loss(
    features: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    n = int(features.size(0))
    if n < 2:
        return features.sum() * 0.0
    device = features.device
    z = F.normalize(features, dim=-1)
    tau = max(float(temperature), 1e-4)
    sim = (z @ z.T) / tau
    labels = labels.contiguous().view(-1, 1)
    mask_pos = torch.eq(labels, labels.T).float().to(device)
    logits_mask = torch.ones_like(mask_pos) - torch.eye(n, device=device)
    mask_pos = mask_pos * logits_mask
    exp_logits = torch.exp(sim) * logits_mask
    log_prob = sim - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-8)
    pos_per_row = mask_pos.sum(dim=1)
    valid = pos_per_row > 0
    if not bool(valid.any()):
        return features.sum() * 0.0
    mean_log_prob_pos = (mask_pos * log_prob).sum(dim=1) / pos_per_row.clamp(min=1.0)
    return -mean_log_prob_pos[valid].mean()


def phonological_supervised_contrast(
    stem_layer: torch.Tensor,
    batch: Batch,
    pad_id: int,
    temperature: float,
    proj: nn.Module | None = None,
    slot_weights: tuple[float, float, float] = (1.0, 1.0, 0.0),
) -> torch.Tensor:
    node_type = batch.node_type
    ref_token_id = batch.ref_token_id
    losses = []
    weights = []
    for slot_type, w in zip((1, 2, 3), slot_weights):
        if w <= 0.0:
            continue
        mask = (node_type == int(slot_type)) & (ref_token_id != int(pad_id))
        if not bool(mask.any()):
            continue
        x_slot = stem_layer[mask]
        labels = ref_token_id[mask]
        z = proj(x_slot) if proj is not None else x_slot
        li = supervised_contrastive_loss(z, labels, temperature=temperature)
        losses.append(li * float(w))
        weights.append(float(w))
    if not losses:
        return stem_layer.sum() * 0.0
    return torch.stack(losses).sum() / sum(weights)
