"""Frozen old CF-Clip equation; no learned parameters."""

from __future__ import annotations

GAMMA = 9.565910959243753


def cf_clip_logits(full_logits, full_logp, drop_logp, gamma: float = GAMMA):
    """Apply vocabulary-wise counterfactual source-influence clipping."""
    import torch

    if full_logits.ndim != 1 or full_logp.ndim != 1:
        raise ValueError("full_logits and full_logp must be [vocabulary]")
    if drop_logp.ndim != 2 or drop_logp.shape[1] != full_logp.shape[0]:
        raise ValueError("drop_logp must be [documents, vocabulary]")
    lifts = torch.clamp(full_logp.unsqueeze(0) - drop_logp, min=0)
    max_lift = lifts.max(dim=0).values
    clip = torch.clamp(max_lift - gamma, min=0)
    return full_logits - clip, max_lift, clip

