"""Frozen old CF-Clip equation; no learned parameters."""
from __future__ import annotations
GAMMA = 9.5659109592

def cf_clip_logits(full_logits, full_logp, drop_logp, gamma=GAMMA):
    import torch
    if drop_logp.ndim != 2: raise ValueError("drop_logp must be [documents,vocabulary]")
    lifts=torch.clamp(full_logp.unsqueeze(0)-drop_logp,min=0)
    max_lift=lifts.max(dim=0).values
    clip=torch.clamp(max_lift-gamma,min=0)
    return full_logits-clip, max_lift, clip
