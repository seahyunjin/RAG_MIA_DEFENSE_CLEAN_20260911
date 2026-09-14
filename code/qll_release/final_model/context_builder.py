"""Canonical context packing used by the frozen final model."""
from qll_source_hide_final import TOTAL_CONTEXT_TOKENS

def equal_redistribution(lengths, hidden=None):
    caps = [0] * len(lengths)
    remaining = TOTAL_CONTEXT_TOKENS
    active = [i for i, length in enumerate(lengths) if i != hidden and length > 0]
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        changed = False
        for index in list(active):
            add = min(share, lengths[index] - caps[index], remaining)
            caps[index] += add
            remaining -= add
            changed |= add > 0
            if caps[index] >= lengths[index]:
                active.remove(index)
            if remaining <= 0:
                break
        if not changed:
            break
    return caps
