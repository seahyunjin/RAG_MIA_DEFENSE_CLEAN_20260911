"""Small, dependency-light helpers for end-to-end privacy tables."""
from e_auc import effective_auc

def privacy_row(name, labels, scores, gate=0.65):
    raw, effective = effective_auc(labels, scores)
    return {"Attack": name, "Raw AUC": raw, "E-AUC": effective,
            "Verdict": "PASS" if effective <= gate else "FAIL"}
