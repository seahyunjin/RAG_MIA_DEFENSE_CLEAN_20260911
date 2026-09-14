"""Membership privacy metrics used throughout the archived experiments."""
import numpy as np
from sklearn.metrics import roc_auc_score

def effective_auc(labels, scores):
    raw = float(roc_auc_score(np.asarray(labels), np.asarray(scores)))
    return raw, max(raw, 1.0 - raw)
