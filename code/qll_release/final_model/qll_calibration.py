#!/usr/bin/env python3
"""Benign-only calibration; this interface intentionally accepts no attack labels."""
import numpy as np
def calibrate(benign_dominance):
    values=np.asarray(benign_dominance,dtype=float)
    if values.ndim!=1 or not len(values): raise ValueError("non-empty 1D benign scores required")
    return float(np.quantile(values,0.95,method="higher"))
