"""Frozen CLD/G4 retrieval intervention."""
from __future__ import annotations
import numpy as np

TAU = 0.0901547372341156

def cld4(scores):
    values=np.asarray(scores,dtype=np.float64)
    if values.shape != (4,): raise ValueError("CLD4 requires exactly four retrieval scores")
    return float(values[0]-values.mean())

def apply_cld(scores, documents, tau=TAU):
    if len(documents)!=4: raise ValueError("Expected four ranked documents")
    risk=cld4(scores); alarm=bool(risk>tau)
    return {"g4":risk,"alarm":alarm,"documents":list(documents[1:] if alarm else documents),"removed":documents[0] if alarm else None}
