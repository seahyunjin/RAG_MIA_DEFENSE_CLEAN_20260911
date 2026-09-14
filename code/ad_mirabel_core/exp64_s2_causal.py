"""Pure statistics and alignment helpers for the frozen Exp64 S² audit."""
from __future__ import annotations
import re
from typing import Mapping,Sequence
import numpy as np
from src.exp44_native import extract_mba_masked_document

def causal_influence(original:float,removed:Sequence[float])->np.ndarray:
    values=np.asarray(removed,dtype=float)
    if values.ndim!=1 or not np.isfinite(values).all() or not np.isfinite(original):
        raise ValueError("finite scalar original and one-dimensional removed scores required")
    return float(original)-values

def concentration(influences:Sequence[float])->dict[str,float]:
    positive=np.maximum(np.asarray(influences,dtype=float),0.0)
    total=float(positive.sum())
    if total<=0:return {"positive_total":0.0,"c1":0.0,"c2":0.0}
    ordered=np.sort(positive)[::-1]
    return {"positive_total":total,"c1":float(ordered[0]/total),
            "c2":float(ordered[:min(2,len(ordered))].sum()/total)}

def best_indices(influences:Sequence[float],n:int=1)->list[int]:
    values=np.asarray(influences,dtype=float)
    if values.ndim!=1 or not len(values):raise ValueError("nonempty vector required")
    return np.argsort(-values,kind="stable")[:n].astype(int).tolist()

def recover_mba_spans(query:str,document_content:str)->Mapping[int,str]:
    """Align released MBA masks to an exact contiguous source substring.

    The released query may mask a sentence/title rather than the entire source
    document, so ordered literal anchors are searched within the canonical
    title+text.  No response or membership label is used.
    """
    masked=extract_mba_masked_document(query)
    parts=re.split(r"\[Mask_(\d+)\]",masked)
    if len(parts)<3 or len(parts)%2==0:raise ValueError("malformed MBA placeholder sequence")
    indices=[int(parts[i]) for i in range(1,len(parts),2)]
    if indices!=list(range(len(indices))):raise ValueError("MBA indices are not contiguous")
    pattern=re.escape(parts[0])
    for i in range(1,len(parts),2):pattern+="(.+?)"+re.escape(parts[i+1])
    pattern=pattern.replace(r"\ ",r"\s+")
    target=re.sub(r"\s+"," ",str(document_content)).strip()
    match=re.search(pattern,target)
    if not match:raise ValueError("masked content is not an exact contiguous source substring")
    values={index:value for index,value in zip(indices,match.groups())}
    if any(not str(value).strip() for value in values.values()):raise ValueError("empty MBA mask span")
    return values

def diagnosis_labels(*,causal_recall:float,target_not_causal:float,median_c1:float,
                     median_c2:float,parent_multiplicity:float)->list[str]:
    """Preregistered descriptive labels; thresholds are diagnostic, not gates."""
    labels=[]
    if causal_recall<.65:labels.append("S2_LOCATOR_MISMATCH")
    if target_not_causal>=.25:labels.append("S2_TARGET_NOT_CAUSAL")
    if parent_multiplicity>0 and causal_recall<.65:labels.append("S2_SAME_PARENT_RESIDUAL_LEAKAGE")
    if median_c1<.60 and median_c2>=.80:labels.append("S2_MULTI_PARENT_DISTRIBUTED_LEAKAGE")
    if not labels:labels.append("S2_GENERATOR_SIDE_RESIDUAL_LEAKAGE")
    if len(labels)>1:labels.append("S2_MIXED_FAILURE")
    return labels
