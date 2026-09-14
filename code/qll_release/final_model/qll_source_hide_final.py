#!/usr/bin/env python3
"""Canonical Stateless QLL Source Hide inference path."""
from __future__ import annotations
import numpy as np

TOP_K = 4
TOTAL_CONTEXT_TOKENS = 2048
PER_SOURCE_BASE_TOKENS = 512

def mean_query_log_likelihood(model, tokenizer, source: str, query: str) -> float:
    before = tokenizer("Context:\n", add_special_tokens=False).input_ids
    source_ids = tokenizer(str(source), add_special_tokens=False).input_ids
    bridge = tokenizer("\n\nUser query:\n", add_special_tokens=False).input_ids
    query_ids = tokenizer(str(query), add_special_tokens=False).input_ids
    maximum = int(model.config.max_position_embeddings)
    source_ids = source_ids[:maximum-len(before)-len(bridge)-len(query_ids)]
    sequence = before + source_ids + bridge + query_ids
    import torch
    ids = torch.tensor(sequence, device=model.device).unsqueeze(0)
    with torch.inference_mode(): logits = model(ids).logits[0, -len(query_ids)-1:-1].float()
    targets = ids[0, -len(query_ids):]
    return float(torch.log_softmax(logits, -1).gather(1, targets[:,None]).mean().cpu())

def qll_distribution(scores):
    values = np.asarray(scores, dtype=float)
    values = np.exp(values-values.max())
    return values/values.sum()

def decision(scores, threshold):
    probabilities = qll_distribution(scores)
    index = int(np.argmax(probabilities))
    return {"dominance": float(probabilities[index]), "hide": bool(probabilities[index] > threshold),
            "source_index": index}

def equal_redistribution(lengths, hidden=None):
    caps=[0]*len(lengths); remaining=TOTAL_CONTEXT_TOKENS
    active=[i for i,n in enumerate(lengths) if i!=hidden and n>0]
    while remaining>0 and active:
        share=max(1,remaining//len(active)); changed=False
        for i in list(active):
            add=min(share,lengths[i]-caps[i],remaining); caps[i]+=add; remaining-=add; changed|=add>0
            if caps[i]>=lengths[i]: active.remove(i)
            if remaining<=0: break
        if not changed: break
    return caps

def no_redistribution(lengths, hidden=None):
    caps=[min(max(0,int(n)),PER_SOURCE_BASE_TOKENS) for n in lengths]
    if hidden is not None: caps[int(hidden)]=0
    return caps

def defend(retrieved_sources, query, qll_scorer, threshold, tokenizer, generator, system_prompt,
           context_policy="EQUAL_REDISTRIBUTION"):
    if len(retrieved_sources)!=TOP_K: raise ValueError("exactly four retrieved sources required")
    scores=[qll_scorer(source,query) for source in retrieved_sources]
    action=decision(scores,threshold); hidden=action["source_index"] if action["hide"] else None
    lengths=[len(tokenizer(s,add_special_tokens=False).input_ids) for s in retrieved_sources]
    allocator=no_redistribution if context_policy=="NO_REDISTRIBUTION" and hidden is not None else equal_redistribution
    caps=allocator(lengths,hidden)
    visible=[tokenizer.decode(tokenizer(s,add_special_tokens=False).input_ids[:cap],skip_special_tokens=True)
             for s,cap in zip(retrieved_sources,caps) if cap]
    answer=generator(query=query,documents=visible,system_prompt=system_prompt)
    return answer, action | {"caps":caps}
