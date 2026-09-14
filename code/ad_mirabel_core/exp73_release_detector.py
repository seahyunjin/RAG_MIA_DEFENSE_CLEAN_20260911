"""Frozen Exp73 LoLA-v1 + Running-MinP + Mirabel rank-1 defense runtime.

This file reproduces the evaluated, domain-calibrated known-domain policy.
Read FINAL_VERDICT.json before deployment: Exp73 did not establish universal
cross-domain validity, and this runtime must not be described as universal.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np


def mirabel(scores, confidence=.95):
    values = np.asarray(scores, dtype=float)
    if values.ndim != 1 or len(values) < 3 or not np.isfinite(values).all():
        return {"margin": float("nan"), "threshold": float("nan"), "candidate_index": None}
    if np.any(values[:-1] < values[1:]):
        raise ValueError("scores must be in deterministic descending order")
    background = values[1:]
    root = math.sqrt(2 * math.log(len(values)))
    quantile = -math.log(-math.log(confidence))
    threshold = float(background.mean() + background.std(ddof=0) * root +
                      quantile * background.std(ddof=0) / root)
    return {"margin": float(values[0] - threshold), "threshold": threshold, "candidate_index": 0}


def normalize_ranking(ranked):
    output, seen = [], set()
    for item in ranked:
        if not isinstance(item, dict) or "id" not in item or "score" not in item:
            raise ValueError("each retrieval item requires id and score metadata")
        key, score = str(item["id"]), float(item["score"])
        if key in seen:
            continue
        if not math.isfinite(score):
            raise ValueError("retrieval scores must be finite")
        output.append(dict(item, id=key, score=score)); seen.add(key)
    output.sort(key=lambda item: (-item["score"], item["id"]))
    return output


def static_context(ranked, alarm, top_k=3):
    ordered = normalize_ranking(ranked)
    hidden = [ordered[0]["id"]] if alarm and ordered else []
    safe = [item for item in ordered if item["id"] not in set(hidden)][:top_k]
    return safe, hidden, len(safe) == top_k


@dataclass
class SessionState:
    turn: int = 0
    minimum_p: float = 1.0
    alarm: bool = False
    alarm_turn: int | None = None


class LoLADetector:
    def __init__(self, package_dir=None, device="cpu"):
        import torch
        from peft import LoraConfig, get_peft_model
        from safetensors.torch import load_file
        from transformers import AutoModel, AutoTokenizer

        self.torch, self.device = torch, str(device)
        self.directory = Path(package_dir or Path(__file__).resolve().parents[1])
        self.config = json.loads((self.directory / "model/config.json").read_text())
        self.session_config = json.loads((self.directory / "model/session_calibration.json").read_text())
        self.sessions: dict[str, SessionState] = {}
        base = os.environ.get("LOLA_BASE_MODEL_PATH", self.config["base_model_id"])
        local, revision = Path(base).exists(), self.config["base_model_revision"]
        self.tokenizer = AutoTokenizer.from_pretrained(base, revision=revision, local_files_only=local)
        encoder = AutoModel.from_pretrained(base, revision=revision, local_files_only=local)
        for parameter in encoder.parameters():
            parameter.requires_grad = False
        adapter = LoraConfig(r=8, lora_alpha=16, lora_dropout=.05, target_modules=["q", "v"],
            layers_to_transform=[10, 11], layers_pattern="layer", bias="none")
        self.encoder = get_peft_model(encoder, adapter).to(self.device).eval()
        self.head = torch.nn.Linear(768, 1).to(self.device).eval()
        state = load_file(str(self.directory / "model/intent_adapter.safetensors"))
        named = dict(self.encoder.named_parameters())
        with torch.no_grad():
            for key, value in state.items():
                if key.startswith("encoder."):
                    target = key[len("encoder."):]
                    if target not in named:
                        raise RuntimeError(f"checkpoint tensor not found: {target}")
                    named[target].copy_(value.to(named[target].device))
            self.head.weight.copy_(state["intent_head.weight"].to(self.head.weight.device))
            self.head.bias.copy_(state["intent_head.bias"].to(self.head.bias.device))

    def score_intent(self, query):
        encoded = self.tokenizer([str(query)], padding=True, truncation=True, max_length=256,
                                 return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            hidden = self.encoder(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).to(hidden.dtype)
            pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1e-9)
            pooled = self.torch.nn.functional.normalize(pooled.float(), dim=1)
            return float(self.head(pooled).squeeze().cpu())

    def _empirical_p(self, score: float, domain: str) -> float:
        reference = np.asarray(self.session_config["reference_scores"][domain], dtype=float)
        greater_equal = len(reference) - int(np.searchsorted(reference, score, side="left"))
        return float((greater_equal + 1.0) / (len(reference) + 1.0))

    def decide_score(self, score: float, domain: str, session_id: str | None = None) -> dict[str, Any]:
        query_threshold = float(self.config["intent_thresholds"].get(domain, self.config["default_threshold"]))
        stateless_alarm = float(score) > query_threshold
        if session_id is None:
            return {"alarm": stateless_alarm, "stateless_alarm": stateless_alarm, "turn": 1,
                    "running_minp": None, "running_minp_risk": None, "alarm_turn": 1 if stateless_alarm else None}
        state = self.sessions.setdefault(str(session_id), SessionState())
        state.turn += 1
        state.minimum_p = min(state.minimum_p, self._empirical_p(float(score), domain))
        risk = -math.log10(max(state.minimum_p, np.finfo(float).tiny))
        horizon = min(state.turn, max(map(int, self.session_config["horizon_thresholds"])))
        threshold = float(self.session_config["horizon_thresholds"][str(horizon)])
        current = stateless_alarm if state.turn == 1 else risk > threshold
        if current and not state.alarm:
            state.alarm_turn = state.turn
        state.alarm = state.alarm or current
        return {"alarm": state.alarm, "stateless_alarm": stateless_alarm, "turn": state.turn,
                "running_minp": state.minimum_p, "running_minp_risk": risk,
                "session_threshold": threshold, "alarm_turn": state.alarm_turn}

    def reset_session(self, session_id: str) -> None:
        self.sessions.pop(str(session_id), None)

    def defend_query(self, query, retriever, corpus, session_id=None):
        ranked = normalize_ranking(list(retriever(query, corpus, return_all_scores=True)))
        domain = str(getattr(retriever, "domain", "default"))
        score = self.score_intent(query)
        decision = self.decide_score(score, domain, session_id=session_id)
        safe, hidden, complete = static_context(ranked, decision["alarm"], self.config["top_k"])
        stats = mirabel([item["score"] for item in ranked])
        audit = decision | {"intent_score": score, "hidden_document_id": hidden[0] if hidden else None,
            "retrieved_doc_ids_before": [x["id"] for x in ranked[:self.config["top_k"]]],
            "retrieved_doc_ids_after": [x["id"] for x in safe], "backfill_complete": complete,
            "mirabel_margin": stats["margin"], "ranking_recomputed_after_hide": False,
            "explicit_refusal": False}
        return safe, audit

    def close(self):
        del self.encoder, self.head
        if self.device.startswith("cuda") and self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def defend_query(query, retriever, corpus, package_dir=None, device="cpu", session_id=None):
    detector = LoLADetector(package_dir=package_dir, device=device)
    try:
        return detector.defend_query(query, retriever, corpus, session_id=session_id)
    finally:
        detector.close()
