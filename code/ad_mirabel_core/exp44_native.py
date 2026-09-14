"""Locked native-MIA utilities used by Exp44.

This module deliberately contains no HTTP/API client.  It separates target
responses from detector decisions and exposes only the observations permitted
by the Exp44 block-policy lock.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
from typing import Iterable, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BLOCKED_RESPONSE = "REQUEST_BLOCKED"
SIDE_FEATURES = (
    "native_score", "native_score_missing", "blocked_count",
    "blocked_fraction", "first_block_turn", "terminal_block",
)


def sha256_text(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def normalize_yes_no(value: str) -> str:
    """Official IA-compatible normalization to Yes/No/unknown."""
    text = str(value or "").strip().lower()
    unknown = (
        "i don't know", "i do not know", "don't know", "do not know",
        "not sure", "cannot find", "can't find", "unable to find",
    )
    if value == BLOCKED_RESPONSE or any(item in text for item in unknown):
        return "UNKNOWN"
    if text.startswith("yes") or re.search(r"\byes\b", text):
        return "YES"
    if text.startswith("no") or re.search(r"\bno\b", text):
        return "NO"
    return "UNKNOWN"


def rag_mia_score(response: str) -> float:
    """RAG-MIA/DCMI first explicit Yes/No response score."""
    if response == BLOCKED_RESPONSE:
        return 0.0
    match = re.search(r"\b(yes|no)\b", str(response or ""), re.I)
    return float(bool(match) and match.group(1).casefold() == "yes")


def dcmi_score(original_response: str, perturbed_response: str) -> float:
    """Official calibrated direct-query difference in {-1,0,1}."""
    return rag_mia_score(original_response) - rag_mia_score(perturbed_response)


def ia_turn_score(rag_response: str, ground_truth_response: str, penalty: float = 1.0) -> float | None:
    """Official IA-MIA per-question score; unknown GT questions are skipped."""
    rag = normalize_yes_no(rag_response)
    ground_truth = normalize_yes_no(ground_truth_response)
    if ground_truth not in {"YES", "NO"}:
        return None
    if rag == ground_truth:
        return 1.0
    if rag == "UNKNOWN":
        return -float(penalty)
    return 0.0


def ia_mia_score(rag_responses: Sequence[str], ground_truth_responses: Sequence[str]) -> float:
    values = [ia_turn_score(rag, gt) for rag, gt in zip(rag_responses, ground_truth_responses)]
    kept = [float(value) for value in values if value is not None]
    if not kept:
        raise ValueError("IA-MIA has no Yes/No ground-truth answers")
    return float(np.mean(kept))


def parse_mask_predictions(response: str) -> dict[int, str]:
    return {
        int(index): value.strip()
        for index, value in re.findall(r"\[Mask_(\d+)\]:\s*([^\n]+)", str(response), re.I)
    }


def mba_accuracy(mask_values: Mapping[int, str] | Sequence[str], response: str) -> float:
    truth = dict(mask_values) if isinstance(mask_values, Mapping) else dict(enumerate(mask_values))
    if not truth:
        raise ValueError("MBA mask_values are empty")
    predictions = parse_mask_predictions(response)
    correct = sum(
        index in predictions and predictions[index].strip().casefold() == value.strip().casefold()
        for index, value in truth.items()
    )
    return float(correct / len(truth))


def extract_mba_masked_document(query: str) -> str:
    prefix = "Predict the answers of the masks in the given texts:"
    suffix = "You should strictly follow"
    if prefix not in query or suffix not in query:
        raise ValueError("not an original MBA query template")
    return re.sub(r"\s+", " ", query.split(prefix, 1)[1].split(suffix, 1)[0]).strip()


def recover_mba_mask_values(query: str, target_document: str) -> dict[int, str]:
    """Recover exact released MBA mask spans by alignment to its source document.

    Only exact whitespace-normalized matches are accepted.  Malformed released
    queries (missing indices, extra brackets, or source drift) raise instead of
    being repaired heuristically.
    """
    masked = extract_mba_masked_document(query)
    parts = re.split(r"\[Mask_(\d+)\]", masked)
    if len(parts) < 3 or len(parts) % 2 == 0:
        raise ValueError("malformed MBA placeholder sequence")
    indices = [int(parts[index]) for index in range(1, len(parts), 2)]
    if indices != list(range(len(indices))):
        raise ValueError("MBA placeholder indices are not contiguous")
    pattern = re.escape(parts[0])
    for index in range(1, len(parts), 2):
        pattern += "(.+?)" + re.escape(parts[index + 1])
    pattern = pattern.replace(r"\ ", r"\s+")
    target = re.sub(r"\s+", " ", target_document).strip()
    match = re.fullmatch(pattern, target)
    if not match:
        raise ValueError("MBA masked document is not an exact source-document transform")
    values = {index: value for index, value in zip(indices, match.groups())}
    if any(not value.strip() for value in values.values()):
        raise ValueError("empty MBA mask span")
    return values


def exact_selected_token_logprobs(logits, selected_token_ids):
    """Return exact checkpoint log P(selected token) from causal-LM logits."""
    import torch

    if logits.shape[:-1] != selected_token_ids.shape:
        raise ValueError("logits/token shape mismatch")
    return torch.log_softmax(logits, dim=-1).gather(-1, selected_token_ids.unsqueeze(-1)).squeeze(-1)


def s2_semantic_scores(knowledge: Sequence[str], responses: Sequence[str], embedding_model) -> np.ndarray:
    """Official S2 main score: normalized cosine semantic similarity."""
    if len(knowledge) != len(responses):
        raise ValueError("S2 input length mismatch")
    left = embedding_model.encode(list(knowledge), convert_to_numpy=True, normalize_embeddings=True,
                                  batch_size=128, show_progress_bar=False)
    right = embedding_model.encode(list(responses), convert_to_numpy=True, normalize_embeddings=True,
                                   batch_size=128, show_progress_bar=False)
    return (np.sum(left * right, axis=1) + 1.0) / 2.0


def safe_auc(labels: Sequence[int], scores: Sequence[float]) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if len(np.unique(labels)) != 2 or not np.isfinite(scores).all():
        raise ValueError("AUROC requires finite scores and both labels")
    return float(roc_auc_score(labels, scores))


def tpr_at_fpr(labels: Sequence[int], scores: Sequence[float], target: float) -> tuple[float, float]:
    fpr, tpr, _ = roc_curve(np.asarray(labels, dtype=int), np.asarray(scores, dtype=float))
    valid = np.flatnonzero(fpr <= float(target) + 1e-15)
    if not len(valid):
        return 0.0, 0.0
    index = valid[np.argmax(tpr[valid])]
    return float(tpr[index]), float(fpr[index])


def neutral_evidence(labels: Sequence[int], scores: Sequence[float]) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    selected = scores[labels == 0]
    if not len(selected) or not np.isfinite(selected).all():
        raise ValueError("neutral evidence requires finite nonmember meta-calibration scores")
    return float(np.median(selected))


def masked_native_observation(native_score: float, *, blocked: bool, neutral: float) -> dict:
    """Prevent target response/logprob leakage after a block."""
    return {
        "native_score": float(neutral if blocked else native_score),
        "native_score_missing": int(bool(blocked)),
        "response_text": BLOCKED_RESPONSE if blocked else None,
        "token_logprobs": None,
    }


def side_channel_row(native_score: float, native_missing: int, first_block: int | None,
                     native_budget: int) -> list[float]:
    blocked_count = 0 if first_block is None else native_budget - int(first_block) + 1
    return [
        float(native_score), float(native_missing), float(blocked_count),
        float(blocked_count / native_budget),
        float(0 if first_block is None else first_block / native_budget),
        float(first_block is not None and first_block <= native_budget),
    ]


@dataclass
class SideChannelAttacker:
    model: object

    @classmethod
    def fit(cls, features: Sequence[Sequence[float]], labels: Sequence[int]):
        labels = np.asarray(labels, dtype=int)
        if len(np.unique(labels)) != 2:
            raise ValueError("side-channel meta-calibration requires both labels")
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=1.0, penalty="l2", solver="liblinear", random_state=44, max_iter=1000),
        )
        model.fit(np.asarray(features, dtype=float), labels)
        return cls(model)

    def score(self, features: Sequence[Sequence[float]]) -> np.ndarray:
        return self.model.predict_proba(np.asarray(features, dtype=float))[:, 1]


def grouped_meta_split(group_key: str) -> str:
    """Frozen 30/70 split based only on a stable source/query-set group key."""
    return "attack_meta_calibration" if int(sha256_text("exp44-meta|" + group_key)[:16], 16) % 10 < 3 else "attack_final_evaluation"


def gate_f_verdict(gates: Mapping[str, bool | None]) -> str:
    """Map pre-registered F0--F7 results to one allowed Exp44 status."""
    if gates.get("F0") is not True or gates.get("F7") is not True:
        return "GATE_F_NOT_EVALUABLE"
    if gates.get("F1") and gates.get("F2") and not gates.get("F3"):
        return "GATE_F_NATIVE_PASSED_SIDE_CHANNEL_FAILED"
    if gates.get("F1") is True and gates.get("F2") is False:
        return "GATE_F_IMPROVES_BUT_ABSOLUTE_REDUCTION_INSUFFICIENT"
    if all(gates.get(name) is True for name in ("F0", "F1", "F2", "F3", "F4", "F5", "F6", "F7")):
        return "GATE_F_PASSED"
    return "GATE_F_FAILED"


def json_hash(value: object) -> str:
    return sha256_text(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")))
