"""S²-MIA-T membership features and threshold decision."""

from __future__ import annotations

import math
from collections import Counter

from protocols.common import ProtocolInputError, normalized_words, perplexity_from_logprobs


SCORE_POLARITY = "higher_bleu_and_lower_perplexity_are_more_member"


def _ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))


def sentence_bleu(reference: str, candidate: str, max_order: int = 4) -> float:
    """Compute unsmoothed sentence BLEU with brevity penalty.

    This matches the paper formula family without substituting embedding
    cosine.  Empty or too-short candidates legitimately receive zero.
    """

    ref = normalized_words(reference)
    cand = normalized_words(candidate)
    if not ref or not cand:
        return 0.0
    precisions: list[float] = []
    for n in range(1, max_order + 1):
        candidate_counts = _ngrams(cand, n)
        denominator = sum(candidate_counts.values())
        if denominator == 0:
            return 0.0
        reference_counts = _ngrams(ref, n)
        clipped = sum(min(count, reference_counts[gram]) for gram, count in candidate_counts.items())
        if clipped == 0:
            return 0.0
        precisions.append(clipped / denominator)
    brevity_penalty = 1.0 if len(cand) > len(ref) else math.exp(1.0 - len(ref) / len(cand))
    return brevity_penalty * math.exp(sum(math.log(p) for p in precisions) / max_order)


def extract_features(reference_text: str, generated_text: str, token_logprobs: list[float]) -> dict:
    if not isinstance(generated_text, str):
        raise ProtocolInputError("generated_text must be a string")
    return {
        "bleu": sentence_bleu(reference_text, generated_text),
        "perplexity": perplexity_from_logprobs(token_logprobs),
    }


def threshold_decision(features: dict, semantic_threshold: float, perplexity_threshold: float) -> int:
    try:
        bleu = float(features["bleu"])
        perplexity = float(features["perplexity"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ProtocolInputError("S2 features must contain numeric bleu and perplexity") from exc
    if not all(math.isfinite(v) for v in (bleu, perplexity, semantic_threshold, perplexity_threshold)):
        raise ProtocolInputError("S2 thresholds/features must be finite")
    return int(bleu >= semantic_threshold and perplexity <= perplexity_threshold)

