"""Frozen native-budget metadata for the six locally available RAG MIA attacks."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AttackProtocol:
    name: str
    native_budget: int
    primary_common_budget: int
    query_mechanism: str
    membership_mechanism: str
    response_required_for_native_utility: bool


PROTOCOLS = {
    "RAG-MIA": AttackProtocol(
        "RAG-MIA", 1, 1, "direct target-document context question",
        "binary response-based membership score", True,
    ),
    "S2-MIA": AttackProtocol(
        "S2-MIA", 1, 1, "semantic-sensitivity prompt",
        "paired semantic response change", True,
    ),
    "MBA": AttackProtocol(
        "MBA", 1, 1, "masked target-document reconstruction prompt",
        "mask reconstruction likelihood/accuracy", True,
    ),
    "IA-MIA": AttackProtocol(
        "IA-MIA", 15, 5, "iterative document-specific question sequence",
        "aggregated answer leakage", True,
    ),
    "DCMI": AttackProtocol(
        "DCMI", 2, 2, "original and perturbed document pair",
        "response sensitivity to document perturbation", True,
    ),
    "MEntA": AttackProtocol(
        "MEntA", 5, 5, "five document-specific generated questions",
        "answer entailment leakage", True,
    ),
}


def protocol(name: str) -> AttackProtocol:
    try:
        return PROTOCOLS[str(name)]
    except KeyError as error:
        raise KeyError(f"unknown attack family: {name}") from error
