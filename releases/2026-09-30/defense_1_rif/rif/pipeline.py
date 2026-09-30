"""Composition of frozen CLD/G4 and old CF-Clip."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, TypeVar

from .cfclip import GAMMA
from .cld import TAU, apply_cld
from .serial_kvcache import SerialKVCacheCFClip

T = TypeVar("T")


class RIFDefense:
    def __init__(self, model, tokenizer, tau: float = TAU, gamma: float = GAMMA):
        if tau != TAU or gamma != GAMMA:
            raise ValueError("This release exposes only the frozen tau and gamma")
        self.tau = tau
        self.decoder = SerialKVCacheCFClip(model, tokenizer, gamma=gamma)

    def generate(
        self,
        scores: Sequence[float],
        documents: Sequence[T],
        prompt_builder: Callable[[Sequence[T]], Sequence[int]],
        max_new_tokens: int = 64,
    ) -> dict[str, Any]:
        decision = apply_cld(scores, documents, self.tau)
        context = list(decision.documents)
        if not context:
            raise ValueError("CLD removed the only generator-context document")
        branches = [list(prompt_builder(context))]
        for index in range(len(context)):
            dropped = context[:index] + context[index + 1 :]
            branches.append(list(prompt_builder(dropped)))
        decoded = self.decoder.decode(branches, max_new_tokens=max_new_tokens)
        return {
            **decoded,
            "g4": decision.g4,
            "cld_alarm": decision.alarm,
            "removed_document": decision.removed,
            "context_documents": context,
            "engine": "SERIAL_KVCACHE",
            "tau": TAU,
            "gamma": GAMMA,
        }

