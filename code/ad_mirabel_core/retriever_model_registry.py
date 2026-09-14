"""Retriever-specific frozen model registry for Experiment 40.

The registry branches only on the deployed retriever geometry.  It never
branches on an attack family, query budget, ordering, dataset, or label.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from .detector_artifact import load_detector_artifact, validate_detector_artifact


class UnsupportedRetrieverError(KeyError):
    """Raised when no calibrated detector exists for a retriever."""


@dataclass(frozen=True)
class RetrieverArtifact:
    retriever_id: str
    artifact_path: Path
    retrieval_model_version: str
    response_model_version: str
    calibration_version: str
    threshold_version: str


class RetrieverModelRegistry:
    """Load and validate one frozen Adaptive-Fisher artifact per retriever."""

    def __init__(self, entries: Mapping[str, RetrieverArtifact]) -> None:
        self._entries = {str(key): value for key, value in entries.items()}
        if not self._entries:
            raise ValueError("at least one retriever artifact is required")
        for retriever_id, entry in self._entries.items():
            if retriever_id != entry.retriever_id:
                raise ValueError("registry key and artifact retriever_id disagree")
            audit = validate_detector_artifact(entry.artifact_path)
            if audit["retriever_id"] != retriever_id:
                raise ValueError("serialized artifact retriever_id mismatch")

    @classmethod
    def exp40_default(cls, project_root: str | Path, *, seed: int = 42) -> "RetrieverModelRegistry":
        root = Path(project_root).resolve()
        gte = root / "experiments/38_ROBUST_LDF_VS_LEGACY_V16/MODEL_ARTIFACTS/Robust-LDF-Linear" / f"seed_{seed}/gte"
        mpnet = root / "experiments/39_LDF_SESSION_DRO_RESPONSE_DIAGNOSTIC/MODEL_ARTIFACTS/LDF-SessionDRO-Linear" / f"seed_{seed}/mpnet"
        entries = {}
        for retriever_id, path, retrieval_version in (
            ("gte", gte, "thenlper/gte-large@4bef63f39fcc"),
            ("mpnet", mpnet, "sentence-transformers/all-mpnet-base-v2@e8c3b32edf54"),
        ):
            calibration = load_detector_artifact(path)
            entries[retriever_id] = RetrieverArtifact(
                retriever_id=retriever_id,
                artifact_path=path,
                retrieval_model_version=retrieval_version,
                response_model_version="none-retrieval-only",
                calibration_version=calibration.model_version,
                threshold_version=calibration.threshold_version,
            )
        return cls(entries)

    def entry(self, retriever_id: str) -> RetrieverArtifact:
        try:
            return self._entries[str(retriever_id)]
        except KeyError as error:
            raise UnsupportedRetrieverError(str(retriever_id)) from error

    def calibration(self, retriever_id: str):
        return load_detector_artifact(self.entry(retriever_id).artifact_path)

    def manifest(self) -> list[dict[str, object]]:
        return [
            {
                "retriever_id": key,
                "artifact_path": str(entry.artifact_path),
                "retrieval_model_version": entry.retrieval_model_version,
                "response_model_version": entry.response_model_version,
                "calibration_version": entry.calibration_version,
                "threshold_version": entry.threshold_version,
            }
            for key, entry in sorted(self._entries.items())
        ]
