"""Safe, versioned serialization for Adaptive Fisher detector artifacts."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .detector_calibration import (
    AdaptiveFisherCalibration,
    HorizonCalibration,
    QueryEvidenceModel,
)


ARTIFACT_SCHEMA_VERSION = "ad_mirabel_adaptive_fisher_artifact_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_detector_artifact(
    directory: str | Path,
    calibration: AdaptiveFisherCalibration,
    *,
    demonstration_only: bool = False,
) -> dict[str, Any]:
    target = Path(directory)
    target.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {
        "query_score_reference": np.asarray(calibration.query_model.query_score_reference),
    }
    for index, reference in enumerate(calibration.query_model.feature_references):
        arrays[f"feature_reference_{index}"] = np.asarray(reference)
    for name in ("scaler_mean", "scaler_scale", "coefficients"):
        value = getattr(calibration.query_model, name)
        if value is not None:
            arrays[name] = np.asarray(value)
    horizons: dict[str, dict[str, Any]] = {}
    for horizon, item in sorted(calibration.horizons.items()):
        for index, reference in enumerate(item.partial_fisher_references, 1):
            arrays[f"h{horizon}_partial_{index}"] = np.asarray(reference)
        arrays[f"h{horizon}_adaptive"] = np.asarray(item.adaptive_score_reference)
        horizons[str(horizon)] = {
            "threshold": item.threshold,
            "target_fpr": item.target_fpr,
            "partial_reference_sessions": item.partial_reference_sessions,
            "adaptive_reference_sessions": item.adaptive_reference_sessions,
            "threshold_sessions": item.threshold_sessions,
        }
    arrays_path = target / "arrays.npz"
    np.savez_compressed(arrays_path, **arrays)
    metadata = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "demonstration_only": bool(demonstration_only),
        "warning": (
            "FOR DEMONSTRATION ONLY; NOT A DEPLOYMENT THRESHOLD; "
            "NOT TRAINED ON REAL ATTACK DATA"
            if demonstration_only else None
        ),
        "query_model": {
            "kind": calibration.query_model.kind,
            "common_top_k": calibration.query_model.common_top_k,
            "feature_reference_count": len(calibration.query_model.feature_references),
            "intercept": calibration.query_model.intercept,
            "logistic_c": calibration.query_model.logistic_c,
        },
        "horizons": horizons,
        "model_version": calibration.model_version,
        "threshold_version": calibration.threshold_version,
        "feature_schema_version": calibration.feature_schema_version,
        "retriever_id": calibration.retriever_id,
        "retriever_schema_version": calibration.retriever_schema_version,
        "mirabel_formula_version": calibration.mirabel_formula_version,
        "arrays_sha256": sha256_file(arrays_path),
    }
    metadata_path = target / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "metadata_sha256": sha256_file(metadata_path),
        "arrays_sha256": sha256_file(arrays_path),
    }
    (target / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def load_detector_artifact(directory: str | Path) -> AdaptiveFisherCalibration:
    source = Path(directory)
    metadata_path = source / "metadata.json"
    arrays_path = source / "arrays.npz"
    manifest_path = source / "MANIFEST.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if metadata.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError("detector artifact schema mismatch")
    if sha256_file(metadata_path) != manifest.get("metadata_sha256"):
        raise ValueError("detector artifact metadata hash mismatch")
    if sha256_file(arrays_path) != manifest.get("arrays_sha256"):
        raise ValueError("detector artifact array hash mismatch")
    if manifest.get("arrays_sha256") != metadata.get("arrays_sha256"):
        raise ValueError("detector artifact cross-hash mismatch")
    with np.load(arrays_path, allow_pickle=False) as archive:
        arrays = {name: np.asarray(archive[name], dtype=float) for name in archive.files}
    query = metadata["query_model"]
    feature_references = tuple(
        arrays[f"feature_reference_{index}"]
        for index in range(int(query["feature_reference_count"]))
    )
    query_model = QueryEvidenceModel(
        kind=str(query["kind"]),
        common_top_k=int(query["common_top_k"]),
        feature_references=feature_references,
        query_score_reference=arrays["query_score_reference"],
        scaler_mean=arrays.get("scaler_mean"),
        scaler_scale=arrays.get("scaler_scale"),
        coefficients=arrays.get("coefficients"),
        intercept=query.get("intercept"),
        logistic_c=query.get("logistic_c"),
    )
    horizons = {}
    for key, item in metadata["horizons"].items():
        horizon = int(key)
        horizons[horizon] = HorizonCalibration(
            horizon=horizon,
            partial_fisher_references=tuple(
                arrays[f"h{horizon}_partial_{index}"] for index in range(1, horizon + 1)
            ),
            adaptive_score_reference=arrays[f"h{horizon}_adaptive"],
            threshold=float(item["threshold"]),
            target_fpr=float(item["target_fpr"]),
            partial_reference_sessions=int(item["partial_reference_sessions"]),
            adaptive_reference_sessions=int(item["adaptive_reference_sessions"]),
            threshold_sessions=int(item["threshold_sessions"]),
        )
    return AdaptiveFisherCalibration(
        query_model=query_model,
        horizons=horizons,
        model_version=str(metadata["model_version"]),
        threshold_version=str(metadata["threshold_version"]),
        feature_schema_version=str(metadata["feature_schema_version"]),
        retriever_id=str(metadata["retriever_id"]),
        retriever_schema_version=str(metadata["retriever_schema_version"]),
        mirabel_formula_version=str(metadata["mirabel_formula_version"]),
    )


def validate_detector_artifact(directory: str | Path) -> dict[str, Any]:
    calibration = load_detector_artifact(directory)
    return {
        "valid": True,
        "model_version": calibration.model_version,
        "threshold_version": calibration.threshold_version,
        "retriever_id": calibration.retriever_id,
        "horizons": sorted(calibration.horizons),
    }
