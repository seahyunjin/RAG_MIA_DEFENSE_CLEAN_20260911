"""Deterministic connected-component splitting for Exp76-R."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import re
from typing import Iterable, Sequence

import numpy as np
import pandas as pd


NORMALIZATION_VERSION = "exp76-v1-casefold-punctuation-space-collapse"


def normalize_query(value: object) -> str:
    """Exact normalization used by the frozen Exp76 duplicate audit.

    Python's Unicode-aware ``casefold`` is applied. Punctuation is replaced by
    spaces and runs of whitespace are collapsed. No NFC/NFKC transform is used.
    """

    text = str(value).casefold()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def query_group_id(value: object) -> str:
    return sha256_text(normalize_query(value))


class UnionFind:
    def __init__(self, size: int):
        self.parent = list(range(size))
        self.weight = [1] * size

    def find(self, value: int) -> int:
        while self.parent[value] != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = self.parent[value]
        return value

    def union(self, left: int, right: int) -> None:
        left, right = self.find(left), self.find(right)
        if left == right:
            return
        if self.weight[left] < self.weight[right]:
            left, right = right, left
        self.parent[right] = left
        self.weight[left] += self.weight[right]


def _available_identity_columns(frame: pd.DataFrame, optional: Sequence[str] = ()) -> list[str]:
    requested = ["query_group_id", "target_document_id", "source_document_id", *optional]
    return [column for column in requested if column in frame.columns and frame[column].notna().any()]


def connected_component_manifest(
    frame: pd.DataFrame,
    optional_identity_columns: Sequence[str] = (),
) -> tuple[pd.DataFrame, dict[str, object]]:
    if frame.row_id.duplicated().any():
        raise ValueError("row_id must be unique before component construction")
    value = frame.copy().reset_index(drop=True)
    value["normalized_query"] = value["query"].map(normalize_query)
    value["query_group_id"] = value["normalized_query"].map(sha256_text)
    identities = _available_identity_columns(value, optional_identity_columns)
    union = UnionFind(len(value))
    for column in identities:
        first: dict[str, int] = {}
        for index, item in enumerate(value[column]):
            if pd.isna(item):
                continue
            identity = str(item)
            if not identity or identity.casefold() == "nan":
                continue
            if identity in first:
                union.union(index, first[identity])
            else:
                first[identity] = index
    members: defaultdict[int, list[str]] = defaultdict(list)
    for index, row_id in enumerate(value.row_id.astype(str)):
        members[union.find(index)].append(row_id)
    root_to_id = {
        root: "cc-" + sha256_text("\n".join(sorted(row_ids)))
        for root, row_ids in members.items()
    }
    value["split_component_id"] = [root_to_id[union.find(index)] for index in range(len(value))]
    sizes = value.groupby("split_component_id").size()
    stats = {
        "rows": int(len(value)),
        "components": int(len(sizes)),
        "largest_component_size": int(sizes.max()),
        "component_size_distribution": {str(key): int(count) for key, count in Counter(sizes).items()},
        "identity_columns": identities,
        "kind_counts": {str(key): int(count) for key, count in value.kind.value_counts().items()},
        "component_kind_counts": {
            kind: int(value.loc[value.kind.eq(kind), "split_component_id"].nunique())
            for kind in sorted(value.kind.unique())
        },
    }
    return value, stats


def _stable_order_key(component_id: str, seed: int) -> str:
    return sha256_text(f"exp76r|{seed}|{component_id}")


def assign_components(
    manifest: pd.DataFrame,
    ratios: dict[str, float],
    seed: int,
) -> pd.DataFrame:
    """Greedily match precommitted multi-stratum targets at component level."""

    if not np.isclose(sum(ratios.values()), 1.0):
        raise ValueError("split ratios must sum to one")
    splits = tuple(ratios)
    feature_names = ["total", "benign", "attack"]
    feature_names += [f"family={value}" for value in sorted(manifest.loc[manifest.kind.eq("attack"), "family"].dropna().unique())]
    feature_names += [f"member={int(value)}" for value in sorted(manifest.loc[manifest.kind.eq("attack"), "member"].dropna().unique())]
    feature_names += [f"domain={value}" for value in sorted(manifest.domain.dropna().unique())]

    def vector(cell: pd.DataFrame) -> np.ndarray:
        values = [len(cell), int(cell.kind.eq("benign").sum()), int(cell.kind.eq("attack").sum())]
        values += [int((cell.kind.eq("attack") & cell.family.eq(name.split("=", 1)[1])).sum())
                   for name in feature_names if name.startswith("family=")]
        values += [int((cell.kind.eq("attack") & cell.member.eq(int(name.split("=", 1)[1]))).sum())
                   for name in feature_names if name.startswith("member=")]
        values += [int(cell.domain.eq(name.split("=", 1)[1]).sum())
                   for name in feature_names if name.startswith("domain=")]
        return np.asarray(values, dtype=float)

    grouped = []
    for component_id, cell in manifest.groupby("split_component_id", sort=False):
        grouped.append((str(component_id), vector(cell), len(cell)))
    grouped.sort(key=lambda row: (-row[2], _stable_order_key(row[0], seed)))
    total = vector(manifest)
    targets = {split: total * float(ratios[split]) for split in splits}
    assigned = {split: np.zeros_like(total) for split in splits}
    component_split: dict[str, str] = {}
    scale = np.maximum(total, 1.0)

    for component_id, values, _ in grouped:
        choices = []
        for split in splits:
            hypothetical = {name: assigned[name].copy() for name in splits}
            hypothetical[split] += values
            error = sum(float(np.square((hypothetical[name] - targets[name]) / scale).sum()) for name in splits)
            tie = _stable_order_key(f"{component_id}|{split}", seed)
            choices.append((error, tie, split))
        selected = min(choices)[2]
        component_split[component_id] = selected
        assigned[selected] += values

    result = manifest.copy()
    result["split"] = result.split_component_id.map(component_split)
    if result.groupby("split_component_id").split.nunique().max() != 1:
        raise RuntimeError("component assignment was not atomic")
    return result


def disjoint_audit(manifest: pd.DataFrame) -> dict[str, object]:
    split_names = sorted(manifest.split.unique())

    def overlap(column: str) -> int | None:
        if column not in manifest.columns:
            return None
        owners: dict[str, set[str]] = defaultdict(set)
        for split, item in manifest[["split", column]].itertuples(index=False):
            if pd.isna(item) or not str(item) or str(item).casefold() == "nan":
                continue
            owners[str(item)].add(str(split))
        return sum(len(splits) > 1 for splits in owners.values())

    checks = {
        "normalized_query_overlap": overlap("normalized_query"),
        "query_group_overlap": overlap("query_group_id"),
        "split_component_overlap": overlap("split_component_id"),
        "target_document_overlap": overlap("target_document_id"),
        "source_document_overlap": overlap("source_document_id"),
        "exact_query_hash_overlap": overlap("query_sha256"),
    }
    attack = manifest[manifest.kind.eq("attack")]
    benign = manifest[manifest.kind.eq("benign")]
    checks["attack_exact_duplicate_overlap"] = overlap("query_sha256") if len(attack) == len(manifest) else _subset_overlap(attack, "query_sha256")
    checks["benign_exact_duplicate_overlap"] = _subset_overlap(benign, "query_sha256")
    passed = all(value in (0, None) for value in checks.values())
    return {"status": "PASS" if passed else "FAIL", "splits": split_names, **checks}


def _subset_overlap(frame: pd.DataFrame, column: str) -> int | None:
    if column not in frame.columns:
        return None
    owners: defaultdict[str, set[str]] = defaultdict(set)
    for split, item in frame[["split", column]].itertuples(index=False):
        if pd.notna(item) and str(item) and str(item).casefold() != "nan":
            owners[str(item)].add(str(split))
    return sum(len(splits) > 1 for splits in owners.values())
