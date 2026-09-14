#!/usr/bin/env python3
"""Retraining-free V0/V10/V25/V50 database-churn validation."""
from __future__ import annotations

import csv
import gzip
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from common import BGE, DOMAINS, EXP, K_LOCAL, PARENT, RAW, atomic_json, atomic_text, checkpoint, empirical_upper_tail, now, read_jsonl, sha_file, sha_text, write_csv, write_jsonl

sys.path.insert(0, str(PARENT.parents[1] / "code"))
from ad_mirabel_core.canonical_mirabel import canonical_mirabel_from_moments  # noqa: E402


PRECOMMIT = EXP / "configs/LC_MIRABEL_LARGE_V1_PRECOMMIT.json"
VERSIONS = {"V0": 0.0, "V10": .10, "V25": .25, "V50": .50}


def verify() -> dict:
    digest = PRECOMMIT.with_suffix(".sha256").read_text().split()[0]
    pre = json.loads(PRECOMMIT.read_text())
    if sha_file(PRECOMMIT) != digest or pre["lineage"]["code_sha256"].get(Path(__file__).name) != sha_file(Path(__file__)):
        raise RuntimeError("precommit/code checksum mismatch")
    if json.loads((EXP / "E2E_RESULT.json").read_text()).get("verdict") != "LC_MIRABEL_E2E_PASS":
        raise RuntimeError("churn forbidden because E2E gate did not pass")
    return pre


def construct_versions() -> dict[str, list[dict]]:
    original = read_jsonl(PARENT / "inputs/CLEAN_CORE3_PROTECTED_DB.jsonl")
    target_rows = list(csv.DictReader((EXP / "inputs/LARGE_SHARED_TARGETS.csv").open(encoding="utf-8")))
    member_ids = {row["document_id"] for row in target_rows if row["membership"] == "member"}
    nonmember_ids = {row["document_id"] for row in target_rows if row["membership"] == "nonmember"}
    target_hashes = {row["normalized_text_hash"] for row in target_rows}
    original_ids = {row["document_id"] for row in original}
    original_hashes = {row["normalized_text_hash"] for row in original}
    background = [row for row in original if row["document_id"] not in member_ids]
    if len(member_ids) != 1000 or not member_ids <= original_ids or nonmember_ids & original_ids:
        raise RuntimeError("membership invariant failed before churn")
    candidates: dict[str, list[dict]] = defaultdict(list)
    with gzip.open(PARENT / "inputs/COMMON_ELIGIBLE_POOL.csv.gz", "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["document_id"] in original_ids or row["document_id"] in nonmember_ids or row["normalized_text_hash"] in original_hashes | target_hashes:
                continue
            candidates[row["domain"]].append(row)
    for domain in DOMAINS:
        candidates[domain].sort(key=lambda row: (sha_text("LC_CHURN_ADD||"+row["document_id"]+"||"+row["normalized_text_hash"]), row["document_id"]))
    removal_order = sorted(background, key=lambda row: (sha_text("LC_CHURN_REMOVE||"+row["document_id"]), row["document_id"]))
    versions: dict[str, list[dict]] = {"V0": original}
    for version, fraction in list(VERSIONS.items())[1:]:
        remove_n = round(len(background)*fraction)
        removed = removal_order[:remove_n]
        remove_ids = {row["document_id"] for row in removed}
        need = defaultdict(int)
        for row in removed:
            need[row["domain"]] += 1
        replacement_meta = {domain: candidates[domain][:need[domain]] for domain in DOMAINS}
        replacement_ids = {row["document_id"] for values in replacement_meta.values() for row in values}
        raw_needed = {domain: {row["local_document_id"] for row in replacement_meta[domain]} for domain in DOMAINS}
        replacements = []
        for domain in DOMAINS:
            with (RAW / domain / "corpus.jsonl").open(encoding="utf-8") as handle:
                for line in handle:
                    row = json.loads(line)
                    local = str(row.get("_id") or row.get("id"))
                    if local not in raw_needed[domain]:
                        continue
                    gid = f"BeIR_{domain}::{local}"
                    meta = next(value for value in replacement_meta[domain] if value["document_id"] == gid)
                    title, text = str(row.get("title") or "").strip(), str(row.get("text") or "").strip()
                    replacements.append({"document_id": gid, "local_document_id": local, "domain": domain,
                                         "title": title, "text": text, "source_text": "\n".join(x for x in (title, text) if x),
                                         "normalized_text_hash": meta["normalized_text_hash"]})
        rows = [row for row in original if row["document_id"] not in remove_ids] + replacements
        rows.sort(key=lambda row: row["document_id"])
        ids, hashes = {row["document_id"] for row in rows}, {row["normalized_text_hash"] for row in rows}
        if len(rows) != 3000 or len(ids) != 3000 or not member_ids <= ids or nonmember_ids & ids or len(hashes) != 3000:
            raise RuntimeError(f"churn invariant failed: {version}")
        versions[version] = rows
        path = EXP / f"runtime/churn/{version}_PROTECTED_DB.jsonl"
        write_jsonl(path, rows)
        atomic_json(EXP / f"manifests/{version}_DB_MANIFEST.json", {
            "version": version, "background_fraction_replaced": fraction, "background_replaced": remove_n,
            "documents": 3000, "member_preserved": len(member_ids & ids), "nonmember_still_excluded": len(nonmember_ids-ids),
            "domain_counts": {domain: sum(row["domain"] == domain for row in rows) for domain in DOMAINS},
            "db_sha256": sha_file(path), "removed_id_sha256": sha_text("\n".join(sorted(remove_ids))),
            "replacement_id_sha256": sha_text("\n".join(sorted(replacement_ids)))})
    return versions


def main() -> None:
    verify()
    versions = construct_versions()
    rows = read_jsonl(EXP / "cache/LARGE_DETECTION_SCORES.jsonl")
    embeddings = np.asarray(np.load(EXP / "cache/QUERY_EMBEDDINGS.float16.npy"), dtype=np.float32)
    if len(rows) != len(embeddings):
        raise RuntimeError("query embedding order/size mismatch")
    ref_indices = [i for i, row in enumerate(rows) if row["split"] == "REFERENCE"]
    eval_indices = [i for i, row in enumerate(rows) if row["split"] != "REFERENCE"]
    ref_embeddings = embeddings[ref_indices]
    neighbors = []
    for start in range(0, len(eval_indices), 256):
        sims = embeddings[eval_indices[start:start+256]] @ ref_embeddings.T
        for similarity in sims:
            selected = np.argpartition(-similarity, K_LOCAL)[:K_LOCAL]
            neighbors.append(selected[np.argsort(-similarity[selected], kind="stable")])
    model = SentenceTransformer(str(BGE), device="cuda")
    model.max_seq_length = 512
    output = []
    v0_tpr = None
    for version, documents in versions.items():
        checkpoint("DB_CHURN_VERSION_STARTED", version=version, documents=len(documents))
        started = time.monotonic()
        doc_embeddings = np.asarray(model.encode([row["source_text"] for row in documents], batch_size=16,
            show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True), dtype=np.float32)
        margins = np.empty(len(rows), dtype=float)
        for start in range(0, len(rows), 128):
            batch = embeddings[start:start+128] @ doc_embeddings.T
            for offset, scores in enumerate(batch):
                stats = canonical_mirabel_from_moments(float(scores.max()), float(scores.sum(dtype=np.float64)),
                    float(np.square(scores, dtype=np.float64).sum(dtype=np.float64)), len(documents), .95)
                margins[start+offset] = stats.margin
        ref_margins = margins[ref_indices]
        global_sorted = sorted(map(float, ref_margins))
        evaluated = []
        for position, row_index in enumerate(eval_indices):
            local_sorted = sorted(float(ref_margins[int(index)]) for index in neighbors[position])
            p_local = empirical_upper_tail(local_sorted, float(margins[row_index]))
            p_global = empirical_upper_tail(global_sorted, float(margins[row_index]))
            evaluated.append({**rows[row_index], "M_churn": float(margins[row_index]),
                              "p_local_churn": p_local, "p_global_churn": p_global})
        holdout = [row for row in evaluated if row["split"] == "HOLDOUT"]
        version_rows = []
        for method, alarm in (("Original MIRABEL", lambda row: row["M_churn"] > 0),
                              ("Global BC-MIRABEL", lambda row: row["p_global_churn"] <= .03),
                              ("LC-MIRABEL", lambda row: row["p_local_churn"] <= .03)):
            fpr = sum(alarm(row) for row in holdout)/len(holdout)
            for attack in ("MEntA", "MBA", "RAG-MIA"):
                member = [row for row in evaluated if row.get("attack") == attack and row.get("membership") == "member"]
                tpr = sum(alarm(row) for row in member)/len(member)
                version_rows.append({"db": version, "method": method, "attack": attack, "benign_fpr": fpr,
                                     "member_tpr": tpr, "calibration_seconds": time.monotonic()-started,
                                     "retraining_required": False, "training_steps": 0})
        output.extend(version_rows)
        if version == "V0":
            v0_tpr = {row["attack"]: row["member_tpr"] for row in version_rows if row["method"] == "LC-MIRABEL"}
        checkpoint("DB_CHURN_VERSION_COMPLETE", version=version, seconds=round(time.monotonic()-started, 2))
    del model
    torch.cuda.empty_cache()
    write_csv(EXP / "tables/DB_CHURN.csv", output)
    assert v0_tpr is not None
    lc = [row for row in output if row["method"] == "LC-MIRABEL"]
    checks = {"fpr_all_versions_le_5pct": all(row["benign_fpr"] <= .05 for row in lc),
              "attack_tpr_degradation_all_le_5pp": all(v0_tpr[row["attack"]]-row["member_tpr"] <= .05+1e-12 for row in lc)}
    passed = all(checks.values())
    final = {"campaign": "LC_MIRABEL_LARGE_V1", "verdict": "LC_MIRABEL_LARGE_PASS" if passed else "LC_MIRABEL_CHURN_FAILED",
             "completed_utc": now(), "checks": checks, "db_churn": output,
             "training_steps": 0, "trainable_parameters_updated": 0}
    atomic_json(EXP / "RESULT.json", final)
    lines = ["# LC-MIRABEL Final Scientific Verdict", "", f"- 판정: `{final['verdict']}`", "",
             "| DB | Method | FPR | MEntA TPR | MBA TPR | RAG-MIA TPR | Retraining |", "|---|---|---:|---:|---:|---:|---|"]
    for version in VERSIONS:
        for method in ("Original MIRABEL", "Global BC-MIRABEL", "LC-MIRABEL"):
            group = [row for row in output if row["db"] == version and row["method"] == method]
            values = {row["attack"]: row["member_tpr"] for row in group}
            lines.append(f"| {version} | {method} | {group[0]['benign_fpr']:.4f} | {values['MEntA']:.4f} | {values['MBA']:.4f} | {values['RAG-MIA']:.4f} | No |")
    atomic_text(EXP / "reports/LC_MIRABEL_FINAL_REPORT_KO.md", "\n".join(lines)+"\n")
    checkpoint(final["verdict"], training_steps=0)
    print(json.dumps(final, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
