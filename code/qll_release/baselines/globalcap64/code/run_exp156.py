#!/usr/bin/env python3
"""Exp156 clean confirmation of the frozen, detector-free GlobalCap64 policy."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROJECT = Path("/home/traffic_3/workspace/workspace/SH/LoRA-mirabel-Decter")
ROOT = PROJECT / "exp156_global_cap64_clean_3k_20260826"
EXP152 = PROJECT / "exp152_exp150_clean_3k_confirmation_20260825"
EXP153 = PROJECT / "exp153_native_dcmi2_ia15_20260826"
EXP91 = PROJECT / "exp91r_conditional_epd_rescue_repair_20260822"
MANIFEST = EXP152 / "private/EXP152_CLEAN_3K_MANIFEST.private.csv.gz"
BUDGETED = EXP152 / "private/EXP152_BUDGETED_GENERATIONS.private.csv.gz"
BASELINES = EXP152 / "private/EXP152_BASELINE_GENERATIONS.private.csv.gz"
NORMAL_MIRABEL = EXP153 / "private/EXP153_MIRABEL_NORMAL_FPR.private.csv.gz"
PRECOMMIT = ROOT / "configs/PRECOMMIT.json"
MPNET = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--sentence-transformers--all-mpnet-base-v2/snapshots/e8c3b32edf5434bc2275fc9bab85f82640a19130")
EMBED_ROOT = Path("/home/traffic_3/workspace/workspace/SH/AD-test-LLM2/outputs/exp1_mirabel_repro")
PARTITION = "EXP156_CLEAN_Q1"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value); handle.flush(); os.fsync(handle.fileno())
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n")


def atomic_csv(frame: pd.DataFrame, path: Path, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=".csv.gz" if compression else ".csv", dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(tmp, index=False, compression=compression); os.replace(tmp, path)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""): digest.update(chunk)
    return digest.hexdigest()


def checkpoint(stage: str, **details: object) -> None:
    payload = {"experiment": "Exp156", "stage": stage, "updated_utc": now(), "pid": os.getpid(), **details}
    atomic_json(ROOT / "HEARTBEAT.json", payload)
    atomic_json(ROOT / "checkpoints" / f"{stage}.json", payload)
    with (ROOT / "logs/pipeline.log").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    lines = ["# Exp156 status", "", f"- Stage: **{stage}**", f"- Updated UTC: {payload['updated_utc']}",
             f"- PID: `{payload['pid']}`"] + [f"- {key}: {value}" for key, value in details.items()] + [""]
    atomic_text(ROOT / "STATUS.md", "\n".join(lines))


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def initialize():
    required = [PRECOMMIT, MANIFEST, BUDGETED, BASELINES, NORMAL_MIRABEL, MPNET / "config.json"]
    required += [EMBED_ROOT / domain / "corpus_embeddings.npy"
                 for domain in ("BeIR_nfcorpus", "BeIR_scidocs", "BeIR_trec-covid")]
    missing = [str(path) for path in required if not path.exists()]
    if missing: raise RuntimeError(f"missing inputs: {missing}")
    protocol = json.loads(PRECOMMIT.read_text(encoding="utf-8"))
    if not protocol.get("written_before_exp156_metrics") or protocol.get("candidate") != "GLOBAL_CAP64":
        raise RuntimeError("invalid precommit")
    frame = pd.read_csv(MANIFEST, keep_default_na=False, low_memory=False)
    counts = frame.cohort.value_counts().to_dict()
    if len(frame) != 3000 or counts != {"MEMBER_ATTACK": 1000, "NONMEMBER_ATTACK": 1000, "NORMAL_USER": 1000}:
        raise RuntimeError(f"cohort contract failure: {counts}")
    budgeted = pd.read_csv(BUDGETED, keep_default_na=False)
    baselines = pd.read_csv(BASELINES, keep_default_na=False)
    if len(budgeted) != 9000 or len(baselines) != 6000:
        raise RuntimeError("generation cache contract failure")
    atomic_csv(pd.DataFrame([{"path": str(path), "sha256": sha256_file(path), "access": "READ_ONLY"}
                             for path in required]), ROOT / "provenance/FROZEN_INPUTS.csv")
    checkpoint("STAGE_01_INPUTS_FROZEN", rows=3000, member=1000, nonmember=1000, normal=1000,
               candidate="GLOBAL_CAP64", normal_calibration_rows=0, attack_calibration_rows=0)
    return protocol, frame, budgeted, baselines


def canonical_mirabel(frame: pd.DataFrame) -> pd.DataFrame:
    destination = ROOT / "private/EXP156_CANONICAL_MIRABEL.private.csv.gz"
    if destination.exists():
        cached = pd.read_csv(destination, keep_default_na=False)
        if len(cached) == 3000 and cached.row_id.nunique() == 3000:
            checkpoint("STAGE_02_MIRABEL_REUSED", rows=3000); return cached
    from sentence_transformers import SentenceTransformer
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = SentenceTransformer(str(MPNET), device=device, local_files_only=True)
    records = []
    rho = 0.05; c = -math.log(-math.log(1-rho))
    for domain, cell in frame.groupby("domain", sort=True):
        corpus_np = np.load(EMBED_ROOT / domain / "corpus_embeddings.npy").astype(np.float32, copy=False)
        queries_np = encoder.encode(cell["query"].astype(str).tolist(), normalize_embeddings=True,
                                    convert_to_numpy=True, batch_size=128, show_progress_bar=False)
        corpus = torch.from_numpy(corpus_np).to(device)
        queries = torch.from_numpy(queries_np.astype(np.float32, copy=False)).to(device)
        for offset in range(0, len(queries), 128):
            scores = queries[offset:offset+128] @ corpus.T
            maximum, index = scores.max(dim=1); n = scores.shape[1]
            total = scores.sum(dim=1); squares = (scores.double() ** 2).sum(dim=1)
            mean = (total.double() - maximum.double()) / (n-1)
            variance = torch.clamp((squares - maximum.double() ** 2)/(n-1) - mean ** 2, min=0)
            std = torch.sqrt(variance); root = math.sqrt(2*math.log(n))
            threshold = mean + std*root + c*std/root
            for source, top, idx, mu, sigma, tau in zip(cell.iloc[offset:offset+128].to_dict("records"),
                    maximum.cpu().numpy(), index.cpu().numpy(), mean.cpu().numpy(), std.cpu().numpy(), threshold.cpu().numpy()):
                records.append({"row_id": str(source["row_id"]), "domain": domain, "cohort": source["cohort"],
                                "family": source["family"], "top1": float(top), "top1_index": int(idx),
                                "background_mean": float(mu), "background_std": float(sigma), "corpus_size": int(n),
                                "threshold": float(tau), "margin": float(top-tau), "mirabel_alarm": bool(top > tau)})
        checkpoint("STAGE_02_MIRABEL_DOMAIN_DONE", domain=domain, rows=len(cell), device=device)
        del corpus, queries
    output = pd.DataFrame(records)
    if len(output) != 3000 or output.row_id.nunique() != 3000: raise RuntimeError("Mirabel audit contract failure")
    atomic_csv(output, destination, "gzip")
    normal = output[output.cohort.eq("NORMAL_USER")]
    historical = pd.read_csv(NORMAL_MIRABEL, keep_default_na=False)
    comparison = normal[["row_id", "mirabel_alarm", "margin"]].merge(
        historical[["row_id", "mirabel_alarm", "margin"]], on="row_id", suffixes=("_new", "_old"), validate="one_to_one")
    agreement = float((comparison.mirabel_alarm_new.astype(str) == comparison.mirabel_alarm_old.astype(str)).mean())
    if agreement != 1.0: raise RuntimeError(f"canonical Mirabel replay disagreement: {agreement}")
    atomic_json(ROOT / "audits/EXP156_CANONICAL_MIRABEL_AUDIT.json",
                {"status": "PASS", "rows": 3000, "normal_replay_alarm_agreement": agreement,
                 "normal_fpr": float(normal.mirabel_alarm.mean()), "rho": rho, "strict_greater_than": True,
                 "device": device})
    del encoder; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    checkpoint("STAGE_02_MIRABEL_COMPLETE", rows=3000, normal_fpr=float(normal.mirabel_alarm.mean()),
               normal_replay_alarm_agreement=agreement)
    return output


def build_responses(frame: pd.DataFrame, budgeted: pd.DataFrame, baselines: pd.DataFrame,
                    mirabel: pd.DataFrame) -> pd.DataFrame:
    b64 = budgeted[budgeted.rank1_budget.astype(int).eq(64)].set_index("row_id").response.to_dict()
    baseline = baselines.set_index(["row_id", "condition"]).response.to_dict()
    alarms = mirabel.set_index("row_id").mirabel_alarm.astype(bool).to_dict()
    rows = []
    for source in frame.itertuples(index=False):
        row_id = str(source.row_id); alarm = alarms[row_id]
        no_defense = str(baseline[(row_id, "NO_DEFENSE")])
        hidden = str(baseline[(row_id, "ORIGINAL_MIRABEL")])
        for condition, response, action in (
                ("GLOBAL_CAP64", str(b64[row_id]), "SOFT_CAP"),
                ("NO_DEFENSE", no_defense, "PASS"),
                ("ORIGINAL_MIRABEL", hidden if alarm else no_defense, "MIRABEL" if alarm else "PASS"),
                ("ALWAYS_HIDE_RANK1_ORACLE", hidden, "MIRABEL")):
            rows.append({"exp87_row_id": row_id, "condition": condition, "response": response,
                         "A_R": no_defense, "action": action, "lola_alarm": alarm,
                         "lola_score": float(mirabel.loc[mirabel.row_id.eq(row_id), "margin"].iloc[0]), "beta": np.nan})
    output = pd.DataFrame(rows)
    if len(output) != 12000 or output.duplicated(["exp87_row_id", "condition"]).any():
        raise RuntimeError("response contract failure")
    atomic_csv(output, ROOT / "private/EXP156_RESPONSES.private.csv.gz", "gzip")
    checkpoint("STAGE_03_RESPONSES_COMPLETE", rows=len(output), conditions=4,
               global_cap_full_document_deletions=0, request_rejections=0)
    return output


def utility_metrics(frame: pd.DataFrame, responses: pd.DataFrame):
    from sentence_transformers import SentenceTransformer
    import torch
    sys.path.insert(0, str(EXP91 / "code")); import exp91r_evaluation as evaluation
    normal = frame[frame.cohort.eq("NORMAL_USER")][["row_id", "subgroup", "domain", "session_id"]]
    detail = responses.merge(normal, left_on="exp87_row_id", right_on="row_id", validate="many_to_one")
    model = SentenceTransformer(str(MPNET), device="cuda" if torch.cuda.is_available() else "cpu", local_files_only=True)
    left = model.encode(detail.A_R.astype(str).tolist(), normalize_embeddings=True, convert_to_numpy=True,
                        batch_size=128, show_progress_bar=False)
    right = model.encode(detail.response.astype(str).tolist(), normalize_embeddings=True, convert_to_numpy=True,
                         batch_size=128, show_progress_bar=False)
    detail["response_preservation"] = np.clip(np.sum(left*right, axis=1), 0, 1)
    detail["exact_response_unchanged"] = detail.response.astype(str).eq(detail.A_R.astype(str))
    detail["refusal"] = detail.response.map(evaluation.refusal)
    ordinary = detail.subgroup.eq("ORDINARY")
    summary = detail.groupby(["condition", "subgroup"], as_index=False).agg(
        rows=("exp87_row_id", "size"), response_preservation=("response_preservation", "mean"),
        exact_response_unchanged=("exact_response_unchanged", "mean"), refusal_rate=("refusal", "mean"))
    aggregate = []
    for condition, cell in detail.groupby("condition"):
        aggregate.append({"condition": condition,
                          "ordinary_response_preservation": float(cell.loc[cell.subgroup.eq("ORDINARY"), "response_preservation"].mean()),
                          "hard_response_preservation": float(cell.loc[~cell.subgroup.eq("ORDINARY"), "response_preservation"].mean()),
                          "normal_response_change_rate": float((~cell.exact_response_unchanged).mean()),
                          "normal_refusal_rate": float(cell.refusal.mean()), "request_rejection_rate": 0.0})
    atomic_csv(summary, ROOT / "tables/TABLE_156_01_UTILITY_BY_GROUP.csv")
    atomic_csv(pd.DataFrame(aggregate), ROOT / "tables/TABLE_156_02_UTILITY_SUMMARY.csv")
    atomic_csv(detail, ROOT / "private/EXP156_UTILITY_DETAIL.private.csv.gz", "gzip")
    del model; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    checkpoint("STAGE_04_UTILITY_COMPLETE", normal_rows=len(normal))
    return pd.DataFrame(aggregate)


def documents_for(frame: pd.DataFrame):
    helper = load_module("exp152_helpers156", EXP152 / "code/run_confirmation.py")
    helper.ROOT = ROOT; helper.checkpoint = checkpoint; helper.atomic_csv = atomic_csv
    return helper.documents_for(frame)


def privacy_metrics(frame: pd.DataFrame, responses: pd.DataFrame, documents):
    metadata = frame.copy(); metadata["exp87_row_id"] = metadata.row_id.astype(str)
    metadata["kind"] = np.where(metadata.cohort.eq("NORMAL_USER"), "benign", "attack")
    metadata["turn"] = 1; metadata["exp91_partition"] = PARTITION
    sys.path.insert(0, str(EXP91 / "code")); import exp91r_evaluation as evaluation
    evaluation.ROOT = ROOT / "evaluation_work"; evaluation.heartbeat = lambda stage, **details: checkpoint(f"EVAL_{stage}", **details)
    checkpoint("STAGE_05_Q1_NATIVE_EVALUATION_STARTED", conditions=4,
               scope="Q1 supports RAG-MIA/S2-MIA/MBA/MEntA; not DCMI-Q2/IA-Q15")
    native = evaluation.native_scores(metadata, responses, documents, PARTITION)
    privacy = evaluation.privacy_metrics(native, PARTITION)
    atomic_csv(native, ROOT / "private/EXP156_Q1_NATIVE_SCORES.private.csv.gz", "gzip")
    atomic_csv(privacy, ROOT / "tables/TABLE_156_03_Q1_PRIVACY.csv")
    return native, privacy


def bootstrap(native: pd.DataFrame, resamples: int = 5000, seed: int = 20260826):
    rng = np.random.default_rng(seed); records = []; distributions = defaultdict(list)
    for (condition, family), cell in native.dropna(subset=["attack_score"]).groupby(["condition", "attack_family"]):
        member = cell[cell.member.eq(1)].attack_score.to_numpy(float)
        nonmember = cell[cell.member.eq(0)].attack_score.to_numpy(float)
        if not len(member) or not len(nonmember): continue
        values = np.empty(resamples)
        for index in range(resamples):
            m = member[rng.integers(0, len(member), len(member))]
            n = nonmember[rng.integers(0, len(nonmember), len(nonmember))]
            auc = roc_auc_score(np.r_[np.ones(len(m)), np.zeros(len(n))], np.r_[m, n])
            values[index] = max(auc, 1-auc)
        point = roc_auc_score(np.r_[np.ones(len(member)), np.zeros(len(nonmember))], np.r_[member, nonmember])
        lo, med, hi = np.quantile(values, [.025, .5, .975]); distributions[condition].append(values)
        records.append({"condition": condition, "attack_family": family, "members": len(member),
                        "nonmembers": len(nonmember), "effective_auc": max(point, 1-point),
                        "bootstrap_median": med, "ci95_low": lo, "ci95_high": hi,
                        "resamples": resamples, "unit": "session"})
    family = pd.DataFrame(records); worst = []
    for condition, values in distributions.items():
        dist = np.max(np.stack(values), axis=0); lo, med, hi = np.quantile(dist, [.025, .5, .975])
        worst.append({"condition": condition, "supported_families": len(values),
                      "worst_bootstrap_median": med, "simultaneous_ci95_low": lo,
                      "simultaneous_ci95_high": hi, "resamples": resamples})
    atomic_csv(family, ROOT / "tables/TABLE_156_04_Q1_BOOTSTRAP_CI.csv")
    atomic_csv(pd.DataFrame(worst), ROOT / "tables/TABLE_156_05_Q1_WORST_SIMULTANEOUS_CI.csv")


def summarize(protocol: dict, mirabel: pd.DataFrame, utility: pd.DataFrame, privacy: pd.DataFrame):
    supported = privacy[privacy.status.eq("SUPPORTED")]
    aggregate = supported.groupby("condition").effective_auc.agg(
        supported_q1_families="size", mean_effective_auc="mean", worst_effective_auc="max").reset_index()
    summary = aggregate.merge(utility, on="condition", how="left")
    baseline = summary[summary.condition.eq("ORIGINAL_MIRABEL")].iloc[0]
    candidate = summary.condition.eq("GLOBAL_CAP64")
    summary["privacy_mean_gate"] = candidate & summary.mean_effective_auc.le(baseline.mean_effective_auc)
    summary["privacy_worst_gate"] = candidate & summary.worst_effective_auc.le(baseline.worst_effective_auc + 0.02)
    summary["ordinary_utility_gate"] = candidate & summary.ordinary_response_preservation.ge(0.90)
    summary["hard_utility_gate"] = candidate & summary.hard_response_preservation.ge(0.60)
    summary["no_rejection_gate"] = candidate & summary.request_rejection_rate.eq(0)
    gates = ["privacy_mean_gate", "privacy_worst_gate", "ordinary_utility_gate", "hard_utility_gate", "no_rejection_gate"]
    summary["confirmation_gate"] = summary[gates].all(axis=1)
    atomic_csv(summary, ROOT / "tables/TABLE_156_06_SUMMARY_AND_GATES.csv")
    row = summary[summary.condition.eq("GLOBAL_CAP64")].iloc[0]
    passed = bool(row.confirmation_gate)
    verdict = "GLOBAL_CAP64_CLEAN_3K_CONFIRMED" if passed else "GLOBAL_CAP64_CLEAN_3K_FAILED"
    result = {"experiment": "Exp156", "verdict": verdict, "confirmation_gate": passed,
              "member": 1000, "nonmember": 1000, "normal": 1000,
              "supported_q1_families": int(row.supported_q1_families), "fresh_blind": False,
              "runtime_detector": False, "runtime_learned_components": 0, "request_rejections": 0,
              "canonical_mirabel_normal_fpr": float(mirabel.loc[mirabel.cohort.eq("NORMAL_USER"), "mirabel_alarm"].mean())}
    atomic_json(ROOT / "FINAL_RESULT.json", result)
    lines = ["# Exp156 GlobalCap64 clean 3K 확인 결과", "", f"- 판정: **{verdict}**",
             "- 표본: member 1,000 / nonmember 1,000 / normal user 1,000",
             f"- 공식 Mirabel 정상 이용자 FPR: **{result['canonical_mirabel_normal_fpr']:.3%}**",
             "- GlobalCap64는 탐지기가 아니므로 detector FPR이 아니라 정상 답변 보존·변경·거부율로 평가",
             "- Q1에서 DCMI-Q2와 IA-Q15는 지원하지 않으므로 별도 native session 실험 필요", "",
             "이 확인을 통과해도 Qwen/MPNet의 clean Q1 결과이다. 두 generator/retriever, native session, 새로운 공격 및 fresh blind 전에는 general defense로 부르지 않는다.", ""]
    atomic_text(ROOT / "reports/EXP156_FINAL_REPORT_KO.md", "\n".join(lines))
    checkpoint("COMPLETE", verdict=verdict, confirmation_gate=passed,
               mean_effective_auc=float(row.mean_effective_auc), worst_effective_auc=float(row.worst_effective_auc))


def main():
    protocol, frame, budgeted, baselines = initialize()
    mirabel = canonical_mirabel(frame)
    responses = build_responses(frame, budgeted, baselines, mirabel)
    utility = utility_metrics(frame, responses)
    documents = documents_for(frame)
    native, privacy = privacy_metrics(frame, responses, documents)
    bootstrap(native, protocol["bootstrap"]["resamples"], protocol["bootstrap"]["seed"])
    summarize(protocol, mirabel, utility, privacy)


if __name__ == "__main__":
    main()
