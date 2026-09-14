#!/usr/bin/env python3
"""Post-transfer, apples-to-apples low-FPR detector diagnostic.

This script does not select or modify the Exp212 defense.  It compares the
continuous canonical Mirabel margin and the frozen Stateless QLL dominance on
the same retriever, benign cohort, attack cohort, and session aggregation.
"""
from __future__ import annotations

from datetime import datetime, timezone
import gzip
import hashlib
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
ROOT = PROJECT / "exp212_retriever_transfer_stateless_qll_20260831"
AD_ROOT = Path("/home/traffic_3/workspace/workspace/SH/AD-test-LLM3")
EXP68 = AD_ROOT / "artifacts/exp68_true_fresh_blind_final"
EXP166 = PROJECT / "exp166_topiocqa_gold_utility_20260827"
RETRIEVERS = {
    "GTE": Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--thenlper--gte-large/snapshots/4bef63f39fcc5e2d6b0aae83089f307af4970164"),
    "BGE": Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--BAAI--bge-m3/snapshots/5617a9f61b028005a4858fdac845db406aefb181"),
}
TARGET_FPRS = (0.01, 0.03, 0.05)
CONFIDENCE = 0.95
CANONICAL_CODE = AD_ROOT / "src/canonical_mirabel.py"
PROTOCOL_PATH = ROOT / "configs/LOW_FPR_POST_TRANSFER_PROTOCOL_V2.json"

sys.path.insert(0, str(AD_ROOT))
from src.canonical_mirabel import (  # noqa: E402
    MIRABEL_FORMULA_VERSION,
    canonical_mirabel_from_moments,
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def atomic_csv(frame: pd.DataFrame, path: Path, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".csv.gz" if str(path).endswith(".gz") else ".csv"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=suffix, dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False, compression=compression)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def checkpoint(stage: str, **details: object) -> None:
    payload = {"stage": stage, "updated_utc": now(), "pid": os.getpid(), **details}
    atomic_json(ROOT / "checkpoints" / f"LOW_FPR_{stage}.json", payload)
    atomic_json(ROOT / "LOW_FPR_HEARTBEAT.json", payload)
    lines = ["# Exp212 post-transfer low-FPR status", "", f"- Stage: **{stage}**",
             f"- Updated UTC: `{payload['updated_utc']}`", f"- PID: `{os.getpid()}`"]
    lines.extend(f"- {key}: `{value}`" for key, value in details.items())
    atomic_text(ROOT / "LOW_FPR_STATUS.md", "\n".join(lines) + "\n")
    with (ROOT / "logs/low_fpr_pipeline.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")


def strict_threshold(benign_scores: np.ndarray, target_fpr: float) -> tuple[float, float, int]:
    """Closest deterministic strict-'> threshold whose empirical FPR is <= target."""
    values = np.asarray(benign_scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("benign scores must be a finite nonempty vector")
    allowed = int(math.floor(float(target_fpr) * values.size + 1e-12))
    if allowed >= values.size:
        threshold = float(np.nextafter(values.min(), -np.inf))
    else:
        threshold = float(np.sort(values)[::-1][allowed])
    false_positives = int(np.sum(values > threshold))
    if false_positives > allowed:
        raise RuntimeError("strict low-FPR threshold contract violated")
    return threshold, false_positives / values.size, false_positives


def wilson(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return math.nan, math.nan
    rate = successes / total
    denominator = 1.0 + z * z / total
    center = (rate + z * z / (2.0 * total)) / denominator
    radius = z * math.sqrt(rate * (1.0 - rate) / total + z * z / (4.0 * total * total)) / denominator
    return max(0.0, center - radius), min(1.0, center + radius)


def corpus_payloads() -> dict[str, tuple[list[str], list[str]]]:
    """Reconstruct the exact ordered corpus IDs and texts used by Exp212.

    Exp212 saved a float16 corpus cache but performed the first retrieval from
    the freshly encoded float32 matrix.  Mirabel moment recomputation therefore
    has to recreate that float32 matrix; loading and renormalizing the float16
    cache can change top-1 on near ties and is not apples-to-apples.
    """
    result: dict[str, tuple[list[str], list[str]]] = {}
    topi = pd.read_csv(EXP166 / "private/TOPIOCQA_CORPUS.private.csv.gz",
                       keep_default_na=False, dtype={"document_id": str})
    result["TopiOCQA"] = (
        topi.document_id.astype(str).tolist(),
        (topi.title.astype(str) + "\n" + topi.text.astype(str)).tolist(),
    )
    with gzip.open(EXP68 / "private/fresh_source.private.json.gz", "rt", encoding="utf-8") as handle:
        source = json.load(handle)
    with gzip.open(EXP68 / "private/fresh_attack_sessions.private.jsonl.gz", "rt", encoding="utf-8") as handle:
        sessions = [json.loads(line) for line in handle]
    nonmember = {str(item["target_document_id"]) for item in sessions if int(item["member_label"]) == 0}
    fiqa_ids: list[str] = []
    fiqa_texts: list[str] = []
    for document_id, document in source["documents"].items():
        if str(document_id) in nonmember:
            continue
        fiqa_ids.append(str(document_id))
        fiqa_texts.append("\n".join(value for value in (
            str(document.get("title", "")), str(document.get("text", ""))) if value))
    result["FiQA-2018"] = (fiqa_ids, fiqa_texts)
    for domain in ("BeIR_nfcorpus", "BeIR_scidocs", "BeIR_trec-covid"):
        ids: list[str] = []
        texts: list[str] = []
        with (AD_ROOT / f"data/beir/{domain}/corpus_member.jsonl").open(encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    item = json.loads(line)
                    ids.append(str(item.get("_id", item.get("id"))))
                    texts.append(str(item.get("text", "")))
        result[domain] = (ids, texts)
    return result


def compute_mirabel_scores(
    retriever: str, corpora_by_dataset: dict[str, tuple[list[str], list[str]]]
) -> pd.DataFrame:
    destination = ROOT / f"private/{retriever}_MIRABEL_LOW_FPR_SCORES.private.csv.gz"
    if destination.exists():
        return pd.read_csv(destination, keep_default_na=False, dtype={"case_id": str, "session_id": str})
    import torch
    from sentence_transformers import SentenceTransformer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full-corpus Mirabel moment recomputation")
    retrieval_path = ROOT / f"private/{retriever}_RETRIEVAL.private.csv.gz"
    retrieval = pd.read_csv(retrieval_path, keep_default_na=False, low_memory=False,
                            dtype={"case_id": str, "session_id": str, "target_document_id": str})
    model = SentenceTransformer(str(RETRIEVERS[retriever]), device="cuda", local_files_only=True)
    model.max_seq_length = 512
    rows: list[dict[str, object]] = []
    for dataset, subset0 in retrieval.groupby("dataset", sort=False):
        subset = subset0.reset_index(drop=True)
        ids, texts = corpora_by_dataset[str(dataset)]
        embedding_path = ROOT / (
            f"private/{retriever}_{str(dataset).replace('/', '_')}_MIRABEL_RECOMPUTE.float32.npy"
        )
        if embedding_path.exists():
            documents = np.asarray(np.load(embedding_path, mmap_mode="r"), dtype=np.float32)
        else:
            checkpoint("MIRABEL_CORPUS_FP32", retriever=retriever, dataset=dataset,
                       completed=0, total=len(texts))
            batch_size = 16 if retriever == "BGE" else 96
            documents = model.encode(
                texts, normalize_embeddings=True, convert_to_numpy=True,
                batch_size=batch_size, show_progress_bar=True,
            ).astype(np.float32)
            fd, temporary = tempfile.mkstemp(
                prefix=f".{embedding_path.name}.", suffix=".npy", dir=embedding_path.parent
            )
            os.close(fd)
            try:
                np.save(temporary, documents)
                os.replace(temporary, embedding_path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            checkpoint("MIRABEL_CORPUS_FP32", retriever=retriever, dataset=dataset,
                       completed=len(texts), total=len(texts))
        if len(ids) != len(documents):
            raise RuntimeError(f"{retriever}/{dataset} corpus identity mismatch: {len(ids)} != {len(documents)}")
        queries = model.encode(subset["query"].astype(str).tolist(), normalize_embeddings=True,
                               convert_to_numpy=True, batch_size=64 if retriever == "BGE" else 128,
                               show_progress_bar=False).astype(np.float32)
        document_tensor = torch.as_tensor(documents, device="cuda", dtype=torch.float32)
        for start in range(0, len(subset), 64):
            query_tensor = torch.as_tensor(queries[start:start + 64], device="cuda", dtype=torch.float32)
            with torch.inference_mode():
                scores = query_tensor @ document_tensor.T
                top_values, top_indices = scores.max(dim=1)
                sums = scores.sum(dim=1)
                sumsqs = (scores * scores).sum(dim=1)
            top_values_np = top_values.cpu().numpy()
            top_indices_np = top_indices.cpu().numpy()
            sums_np = sums.cpu().numpy()
            sumsqs_np = sumsqs.cpu().numpy()
            for offset in range(len(top_values_np)):
                item = subset.iloc[start + offset]
                top1 = float(top_values_np[offset])
                stats = canonical_mirabel_from_moments(
                    top1=top1, sum_all=float(sums_np[offset]), sumsq_all=float(sumsqs_np[offset]),
                    corpus_size=len(documents), confidence=CONFIDENCE)
                stored_scores = json.loads(item.retrieved_scores)
                stored_ids = json.loads(item.retrieved_document_ids)
                recomputed_id = str(ids[int(top_indices_np[offset])])
                rows.append({
                    "case_id": str(item.case_id), "session_id": str(item.session_id), "turn": int(item.turn),
                    "kind": str(item.kind), "attack_family": str(item.attack_family), "member": int(item.member),
                    "dataset": str(dataset), "mirabel_margin": float(stats.margin),
                    "mirabel_official_alarm": bool(stats.margin > 0.0),
                    "recomputed_top1_score": top1, "exp212_stored_top1_score": float(stored_scores[0]),
                    "top1_score_abs_diff": abs(top1 - float(stored_scores[0])),
                    "top1_id_match": recomputed_id == str(stored_ids[0]),
                    "mirabel_corpus_size": len(documents), "mirabel_formula_version": MIRABEL_FORMULA_VERSION,
                })
            checkpoint("MIRABEL_PROGRESS", retriever=retriever, dataset=dataset,
                       completed=min(start + 64, len(subset)), total=len(subset))
        del document_tensor, documents, queries
        torch.cuda.empty_cache()
    del model
    torch.cuda.empty_cache()
    output = pd.DataFrame(rows)
    if len(output) != len(retrieval) or output.case_id.nunique() != len(retrieval):
        raise RuntimeError(f"{retriever} Mirabel score identity failure")
    atomic_csv(output, destination, "gzip")
    audit_rows = []
    for dataset, group in output.groupby("dataset", sort=True):
        audit_rows.append({"retriever": retriever, "dataset": dataset, "queries": len(group),
                           "top1_id_match_rate": float(group.top1_id_match.mean()),
                           "mean_top1_score_abs_diff": float(group.top1_score_abs_diff.mean()),
                           "max_top1_score_abs_diff": float(group.top1_score_abs_diff.max())})
    atomic_csv(pd.DataFrame(audit_rows), ROOT / f"audits/{retriever}_MIRABEL_RECOMPUTE_AUDIT.csv")
    # A tied maximum can select a different document ID while preserving the exact
    # score consumed by the canonical Mirabel margin. Reject only a different-ID
    # maximum whose score also differs, or any material score drift.
    bad_id_mismatch = (~output.top1_id_match.astype(bool)) & output.top1_score_abs_diff.gt(1e-8)
    if bad_id_mismatch.any() or float(output.top1_score_abs_diff.max()) > 0.005:
        raise RuntimeError(f"{retriever} recomputed retrieval is not aligned with Exp212")
    return output


def query_scores(retriever: str, mirabel: pd.DataFrame) -> pd.DataFrame:
    qll_path = ROOT / f"private/{retriever}_QLL_CASES.private.csv.gz"
    qll = pd.read_csv(qll_path, keep_default_na=False, low_memory=False,
                      dtype={"case_id": str, "session_id": str, "target_document_id": str})
    keep = ["case_id", "session_id", "turn", "kind", "attack_family", "member", "dominance"]
    merged = qll[keep].merge(mirabel[["case_id", "mirabel_margin"]], on="case_id", validate="one_to_one")
    if len(merged) != len(qll) or not np.isfinite(merged[["dominance", "mirabel_margin"]].to_numpy(float)).all():
        raise RuntimeError(f"{retriever} comparison score merge failure")
    merged = merged.rename(columns={"dominance": "stateless_qll_score"})
    merged["retriever"] = retriever
    return merged


def session_scores(query: pd.DataFrame) -> pd.DataFrame:
    return query.groupby(["retriever", "kind", "attack_family", "session_id"], as_index=False).agg(
        member=("member", "first"), turns=("turn", "nunique"),
        stateless_qll_score=("stateless_qll_score", "max"), mirabel_margin=("mirabel_margin", "max"))


def evaluate_unit(table: pd.DataFrame, unit: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    detectors = {"Original Mirabel": "mirabel_margin", "Stateless QLL Source Hide": "stateless_qll_score"}
    benign = table[table.kind.eq("BENIGN")]
    attack = table[table.kind.ne("BENIGN")]
    threshold_rows: list[dict[str, object]] = []
    metric_rows: list[dict[str, object]] = []
    for detector, score_column in detectors.items():
        for target_fpr in TARGET_FPRS:
            threshold, actual_fpr, false_positives = strict_threshold(benign[score_column].to_numpy(float), target_fpr)
            threshold_rows.append({"retriever": str(table.retriever.iloc[0]), "unit": unit, "detector": detector,
                                   "target_fpr": target_fpr, "threshold": threshold, "comparison": "strict >",
                                   "n_benign": len(benign), "false_positives": false_positives,
                                   "realized_fpr": actual_fpr, "attack_examples_used_for_threshold": 0})
            families = ["ALL_ATTACKS_MICRO"] + sorted(attack.attack_family.unique().tolist())
            for family in families:
                family_group = attack if family == "ALL_ATTACKS_MICRO" else attack[attack.attack_family.eq(family)]
                for scope, member_value in (("ALL", None), ("MEMBER", 1), ("NONMEMBER", 0)):
                    group = family_group if member_value is None else family_group[family_group.member.eq(member_value)]
                    auc_labels = np.concatenate((np.zeros(len(benign), dtype=int),
                                                 np.ones(len(group), dtype=int)))
                    auc_scores = np.concatenate((benign[score_column].to_numpy(float),
                                                 group[score_column].to_numpy(float)))
                    auroc = float(roc_auc_score(auc_labels, auc_scores)) if len(group) else math.nan
                    successes = int(np.sum(group[score_column].to_numpy(float) > threshold))
                    low, high = wilson(successes, len(group))
                    metric_rows.append({"retriever": str(table.retriever.iloc[0]), "unit": unit,
                                        "target_fpr": target_fpr, "detector": detector, "family": family,
                                        "membership_scope": scope, "n_attack": len(group), "true_positives": successes,
                                        "tpr": successes / len(group) if len(group) else math.nan,
                                        "tpr_wilson_low": low, "tpr_wilson_high": high,
                                        "auroc": auroc,
                                        "threshold": threshold, "n_benign": len(benign),
                                        "false_positives": false_positives, "realized_fpr": actual_fpr})
    return pd.DataFrame(threshold_rows), pd.DataFrame(metric_rows)


def paired_table(metrics: pd.DataFrame) -> pd.DataFrame:
    keys = ["retriever", "transfer_level", "transfer_verdict", "unit", "target_fpr", "family",
            "membership_scope", "n_attack", "n_benign"]
    values = ["tpr", "tpr_wilson_low", "tpr_wilson_high", "auroc", "realized_fpr", "threshold"]
    wide = metrics.pivot(index=keys, columns="detector", values=values).reset_index()
    wide.columns = ["_".join(part for part in column if part).strip("_") if isinstance(column, tuple) else column
                    for column in wide.columns]
    rename = {}
    for column in wide.columns:
        rename[column] = column.replace("Original Mirabel", "mirabel").replace(
            "Stateless QLL Source Hide", "stateless_qll").replace(" ", "_").lower()
    wide = wide.rename(columns=rename)
    wide["tpr_delta_qll_minus_mirabel"] = wide["tpr_stateless_qll"] - wide["tpr_mirabel"]
    return wide.sort_values(["retriever", "unit", "target_fpr", "family", "membership_scope"])


def write_report(comparison: pd.DataFrame, audits: list[dict[str, object]], transfer: dict[str, dict[str, object]]) -> None:
    headline = comparison[(comparison.unit.eq("session_any_turn")) &
                          (comparison.family.eq("ALL_ATTACKS_MICRO")) &
                          (comparison.membership_scope.eq("ALL"))].copy()
    columns = ["retriever", "target_fpr", "realized_fpr_mirabel", "tpr_mirabel", "auroc_mirabel",
               "realized_fpr_stateless_qll", "tpr_stateless_qll", "auroc_stateless_qll",
               "tpr_delta_qll_minus_mirabel"]
    family = comparison[(comparison.unit.eq("session_any_turn")) & (comparison.target_fpr.eq(0.03)) &
                        (comparison.membership_scope.eq("ALL")) &
                        (~comparison.family.eq("ALL_ATTACKS_MICRO"))].copy()
    family_columns = ["retriever", "family", "n_attack", "tpr_mirabel", "tpr_stateless_qll",
                      "auroc_mirabel", "auroc_stateless_qll",
                      "tpr_delta_qll_minus_mirabel"]
    transfer_table = pd.DataFrame([{"retriever": key, **value} for key, value in transfer.items()])
    lines = ["# Exp212 전이 후 Mirabel–Stateless QLL 동일 FPR 비교", "",
             "이 결과는 Exp212의 GTE·BGE 평가가 끝난 뒤 성공 여부와 무관하게 수행한 **별도 진단**이다. 모델·threshold 선택이나 Exp212 판정에는 사용하지 않았다.", "",
             "## 전이 판정", "", transfer_table.to_markdown(index=False), "",
             "## 세션 기준 전체 공격", "", headline[columns].to_markdown(index=False, floatfmt=".4f"), "",
             "## 공격 family별 TPR (목표 FPR 3%)", "", family[family_columns].to_markdown(index=False, floatfmt=".4f"), "",
             "## 해석", "",
             "- 정상 이용자 1,000개가 두 모델의 공통 음성 집단이다.",
             "- 각 모델의 threshold는 정상 점수만으로 목표 FPR을 넘지 않도록 정했고 `score > threshold`를 적용했다.",
             "- 세션 기준은 여러 질의 중 최대 점수로 판정한다. 단일 질의 결과도 CSV에 함께 보존했다.",
             "- TPR은 공격 트래픽 flag 비율이다. 생성 답변의 membership privacy를 뜻하는 E-AUC와는 다른 지표다.",
             "- Original Mirabel 점수는 full-corpus background moments를 쓰는 canonical Gumbel margin이다.",
             "- Mirabel 재계산은 Exp212와 동일한 모델·문서 순서·batch 설정으로 float32 corpus embedding을 재생성했다. 동점 문서의 ID가 달라도 top-1 점수가 정확히 같으면 score-aligned tie로 기록하며, ID 일치율과 점수 차이를 모두 공개한다.", "",
             "## 재계산 정합성", "", pd.DataFrame(audits).to_markdown(index=False, floatfmt=".6f"), ""]
    atomic_text(ROOT / "reports/LOW_FPR_MIRABEL_VS_STATELESS_QLL_KO.md", "\n".join(lines) + "\n")


def main() -> None:
    final_path = ROOT / "FINAL_RESULT.json"
    if not final_path.exists():
        raise RuntimeError("Exp212 FINAL_RESULT.json is absent")
    final = json.loads(final_path.read_text())
    transfer = {retriever: {"transfer_level": str(final["retrievers"][retriever]["transfer_level"]),
                            "transfer_verdict": str(final["retrievers"][retriever]["verdict"])}
                for retriever in RETRIEVERS}
    checkpoint("COMPARISON_STARTED", overall_transfer_gate_passed=bool(final.get("exp213_may_open")),
               gte_transfer=transfer["GTE"]["transfer_level"], bge_transfer=transfer["BGE"]["transfer_level"])
    expected_protocol = {
        "purpose": "post-transfer diagnostic only", "retrievers": list(RETRIEVERS),
        "negative_cohort": "same frozen 1000 TopiOCQA benign sessions",
        "positive_cohort": "same frozen Exp212 native and external attack sessions",
        "units": ["query", "session_any_turn"], "session_score": "maximum query score",
        "target_fprs": list(TARGET_FPRS), "threshold_source": "benign only",
        "threshold_policy": "closest deterministic strict-> threshold with empirical FPR <= target",
        "mirabel_score": f"{MIRABEL_FORMULA_VERSION} full-corpus Gumbel margin, confidence={CONFIDENCE}",
        "stateless_qll_score": "frozen QLL source dominance", "attack_examples_used_for_threshold": 0,
        "changes_exp212_verdict": False,
        "execution_gate": "run after both GTE and BGE transfer evaluations finish, regardless of pass/fail",
        "transfer_reporting": "attach each retriever's Exp212 transfer verdict to every output row",
    }
    protocol = json.loads(PROTOCOL_PATH.read_text())
    if protocol != expected_protocol:
        raise RuntimeError("precommitted low-FPR protocol drift")
    corpora_by_dataset = corpus_payloads()
    all_query, all_thresholds, all_metrics, audit_summary = [], [], [], []
    protected = [final_path, ROOT / "configs/PRECOMMIT.json"]
    hashes_before = {str(path): sha256_file(path) for path in protected}
    for retriever in RETRIEVERS:
        mirabel = compute_mirabel_scores(retriever, corpora_by_dataset)
        query = query_scores(retriever, mirabel)
        query["transfer_level"] = transfer[retriever]["transfer_level"]
        query["transfer_verdict"] = transfer[retriever]["transfer_verdict"]
        session = session_scores(query)
        all_query.append(query)
        for unit, table in (("query", query), ("session_any_turn", session)):
            thresholds, metrics = evaluate_unit(table, unit)
            thresholds["transfer_level"] = transfer[retriever]["transfer_level"]
            thresholds["transfer_verdict"] = transfer[retriever]["transfer_verdict"]
            metrics["transfer_level"] = transfer[retriever]["transfer_level"]
            metrics["transfer_verdict"] = transfer[retriever]["transfer_verdict"]
            all_thresholds.append(thresholds)
            all_metrics.append(metrics)
        audit = pd.read_csv(ROOT / f"audits/{retriever}_MIRABEL_RECOMPUTE_AUDIT.csv")
        audit_summary.append({"retriever": retriever, "queries": int(audit.queries.sum()),
                              "top1_id_match_rate": float(np.average(audit.top1_id_match_rate, weights=audit.queries)),
                              "max_top1_score_abs_diff": float(audit.max_top1_score_abs_diff.max())})
    query_scores_all = pd.concat(all_query, ignore_index=True)
    thresholds_all = pd.concat(all_thresholds, ignore_index=True)
    metrics_all = pd.concat(all_metrics, ignore_index=True)
    comparison = paired_table(metrics_all)
    atomic_csv(query_scores_all, ROOT / "private/LOW_FPR_QUERY_SCORES.private.csv.gz", "gzip")
    atomic_csv(thresholds_all, ROOT / "tables/LOW_FPR_THRESHOLDS.csv")
    atomic_csv(metrics_all, ROOT / "tables/LOW_FPR_TPR_LONG.csv")
    atomic_csv(comparison, ROOT / "tables/TPR_AT_LOW_FPR_MIRABEL_VS_STATELESS_QLL.csv")
    write_report(comparison, audit_summary, transfer)
    hashes_after = {str(path): sha256_file(path) for path in protected}
    if hashes_before != hashes_after:
        raise RuntimeError("frozen Exp212 artifacts changed during post-transfer diagnostic")
    result = {"verdict": "POST_TRANSFER_LOW_FPR_COMPARISON_COMPLETE", "updated_utc": now(),
              "exp212_transfer_gate_passed": bool(final.get("exp213_may_open")), "retrievers": list(RETRIEVERS),
              "retriever_transfer_verdicts": transfer,
              "target_fprs": list(TARGET_FPRS), "benign_sessions": 1000,
              "auroc_reported_from_same_frozen_scores": True,
              "canonical_mirabel_formula": MIRABEL_FORMULA_VERSION,
              "canonical_mirabel_code_sha256": sha256_file(CANONICAL_CODE),
              "protocol_sha256": sha256_file(PROTOCOL_PATH),
              "comparison_table": str(ROOT / "tables/TPR_AT_LOW_FPR_MIRABEL_VS_STATELESS_QLL.csv"),
              "report": str(ROOT / "reports/LOW_FPR_MIRABEL_VS_STATELESS_QLL_KO.md"),
              "frozen_artifacts_unchanged": True, "changes_exp212_verdict": False}
    atomic_json(ROOT / "LOW_FPR_COMPARISON_RESULT.json", result)
    checkpoint("COMPARISON_COMPLETE", verdict=result["verdict"], benign_sessions=1000)


if __name__ == "__main__":
    main()
