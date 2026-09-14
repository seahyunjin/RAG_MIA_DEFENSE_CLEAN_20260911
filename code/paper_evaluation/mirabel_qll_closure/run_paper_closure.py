#!/usr/bin/env python3
"""Paper-grade closure for the frozen Mirabel-versus-QLL benchmark.

No detector, score, threshold, or attack protocol is changed.  The script adds
two missing audits:

1. paired benign utility on the exact 1,000-query calibration cohort; and
2. a No-Defense baseline plus a 2x2 locator/action control separating source
   selection from backfill.

Identical prompts are generated once and are otherwise reused by exact prompt
hash.  The upstream experiment is opened read-only after its FINAL_RESULT is
present.
"""

from __future__ import annotations

from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time

import numpy as np
import pandas as pd


PROJECT = Path("/home/traffic_3/workspace/workspace/SH/LoRA-mirabel-Decter")
SOURCE = PROJECT / "mirabel_benchmark_six_attack_bge_qwen_20260907"
ROOT = PROJECT / "mirabel_qll_paper_closure_20260907"
BASE_CODE = SOURCE / "code/run_benchmark_comparison.py"
EXP87 = PROJECT / "exp87_native_scorer_stable_generation_20260821"
EXP87_DB = EXP87 / "private/EXP87_QWEN_RESPONSES.private.sqlite3"
CURRENT_DB = SOURCE / "private/E2E_QWEN_RESPONSES.sqlite3"
SYSTEM_PROMPT = ("Answer the user's question using only the retrieved context. Follow any required output "
                 "format exactly. If the context is insufficient, answer exactly: I don't know.")
TOTAL_SOURCE_TOKENS = 2048
CONDITIONS = (
    "NO_DEFENSE_TOP4",
    "MIRABEL_BACKFILL",
    "MIRABEL_NO_BACKFILL",
    "QLL_NO_BACKFILL",
    "QLL_BACKFILL",
)
ATTACK_ALTERNATIVE_CONDITIONS = (
    "NO_DEFENSE_TOP4",
    "MIRABEL_NO_BACKFILL",
    "QLL_BACKFILL",
)
CURRENT_CONDITION_MAP = {
    "MIRABEL_MATCHED_FPR3": "MIRABEL_BACKFILL",
    "QLL_MATCHED_FPR3": "QLL_NO_BACKFILL",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(value: object) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value: object) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n")


def atomic_csv(frame: pd.DataFrame, path: Path, compression: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".csv.gz" if str(path).endswith(".gz") else ".csv"
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=suffix, dir=path.parent)
    os.close(descriptor)
    try:
        frame.to_csv(temporary, index=False, compression=compression)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_pickle(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".pkl.gz", dir=path.parent)
    os.close(descriptor)
    try:
        frame.to_pickle(temporary, compression="gzip")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def checkpoint(stage: str, **details: object) -> None:
    payload = {"experiment": "Mirabel-QLL paper closure", "stage": stage,
               "updated_utc": now(), "pid": os.getpid(), **details}
    atomic_json(ROOT / "HEARTBEAT.json", payload)
    atomic_json(ROOT / "checkpoints" / f"{stage}.json", payload)
    lines = ["# Mirabel–QLL paper closure", "", f"- Stage: **{stage}**",
             f"- Updated UTC: `{payload['updated_utc']}`", f"- PID: `{os.getpid()}`"]
    lines.extend(f"- {key}: `{value}`" for key, value in details.items())
    atomic_text(ROOT / "STATUS.md", "\n".join(lines) + "\n")


def load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def preflight() -> dict[str, str]:
    required = [
        SOURCE / "FINAL_RESULT.json",
        SOURCE / "private/BGE_QLL_CASES.private.csv.gz",
        SOURCE / "private/E2E_PACKING.private.pkl.gz",
        SOURCE / "private/E2E_QWEN_RESPONSES.private.csv.gz",
        SOURCE / "private/E2E_ATTACK_SCORES.private.csv.gz",
        SOURCE / "tables/END_TO_END_EFFECTIVE_AUC.csv",
        SOURCE / "results/PRIMARY_FPR3_THRESHOLDS.json",
        BASE_CODE,
        CURRENT_DB,
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"upstream experiment incomplete: {missing}")
    hashes = {str(path): sha256_file(path) for path in required}
    atomic_json(ROOT / "provenance/UPSTREAM_READ_ONLY_HASHES.json", hashes)
    checkpoint("PREFLIGHT_COMPLETE", upstream="COMPLETE", immutable_files=len(required))
    return hashes


def waterfill(lengths: list[int], hidden: int | None = None) -> list[int]:
    values = np.asarray(lengths, dtype=int)
    caps = np.zeros(len(values), dtype=int)
    remaining = TOTAL_SOURCE_TOKENS
    active = [index for index, length in enumerate(values) if index != hidden and length > 0]
    while remaining > 0 and active:
        share = max(1, remaining // len(active))
        changed = False
        for index in list(active):
            add = min(share, int(values[index] - caps[index]), remaining)
            if add > 0:
                caps[index] += add
                remaining -= add
                changed = True
            if caps[index] >= values[index]:
                active.remove(index)
            if remaining <= 0:
                break
        if not changed:
            break
    return caps.tolist()


def decision(item, condition: str, thresholds: dict[str, float]) -> tuple[bool, str, list[str], str]:
    top10 = list(map(str, json.loads(item.retrieved_document_ids)))
    top4 = top10[:4]
    if condition == "NO_DEFENSE_TOP4":
        return False, "", top4, "PASS_TOP4"
    if condition.startswith("MIRABEL"):
        alarm = float(item.mirabel_margin) > float(thresholds["Original Mirabel"])
        selected = top4[0]
        if not alarm:
            visible = top4
        elif condition == "MIRABEL_BACKFILL":
            visible = top10[1:5]
        else:
            visible = top4[1:]
    elif condition.startswith("QLL"):
        alarm = float(item.dominance) > float(thresholds["Stateless QLL Source Hide"])
        selected = str(item.qll_top1_source)
        if not alarm:
            visible = top4
        else:
            visible = [source for source in top4 if source != selected]
            if condition == "QLL_BACKFILL":
                replacement = next((source for source in top10[4:] if source not in visible), None)
                if replacement is None:
                    raise RuntimeError(f"no QLL backfill source for {item.case_id}")
                visible.append(replacement)
    else:
        raise KeyError(condition)
    action = "PASS" if not alarm else f"HIDE_{selected}_{'BACKFILL' if condition.endswith('BACKFILL') and not condition.endswith('NO_BACKFILL') else 'NO_BACKFILL'}"
    return bool(alarm), selected if alarm else "", visible, action


def build_packing(cases: pd.DataFrame, documents: dict[tuple[str, str], str],
                  thresholds: dict[str, float], conditions: tuple[str, ...],
                  tokenizer, normal_prompt, normal: bool) -> pd.DataFrame:
    rows = []
    for item in cases.itertuples(index=False):
        for condition in conditions:
            alarm, selected, visible_sources, action = decision(item, condition, thresholds)
            token_ids = [tokenizer(documents[(str(item.dataset), source)], add_special_tokens=False).input_ids
                         for source in visible_sources]
            caps = waterfill([len(value) for value in token_ids])
            visible = [tokenizer.decode(value[:cap], skip_special_tokens=True).strip()
                       for value, cap in zip(token_ids, caps) if cap]
            prompt = normal_prompt(str(item.query), visible)
            max_new = 128 if normal else int({"RAG-MIA": 12, "S²-MIA": 96, "MBA": 160,
                                               "DCMI": 12, "MEntA": 96, "IA": 32}[str(item.family)])
            rows.append({**item._asdict(), "condition": condition, "intervened": alarm,
                         "selected_source_id": selected, "action": action,
                         "source_ids_used": json.dumps(visible_sources), "source_caps": json.dumps(caps),
                         "total_context_tokens": int(sum(caps)), "prompt": prompt,
                         "system_prompt": SYSTEM_PROMPT, "max_new_tokens": max_new,
                         "prompt_identity": sha256_text(f"{SYSTEM_PROMPT}\0{max_new}\0{prompt}")})
    output = pd.DataFrame(rows)
    expected = len(cases) * len(conditions)
    if len(output) != expected or output.total_context_tokens.gt(TOTAL_SOURCE_TOKENS).any():
        raise RuntimeError(f"packing contract failed: {len(output)}/{expected}")
    return output


def generate_unique(packing: pd.DataFrame, store_path: Path, cache_paths: list[Path],
                    stage: str, models) -> pd.DataFrame:
    output_path = store_path.with_suffix(".responses.csv.gz")
    if output_path.exists():
        result = pd.read_csv(output_path, keep_default_na=False, low_memory=False)
        if len(result) == len(packing) and result.response.astype(str).ne("").all():
            return result
    unique = packing.drop_duplicates("prompt_identity", keep="first").copy()
    tasks = [models.make_task(task_type="MIRABEL_QLL_PAPER_CLOSURE",
                              row_id=str(item.prompt_identity), prompt=str(item.prompt),
                              system_prompt=str(item.system_prompt), max_new_tokens=int(item.max_new_tokens))
             for item in unique.itertuples(index=False)]
    models.ROOT = ROOT
    models.heartbeat = lambda name, **details: checkpoint(f"{stage}_{name}",
                                                           logical_rows=len(packing),
                                                           unique_prompts=len(unique), **details)
    cache = models.ExactPromptCache(cache_paths)
    answers = models.run_generation(tasks, store_path, cache, stage)
    result = packing.copy()
    result["response"] = result.prompt_identity.map(answers)
    if result.response.isna().any() or result.response.astype(str).eq("").any():
        raise RuntimeError(f"{stage} response mapping incomplete")
    atomic_csv(result, output_path, "gzip")
    return result


def normalize_tokens(text: object) -> list[str]:
    return re.findall(r"\w+", str(text).casefold(), flags=re.UNICODE)


def token_f1(left: object, right: object) -> float:
    from collections import Counter
    a, b = Counter(normalize_tokens(left)), Counter(normalize_tokens(right))
    overlap = sum((a & b).values())
    if not a and not b:
        return 1.0
    if not overlap:
        return 0.0
    precision, recall = overlap / max(sum(a.values()), 1), overlap / max(sum(b.values()), 1)
    return 2 * precision * recall / (precision + recall)


def refusal(text: object) -> bool:
    value = str(text).strip().casefold()
    patterns = ("i don't know", "i do not know", "cannot determine", "can't determine",
                "insufficient context", "not enough information", "unable to answer")
    return any(pattern in value for pattern in patterns)


def benign_utility(responses: pd.DataFrame, mpnet_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = responses[responses.condition.eq("NO_DEFENSE_TOP4")].set_index("case_id")["response"]
    work = responses.copy()
    work["baseline_response"] = work.case_id.map(base)
    work["token_f1"] = [token_f1(a, b) for a, b in zip(work.response, work.baseline_response)]
    work["answer_changed"] = work.response.astype(str) != work.baseline_response.astype(str)
    work["refusal"] = work.response.map(refusal)
    work["baseline_refusal"] = work.baseline_response.map(refusal)
    work["new_refusal"] = work.refusal & ~work.baseline_refusal
    work["answer_length"] = work.response.astype(str).str.len()

    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(str(mpnet_path), device="cuda", local_files_only=True)
    left = model.encode(work.response.astype(str).tolist(), normalize_embeddings=True,
                        convert_to_numpy=True, batch_size=128, show_progress_bar=False)
    right = model.encode(work.baseline_response.astype(str).tolist(), normalize_embeddings=True,
                         convert_to_numpy=True, batch_size=128, show_progress_bar=False)
    work["semantic_similarity"] = np.sum(left * right, axis=1)
    del model, left, right
    gc.collect()
    import torch
    torch.cuda.empty_cache()

    overall = work.groupby("condition", as_index=False).agg(
        queries=("case_id", "size"), intervention_rate=("intervened", "mean"),
        mean_token_f1=("token_f1", "mean"), mean_semantic_similarity=("semantic_similarity", "mean"),
        answer_change_rate=("answer_changed", "mean"), refusal_rate=("refusal", "mean"),
        new_refusal_rate=("new_refusal", "mean"), mean_answer_length=("answer_length", "mean"))
    hidden = work[work.intervened].groupby("condition", as_index=False).agg(
        hidden_queries=("case_id", "size"), hidden_token_f1=("token_f1", "mean"),
        hidden_semantic_similarity=("semantic_similarity", "mean"),
        hidden_answer_change_rate=("answer_changed", "mean"),
        hidden_refusal_rate=("refusal", "mean"), hidden_new_refusal_rate=("new_refusal", "mean"),
        hidden_mean_answer_length=("answer_length", "mean"))
    atomic_csv(work.drop(columns=["prompt", "system_prompt"], errors="ignore"),
               ROOT / "tables/NORMAL_QUERY_RESPONSE_AUDIT.csv.gz", "gzip")
    atomic_csv(overall, ROOT / "tables/NORMAL_UTILITY_SAME_COHORT.csv")
    atomic_csv(hidden, ROOT / "tables/HIDDEN_BENIGN_UTILITY.csv")
    return overall, hidden


def summarize_privacy(scores: pd.DataFrame, base) -> pd.DataFrame:
    rows = []
    for condition in sorted(scores.condition.unique()):
        for family in base.FAMILIES:
            frame = scores[scores.condition.eq(condition) & scores.attack_family.eq(family)]
            if frame.member.nunique() < 2:
                continue
            raw, effective = base.effective_auc(frame.member, frame.attack_score)
            raw_low, raw_high, effective_low, effective_high = base.bootstrap_auc(
                frame.member, frame.attack_score, iterations=5000,
                seed=int(sha256_text(f"{condition}|{family}")[:8], 16))
            rows.append({"condition": condition, "attack_family": family, "sessions": len(frame),
                         "members": int(frame.member.eq(1).sum()),
                         "nonmembers": int(frame.member.eq(0).sum()), "raw_auc": raw,
                         "effective_auc": effective, "raw_auc_ci95_low": raw_low,
                         "raw_auc_ci95_high": raw_high, "effective_auc_ci95_low": effective_low,
                         "effective_auc_ci95_high": effective_high})
    output = pd.DataFrame(rows)
    atomic_csv(output, ROOT / "tables/END_TO_END_BASELINE_AND_LOCATOR_ACTION.csv")
    return output


def paired_bootstrap_differences(scores: pd.DataFrame, base,
                                 iterations: int = 5000) -> pd.DataFrame:
    """Paired session bootstrap for pre-specified E-AUC contrasts.

    Member and nonmember sessions are resampled separately, and the same
    sampled session indices are used for both conditions.  A negative delta
    means the candidate leaks less membership information than the reference.
    """
    contrasts = (
        ("PRIMARY", "QLL_NO_BACKFILL", "MIRABEL_BACKFILL"),
        ("LOCATOR_WITH_BACKFILL", "QLL_BACKFILL", "MIRABEL_BACKFILL"),
        ("LOCATOR_WITHOUT_BACKFILL", "QLL_NO_BACKFILL", "MIRABEL_NO_BACKFILL"),
        ("MIRABEL_ACTION", "MIRABEL_NO_BACKFILL", "MIRABEL_BACKFILL"),
        ("QLL_ACTION", "QLL_BACKFILL", "QLL_NO_BACKFILL"),
        ("MIRABEL_VS_NO_DEFENSE", "MIRABEL_BACKFILL", "NO_DEFENSE_TOP4"),
        ("QLL_VS_NO_DEFENSE", "QLL_NO_BACKFILL", "NO_DEFENSE_TOP4"),
    )
    from scipy.stats import rankdata

    def batched_effective_auc(positive: np.ndarray, negative: np.ndarray,
                              positive_indices: np.ndarray,
                              negative_indices: np.ndarray) -> np.ndarray:
        output = np.empty(len(positive_indices), dtype=float)
        batch_size = 250
        positive_n = positive_indices.shape[1]
        negative_n = negative_indices.shape[1]
        offset_constant = positive_n * (positive_n + 1) / 2
        for offset in range(0, len(output), batch_size):
            stop = min(len(output), offset + batch_size)
            values = np.concatenate([
                positive[positive_indices[offset:stop]],
                negative[negative_indices[offset:stop]],
            ], axis=1)
            ranks = rankdata(values, method="average", axis=1)
            raw = (ranks[:, :positive_n].sum(axis=1) - offset_constant) / (
                positive_n * negative_n
            )
            output[offset:stop] = np.maximum(raw, 1.0 - raw)
        return output

    rows = []
    for contrast, candidate, reference in contrasts:
        for family in base.FAMILIES:
            frame = scores[scores.attack_family.eq(family)].pivot_table(
                index=["session_id", "member"], columns="condition",
                values="attack_score", aggfunc="first"
            ).dropna(subset=[candidate, reference]).reset_index()
            members = frame[frame.member.eq(1)].reset_index(drop=True)
            nonmembers = frame[frame.member.eq(0)].reset_index(drop=True)
            if members.empty or nonmembers.empty:
                continue
            observed = (
                base.effective_auc(frame.member, frame[candidate])[1]
                - base.effective_auc(frame.member, frame[reference])[1]
            )
            rng = np.random.default_rng(int(sha256_text(f"{contrast}|{family}")[:8], 16))
            mi = rng.integers(0, len(members), size=(iterations, len(members)))
            ni = rng.integers(0, len(nonmembers), size=(iterations, len(nonmembers)))
            candidate_auc = batched_effective_auc(
                members[candidate].to_numpy(float), nonmembers[candidate].to_numpy(float), mi, ni
            )
            reference_auc = batched_effective_auc(
                members[reference].to_numpy(float), nonmembers[reference].to_numpy(float), mi, ni
            )
            deltas = candidate_auc - reference_auc
            p_two_sided = min(1.0, 2.0 * min(
                (np.count_nonzero(deltas <= 0) + 1) / (iterations + 1),
                (np.count_nonzero(deltas >= 0) + 1) / (iterations + 1),
            ))
            rows.append({
                "contrast": contrast,
                "attack_family": family,
                "candidate": candidate,
                "reference": reference,
                "paired_sessions": len(frame),
                "candidate_minus_reference_eauc": observed,
                "ci95_low": float(np.quantile(deltas, 0.025)),
                "ci95_high": float(np.quantile(deltas, 0.975)),
                "p_two_sided": p_two_sided,
            })
    output = pd.DataFrame(rows)
    output["p_holm_within_contrast"] = np.nan
    for contrast, indices in output.groupby("contrast").groups.items():
        ordered = output.loc[list(indices), "p_two_sided"].sort_values()
        adjusted = {}
        running = 0.0
        count = len(ordered)
        for rank, (row_index, value) in enumerate(ordered.items()):
            running = max(running, min(1.0, float(value) * (count - rank)))
            adjusted[row_index] = running
        for row_index, value in adjusted.items():
            output.loc[row_index, "p_holm_within_contrast"] = value
    atomic_csv(output, ROOT / "tables/PAIRED_EAUC_CONTRASTS.csv")
    return output


def response_audit(cases: pd.DataFrame, packing: pd.DataFrame) -> None:
    index = ["case_id", "family", "dataset", "member", "session_id", "turn", "native_budget",
             "query", "query_sha256", "target_document_id", "target_rank", "mirabel_margin",
             "dominance", "qll_top1_source"]
    metadata = cases[index].drop_duplicates("case_id")
    answers = packing.pivot_table(index="case_id", columns="condition", values="response", aggfunc="first")
    answers.columns = [f"answer__{column}" for column in answers.columns]
    actions = packing.pivot_table(index="case_id", columns="condition", values="action", aggfunc="first")
    actions.columns = [f"action__{column}" for column in actions.columns]
    output = metadata.merge(answers.reset_index(), on="case_id", validate="one_to_one")
    output = output.merge(actions.reset_index(), on="case_id", validate="one_to_one")
    atomic_csv(output, ROOT / "tables/ATTACK_QUERY_RESPONSE_AUDIT.csv.gz", "gzip")


def runtime_audit() -> pd.DataFrame:
    records = []
    log_path = SOURCE / "logs/pipeline.jsonl"
    for line in log_path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        stage = str(item.get("stage", ""))
        if stage == "QLL_COMPLETE":
            cases = int(item["cases"])
            records.append({
                "component": "QLL dominance scoring",
                "queries": cases,
                "source_forward_pairs": int(item["unique_source_pairs"]),
                "elapsed_seconds": float(item["seconds"]),
                "batched_seconds_per_query": float(item["seconds"]) / cases,
                "interpretation": "four source-conditioned query likelihoods per query",
            })
        elif stage == "E2E_GENERATION_COMPLETE":
            responses = int(item["responses"])
            records.append({
                "component": "defended answer generation",
                "queries": responses,
                "source_forward_pairs": 0,
                "elapsed_seconds": float(item["seconds"]),
                "batched_seconds_per_query": float(item["seconds"]) / responses,
                "interpretation": "one answer generation per detector condition",
            })
    output = pd.DataFrame(records)
    atomic_csv(output, ROOT / "tables/RUNTIME_AUDIT.csv")
    return output


def main() -> None:
    upstream_hashes = preflight()
    base = load_module("paper_closure_base", BASE_CODE)
    base.ROOT = ROOT
    cases = pd.read_csv(SOURCE / "private/BGE_QLL_CASES.private.csv.gz", keep_default_na=False,
                        low_memory=False, dtype={"case_id": str, "session_id": str,
                                                 "target_document_id": str})
    thresholds = json.loads((SOURCE / "results/PRIMARY_FPR3_THRESHOLDS.json").read_text())
    corpora, member_documents = base.load_corpora()
    documents = base.load_target_documents(cases, member_documents)

    sys.path.insert(0, str(EXP87 / "code"))
    import exp87_models as models
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(base.QWEN, local_files_only=True)

    attacks = cases[cases.kind.eq("ATTACK")].copy()
    alternatives = build_packing(attacks, documents, thresholds,
                                 ATTACK_ALTERNATIVE_CONDITIONS,
                                 tokenizer, models.normal_prompt, normal=False)
    atomic_pickle(alternatives, ROOT / "private/ATTACK_ALTERNATIVE_PACKING.private.pkl.gz")
    checkpoint("ATTACK_CAUSAL_GENERATION_STARTED", logical_rows=len(alternatives),
               unique_prompts=alternatives.prompt_identity.nunique())
    alternatives = generate_unique(alternatives, ROOT / "private/ATTACK_ALTERNATIVE_RESPONSES.sqlite3",
                                   [CURRENT_DB], "ATTACK_CAUSAL", models)

    current = pd.read_csv(SOURCE / "private/E2E_QWEN_RESPONSES.private.csv.gz",
                          keep_default_na=False, low_memory=False,
                          dtype={"case_id": str, "session_id": str, "target_document_id": str})
    current["condition"] = current.condition.map(CURRENT_CONDITION_MAP)
    combined_responses = pd.concat([current, alternatives], ignore_index=True, sort=False)
    response_audit(attacks, combined_responses)

    ia_truth = base.load_ia_ground_truth(cases)
    # The imported scorer memoizes to ``base.ROOT/private/E2E_ATTACK_SCORES``.
    # Keep the final three-condition alternative scoring isolated from the
    # historical two-condition diagnostic so a stale cache cannot silently
    # omit the No-Defense baseline.
    scoring_root = ROOT / "work/alternative_scoring_final"
    scoring_root.mkdir(parents=True, exist_ok=True)
    base.ROOT = scoring_root
    alternative_scores = base.score_end_to_end(cases, alternatives, documents, ia_truth)
    base.ROOT = ROOT
    expected_alternatives = set(ATTACK_ALTERNATIVE_CONDITIONS)
    actual_alternatives = set(alternative_scores.condition.astype(str).unique())
    if actual_alternatives != expected_alternatives:
        raise RuntimeError(
            f"alternative scorer condition mismatch: {sorted(actual_alternatives)} != "
            f"{sorted(expected_alternatives)}"
        )
    expected_family_counts = {
        "RAG-MIA": 360, "S²-MIA": 360, "MBA": 327,
        "DCMI": 200, "MEntA": 360, "IA": 360,
    }
    observed_counts = alternative_scores.groupby(
        ["condition", "attack_family"]
    ).size().to_dict()
    mismatches = []
    for condition in ATTACK_ALTERNATIVE_CONDITIONS:
        for family, expected_count in expected_family_counts.items():
            observed = int(observed_counts.get((condition, family), 0))
            if observed != expected_count:
                mismatches.append({"condition": condition, "family": family,
                                   "expected": expected_count, "observed": observed})
    atomic_json(ROOT / "audits/ALTERNATIVE_SCORER_COVERAGE_AUDIT.json", {
        "status": "PASS" if not mismatches else "FAIL",
        "conditions": list(ATTACK_ALTERNATIVE_CONDITIONS),
        "expected_family_counts_per_condition": expected_family_counts,
        "observed_rows": len(alternative_scores),
        "mismatches": mismatches,
        "note": "The imported two-condition IA audit has a fixed 720-row expectation; "
                "this closure validates the three-condition 1,080-row IA total explicitly.",
    })
    if mismatches:
        raise RuntimeError(f"alternative scorer coverage mismatch: {mismatches}")
    current_scores = pd.read_csv(SOURCE / "private/E2E_ATTACK_SCORES.private.csv.gz",
                                 keep_default_na=False, low_memory=False,
                                 dtype={"session_id": str, "target_document_id": str})
    current_scores["condition"] = current_scores.condition.map(CURRENT_CONDITION_MAP)
    all_scores = pd.concat([current_scores, alternative_scores], ignore_index=True, sort=False)
    atomic_csv(all_scores, ROOT / "private/ALL_ATTACK_SCORES.private.csv.gz", "gzip")
    privacy = summarize_privacy(all_scores, base)
    contrasts = paired_bootstrap_differences(all_scores, base)
    checkpoint("ATTACK_CAUSAL_SCORING_COMPLETE", conditions=5, score_rows=len(all_scores),
               paired_contrasts=len(contrasts))

    normals = cases[cases.kind.eq("NORMAL")].copy()
    normal_packing = build_packing(normals, documents, thresholds, CONDITIONS,
                                   tokenizer, models.normal_prompt, normal=True)
    atomic_pickle(normal_packing, ROOT / "private/NORMAL_PACKING.private.pkl.gz")
    checkpoint("NORMAL_UTILITY_GENERATION_STARTED", logical_rows=len(normal_packing),
               unique_prompts=normal_packing.prompt_identity.nunique())
    normal_responses = generate_unique(
        normal_packing, ROOT / "private/NORMAL_RESPONSES.sqlite3", [CURRENT_DB, EXP87_DB],
        "NORMAL_UTILITY", models)
    utility, hidden = benign_utility(normal_responses, base.MPNET)
    checkpoint("NORMAL_UTILITY_COMPLETE", normal_queries=1000, conditions=len(CONDITIONS))
    runtime = runtime_audit()

    mean_privacy = privacy.groupby("condition").effective_auc.mean().sort_values()
    worst_privacy = privacy.groupby("condition").effective_auc.max().sort_values()
    lines = [
        "# Mirabel–QLL 논문 제출 전 검증", "",
        "## 무엇을 고정했나", "",
        "- 동일 BEIR 3개 데이터, 동일 공격 세션 2,000개, 동일 정상 질문 1,000개.",
        "- Mirabel/QLL score 및 FPR 3% threshold는 변경하지 않았다.",
        "- 생성기·검색기·prompt·2048 source-token budget도 변경하지 않았다.",
        "- 추가한 것은 locator와 backfill action을 분리한 2×2 대조 및 동일 정상 cohort 생성뿐이다.", "",
        "## Locator × action E-AUC", "", privacy.to_markdown(index=False, floatfmt=".4f"), "",
        "## 조건별 평균/최악 E-AUC", "",
        pd.DataFrame({"mean_eauc": mean_privacy, "worst_eauc": worst_privacy}).to_markdown(floatfmt=".4f"), "",
        "## 사전 지정 paired E-AUC 대조", "", contrasts.to_markdown(index=False, floatfmt=".4f"), "",
        "## 동일 정상 1,000개 utility", "", utility.to_markdown(index=False, floatfmt=".4f"), "",
        "## 실제로 문서를 숨긴 정상 질문", "", hidden.to_markdown(index=False, floatfmt=".4f"), "",
        "## 실행 비용", "", runtime.to_markdown(index=False, floatfmt=".4f"), "",
        "## 해석 제한", "",
        "- 정상 utility는 paired No-Defense 답변 보존율이며 gold 정답률과 같지 않다.",
        "- hidden-benign subset 손상은 전체 평균과 별도로 보고해야 한다.",
        "- 이 실험은 frozen 후보의 인과 대조이지 새 threshold/model search가 아니다.",
    ]
    atomic_text(ROOT / "reports/FINAL_REPORT_KO.md", "\n".join(lines) + "\n")

    after = {path: sha256_file(Path(path)) for path in upstream_hashes}
    unchanged = upstream_hashes == after
    atomic_json(ROOT / "provenance/UPSTREAM_POSTRUN_HASHES.json", after)
    if not unchanged:
        raise RuntimeError("upstream frozen artifact changed")
    result = {
        "verdict": "MIRABEL_QLL_PAPER_CLOSURE_COMPLETE",
        "updated_utc": now(), "upstream_unchanged": True,
        "attack_sessions": 2000, "attack_query_turns": 8680, "normal_queries": 1000,
        "privacy_conditions": sorted(privacy.condition.unique()),
        "utility_conditions": list(CONDITIONS),
        "mean_eauc": mean_privacy.to_dict(), "worst_eauc": worst_privacy.to_dict(),
        "privacy_table": str(ROOT / "tables/END_TO_END_BASELINE_AND_LOCATOR_ACTION.csv"),
        "paired_contrast_table": str(ROOT / "tables/PAIRED_EAUC_CONTRASTS.csv"),
        "utility_table": str(ROOT / "tables/NORMAL_UTILITY_SAME_COHORT.csv"),
        "hidden_utility_table": str(ROOT / "tables/HIDDEN_BENIGN_UTILITY.csv"),
        "runtime_table": str(ROOT / "tables/RUNTIME_AUDIT.csv"),
        "query_response_audit": str(ROOT / "tables/ATTACK_QUERY_RESPONSE_AUDIT.csv.gz"),
        "report": str(ROOT / "reports/FINAL_REPORT_KO.md"),
    }
    atomic_json(ROOT / "FINAL_RESULT.json", result)
    checkpoint("COMPLETE", verdict=result["verdict"], upstream_unchanged=True)


if __name__ == "__main__":
    main()
