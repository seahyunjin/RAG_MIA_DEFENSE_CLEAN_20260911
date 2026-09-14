#!/usr/bin/env python3
"""Exp187: frozen Exp186 DC-MCEL transfer screen on RAGLeak/BudgetLeak-Z."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile

import numpy as np
import pandas as pd


PROJECT = Path("/home/traffic_3/workspace/workspace/SH/LoRA-mirabel-Decter")
ROOT = PROJECT / "exp187_dc_mcel_external_attacks_20260828"
EXP87 = PROJECT / "exp87_native_scorer_stable_generation_20260821"
EXP160 = PROJECT / "exp160_global_cap64_ragleak_budgetleak_20260826"
EXP160E = PROJECT / "exp160e_global_cap64_external_attacks_exploratory_20260826"
EXP179 = PROJECT / "exp179_cross_family_qwen_source_influence_20260828"
EXP179B = PROJECT / "exp179b_qll_soft64_response_audit_20260828"
EXP180R = PROJECT / "exp180r_continuous_qll_budget_20260828"
EXP182 = PROJECT / "exp182_minimax_inverse_qll_20260828"
EXP183R = PROJECT / "exp183r_claim_level_source_attribution_20260828"
EXP186 = PROJECT / "exp186_dual_channel_counterfactual_ledger_20260828"
PRECOMMIT = ROOT / "configs/PRECOMMIT.json"
AMENDMENT = ROOT / "configs/PRECOMMIT_AMENDMENT_01.json"
SAMPLE = EXP179B / "private/EXP179B_SAMPLE.private.csv.gz"
QLL = EXP179 / "tables/TABLE_179_01_SOURCE_QLL.csv"
COHORT = EXP160E / "private/EXP160E_ATTACK_COHORT.private.csv.gz"
RETRIEVAL = EXP160E / "private/EXP160_RETRIEVAL.private.csv.gz"
OLD_GENERATIONS = EXP160E / "private/EXP160_GENERATIONS.private.csv.gz"
QWEN = Path("/home/traffic_3/workspace/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1")
DB = ROOT / "private/EXP187_CLAIM_LOO.sqlite3"
BUDGETS = tuple(range(10, 271, 20))
BASE = "MINIMAX_INVERSE_QLL"
PRIMARY = "DC_MCEL_SESSION_T128_D16"
RESET = "DC_MCEL_RESET_T128_D16"
TOTAL_BUDGET = 128.0
DECISION_BUDGET = 16.0


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                 default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")


def atomic_csv(frame, path, compression=None):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    suffix = ".csv.gz" if compression == "gzip" else ".csv"
    fd, temporary = tempfile.mkstemp(prefix=f".{path.stem}.", suffix=suffix, dir=path.parent); os.close(fd)
    try:
        frame.to_csv(temporary, index=False, compression=compression); os.replace(temporary, path)
    finally:
        if os.path.exists(temporary): os.unlink(temporary)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()


def sha256_text(value):
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); assert spec.loader is not None
    spec.loader.exec_module(module); return module


def checkpoint(stage, **details):
    payload = {"experiment": "Exp187", "stage": stage, "updated_utc": now(), "pid": os.getpid(),
               "candidate": PRIMARY, "paid_api_calls": 0, "attack_labels_used_by_policy": False,
               "membership_labels_used_by_policy": False, **details}
    atomic_json(ROOT / "HEARTBEAT.json", payload)
    atomic_json(ROOT / "checkpoints" / f"{stage}.json", payload)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    with (ROOT / "logs/pipeline.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, default=str) + "\n")
    lines = ["# Exp187 Status", "", f"- Stage: **{stage}**", f"- Updated UTC: `{payload['updated_utc']}`",
             f"- PID: `{payload['pid']}`"] + [f"- {key}: `{value}`" for key, value in details.items()]
    atomic_text(ROOT / "STATUS.md", "\n".join(lines) + "\n")


def verify_inputs():
    config = json.loads(PRECOMMIT.read_text(encoding="utf-8"))
    if not config.get("written_before_exp187_metrics") or config.get("frozen_candidate") != "DC_MCEL_T128_D16":
        raise RuntimeError("invalid Exp187 precommit")
    result = json.loads((EXP186 / "FINAL_RESULT.json").read_text(encoding="utf-8"))
    if result.get("verdict") != "DC_MCEL_NATIVE_SESSION_GO" or result.get("selected_development_condition") != "DC_MCEL_T128_D16":
        raise RuntimeError("Exp186 selected model mismatch")
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    if not amendment.get("written_before_generation_or_metrics") or amendment.get("selection_change"):
        raise RuntimeError("invalid cohort-count amendment")
    paths = [PRECOMMIT, AMENDMENT, EXP186 / "FINAL_RESULT.json", EXP186 / "configs/PRECOMMIT.json", SAMPLE, QLL,
             COHORT, RETRIEVAL, OLD_GENERATIONS, EXP182 / "code/run_exp182.py",
             EXP183R / "code/run_exp183r.py", EXP160 / "code/run_exp160.py", QWEN / "config.json"]
    rows = [{"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size, "access": "READ_ONLY"}
            for path in paths]
    atomic_csv(pd.DataFrame(rows), ROOT / "provenance/FROZEN_INPUTS.csv")
    atomic_text(ROOT / "configs/PRECOMMIT.sha256", sha256_file(PRECOMMIT) + "  PRECOMMIT.json\n")
    checkpoint("INPUTS_FROZEN", inputs=len(rows), parent_verdict=result["verdict"])
    return config


def frozen_cohort():
    sample = pd.read_csv(SAMPLE, keep_default_na=False, low_memory=False)
    sample = sample[sample.attack_family.isin(["RAGLeak", "BudgetLeak-Z"])].copy()
    expected = {"RAGLeak": 100, "BudgetLeak-Z": 100}
    if sample.attack_family.value_counts().to_dict() != expected:
        raise RuntimeError(f"sample contract mismatch: {sample.attack_family.value_counts().to_dict()}")
    counts = sample.groupby(["attack_family", "member"]).size().to_dict()
    expected_counts = {("RAGLeak", 0): 48, ("RAGLeak", 1): 52,
                       ("BudgetLeak-Z", 0): 36, ("BudgetLeak-Z", 1): 64}
    if counts != expected_counts:
        raise RuntimeError(f"membership balance mismatch: {counts}")
    sample["row_id"] = sample.case_id.str.split("|", n=1).str[1]
    if sample.case_id.nunique() != 200 or sample.row_id.nunique() != 196:
        raise RuntimeError("case/target identity mismatch")
    atomic_csv(sample, ROOT / "private/EXP187_COHORT.private.csv.gz", "gzip")
    audit = {"cases": len(sample), "targets": int(sample.row_id.nunique()),
             "RAGLeak": int((sample.attack_family == "RAGLeak").sum()),
             "BudgetLeak-Z": int((sample.attack_family == "BudgetLeak-Z").sum()),
             "member": int(sample.member.sum()), "nonmember": int((sample.member == 0).sum()),
             "case_id_hash": sha256_text("\n".join(sorted(sample.case_id)) + "\n")}
    atomic_json(ROOT / "audits/COHORT_AUDIT.json", audit)
    checkpoint("COHORT_FROZEN", **audit)
    return sample


def prepare_generation(sample):
    destination = ROOT / "private/EXP187_BASE_GENERATIONS.private.csv.gz"
    if destination.exists():
        frame = pd.read_csv(destination, keep_default_na=False, low_memory=False)
        if len(frame) == 1500: checkpoint("BASE_GENERATION_REUSED", rows=len(frame)); return frame
    qll = pd.read_csv(QLL, keep_default_na=False, low_memory=False)
    qll = qll[qll.case_id.isin(sample.case_id)].copy()
    if len(qll) != 800 or not qll.groupby("case_id").size().eq(4).all():
        raise RuntimeError("QLL source contract mismatch")
    exp180 = load_module("exp187_docs", EXP180R / "code/run_exp180r.py")
    exp180.ROOT = ROOT; exp180.checkpoint = checkpoint
    documents = exp180.load_documents(sample)
    exp182 = load_module("exp187_alloc", EXP182 / "code/run_exp182.py")
    sys.path.insert(0, str(EXP87 / "code")); import exp87_models as models
    models.ROOT = ROOT; models.heartbeat = lambda stage, **details: checkpoint(stage, **details)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    by_case = {str(case): cell.sort_values(["source_rank", "source_id"]).to_dict("records")
               for case, cell in qll.groupby("case_id", sort=False)}
    tasks, metadata, packing = [], [], []
    for row in sample.sort_values(["attack_family", "selection_key"]).itertuples(index=False):
        source_ids = list(dict.fromkeys(map(str, json.loads(row.retrieved_document_ids))))[:4]
        source_map = {str(item["source_id"]): item for item in by_case[str(row.case_id)]}
        source_rows = [source_map[source] for source in source_ids]
        caps = exp182.minimax_allocations(source_rows)
        texts, used = [], []
        for source in source_ids:
            token_ids = tokenizer(str(documents[(str(row.dataset), source)]), add_special_tokens=False).input_ids[:caps[source]]
            texts.append(tokenizer.decode(token_ids, skip_special_tokens=True).strip()); used.append(len(token_ids))
        prompt = models.normal_prompt(str(row.query), texts)
        prompt_tokens = len(tokenizer(prompt, add_special_tokens=False).input_ids)
        budgets = (128,) if row.attack_family == "RAGLeak" else BUDGETS
        for budget in budgets:
            composite = f"{BASE}|{row.case_id}|B{budget}"
            tasks.append(models.make_task(task_type=f"EXP187_{row.attack_family}", row_id=composite,
                                          prompt=prompt, max_new_tokens=int(budget)))
            metadata.append({"composite_id": composite, "case_id": row.case_id, "row_id": row.row_id,
                             "member": int(row.member), "dataset": row.dataset, "attack": row.attack_family,
                             "condition": BASE, "budget": int(budget), "query": row.query,
                             "target_document_id": row.target_document_id, "target_rank": int(row.target_rank),
                             "retrieved_document_ids": json.dumps(source_ids), "source_token_caps": json.dumps(caps),
                             "source_tokens_used": json.dumps(used), "prompt_tokens": prompt_tokens,
                             "prompt_sha256": sha256_text(prompt)})
        packing.append({"case_id": row.case_id, "row_id": row.row_id, "attack": row.attack_family,
                        "member": int(row.member), "dataset": row.dataset,
                        "retrieved_document_ids": json.dumps(source_ids), "source_token_caps": json.dumps(caps),
                        "source_tokens_used": json.dumps(used), "total_assigned": int(sum(caps.values())),
                        "total_used": int(sum(used)), "prompt_tokens": prompt_tokens, "prompt_sha256": sha256_text(prompt)})
    if len(tasks) != 1500: raise RuntimeError(f"generation task mismatch: {len(tasks)}")
    atomic_csv(pd.DataFrame(packing), ROOT / "tables/TABLE_187_01_PACKING_AUDIT.csv")
    checkpoint("BASE_GENERATION_STARTED", tasks=len(tasks), ragleak=100, budgetleak=1400, device="cuda")
    cache = models.ExactPromptCache([EXP179B / "private/EXP179B_RESPONSES.sqlite3",
                                     EXP180R / "private/EXP180R_RESPONSES.sqlite3"])
    answers = models.run_generation(tasks, ROOT / "private/EXP187_RESPONSES.sqlite3", cache, "EXP187_GENERATION")
    frame = pd.DataFrame(metadata); frame["response"] = frame.composite_id.map(answers)
    if frame.response.isna().any(): raise RuntimeError("base generation incomplete")
    cohort = pd.read_csv(COHORT, keep_default_na=False, low_memory=False)
    references = {}
    for item in cohort[cohort.row_id.isin(frame.row_id)].itertuples(index=False):
        references[(str(item.row_id), "RAGLeak")] = str(item.ragleak_reference)
        references[(str(item.row_id), "BudgetLeak-Z")] = str(item.budgetleak_reference)
    frame["reference"] = [references[(str(row.row_id), str(row.attack))] for row in frame.itertuples(index=False)]
    atomic_csv(frame, destination, "gzip")
    checkpoint("BASE_GENERATION_COMPLETE", rows=len(frame), exact_cache_entries=len(cache.values))
    return frame


def build_claim_tasks(base_generation):
    claims_path = ROOT / "private/EXP187_CLAIMS.private.csv.gz"
    tasks_path = ROOT / "private/EXP187_CLAIM_TASKS.private.pkl.gz"
    if claims_path.exists() and tasks_path.exists():
        claims = pd.read_csv(claims_path, keep_default_na=False, low_memory=False)
        tasks = pd.read_pickle(tasks_path, compression="gzip")
        checkpoint("CLAIM_TASKS_REUSED", claims=len(claims), tasks=len(tasks)); return claims, tasks
    exp183 = load_module("exp187_claim_helpers", EXP183R / "code/run_exp183r.py")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    exp180 = load_module("exp187_claim_docs", EXP180R / "code/run_exp180r.py")
    exp180.ROOT = ROOT; exp180.checkpoint = checkpoint
    sample = pd.read_csv(ROOT / "private/EXP187_COHORT.private.csv.gz", keep_default_na=False, low_memory=False)
    documents = exp180.load_documents(sample)
    claims, tasks = [], []
    for row in base_generation.itertuples(index=False):
        source_ids = list(map(str, json.loads(row.retrieved_document_ids)))
        caps = {str(k): int(v) for k, v in json.loads(row.source_token_caps).items()}
        texts = []
        for source in source_ids:
            ids = tokenizer(str(documents[(str(row.dataset), source)]), add_special_tokens=False).input_ids[:caps[source]]
            texts.append(tokenizer.decode(ids, skip_special_tokens=True).strip())
        full_user = exp183.normal_prompt(str(row.query), list(range(1, len(source_ids) + 1)), texts)
        if sha256_text(full_user) != str(row.prompt_sha256): raise RuntimeError(f"prompt mismatch: {row.composite_id}")
        response = str(row.response)
        for index, (start, end, claim) in enumerate(exp183.claim_spans(response), 1):
            claim_id = sha256_text(f"EXP187\0{row.composite_id}\0{start}\0{end}")
            claims.append({"claim_id": claim_id, "composite_id": row.composite_id, "case_id": row.case_id,
                           "row_id": row.row_id, "attack": row.attack, "budget": int(row.budget),
                           "claim_index": index, "claim_start": start, "claim_end": end,
                           "claim_text": claim, "member": int(row.member), "query": row.query})
            variants = [("FULL", "", source_ids, texts)]
            for removed in source_ids:
                kept = [(source, text) for source, text in zip(source_ids, texts) if source != removed]
                variants.append(("REMOVE", removed, [item[0] for item in kept], [item[1] for item in kept]))
            for condition, removed, kept_ids, kept_texts in variants:
                ranks = [source_ids.index(source) + 1 for source in kept_ids]
                user_prompt = exp183.normal_prompt(str(row.query), ranks, kept_texts)
                rendered = exp183.render_prompt(tokenizer, user_prompt)
                task_id = sha256_text(f"{claim_id}\0{condition}\0{removed}")
                tasks.append({"task_id": task_id, "claim_id": claim_id, "condition": condition,
                              "removed_source_id": removed, "rendered_prompt": rendered,
                              "prior_response": response[:start], "claim_text": claim})
    claims, tasks = pd.DataFrame(claims), pd.DataFrame(tasks)
    if tasks.task_id.nunique() != len(tasks) or len(tasks) != 5 * len(claims):
        raise RuntimeError("claim task identity mismatch")
    atomic_csv(claims, claims_path, "gzip"); tasks.to_pickle(tasks_path, compression="gzip")
    atomic_csv(tasks.drop(columns=["rendered_prompt", "prior_response", "claim_text"]),
               ROOT / "private/EXP187_CLAIM_TASK_INDEX.private.csv.gz", "gzip")
    checkpoint("CLAIM_TASKS_FROZEN", claims=len(claims), tasks=len(tasks))
    return claims, tasks


def score_claims(tasks):
    exp183 = load_module("exp187_claim_scorer", EXP183R / "code/run_exp183r.py")
    exp183.ROOT = ROOT; exp183.DB = DB; exp183.checkpoint = checkpoint
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(QWEN, local_files_only=True)
    scored = exp183.score_tasks(tasks, tokenizer)
    full = scored[scored.condition == "FULL"][["claim_id", "mean_log_probability", "claim_tokens"]].rename(
        columns={"mean_log_probability": "full_mean_log_probability"})
    removed = scored[scored.condition == "REMOVE"].merge(full, on="claim_id", validate="many_to_one")
    if not removed.claim_tokens_x.eq(removed.claim_tokens_y).all():
        raise RuntimeError("full/remove claim token count mismatch")
    removed["delta"] = removed.full_mean_log_probability - removed.mean_log_probability
    removed["positive_delta"] = removed.delta.clip(lower=0)
    removed["claim_tokens"] = removed.claim_tokens_x.astype(int)
    removed["information_cost_nats"] = removed.positive_delta * removed.claim_tokens
    atomic_csv(removed, ROOT / "tables/TABLE_187_02_CLAIM_SOURCE_COST.csv.gz", "gzip")
    checkpoint("CLAIM_COSTS_COMPLETE", source_claim_pairs=len(removed),
               mean_positive_delta=float(removed.positive_delta.mean()))
    return removed


def normalized_claim_hash(value):
    return sha256_text(" ".join(str(value).casefold().split()))


def apply_policy(base_generation, claims, costs, cumulative):
    condition = PRIMARY if cumulative else RESET
    cost_map = {str(claim): cell[["removed_source_id", "information_cost_nats", "positive_delta"]].to_dict("records")
                for claim, cell in costs.groupby("claim_id", sort=False)}
    claim_map = {str(composite): cell.sort_values("claim_index")
                 for composite, cell in claims.groupby("composite_id", sort=False)}
    rows, audits = [], []
    for case_id, session in base_generation.sort_values(["case_id", "budget"]).groupby("case_id", sort=False):
        total_ledger, decision_ledger, released_hashes = defaultdict(float), defaultdict(float), defaultdict(set)
        for turn in session.itertuples(index=False):
            if not cumulative:
                total_ledger, decision_ledger, released_hashes = defaultdict(float), defaultdict(float), defaultdict(set)
            released, blocked = [], 0
            for claim in claim_map[str(turn.composite_id)].itertuples(index=False):
                digest = normalized_claim_hash(claim.claim_text); total_charge, decision_charge = {}, {}
                for item in cost_map[str(claim.claim_id)]:
                    source = str(item["removed_source_id"]); duplicate = digest in released_hashes[source]
                    total_charge[source] = 0.0 if duplicate else float(item["information_cost_nats"])
                    decision_charge[source] = 0.0 if duplicate else float(item["positive_delta"])
                total_ok = all(total_ledger[source] + value <= TOTAL_BUDGET + 1e-12
                               for source, value in total_charge.items())
                decision_ok = all(decision_ledger[source] + value <= DECISION_BUDGET + 1e-12
                                  for source, value in decision_charge.items())
                allowed = total_ok and decision_ok
                if allowed:
                    released.append(str(claim.claim_text)); action = "RELEASE"
                    for source in total_charge:
                        total_ledger[source] += total_charge[source]
                        decision_ledger[source] += decision_charge[source]
                        released_hashes[source].add(digest)
                else:
                    blocked += 1; action = "SUPPRESS"
                audits.append({"condition": condition, "case_id": case_id, "row_id": turn.row_id,
                               "attack": turn.attack, "budget": int(turn.budget), "claim_id": claim.claim_id,
                               "claim_index": int(claim.claim_index), "claim_text": claim.claim_text,
                               "decision": action, "total_gate_pass": total_ok,
                               "decision_gate_pass": decision_ok})
            response = "".join(released).strip() or "I don't know."
            rows.append({**{column: getattr(turn, column) for column in base_generation.columns
                            if column not in {"condition", "response"}},
                         "condition": condition, "response": response,
                         "blocked_claims": blocked, "original_response": str(turn.response)})
    return pd.DataFrame(rows), pd.DataFrame(audits)


def baseline_generations(sample):
    old = pd.read_csv(OLD_GENERATIONS, keep_default_na=False, low_memory=False)
    pairs = sample[["row_id", "attack_family", "query"]].rename(columns={"attack_family": "attack"})
    old = old[old.condition.isin(["NO_DEFENSE", "ORIGINAL_MIRABEL", "GLOBAL_CAP64"])].merge(
        pairs[["row_id", "attack"]], on=["row_id", "attack"], validate="many_to_one")
    if len(old) != 4500: raise RuntimeError(f"baseline generation mismatch: {len(old)}")
    queries = sample.set_index(["row_id", "attack_family"])["query"].to_dict()
    old["query"] = [queries[(str(row.row_id), str(row.attack))] for row in old.itertuples(index=False)]
    old["case_id"] = old.attack + "|" + old.row_id.astype(str)
    return old


def evaluate(sample, base_generation, session, reset, audits, config):
    old = baseline_generations(sample)
    common = ["composite_id", "case_id", "row_id", "member", "domain", "attack", "condition", "budget",
              "reference", "response", "query"]
    candidate = base_generation.rename(columns={"dataset": "domain"}).copy()
    session_eval = session.rename(columns={"dataset": "domain"}).copy()
    reset_eval = reset.rename(columns={"dataset": "domain"}).copy()
    all_generations = pd.concat([old[common], candidate[common], session_eval[common], reset_eval[common]],
                                ignore_index=True)
    if len(all_generations) != 9000 or all_generations.groupby(["condition", "attack"]).size().min() != 100:
        raise RuntimeError(f"evaluation row contract failed: {len(all_generations)}")
    runner = load_module("exp187_external_scorer", EXP160 / "code/run_exp160.py")
    runner.ROOT = ROOT; runner.checkpoint = checkpoint
    metrics = runner.response_metrics(all_generations)
    summary, scores = runner.attack_scores(metrics)
    runner.bootstrap(scores, resamples=int(config["scoring"]["bootstrap_resamples"]), seed=18720260828)
    summary.to_csv(ROOT / "tables/TABLE_187_03_EXTERNAL_PRIVACY.csv", index=False)
    pivot = summary.pivot(index="attack", columns="condition", values="effective_auc")
    checks = []
    for attack in ("RAGLeak", "BudgetLeak-Z"):
        value = float(pivot.loc[attack, PRIMARY]); mirabel = float(pivot.loc[attack, "ORIGINAL_MIRABEL"])
        checks.append({"attack": attack, "candidate_eauc": value, "original_mirabel_eauc": mirabel,
                       "absolute_pass": value <= .65 + 1e-12,
                       "mirabel_noninferior_pass": value <= mirabel + .03 + 1e-12})
    gates = pd.DataFrame(checks); gates["all_pass"] = gates.absolute_pass & gates.mirabel_noninferior_pass
    atomic_csv(gates, ROOT / "tables/TABLE_187_05_GATES.csv")
    actions = audits.groupby(["condition", "attack", "decision"], as_index=False).size().rename(columns={"size": "claims"})
    actions["rate"] = actions.claims / actions.groupby(["condition", "attack"]).claims.transform("sum")
    atomic_csv(actions, ROOT / "tables/TABLE_187_04_ACTIONS.csv")
    # Wide, human-readable query/response packet requested by the user.
    key = ["attack", "row_id", "budget"]
    wide = all_generations.pivot(index=key, columns="condition", values="response").reset_index()
    meta = all_generations.sort_values("condition").drop_duplicates(key)[key + ["member", "domain", "query", "reference"]]
    wide = meta.merge(wide, on=key, validate="one_to_one")
    rank = sample[["row_id", "attack_family", "target_document_id", "target_rank"]].rename(columns={"attack_family": "attack"})
    wide = wide.merge(rank, on=["row_id", "attack"], validate="many_to_one")
    score_wide = scores.pivot(index=["attack", "row_id"], columns="condition", values="attack_score")
    score_wide.columns = [f"attack_score__{column}" for column in score_wide.columns]
    wide = wide.merge(score_wide.reset_index(), on=["attack", "row_id"], validate="many_to_one")
    wide["dc_mcel_changed"] = wide[PRIMARY] != wide[BASE]
    wide["dc_mcel_reset_changed"] = wide[RESET] != wide[BASE]
    atomic_csv(wide, ROOT / "query_response_external_audit.csv")
    atomic_csv(wide, ROOT / "private/QUERY_RESPONSE_EXTERNAL_AUDIT.private.csv.gz", "gzip")
    passed = bool(gates.all_pass.all())
    verdict = "DC_MCEL_EXTERNAL_EXPLORATORY_GO" if passed else "DC_MCEL_EXTERNAL_EXPLORATORY_FAIL"
    result = {"experiment": "Exp187", "status": "COMPLETE", "verdict": verdict,
              "candidate": PRIMARY, "exploratory_gate": passed, "targets_per_attack": 100,
              "class_counts": {"RAGLeak": {"member": 52, "nonmember": 48},
                               "BudgetLeak-Z": {"member": 64, "nonmember": 36}},
              "local_qwen_generation_rows": 1500, "paid_api_calls": 0,
              "fresh_blind": False, "full_900_900_confirmation": False,
              "claim_boundary": config["claim_boundary"], "completed_utc": now(),
              "metrics": summary.to_dict("records")}
    atomic_json(ROOT / "FINAL_RESULT.json", result)
    report = ["# Exp187 — DC-MCEL 외부 공격 탐색 평가", "",
              f"- 판정: **{verdict}**", f"- 동결 후보: **{PRIMARY}**", "",
              "## 공격 성능", "", summary.to_markdown(index=False, floatfmt=".4f"), "",
              "## 사전 동결 게이트", "", gates.to_markdown(index=False, floatfmt=".4f"), "",
              "## 해석 경계", "", config["claim_boundary"], "",
              "BudgetLeak primary는 14개 출력 예산을 같은 공격 세션으로 보고 source별 장부를 누적한다. RESET 조건은 각 출력 예산에서 장부를 초기화하는 진단이다.", "",
              "질의와 실제 응답은 query_response_external_audit.csv에 저장했다.", ""]
    atomic_text(ROOT / "reports/REPORT_187_EXTERNAL_ATTACKS_KO.md", "\n".join(report))
    checkpoint("COMPLETE", verdict=verdict, gate=passed, query_response_rows=len(wide))
    return result


def main():
    config = verify_inputs(); sample = frozen_cohort(); base = prepare_generation(sample)
    claims, tasks = build_claim_tasks(base); costs = score_claims(tasks)
    session, audit_session = apply_policy(base, claims, costs, cumulative=True)
    reset, audit_reset = apply_policy(base, claims, costs, cumulative=False)
    atomic_csv(pd.concat([session, reset], ignore_index=True), ROOT / "private/EXP187_DEFENDED_RESPONSES.private.csv.gz", "gzip")
    audits = pd.concat([audit_session, audit_reset], ignore_index=True)
    atomic_csv(audits, ROOT / "claim_release_audit.csv.gz", "gzip")
    checkpoint("POLICY_RESPONSES_COMPLETE", response_rows=len(session) + len(reset), claim_decisions=len(audits))
    return evaluate(sample, base, session, reset, audits, config)


if __name__ == "__main__":
    try:
        print(json.dumps(main(), ensure_ascii=False, indent=2, default=str), flush=True)
    except Exception as error:
        checkpoint("FAILED", error_type=type(error).__name__, error=str(error))
        raise
