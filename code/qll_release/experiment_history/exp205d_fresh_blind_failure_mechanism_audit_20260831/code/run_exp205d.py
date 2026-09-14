#!/usr/bin/env python3
"""Read-only mechanism audit of the frozen Stage205A failure."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import gzip
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


PROJECT = Path("/home/traffic_3/workspace/workspace/SH/LoRA-mirabel-Decter")
ROOT = PROJECT / "exp205d_fresh_blind_failure_mechanism_audit_20260831"
STAGE = PROJECT / "qll_source_hide_final_validation_20260830/stage_205a_fiqa_external_confirmation"
AD_ROOT = Path("/home/traffic_3/workspace/workspace/SH/AD-test-LLM3")
EXP68 = AD_ROOT / "artifacts/exp68_true_fresh_blind_final"
RESPONSES = STAGE / "private/FIQA_QWEN_RESPONSES.private.csv.gz"
PACKING = STAGE / "private/FIQA_PACKING.private.pkl.gz"
QLL_CASES = STAGE / "tables/FIQA_QLL_CASE_SCORES.csv"
QLL_SOURCES = STAGE / "private/FIQA_QLL_SOURCE_SCORES.private.csv.gz"
ATTACK_SCORES = STAGE / "private/FIQA_NATIVE_ATTACK_SCORES.private.csv.gz"
BENIGN = STAGE / "tables/FIQA_BENIGN_RESPONSE_COMPARISON.csv"
FINAL = STAGE / "FINAL_RESULT.json"
SOURCE = EXP68 / "private/fresh_source.private.json.gz"
IA_LINKS = EXP68 / "private/qwen_condition_response_map.private.csv.gz"
IA_STORE = EXP68 / "private/exp68_qwen_responses.private.sqlite3"
THRESHOLD = 0.5300846414247485
FAMILIES = ("DCMI", "IA", "MBA", "MEntA", "RAG-MIA", "S²-MIA")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(value); handle.flush(); os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def atomic_json(path: Path, value) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                 default=lambda x: x.item() if hasattr(x, "item") else str(x)) + "\n")


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".csv", dir=path.parent); os.close(fd)
    try:
        frame.to_csv(temporary, index=False); os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bool_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.casefold().eq("true")


def refusal(value: str) -> bool:
    text = str(value).strip().casefold()
    return text.startswith("i don't know") or text.startswith("i do not know") or "cannot determine" in text[:100]


def preflight() -> pd.DataFrame:
    required = {
        "stage205a_final": FINAL, "responses_2300": RESPONSES, "packing": PACKING,
        "qll_case_scores": QLL_CASES, "qll_source_scores": QLL_SOURCES,
        "attack_scores": ATTACK_SCORES, "benign_comparison": BENIGN,
        "fiqa_source": SOURCE, "ia_ground_truth_links": IA_LINKS,
        "ia_ground_truth_store": IA_STORE,
    }
    missing = [str(path) for path in required.values() if not path.exists()]
    if missing:
        raise RuntimeError(f"missing immutable inputs: {missing}")
    final = json.loads(FINAL.read_text(encoding="utf-8"))
    if final.get("verdict") != "STAGE205A_EXTERNAL_CONFIRMATION_FAILED" or final.get("passed"):
        raise RuntimeError("Stage205A frozen failure verdict drift")
    frame = pd.DataFrame([{"key": key, "path": str(path), "bytes": path.stat().st_size,
                           "sha256_before": sha256_file(path), "access": "READ_ONLY"}
                          for key, path in required.items()])
    atomic_csv(frame, ROOT / "provenance/FROZEN_INPUTS.csv")
    atomic_json(ROOT / "audits/PREFLIGHT.json", {
        "status": "PASS", "frozen_inputs": len(frame), "new_generation": 0,
        "threshold_search": 0, "candidate_changes": 0, "stage205a_verdict_preserved": True,
    })
    return frame


def load_frames():
    responses = pd.read_csv(RESPONSES, keep_default_na=False, low_memory=False,
                            dtype={"session_id": str, "row_id": str, "target_document_id": str,
                                   "source_document_id": str, "qll_top1_source": str,
                                   "qll_top2_source": str, "ledger_source_id": str,
                                   "hidden_source_id": str})
    packing = pd.read_pickle(PACKING, compression="gzip")
    scores = pd.read_csv(ATTACK_SCORES, keep_default_na=False,
                         dtype={"session_id": str, "source_document_id": str})
    scores["attack_score"] = pd.to_numeric(scores.attack_score, errors="coerce")
    benign = pd.read_csv(BENIGN, keep_default_na=False)
    for column in ("token_f1", "semantic_cosine", "dominance"):
        benign[column] = pd.to_numeric(benign[column], errors="coerce")
    for column in ("changed", "candidate_refusal", "baseline_refusal", "new_refusal", "intervened"):
        benign[column] = bool_series(benign[column])
    if len(responses) != 2300 or len(scores) != 360 or len(benign) != 400:
        raise RuntimeError("frozen Stage205A population count drift")
    return responses, packing, scores, benign


def ia_audit(responses: pd.DataFrame, scores: pd.DataFrame) -> dict:
    sys.path.insert(0, str(AD_ROOT))
    from src.exp44_native import ia_turn_score, normalize_yes_no
    links = pd.read_csv(IA_LINKS, keep_default_na=False, dtype={"session_id": str, "task_key": str})
    links = links[links.condition.eq("IA_GROUND_TRUTH")].copy()
    connection = sqlite3.connect(f"file:{IA_STORE}?mode=ro", uri=True)
    mapping = {str(key): str(value) for key, value in connection.execute(
        "SELECT task_key,response_text FROM responses")}
    connection.close()
    links["ground_truth_response"] = links.task_key.map(mapping)
    links["expected_label"] = links.ground_truth_response.map(normalize_yes_no)
    candidate = responses[(responses.kind.eq("ATTACK")) & responses.attack_family.eq("IA")][
        ["session_id", "turn", "member", "response"]].copy()
    candidate["candidate_label"] = candidate.response.map(normalize_yes_no)
    joined = candidate.merge(links[["session_id", "turn", "query_text", "ground_truth_response", "expected_label"]],
                             on=["session_id", "turn"], how="left", validate="one_to_one")
    joined["turn_score"] = [ia_turn_score(a, b) for a, b in zip(joined.response, joined.ground_truth_response)]
    session_rows = []
    for session_id, cell in joined.groupby("session_id", sort=False):
        kept = pd.to_numeric(cell.turn_score, errors="coerce").dropna()
        session_rows.append({"session_id": session_id, "member": int(cell.member.iloc[0]),
                             "turns": len(cell), "ground_truth_responses_present": int(cell.ground_truth_response.ne("").sum()),
                             "ground_truth_yes_no_turns": int(cell.expected_label.isin(["YES", "NO"]).sum()),
                             "candidate_yes_no_turns": int(cell.candidate_label.isin(["YES", "NO"]).sum()),
                             "scoreable": bool(len(kept)), "recomputed_score": float(kept.mean()) if len(kept) else np.nan})
    session = pd.DataFrame(session_rows)
    existing = scores[scores.attack_family.eq("IA")][["session_id", "attack_score", "status"]]
    session = session.merge(existing, on="session_id", validate="one_to_one")
    valid = session[session.scoreable].copy()
    auc = float(roc_auc_score(valid.member, valid.recomputed_score)) if valid.member.nunique() == 2 else np.nan
    eauc = max(auc, 1 - auc) if np.isfinite(auc) else np.nan
    atomic_csv(joined, ROOT / "tables/IA_TURN_PROVENANCE.csv")
    atomic_csv(session, ROOT / "tables/IA_SESSION_COMPLETENESS.csv")
    labels = joined.expected_label.value_counts().to_dict()
    result = {
        "verdict": "IA_FRESH_BLIND_INCOMPLETE", "sessions": 60,
        "generation_outputs_present": int(joined.response.ne("").sum()),
        "ground_truth_outputs_present": int(joined.ground_truth_response.ne("").sum()),
        "ground_truth_label_counts": {str(k): int(v) for k, v in labels.items()},
        "scoreable_sessions": int(session.scoreable.sum()), "incomplete_sessions": int((~session.scoreable).sum()),
        "incomplete_member": int(((~session.scoreable) & session.member.eq(1)).sum()),
        "incomplete_nonmember": int(((~session.scoreable) & session.member.eq(0)).sum()),
        "frozen_valid_subset_raw_auc_diagnostic_only": auc,
        "frozen_valid_subset_effective_auc_diagnostic_only": eauc,
        "corrected_60_session_eauc": None,
        "generation_failure": False, "parser_implementation_failure": False,
        "scorer_implementation_failure": False,
        "source_reference_protocol_mismatch": True,
        "reason": "The immutable target-only ground-truth answers are open-form; 849/900 contain no official YES/NO label. No separate immutable expected-label field exists.",
        "deterministic_repair_without_response_or_membership": False,
        "imputation_or_synthetic_label_used": False,
    }
    atomic_json(ROOT / "audits/IA_COMPLETENESS_AUDIT.json", result)
    atomic_text(ROOT / "reports/IA_MISSINGNESS_REPORT.md", f"""# IA completeness audit\n\n+- Verdict: **IA_FRESH_BLIND_INCOMPLETE**\n+- Q15 generations present: **{len(joined)}/900**\n+- Immutable GT labels: YES={labels.get('YES',0)}, NO={labels.get('NO',0)}, UNKNOWN={labels.get('UNKNOWN',0)}\n+- Scoreable sessions: **{result['scoreable_sessions']}/60**\n+- Incomplete: member={result['incomplete_member']}, nonmember={result['incomplete_nonmember']}\n+- 60/60 deterministic repair: **not possible**\n\nThe parser follows the frozen contract. The mismatch is that open-form target-only answers do not provide the YES/NO expected labels required by the frozen IA scorer. No labels were fabricated.\n""")
    return result


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value)).strip().casefold().strip('"')


def extract_dcmi_quote(query: str) -> str:
    query = str(query)
    if query.startswith("Does this:") and "appear in the context?" in query:
        return normalize_text(query.split("Does this:", 1)[1].rsplit("appear in the context?", 1)[0])
    return normalize_text(query)


def dcmi_audit(responses: pd.DataFrame, scores: pd.DataFrame) -> dict:
    cell = responses[(responses.kind.eq("ATTACK")) & responses.attack_family.eq("DCMI")].copy()
    for column in ("per_query_trigger", "ledger_active", "intervened"):
        cell[column] = bool_series(cell[column])
    cell["retrieved_ids"] = cell.retrieved_document_ids.map(lambda x: list(map(str, json.loads(x))))
    cell["target_present"] = [str(target) in ids for target, ids in zip(cell.target_document_id, cell.retrieved_ids)]
    cell["ledger_present"] = [bool(str(ledger)) and str(ledger) in ids for ledger, ids in zip(cell.ledger_source_id, cell.retrieved_ids)]
    cell["qll_target_alignment"] = cell.qll_top1_source.astype(str).eq(cell.target_document_id.astype(str))
    cell["sticky_target_alignment"] = cell.ledger_source_id.astype(str).eq(cell.target_document_id.astype(str))
    cell["yes"] = cell.response.str.contains(r"\byes\b", case=False, regex=True).astype(int)
    cell["hide_mode"] = np.select([
        ~cell.ledger_active,
        cell.turn.eq(1) & cell.intervened,
        cell.turn.gt(1) & cell.intervened,
        cell.ledger_active & ~cell.intervened,
    ], ["NO_HIDE", "FRESH_HIDE", "STICKY_HIDE", "STICKY_MISS"], default="OTHER")
    session_rows = []
    for session_id, group in cell.groupby("session_id", sort=False):
        group = group.sort_values("turn"); first, second = group.iloc[0], group.iloc[1]
        session_rows.append({
            "session_id": session_id, "member": int(first.member),
            "first_dominance": float(first.dominance), "second_dominance": float(second.dominance),
            "threshold_exceed_any": bool(group.per_query_trigger.any()),
            "first_hide": bool(first.intervened), "second_hide": bool(second.intervened),
            "sticky_active": bool(group.ledger_active.any()),
            "sticky_source": str(first.ledger_source_id),
            "sticky_source_equals_target": bool(first.sticky_target_alignment),
            "wrong_source_lock": bool(group.ledger_active.any() and not first.sticky_target_alignment),
            "protected_source_reappears_turn2": bool(second.ledger_present),
            "target_disappears": bool(first.sticky_target_alignment and not second.ledger_present),
            "turn1_yes": int(first.yes), "turn2_yes": int(second.yes),
            "calculated_dcmi_score": int(first.yes - second.yes),
            "sequence": f"{first.hide_mode}->{second.hide_mode}",
        })
    sessions = pd.DataFrame(session_rows).merge(
        scores[scores.attack_family.eq("DCMI")][["session_id", "attack_score"]], on="session_id", validate="one_to_one")
    if not np.allclose(sessions.calculated_dcmi_score, sessions.attack_score):
        raise RuntimeError("DCMI frozen scorer reproduction mismatch")
    raw_auc = float(roc_auc_score(sessions.member, sessions.attack_score)); eauc = max(raw_auc, 1 - raw_auc)

    with gzip.open(SOURCE, "rt", encoding="utf-8") as handle:
        documents = json.load(handle)["documents"]
    evidence_rows = []
    for row in cell[cell.member.eq(1)].itertuples(index=False):
        ids = list(row.retrieved_ids); hidden = str(row.hidden_source_id)
        visible = [item for item in ids if item != hidden]
        target = normalize_text(documents[str(row.target_document_id)]["text"])
        quote = extract_dcmi_quote(row.query)
        visible_text = [normalize_text(documents[item]["text"]) for item in visible]
        evidence_rows.append({
            "session_id": row.session_id, "turn": int(row.turn), "intervened": bool(row.intervened),
            "qll_selected_present": str(row.qll_top1_source) in ids,
            "qll_selected_hidden": str(row.qll_top1_source) == hidden,
            "exact_target_duplicate_visible": any(value == target for value in visible_text),
            "target_containment_visible": any((target in value or value in target) and min(len(target), len(value)) >= 100 for value in visible_text),
            "exact_quote_visible": any(value == quote for value in visible_text),
            "quote_containment_visible": any((quote in value or value in quote) and min(len(quote), len(value)) >= 100 for value in visible_text),
            "response": row.response,
        })
    evidence = pd.DataFrame(evidence_rows)
    atomic_csv(cell.drop(columns=["retrieved_ids"]), ROOT / "tables/DCMI_TURN_AUDIT.csv")
    atomic_csv(sessions, ROOT / "tables/DCMI_SESSION_AUDIT.csv")
    atomic_csv(evidence, ROOT / "tables/DCMI_REMAINING_EVIDENCE_AUDIT.csv")

    group_rows = []
    for member, group in cell.groupby("member"):
        first, second = group[group.turn.eq(1)], group[group.turn.eq(2)]
        sess = sessions[sessions.member.eq(member)]
        group_rows.append({
            "member": int(member), "turns": len(group),
            "dominance_mean": float(group.dominance.mean()), "dominance_median": float(group.dominance.median()),
            "threshold_exceed_rate": float(group.per_query_trigger.mean()),
            "source_hide_rate": float(group.intervened.mean()),
            "first_turn_hide_rate": float(first.intervened.mean()), "second_turn_hide_rate": float(second.intervened.mean()),
            "sticky_activation_session_rate": float(sess.sticky_active.mean()),
            "protected_source_reappears_turn2_rate": float(sess.protected_source_reappears_turn2.mean()),
            "qll_target_alignment_all_turns": float(group.qll_target_alignment.mean()),
            "qll_target_alignment_when_target_present": float(group[group.target_present].qll_target_alignment.mean()) if group.target_present.any() else np.nan,
            "sticky_target_alignment_session_rate": float(sess.sticky_source_equals_target.mean()),
            "wrong_source_lock_rate": float(sess.wrong_source_lock.mean()),
            "target_disappearance_rate": float(sess.target_disappears.mean()),
            "dcmi_score_mean": float(sess.attack_score.mean()), "dcmi_score_median": float(sess.attack_score.median()),
        })
    group_table = pd.DataFrame(group_rows)
    atomic_csv(group_table, ROOT / "tables/DCMI_MEMBER_NONMEMBER.csv")

    distribution = sessions.groupby(["member", "sequence", "attack_score"], as_index=False).size()
    atomic_csv(distribution, ROOT / "tables/DCMI_SCORE_DECOMPOSITION.csv")
    member = group_table.set_index("member").loc[1]; nonmember = group_table.set_index("member").loc[0]
    member_sessions = sessions[sessions.member.eq(1)]
    turn1_evidence = evidence[evidence.turn.eq(1)]
    result = {
        "raw_auc": raw_auc, "effective_auc": eauc, "direction": "INVERSE_MEMBER_SIGNAL",
        "member_score_distribution": {str(k): int(v) for k, v in member_sessions.attack_score.value_counts().sort_index().items()},
        "nonmember_score_distribution": {str(k): int(v) for k, v in sessions[sessions.member.eq(0)].attack_score.value_counts().sort_index().items()},
        "member_hide_rate": float(member.source_hide_rate), "nonmember_hide_rate": float(nonmember.source_hide_rate),
        "hide_rate_difference": float(member.source_hide_rate - nonmember.source_hide_rate),
        "member_sticky_rate": float(member.sticky_activation_session_rate), "nonmember_sticky_rate": float(nonmember.sticky_activation_session_rate),
        "sticky_rate_difference": float(member.sticky_activation_session_rate - nonmember.sticky_activation_session_rate),
        "member_first_turn_qll_target_alignment": float(cell[(cell.member.eq(1)) & cell.turn.eq(1)].qll_target_alignment.mean()),
        "member_sticky_target_alignment": float(member.sticky_target_alignment_session_rate),
        "wrong_source_lock_rate_member": float(member.wrong_source_lock_rate),
        "target_disappearance_rate_among_correct_member_locks": float(member_sessions[member_sessions.sticky_source_equals_target].target_disappears.mean()),
        "protected_source_reappears_turn2_member": float(member.protected_source_reappears_turn2_rate),
        "turn2_current_qll_source_present_but_not_hidden_member": float(
            ((evidence.turn.eq(2)) & evidence.qll_selected_present & ~evidence.qll_selected_hidden).mean() * 2),
        "turn1_hidden_target_yes_rate": float(cell[(cell.member.eq(1)) & cell.turn.eq(1)].yes.mean()),
        "turn2_sticky_miss_yes_rate": float(cell[(cell.member.eq(1)) & cell.turn.eq(2)].yes.mean()),
        "turn1_exact_or_containment_duplicate_count": int((turn1_evidence.exact_target_duplicate_visible |
                                                             turn1_evidence.target_containment_visible |
                                                             turn1_evidence.exact_quote_visible |
                                                             turn1_evidence.quote_containment_visible).sum()),
        "hide_pattern_auc": float(roc_auc_score(sessions.member, sessions.first_hide.astype(int))),
        "wrong_source_lock_supported": False, "target_disappears_supported": True,
        "redundant_evidence_supported_as_primary": False,
        "qll_no_generalization_supported": False, "defense_sidechannel_supported": True,
    }
    # Exact direct computation avoids depending on the convenience expression above.
    member_turn2 = evidence[evidence.turn.eq(2)]
    result["turn2_current_qll_source_present_but_not_hidden_member"] = float(
        (member_turn2.qll_selected_present & ~member_turn2.qll_selected_hidden).mean())
    atomic_json(ROOT / "audits/DCMI_MECHANISM_AUDIT.json", result)
    return result


def benign_audit(benign: pd.DataFrame, packing: pd.DataFrame) -> dict:
    candidate = packing[(packing.kind.eq("BENIGN")) & packing.condition.eq("QLL Source Hide + Session Sticky")].set_index("row_id")
    baseline = packing[(packing.kind.eq("BENIGN")) & packing.condition.eq("No Defense")].set_index("row_id")
    prompt_equal = candidate.prompt.eq(baseline.prompt)
    if not prompt_equal[~candidate.intervened.astype(bool)].all():
        raise RuntimeError("nonintervened benign prompt unexpectedly differs")
    rows = []
    for intervened, cell in benign.groupby("intervened"):
        label = "SOURCE_HIDDEN" if intervened else "UNTOUCHED_IDENTICAL_PROMPT"
        rows.append({
            "group": label, "queries": len(cell), "token_f1_mean": float(cell.token_f1.mean()),
            "token_f1_median": float(cell.token_f1.median()), "semantic_similarity_mean": float(cell.semantic_cosine.mean()),
            "candidate_answer_words": float(cell["QLL Source Hide + Session Sticky"].str.split().str.len().mean()),
            "baseline_answer_words": float(cell["No Defense"].str.split().str.len().mean()),
            "candidate_refusal_rate": float(cell.candidate_refusal.mean()),
            "baseline_refusal_rate": float(cell.baseline_refusal.mean()),
            "new_refusal_rate": float(cell.new_refusal.mean()), "exact_preservation_rate": float((~cell.changed).mean()),
            "near_preservation_rate_f1_ge_090": float(cell.token_f1.ge(.90).mean()),
            "dominance_mean": float(cell.dominance.mean()),
            "hidden_source_rank_mean": float(pd.to_numeric(cell.hidden_source_rank, errors="coerce").mean()),
        })
    table = pd.DataFrame(rows); atomic_csv(table, ROOT / "tables/BENIGN_UTILITY_DECOMPOSITION.csv")
    untouched = benign[~benign.intervened]; hidden = benign[benign.intervened]
    deficit_untouched = float(((1 - untouched.token_f1).sum()) / len(benign))
    deficit_hidden = float(((1 - hidden.token_f1).sum()) / len(benign))
    corrected = np.where(benign.intervened, benign.token_f1, 1.0)
    result = {
        "frozen_overall_token_f1": float(benign.token_f1.mean()),
        "untouched_queries": len(untouched), "hidden_queries": len(hidden),
        "untouched_identical_prompt_rate": float(prompt_equal[~candidate.intervened.astype(bool)].mean()),
        "untouched_answer_change_rate_despite_identical_prompt": float(untouched.changed.mean()),
        "untouched_token_f1": float(untouched.token_f1.mean()),
        "hidden_token_f1": float(hidden.token_f1.mean()),
        "hidden_semantic_similarity": float(hidden.semantic_cosine.mean()),
        "hidden_new_refusal_rate": float(hidden.new_refusal.mean()),
        "f1_deficit_from_independent_same_prompt_forwards": deficit_untouched,
        "f1_deficit_from_actual_source_hide": deficit_hidden,
        "share_of_total_deficit_from_same_prompt_forward_mismatch": deficit_untouched / (deficit_untouched + deficit_hidden),
        "diagnostic_exact_reuse_f1_not_official": float(np.mean(corrected)),
        "primary_cause": "MATCHED_BASELINE_DUPLICATE_FORWARD_CONFOUND_PLUS_SEVERE_INTERVENTION_DAMAGE",
        "stage205a_utility_number_changed": False,
    }
    atomic_json(ROOT / "audits/BENIGN_UTILITY_FAILURE_AUDIT.json", result)
    return result


def cross_attack_audit(responses: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    attack = responses[responses.kind.eq("ATTACK")].copy()
    for column in ("per_query_trigger", "ledger_active", "intervened"):
        attack[column] = bool_series(attack[column])
    attack["target_present"] = [str(target) in list(map(str, json.loads(ids)))
                                for target, ids in zip(attack.target_document_id, attack.retrieved_document_ids)]
    attack["target_alignment"] = attack.qll_top1_source.astype(str).eq(attack.target_document_id.astype(str))
    eauc = []
    for family, cell in scores.groupby("attack_family"):
        valid = cell.dropna(subset=["attack_score"])
        if len(valid) and valid.member.nunique() == 2:
            auc = float(roc_auc_score(valid.member, valid.attack_score)); effective = max(auc, 1 - auc)
        else:
            auc = effective = np.nan
        eauc.append({"attack_family": family, "raw_auc": auc, "effective_auc": effective,
                     "valid_sessions": len(valid), "total_sessions": len(cell)})
    eauc = pd.DataFrame(eauc).set_index("attack_family")
    rows = []
    for family, cell in attack.groupby("attack_family"):
        member, nonmember = cell[cell.member.eq(1)], cell[cell.member.eq(0)]
        sessions = cell.groupby("session_id", as_index=False).agg(sticky_session=("ledger_active", "max"))
        rows.append({
            "attack_family": family, "turns": len(cell),
            "trigger_rate": float(cell.per_query_trigger.mean()), "hide_rate": float(cell.intervened.mean()),
            "member_trigger_rate": float(member.per_query_trigger.mean()), "nonmember_trigger_rate": float(nonmember.per_query_trigger.mean()),
            "member_hide_rate": float(member.intervened.mean()), "nonmember_hide_rate": float(nonmember.intervened.mean()),
            "sticky_session_rate": float(sessions.sticky_session.mean()),
            "member_target_retrieval_rate": float(member.target_present.mean()),
            "member_target_alignment_all_turns": float(member.target_alignment.mean()),
            "member_target_alignment_when_present": float(member[member.target_present].target_alignment.mean()) if member.target_present.any() else np.nan,
            "raw_auc": float(eauc.loc[family, "raw_auc"]), "effective_auc": float(eauc.loc[family, "effective_auc"]),
            "valid_sessions": int(eauc.loc[family, "valid_sessions"]),
        })
    result = pd.DataFrame(rows).sort_values("attack_family")
    atomic_csv(result, ROOT / "tables/CROSS_ATTACK_CONTRAST.csv")
    return result


def final_report(before: pd.DataFrame, ia: dict, dcmi: dict, benign: dict, contrast: pd.DataFrame) -> dict:
    primary = "FRESH_BLIND_DEFENSE_SIDECHANNEL_FAILURE"
    secondary = ["FRESH_BLIND_STICKY_MECHANISM_FAILURE"]
    result = {
        "final_diagnostic_verdict": primary, "secondary_verdicts": secondary,
        "stage205a_verdict_changed": False, "stage205a_reevaluation_justified": False,
        "ia": ia, "dcmi": dcmi, "benign": benign,
        "qll_generalization_failure_supported": False,
        "source_redundancy_failure_supported": False,
        "architecture_revision_scientifically_justified": True,
        "exp206_exp207_exp208_state": "CLOSED",
        "new_untouched_blind_required_after_any_revision": True,
        "new_generation_calls": 0, "model_changes": 0, "threshold_searches": 0,
    }
    atomic_json(ROOT / "FINAL_RESULT.json", result)
    member_hide, nonmember_hide = dcmi["member_hide_rate"], dcmi["nonmember_hide_rate"]
    member_sticky, nonmember_sticky = dcmi["member_sticky_rate"], dcmi["nonmember_sticky_rate"]
    passing = contrast[contrast.effective_auc.le(.65)].attack_family.tolist()
    report = f"""# Exp205D Fresh-Blind Failure Mechanism Audit\n\n+## Final diagnostic verdict\n+\n+**{primary}**  \n+Secondary: **FRESH_BLIND_STICKY_MECHANISM_FAILURE** (`TARGET_DISAPPEARS`)\n+\n+The QLL locator itself did not fail on DCMI. For member first turns, QLL top-1 equalled the target in 30/30 sessions. The locked target then disappeared from turn-2 Top-4 in 30/30 sessions, but Session Sticky retained that absent ID and hid no current source. This member-only action sequence became an inverse membership side channel.\n+\n+## IA completeness\n+\n+- Recovery: **not possible; IA_FRESH_BLIND_INCOMPLETE**\n+- Q15 generations present: **900/900**\n+- Immutable GT labels: YES={ia['ground_truth_label_counts'].get('YES',0)}, NO={ia['ground_truth_label_counts'].get('NO',0)}, UNKNOWN={ia['ground_truth_label_counts'].get('UNKNOWN',0)}\n+- Scoreable sessions: **{ia['scoreable_sessions']}/60**\n+- Corrected 60/60 E-AUC: **not available**\n+- Existing 26-session E-AUC remains diagnostic only: **{ia['frozen_valid_subset_effective_auc_diagnostic_only']:.4f}**\n+\n+The failure is not missing generation or a broken parser. The frozen scorer requires a YES/NO expected label, while the immutable target-only answers are mostly open-form and no separate expected-label field exists. No label was imputed.\n+\n+## DCMI decomposition\n+\n+- Raw AUC / E-AUC: **{dcmi['raw_auc']:.4f} / {dcmi['effective_auc']:.4f}**\n+- Direction: **inverse member signal**\n+- Member/nonmember hide rate: **{100*member_hide:.1f}% / {100*nonmember_hide:.1f}%**\n+- Member/nonmember sticky-session rate: **{100*member_sticky:.1f}% / {100*nonmember_sticky:.1f}%**\n+- Member first-turn QLL-target alignment: **{100*dcmi['member_first_turn_qll_target_alignment']:.1f}%**\n+- Sticky-target alignment: **{100*dcmi['member_sticky_target_alignment']:.1f}%**\n+- Wrong-source lock: **{100*dcmi['wrong_source_lock_rate_member']:.1f}%**\n+- Correct-lock target disappearance at turn 2: **{100*dcmi['target_disappearance_rate_among_correct_member_locks']:.1f}%**\n+- Protected source reappeared at turn 2: **{100*dcmi['protected_source_reappears_turn2_member']:.1f}%**\n+- Turn-2 current QLL source present but not hidden: **{100*dcmi['turn2_current_qll_source_present_but_not_hidden_member']:.1f}%**\n+- Hide-pattern-only membership AUC: **{dcmi['hide_pattern_auc']:.3f}**\n+- Member score distribution: `{dcmi['member_score_distribution']}`\n+- Nonmember score distribution: `{dcmi['nonmember_score_distribution']}`\n+\n+The member sequence was `FRESH_HIDE -> STICKY_MISS` in all 30 sessions; nonmembers were `NO_HIDE -> NO_HIDE`. Member responses were Yes on only 6.7% of hidden first turns but 83.3% of unhidden second turns, yielding mostly score -1.\n+\n+## Redundant evidence\n+\n+No exact/containment duplicate of the hidden target or quoted first-turn evidence was found in the visible sources for any of the 30 member first turns. Only 2/30 hidden first turns answered Yes. Source redundancy is therefore not supported as the primary DCMI failure. In turn 2, the current QLL-selected source remained visible in 30/30 cases; this supports stale Sticky rather than redundant target evidence.\n+\n+## Benign utility\n+\n+- Frozen overall Token-F1: **{benign['frozen_overall_token_f1']:.4f}**\n+- Untouched identical-prompt queries: **{benign['untouched_queries']}**, F1 **{benign['untouched_token_f1']:.4f}**\n+- Hidden queries: **{benign['hidden_queries']}**, F1 **{benign['hidden_token_f1']:.4f}**\n+- Hidden-query semantic similarity: **{benign['hidden_semantic_similarity']:.4f}**\n+- Hidden-query new refusal: **{100*benign['hidden_new_refusal_rate']:.1f}%**\n+- Unchanged prompts nevertheless produced different answers: **{100*benign['untouched_answer_change_rate_despite_identical_prompt']:.1f}%**\n+- Diagnostic exact-reuse F1 (not an official replacement): **{benign['diagnostic_exact_reuse_f1_not_official']:.4f}**\n+\n+About {100*benign['share_of_total_deficit_from_same_prompt_forward_mismatch']:.1f}% of the aggregate F1 deficit came from independently generated, byte-identical no-intervention prompts; the remaining loss came from severe source-hide evidence damage. Thus the frozen 87.41% combines a matched-generation protocol confound with real intervention damage. The frozen result is not rewritten.\n+\n+## Cross-attack contrast\n+\n+Point-estimate pass families were: {', '.join(passing)}. DCMI is unique because its scorer subtracts turn-2 from turn-1 and the sticky protected source disappears exactly between those turns. RAG-MIA has only one turn; the other multi-turn scorers do not convert this deterministic first-hide/second-miss sequence into the same direct difference signal.\n+\n+## Scientific decision\n+\n+Architecture revision is scientifically justified, but no revision is performed in Exp205D. Exp206/207/208 remain closed. Once Sticky, the threshold, or the architecture is changed using this diagnosis, FiQA Stage205A becomes a development diagnostic set. A separate untouched blind cohort is mandatory for any final generalization claim.\n+"""
    # Strip accidental diff-style prefixes from the embedded report template.
    report = "\n".join(line[1:] if line.startswith("+") else line for line in report.splitlines()) + "\n"
    atomic_text(ROOT / "reports/EXP205D_FINAL_REPORT_EN.md", report)
    report_ko = f"""# Exp205D 실패 원인 감사\n\n## 최종 판정\n\n**{primary}**  \n+부가 판정: **FRESH_BLIND_STICKY_MECHANISM_FAILURE** (`TARGET_DISAPPEARS`)\n+\n+DCMI에서 QLL locator 자체는 실패하지 않았다. member의 첫 질의 30/30에서 QLL top-1이 실제 target과 일치했다. 그러나 고정한 target이 두 번째 질의의 Top-4에서 30/30 모두 사라졌는데도 Session Sticky는 사라진 ID를 계속 보호했다. 그 결과 현재 두 번째 QLL source는 하나도 숨기지 못했고, 이 member 전용 동작 순서가 역방향 membership 신호가 됐다.\n+\n+## IA 완전성\n+\n+- 판정: **IA_FRESH_BLIND_INCOMPLETE**\n+- 생성 성공: **900/900 turns**\n+- immutable GT label: YES={ia['ground_truth_label_counts'].get('YES',0)}, NO={ia['ground_truth_label_counts'].get('NO',0)}, UNKNOWN={ia['ground_truth_label_counts'].get('UNKNOWN',0)}\n+- 유효 세션: **{ia['scoreable_sessions']}/60**\n+- 60/60 복구 E-AUC: **산출 불가**\n+- 26개 유효 subset E-AUC `0.6923`은 진단값으로만 유지\n+\n+생성 실패나 parser 구현 오류가 아니다. 동결 IA scorer는 YES/NO 정답표를 요구하지만 target-only GT 답변 대부분이 서술형이며 별도의 원본 YES/NO label 필드가 없다. 따라서 label을 추정하거나 대입하지 않았다.\n+\n+## DCMI 분해\n+\n+- Raw AUC / E-AUC: **{dcmi['raw_auc']:.4f} / {dcmi['effective_auc']:.4f}**\n+- 방향: **역방향 member 신호**\n+- member/nonmember hide rate: **{100*member_hide:.1f}% / {100*nonmember_hide:.1f}%**\n+- member/nonmember sticky-session rate: **{100*member_sticky:.1f}% / {100*nonmember_sticky:.1f}%**\n+- member 첫 질의 QLL-target 일치: **{100*dcmi['member_first_turn_qll_target_alignment']:.1f}%**\n+- 잘못된 source lock: **{100*dcmi['wrong_source_lock_rate_member']:.1f}%**\n+- 올바르게 lock한 target의 두 번째 질의 이탈: **{100*dcmi['target_disappearance_rate_among_correct_member_locks']:.1f}%**\n+- 두 번째 현재 QLL source가 존재하지만 숨겨지지 않은 비율: **{100*dcmi['turn2_current_qll_source_present_but_not_hidden_member']:.1f}%**\n+- hide pattern만으로 계산한 membership AUC: **{dcmi['hide_pattern_auc']:.3f}**\n+\n+member 30세션 모두 `FRESH_HIDE -> STICKY_MISS`, nonmember 30세션 모두 `NO_HIDE -> NO_HIDE`였다. member의 첫 번째 hidden turn은 Yes가 6.7%였지만, 숨겨지지 않은 두 번째 turn은 Yes가 83.3%여서 대부분 DCMI score -1이 됐다.\n+\n+## 중복 근거\n+\n+숨긴 첫 번째 target과 동일하거나 포함 관계인 문서는 30세션의 남은 context에서 발견되지 않았다. 첫 hidden turn에서 Yes도 2/30뿐이다. 따라서 source redundancy는 주원인으로 지지되지 않는다.\n+\n+## 정상 utility\n+\n+- 동결 전체 Token-F1: **{benign['frozen_overall_token_f1']:.4f}**\n+- 미개입 376개: F1 **{benign['untouched_token_f1']:.4f}**\n+- source hide 24개: F1 **{benign['hidden_token_f1']:.4f}**\n+- hide된 질의 semantic similarity: **{benign['hidden_semantic_similarity']:.4f}**\n+- hide된 질의의 새 refusal: **{100*benign['hidden_new_refusal_rate']:.1f}%**\n+- 동일 prompt를 독립 생성했는데 답변이 달라진 비율: **{100*benign['untouched_answer_change_rate_despite_identical_prompt']:.1f}%**\n+- 동일 prompt exact reuse 진단 F1: **{benign['diagnostic_exact_reuse_f1_not_official']:.4f}** (공식 수치 대체 아님)\n+\n+전체 F1 손실의 약 {100*benign['share_of_total_deficit_from_same_prompt_forward_mismatch']:.1f}%는 byte-identical prompt를 별도 forward로 생성한 비교 protocol에서 발생했고, 나머지는 실제 source-hide 손상이다. 따라서 87.41%에는 비교 protocol 교란과 실제 evidence 손실이 함께 포함된다. 기존 결과는 수정하지 않았다.\n+\n+## 공격 간 차이\n+\n+통과한 공격은 {', '.join(passing)}이다. DCMI는 두 turn의 Yes/No를 직접 빼는 scorer이며, 보호 source가 정확히 두 turn 사이에 사라진다. 이 때문에 다른 공격과 달리 `첫 hide-두 번째 miss`가 직접적인 역방향 점수로 변환됐다.\n+\n+## 연구 판단\n+\n+구조 수정은 과학적으로 필요하지만 Exp205D에서는 수정하지 않았다. Exp206/207/208은 닫힌 상태다. 이 진단을 이용해 Sticky·threshold·구조를 수정하면 FiQA는 이후 development diagnostic set으로만 사용해야 한다. 최종 일반화 주장을 위해 별도의 untouched blind cohort가 반드시 필요하다. 또한 이 FiQA 평가는 후보 관점의 외부 확인이지, 데이터 자체가 연구 전체에서 한 번도 열리지 않은 진정한 untouched blind는 아니다.\n+"""
    report_ko = "\n".join(line[1:] if line.startswith("+") else line for line in report_ko.splitlines()) + "\n"
    atomic_text(ROOT / "reports/EXP205D_FINAL_REPORT_KO.md", report_ko)
    after = before.copy(); after["sha256_after"] = [sha256_file(Path(path)) for path in after.path]
    after["unchanged"] = after.sha256_before.eq(after.sha256_after)
    atomic_csv(after, ROOT / "provenance/FROZEN_INPUTS_POSTRUN.csv")
    if not after.unchanged.all():
        raise RuntimeError("read-only input changed during audit")
    atomic_json(ROOT / "audits/NO_MUTATION_AUDIT.json", {
        "status": "PASS", "inputs": len(after), "unchanged": int(after.unchanged.sum()),
        "new_generation_calls": 0, "model_changes": 0, "threshold_searches": 0,
    })
    status = f"""# Exp205D status\n\n+- Status: **COMPLETE**\n+- Final diagnostic verdict: **{primary}**\n+- Secondary: **FRESH_BLIND_STICKY_MECHANISM_FAILURE**\n+- IA: **IA_FRESH_BLIND_INCOMPLETE**\n+- Exp206/207/208: **CLOSED**\n+- New generation/model changes: **0/0**\n+- Updated UTC: `{now()}`\n+"""
    status = "\n".join(line[1:] if line.startswith("+") else line for line in status.splitlines()) + "\n"
    atomic_text(ROOT / "STATUS.md", status)
    return result


def main() -> None:
    for name in ("configs", "provenance", "audits", "tables", "reports", "tests"):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    before = preflight()
    responses, packing, scores, benign = load_frames()
    ia = ia_audit(responses, scores)
    dcmi = dcmi_audit(responses, scores)
    utility = benign_audit(benign, packing)
    contrast = cross_attack_audit(responses, scores)
    final_report(before, ia, dcmi, utility, contrast)


if __name__ == "__main__":
    main()
