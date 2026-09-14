#!/usr/bin/env python3
"""Read-only Core6 diagnostic for the frozen BC-MIRABEL simple-hide model.

No detector, threshold, retrieval, answer generation, or attack score is
created here. Families whose member/nonmember labels do not match the frozen
CLEAN protected corpus fail closed rather than being evaluated misleadingly.
"""
from __future__ import annotations

from collections import Counter
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import sqlite3
import tempfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
WORK = Path("/home/traffic_3/workspace/workspace/SH")
CLEAN = WORK / "RAG_MIA_DEFENSE_CLEAN_20260911/experiments/OUTPUT_CONDITIONED_SELECTIVE_LOO_GUARD_CLEAN_V1"
BC = WORK / "RAG_MIA_DEFENSE_CLEAN_20260911/experiments/BC_MIRABEL_COUNTERFACTUAL_GROUNDED_DISCLOSURE"
SOURCE = WORK / "b2_counterfactual_min_20260910/phase_02_core6/QUERY_RESPONSE_REVIEW.csv.gz"
REQUEST = Path("/home/cau_lab/.codex/attachments/649cde99-9bc4-46f7-8bcc-a292d130d726/pasted-text.txt")
FAMILIES = ("MEntA", "IA", "DCMI", "S²-MIA", "MBA", "RAG-MIA")
BUDGETS = {"MEntA": 5, "IA": 15, "DCMI": 2, "S²-MIA": 1, "MBA": 1, "RAG-MIA": 1}
TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(value):
    return hashlib.sha256(str(value).encode()).hexdigest()


def atomic_text(path: Path, value: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(value); f.flush(); os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp): os.unlink(temp)


def atomic_json(path: Path, value):
    atomic_text(path, json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, default=str) + "\n")


def atomic_csv(frame, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def checkpoint(stage, **details):
    payload = {"campaign": "BC_MIRABEL_SIMPLE_HIDE_CORE6_DIAGNOSTIC", "stage": stage,
               "updated_utc": now(), "pid": os.getpid(), "new_generation": 0,
               "new_model_forward": 0, **details}
    atomic_json(ROOT / "HEARTBEAT.json", payload)
    atomic_json(ROOT / "checkpoints" / f"{stage}.json", payload)
    atomic_text(ROOT / "STATUS.md", "# BC-MIRABEL Simple-Hide Core6 Diagnostic\n\n" +
                "\n".join(f"- {k}: `{v}`" for k, v in payload.items()) + "\n")
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def select_core6():
    source = pd.read_csv(SOURCE, keep_default_na=False, dtype=str)
    source["member"] = source.member.astype(int)
    source["turn"] = source.turn.astype(int)
    sessions = []
    for (family, member, session), cell in source.groupby(["family", "member", "session_id"], sort=False):
        ordered = cell.sort_values(["turn", "case_id"])
        if len(ordered) != BUDGETS[family] or ordered.turn.tolist() != list(range(1, BUDGETS[family] + 1)):
            raise RuntimeError(f"native session budget/order mismatch: {session}")
        sessions.append({"family": family, "member": member, "session_id": session,
                         "selection_key": min(sha256_text(x) for x in ordered.case_id.astype(str))})
    sessions = pd.DataFrame(sessions)
    chosen = pd.concat([cell.sort_values(["selection_key", "session_id"]).head(20)
                        for _, cell in sessions.groupby(["family", "member"], sort=True)], ignore_index=True)
    return source.merge(chosen, on=["family", "member", "session_id"], validate="many_to_one").sort_values(
        ["family", "member", "selection_key", "turn", "case_id"]).reset_index(drop=True)


def protected_ids():
    ids = set()
    manifest = []
    for path in sorted((CLEAN / "private").glob("CORPUS_*.csv.gz")):
        frame = pd.read_csv(path, dtype=str)
        ids.update(frame.document_id.astype(str))
        manifest.append({"file": str(path), "rows": len(frame), "sha256": sha256_file(path)})
    return ids, manifest


def preflight():
    selected = select_core6()
    ids, corpus_manifest = protected_ids()
    audit = []
    for (family, member), cell in selected.groupby(["family", "member"], sort=True):
        targets = set(cell.target_id.astype(str))
        present = targets & ids
        required_present = member == 1
        valid = len(present) == (len(targets) if required_present else 0)
        audit.append({"family": family, "label": "member" if member else "nonmember",
                      "sessions": cell.session_id.nunique(), "queries": len(cell), "targets": len(targets),
                      "targets_present_in_clean_db": len(present), "targets_absent_from_clean_db": len(targets-present),
                      "membership_contract_valid": valid})
    audit = pd.DataFrame(audit)
    family_status = []
    for family, cell in audit.groupby("family"):
        valid = bool(cell.membership_contract_valid.all())
        reason = "CLEAN membership labels match actual corpus inclusion/exclusion" if valid else (
            "PROTOCOL_OR_LINEAGE_UNAVAILABLE: frozen CLEAN corpus does not realize this family's member/nonmember labels")
        family_status.append({"family": family, "status": "AVAILABLE" if valid else "PROTOCOL_OR_LINEAGE_UNAVAILABLE",
                              "reason": reason})
    family_status = pd.DataFrame(family_status)
    expected_available = {"MEntA", "S²-MIA", "MBA"}
    actual_available = set(family_status[family_status.status.eq("AVAILABLE")].family)
    existing = pd.read_csv(CLEAN / "private/PHASE1_ATTACK_SELECTION.csv.gz", keep_default_na=False, dtype=str)
    new_valid_ids = set(selected[selected.family.isin(expected_available)].case_id.astype(str))
    exact_existing = new_valid_ids == set(existing.case_id.astype(str))
    if actual_available != expected_available or not exact_existing:
        raise RuntimeError("unexpected CLEAN lineage status; fail closed")
    # IA scorer metadata required by the old benchmark was removed during the
    # approved cleanup; do not regenerate or fabricate it.
    ia_gt = WORK / "mirabel_benchmark_six_attack_bge_qwen_20260907/private/IA_ORIGINAL_GROUND_TRUTH.private.csv.gz"
    checks = [
        {"check": "clean_precommit_hash", "pass": sha256_file(CLEAN/"configs/PRECOMMIT.json") == (CLEAN/"configs/PRECOMMIT.sha256").read_text().split()[0]},
        {"check": "existing_three_cohort_ids_exact", "pass": exact_existing},
        {"check": "existing_three_membership_contract", "pass": actual_available == expected_available},
        {"check": "new_three_membership_contract", "pass": False,
         "detail": "DCMI/IA/RAG-MIA labels mismatch actual frozen CLEAN DB inclusion"},
        {"check": "ia_immutable_ground_truth_present", "pass": ia_gt.exists(), "detail": str(ia_gt)},
        {"check": "legacy_numeric_join", "pass": True, "detail": "none; only valid frozen CLEAN families reused"},
    ]
    atomic_csv(audit, ROOT / "audits/CORE6_MEMBERSHIP_CONTRACT.csv")
    atomic_csv(family_status, ROOT / "audits/FAMILY_PROTOCOL_STATUS.csv")
    atomic_csv(pd.DataFrame(checks), ROOT / "audits/PREFLIGHT_CHECKS.csv")
    result = {"verdict": "BC_MIRABEL_CORE6_PREFLIGHT_PARTIAL_PASS",
              "available_families": sorted(actual_available),
              "unavailable_families": sorted(set(FAMILIES)-actual_available),
              "selected_queries": len(selected), "selected_sessions": selected.session_id.nunique(),
              "clean_existing_ids_exact": exact_existing, "corpus_manifest": corpus_manifest,
              "request_sha256": sha256_file(REQUEST), "source_sha256": sha256_file(SOURCE),
              "attack_selection_sha256": sha256_file(CLEAN/"private/PHASE1_ATTACK_SELECTION.csv.gz"),
              "ia_ground_truth_present": ia_gt.exists(), "updated_utc": now()}
    atomic_json(ROOT / "PREFLIGHT_RESULT.json", result)
    checkpoint(result["verdict"], available=result["available_families"], unavailable=result["unavailable_families"])
    return result, selected


def strict_threshold(values, fpr):
    x = np.sort(np.asarray(values, float))[::-1]
    allowed = int(math.floor(fpr * len(x) + 1e-12))
    tau = float(x[allowed])
    return tau, int(np.sum(np.asarray(values, float) > tau)) / len(x)


def token_f1(a, b):
    x, y = TOKEN_RE.findall(str(a).casefold()), TOKEN_RE.findall(str(b).casefold())
    if not x and not y: return 1.0
    if not x or not y: return 0.0
    c = sum((Counter(x) & Counter(y)).values())
    return 2*c/(len(x)+len(y))


def precommit(preflight_result):
    parent = json.loads((BC / "configs/BC_CGD_PRECOMMIT.json").read_text())
    payload = {"campaign": "BC_MIRABEL_SIMPLE_HIDE_CORE6_DIAGNOSTIC",
               "status": "PRECOMMITTED_READ_ONLY_PARTIAL_DIAGNOSTIC",
               "request": {"path": str(REQUEST), "sha256": sha256_file(REQUEST)},
               "clean_precommit": {"path": str(CLEAN/"configs/PRECOMMIT.json"), "sha256": sha256_file(CLEAN/"configs/PRECOMMIT.json")},
               "bc_parent_precommit": {"path": str(BC/"configs/BC_CGD_PRECOMMIT.json"), "sha256": sha256_file(BC/"configs/BC_CGD_PRECOMMIT.json")},
               "detector": parent["detector"], "hide_action": "alarm -> MIRABEL retrieval rank1 removed -> one frozen ordinary Qwen generation",
               "conditions": ["NO_DEFENSE", "ORIGINAL_MIRABEL", "BC_MIRABEL"],
               "valid_families": preflight_result["available_families"],
               "fail_closed_families": preflight_result["unavailable_families"],
               "fpr_points": [0.01, 0.03, 0.05],
               "forbidden": ["new detector", "threshold tuning", "new generation for invalid family", "legacy result join"],
               "code_sha256": sha256_file(Path(__file__)), "new_generation": 0, "new_model_forward": 0}
    path = ROOT / "configs/BC_MIRABEL_CORE6_DIAGNOSTIC_PRECOMMIT.json"
    atomic_json(path, payload)
    digest = sha256_file(path)
    atomic_text(ROOT / "configs/BC_MIRABEL_CORE6_DIAGNOSTIC_PRECOMMIT.sha256", digest + "\n")
    checkpoint("PRECOMMIT_WRITTEN", sha256=digest)
    return payload


def produce_report(preflight_result, selected):
    # Verify precommit before reading numerical result tables.
    pp = ROOT / "configs/BC_MIRABEL_CORE6_DIAGNOSTIC_PRECOMMIT.json"
    if sha256_file(pp) != (ROOT/"configs/BC_MIRABEL_CORE6_DIAGNOSTIC_PRECOMMIT.sha256").read_text().strip():
        raise RuntimeError("precommit drift")
    responses = pd.read_csv(BC / "private/SMALL_FINAL_RESPONSES.csv.gz", keep_default_na=False, dtype=str)
    responses = responses[responses.condition.isin(["NO_DEFENSE", "ORIGINAL_MIRABEL", "BC_MIRABEL"])].copy()
    responses["alarm"] = responses.alarm.str.lower().eq("true")
    responses["member"] = responses.member.astype(int)
    responses["target_rank"] = responses.target_rank.astype(int)
    privacy = pd.read_csv(BC / "tables/SMALL_PRIVACY_METRICS.csv")
    privacy = privacy[privacy.condition.isin(["NO_DEFENSE", "ORIGINAL_MIRABEL", "BC_MIRABEL"])].copy()
    privacy["metric_provenance"] = privacy.family.map({"MEntA":"frozen MEntA entailment/abstention Q5 scorer",
                                                        "S²-MIA":"frozen S² semantic scorer",
                                                        "MBA":"frozen exact mask-reconstruction scorer"})
    atomic_csv(privacy, ROOT / "tables/CORE6_POST_GENERATION_PRIVACY_PARTIAL.csv")

    scores = pd.read_csv(CLEAN / "private/PHASE1_QUERY_SCORES.csv.gz", keep_default_na=False, dtype=str)
    mir = pd.read_csv(WORK/"RAG_MIA_DEFENSE_CLEAN_20260911/experiments/DV_LOO_V2_PHASE1/private/MIRABEL_GUMBEL_MARGIN.csv.gz",
                      keep_default_na=False, dtype=str)
    benign = pd.read_csv(CLEAN/"private/PHASE1_BENIGN_SELECTION.csv.gz", keep_default_na=False, dtype=str)
    margins = scores[["case_id","domain"]].merge(mir[["case_id","mirabel_gumbel_margin"]],on="case_id").merge(benign[["case_id","split"]],on="case_id")
    margins["mirabel_gumbel_margin"] = margins.mirabel_gumbel_margin.astype(float)
    cal, hold = margins[margins.split.eq("CALIBRATION")], margins[margins.split.eq("HOLDOUT")]
    fpr_rows=[]
    thresholds={"ORIGINAL":0.0}
    for fpr,name in ((.01,"q99"),(.03,"q97"),(.05,"q95")):
        tau,_=strict_threshold(cal.mirabel_gumbel_margin,fpr);thresholds[name]=tau
    for name,tau in thresholds.items():
        for domain,cell in [("ALL",hold),*list(hold.groupby("domain"))]:
            fpr_rows.append({"operating_point":name,"threshold":tau,"domain":domain,"n":len(cell),
                             "actual_holdout_fpr":float((cell.mirabel_gumbel_margin>tau).mean()),
                             "false_positives":int((cell.mirabel_gumbel_margin>tau).sum())})
    fpr_table=pd.DataFrame(fpr_rows);atomic_csv(fpr_table,ROOT/"tables/BENIGN_FPR.csv")

    # Frozen detector/locator decomposition for valid families.
    detail=[]
    for family in ("MEntA","S²-MIA","MBA"):
        cell=responses[(responses.condition.eq("BC_MIRABEL"))&(responses.family.eq(family))]
        for scope,part in (("ALL",cell),("MEMBER",cell[cell.member.eq(1)]),("NONMEMBER",cell[cell.member.eq(0)])):
            hit=part.alarm & part.target_rank.gt(0) & part.target_rank.eq(1)
            detail.append({"family":family,"scope":scope,"queries":len(part),
                           "target_retrieval_at4":float(part.target_rank.gt(0).mean()),
                           "alarm_rate":float(part.alarm.mean()),
                           "locator_hit_given_alarm_and_retrieved":float(part[part.alarm & part.target_rank.gt(0)].target_rank.eq(1).mean()) if (part.alarm & part.target_rank.gt(0)).any() else math.nan,
                           "effective_protection_opportunity":float(hit.mean()),
                           "session_any_alarm":float(part.groupby("session_id").alarm.max().mean())})
    detection=pd.DataFrame(detail);atomic_csv(detection,ROOT/"tables/CORE6_DETECTION_PARTIAL.csv")

    # Full false-positive case audit for q97 holdout.
    fp = responses[(responses.condition.eq("BC_MIRABEL")) & responses.kind.eq("BENIGN") & responses.alarm].copy()
    mm=mir[["case_id","mirabel_gumbel_margin","s_M"]].rename(columns={"s_M":"removed_doc_id"})
    fp=fp.merge(mm,on="case_id",validate="one_to_one")
    fp["answer_changed"] = fp.response != fp.A0
    fp["preservation_token_f1"]=[token_f1(a,b) for a,b in zip(fp.response,fp.A0)]
    fp["length_change"] = fp.response.str.len()-fp.A0.str.len()
    atomic_csv(fp[["case_id","domain","query","mirabel_gumbel_margin","removed_doc_id","A0","response","answer_changed","refusal","preservation_token_f1","length_change"]],ROOT/"audits/BC_FALSE_POSITIVE_CASES.csv")

    # No scoring is permitted for invalid membership populations.
    statuses=pd.read_csv(ROOT/"audits/FAMILY_PROTOCOL_STATUS.csv")
    post=[]
    for family in FAMILIES:
        if family in {"MEntA","S²-MIA","MBA"}:
            p=privacy[(privacy.family.eq(family))&(privacy.condition.eq("BC_MIRABEL"))].iloc[0]
            d=detection[(detection.family.eq(family))&(detection.scope.eq("MEMBER"))].iloc[0]
            if p.effective_auc<=.65: cls="ALREADY_CONTROLLED"
            elif d.target_retrieval_at4<.8: cls="RETRIEVAL_LIMITED"
            elif d.alarm_rate<.8: cls="DETECTION_LIMITED"
            elif d.locator_hit_given_alarm_and_retrieved<.8: cls="LOCATOR_LIMITED"
            else: cls="PROTECTION_LIMITED"
            post.append({"family":family,"status":"VALID_FROZEN_RESULT","native_auc":p.native_auc,
                         "effective_auc_diagnostic":p.effective_auc,"member_retrieval_at4":d.target_retrieval_at4,
                         "member_alarm_rate":d.alarm_rate,"member_effective_opportunity":d.effective_protection_opportunity,
                         "classification":cls})
        else:
            post.append({"family":family,"status":"PROTOCOL_OR_LINEAGE_UNAVAILABLE","classification":"UNAVAILABLE_MEMBERSHIP_SUBSTRATE"})
    bottleneck=pd.DataFrame(post);atomic_csv(bottleneck,ROOT/"tables/BOTTLENECK_MAP.csv")
    result={"verdict":"PARTIAL_DIAGNOSTIC_ONLY_NEW_CORE6_SUBSTRATE_REQUIRED",
            "available_families":["MEntA","S²-MIA","MBA"],"unavailable_families":["DCMI","IA","RAG-MIA"],
            "new_generations":0,"new_model_forwards":0,"full_core6_completed":False,
            "reason":"Frozen CLEAN DB realizes membership labels only for MEntA/S²-MIA/MBA.",
            "next_direction":"Build one new precommitted CLEAN Core6 protected corpus before retrieval/generation; do not mix legacy answers.",
            "precommit_sha256":sha256_file(pp)}
    atomic_json(ROOT/"FINAL_RESULT.json",result)
    lines=["# BC-MIRABEL Simple-Hide Core6 진단", "", f"- Verdict: **{result['verdict']}**", "",
           "## 핵심 preflight 결과", "", statuses.to_markdown(index=False), "",
           "동결 CLEAN DB에서 DCMI/IA/RAG-MIA의 member/nonmember label이 실제 문서 포함 여부와 일치하지 않아 세 공격은 fail-close했다. 이 상태에서 생성하면 membership 공격이 아니라 잘못된 label을 평가하게 된다.", "",
           "## 유효한 동결 결과", "", bottleneck.to_markdown(index=False,floatfmt=".4f"), "",
           "## Benign FPR", "", fpr_table.to_markdown(index=False,floatfmt=".4f"), "",
           "## 결론", "", "새 방어를 만들기 전에 6개 공격의 member target은 포함하고 nonmember target은 제외한 새로운 CLEAN Core6 corpus를 결과 확인 전에 한 번 동결해야 한다. 이후 동일 BC-MIRABEL simple-hide로 여섯 공격을 평가해야 방향을 정할 수 있다."]
    atomic_text(ROOT/"reports/FINAL_DIAGNOSTIC_REPORT_KO.md","\n".join(lines)+"\n")
    checkpoint("PARTIAL_DIAGNOSTIC_COMPLETE", verdict=result["verdict"], available=3, unavailable=3)
    return result


def main():
    result, selected=preflight()
    precommit(result)
    produce_report(result,selected)


if __name__ == "__main__": main()
