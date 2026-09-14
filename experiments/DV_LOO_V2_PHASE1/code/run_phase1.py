#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from common import (BGE, DOMAINS, FAMILIES, PARENT, RHO, ROOT, SEED, TARGET_FPRS,
                    atomic_csv, atomic_json, atomic_text, checkpoint, empirical_cdf,
                    mirabel_margin, sha256_file, strict_threshold)


EXPECTED = {
    "parent_precommit": (PARENT / "configs/PRECOMMIT.json", "479fa2db024ca869ed883e0ee791647340c550fbc67147db671ef9c891d7a016"),
    "query_scores": (PARENT / "private/PHASE1_QUERY_SCORES.csv.gz", "5b843f73ab392b84f16f92f5886b81b23309533c3de909ea7b222903a60bd540"),
    "retrieval": (PARENT / "private/PHASE1_RETRIEVAL.csv.gz", "787302b51f45b7162f7938d5e2307f6c2f7914c9bb67ef01d0531737661d3b4d"),
    "benign_selection": (PARENT / "private/PHASE1_BENIGN_SELECTION.csv.gz", "c7524bcfe6756c619ae1f4cd245e60ac411b7a97b99e4961c68c8c4d86462310"),
    "attack_selection": (PARENT / "private/PHASE1_ATTACK_SELECTION.csv.gz", "6533d76513b5913f864d5e720bb8b7bb035dd9424fb1fdc20f7b8adb7424985c"),
    "corpus_nfcorpus": (PARENT / "private/CORPUS_BeIR_nfcorpus.csv.gz", "cbae4b8a9d3835cc71d69492967028e1dd5e7498cd820ad0689ada24693fbccd"),
    "corpus_scidocs": (PARENT / "private/CORPUS_BeIR_scidocs.csv.gz", "2188bcb095cdff213d269a8dea2eea9ace544f8c3fa704dc073c270e1ee3cd5e"),
    "corpus_trec": (PARENT / "private/CORPUS_BeIR_trec-covid.csv.gz", "597dd005310ca5bc5a17402526ee2b09d9ee4ea72b9a3401123040172efd8e89"),
    "a0_db": (PARENT / "private/A0_GENERATIONS.sqlite3", "9e4835bc7fa6c347cc246e2e528bbd68d0c5ce61c490c5118d280cb19327c9dd"),
    "loo_db": (PARENT / "private/LOO_SCORES.sqlite3", "a7a2077012b9f23f33492ce00d1c1502ea1cf5b3637f0f1c20410951459b5105"),
    "qll_db": (PARENT / "private/QLL_SCORES.sqlite3", "04a64bba7921fdd43bd45023dfb242da051753202d433213e4ad611fb99f53bd"),
}


def verify() -> tuple[pd.DataFrame, pd.DataFrame]:
    precommit = ROOT / "configs/PHASE1_PRECOMMIT.json"
    if not precommit.exists():
        raise RuntimeError("missing phase1 precommit")
    expected_precommit = (ROOT / "configs/PHASE1_PRECOMMIT.sha256").read_text().split()[0]
    if sha256_file(precommit) != expected_precommit:
        raise RuntimeError("phase1 precommit hash mismatch")
    locked = json.loads(precommit.read_text(encoding="utf-8"))
    for relative, digest in locked["code_sha256"].items():
        if sha256_file(ROOT / relative) != digest:
            raise RuntimeError(f"phase1 code drift: {relative}")
    rows = []
    for name, (path, digest) in EXPECTED.items():
        actual = sha256_file(path)
        rows.append({"name": name, "path": str(path), "expected_sha256": digest,
                     "actual_sha256": actual, "match": actual == digest})
    audit = pd.DataFrame(rows)
    atomic_csv(audit, ROOT / "audits/FROZEN_INPUT_VERIFICATION.csv")
    if not audit.match.all():
        raise RuntimeError("frozen parent input drift")
    scores = pd.read_csv(EXPECTED["query_scores"][0], keep_default_na=False,
                         dtype={"case_id": str, "session_id": str, "target_id": str})
    retrieval = pd.read_csv(EXPECTED["retrieval"][0], keep_default_na=False,
                            dtype={"case_id": str, "session_id": str, "target_id": str})
    benign = pd.read_csv(EXPECTED["benign_selection"][0], keep_default_na=False,
                         dtype={"case_id": str})[["case_id", "split"]]
    if len(scores) != 780 or len(retrieval) != 780 or scores.case_id.nunique() != 780:
        raise RuntimeError("parent cohort count drift")
    if set(scores.case_id) != set(retrieval.case_id):
        raise RuntimeError("score/retrieval ID mismatch")
    scores = scores.drop(columns=["split"], errors="ignore").merge(benign, on="case_id", how="left", validate="one_to_one")
    scores.loc[scores.kind.eq("ATTACK"), "split"] = "ATTACK"
    query = retrieval[["case_id", "query", "retrieval_scores", "retrieval_top1_source"]]
    scores = scores.merge(query, on="case_id", validate="one_to_one")
    checkpoint("PHASE1_PREFLIGHT_PASS", queries=len(scores), benign=int(scores.kind.eq("BENIGN").sum()),
               attack=int(scores.kind.eq("ATTACK").sum()), parent_hashes=len(audit))
    return scores, retrieval


def reconstruct_mirabel(scores: pd.DataFrame) -> pd.DataFrame:
    output = ROOT / "private/MIRABEL_GUMBEL_MARGIN.csv.gz"
    if output.exists():
        cached = pd.read_csv(output, keep_default_na=False, dtype={"case_id": str})
        if len(cached) == len(scores) and set(cached.case_id) == set(scores.case_id):
            checkpoint("MIRABEL_RECONSTRUCTION_REUSED", rows=len(cached), sha256=sha256_file(output))
            return cached
    import torch
    from sentence_transformers import SentenceTransformer
    checkpoint("MIRABEL_RECONSTRUCTION_STARTED", device="cuda", rho=RHO, queries=len(scores),
               documents_per_domain=1000, full_corpus=True)
    model = SentenceTransformer(str(BGE), device="cuda", local_files_only=True)
    model.max_seq_length = 512
    rows = []
    for domain in DOMAINS:
        corpus = pd.read_csv(PARENT / "private" / f"CORPUS_{domain}.csv.gz", keep_default_na=False,
                             dtype={"document_id": str}).sort_values("corpus_order")
        subset = scores[scores.domain.eq(domain)].sort_values("case_id").reset_index(drop=True)
        checkpoint("MIRABEL_DOMAIN_STARTED", domain=domain, documents=len(corpus), queries=len(subset))
        docs = model.encode(corpus.text.astype(str).tolist(), batch_size=32, convert_to_numpy=True,
                            normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        queries = model.encode(subset["query"].astype(str).tolist(), batch_size=32, convert_to_numpy=True,
                               normalize_embeddings=True, show_progress_bar=False).astype(np.float32)
        matrix = queries @ docs.T
        ids = corpus.document_id.astype(str).tolist()
        for index, item in enumerate(subset.itertuples(index=False)):
            top, smax, mu, sigma, tau, margin = mirabel_margin(matrix[index], RHO)
            stored_scores = json.loads(item.retrieval_scores)
            rows.append({"case_id": str(item.case_id), "domain": domain, "s_M": ids[top],
                         "s_max": smax, "mu_q": mu, "sigma_q": sigma, "tau_q": tau,
                         "mirabel_gumbel_margin": margin, "original_mirabel_alarm": bool(margin > 0),
                         "stored_top1_source": str(item.retrieval_top1_source),
                         "stored_top1_similarity": float(stored_scores[0]),
                         "top1_match": ids[top] == str(item.retrieval_top1_source),
                         "top1_similarity_abs_error": abs(smax-float(stored_scores[0]))})
        checkpoint("MIRABEL_DOMAIN_COMPLETE", domain=domain, queries=len(subset))
        del docs, queries, matrix
        gc.collect(); torch.cuda.empty_cache()
    del model; gc.collect(); torch.cuda.empty_cache()
    frame = pd.DataFrame(rows).sort_values("case_id").reset_index(drop=True)
    atomic_csv(frame, output, "gzip")
    mismatch = int((~frame.top1_match).sum())
    maximum_error = float(frame.top1_similarity_abs_error.max())
    audit = {"queries": len(frame), "top1_matches": int(frame.top1_match.sum()),
             "top1_mismatches": mismatch, "top1_match_rate": float(frame.top1_match.mean()),
             "max_top1_similarity_abs_error": maximum_error, "tolerance": 5e-5,
             "rho": RHO, "full_corpus": True, "score": "s_max-tau_q"}
    atomic_json(ROOT / "audits/MIRABEL_RECONSTRUCTION_AUDIT.json", audit)
    if mismatch or maximum_error > 5e-5:
        checkpoint("PHASE1_MIRABEL_RECONSTRUCTION_FAILED", **audit)
        raise RuntimeError("Mirabel reconstruction mismatch")
    checkpoint("MIRABEL_RECONSTRUCTION_PASS", **audit, sha256=sha256_file(output))
    return frame


def bootstrap_rate(values, iterations=10000):
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(SEED)
    means = np.asarray([rng.choice(array, len(array), replace=True).mean() for _ in range(iterations)])
    return float(np.quantile(means, .025)), float(np.quantile(means, .975))


def bootstrap_auc(positive, negative, iterations=5000):
    pos, neg = np.asarray(positive, float), np.asarray(negative, float)
    point = float(roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[pos, neg]))
    rng = np.random.default_rng(SEED); values=[]
    for _ in range(iterations):
        p=rng.choice(pos,len(pos),replace=True); n=rng.choice(neg,len(neg),replace=True)
        values.append(roc_auc_score(np.r_[np.ones(len(p)),np.zeros(len(n))],np.r_[p,n]))
    return point, float(np.quantile(values,.025)), float(np.quantile(values,.975))


def add_risks(scores: pd.DataFrame, mirabel: pd.DataFrame) -> pd.DataFrame:
    frame = scores.merge(mirabel[["case_id", "s_M", "mirabel_gumbel_margin", "original_mirabel_alarm"]],
                         on="case_id", validate="one_to_one")
    if not frame.s_M.eq(frame.mirabel_source_id.astype(str)).all():
        raise RuntimeError("new Mirabel locator differs from parent MIRABEL-selected LOO branch")
    calibration = frame[frame.split.eq("CALIBRATION")]
    frame["U_DIR"] = empirical_cdf(calibration.mirabel_gumbel_margin, frame.mirabel_gumbel_margin)
    frame["U_M"] = empirical_cdf(calibration.risk_mirabel, frame.risk_mirabel)
    frame["U_Q"] = empirical_cdf(calibration.risk_qll, frame.risk_qll)
    frame["U_LOO"] = frame[["U_M", "U_Q"]].max(axis=1)
    frame["DV_LOO_V2"] = frame[["U_DIR", "U_LOO"]].max(axis=1)
    atomic_csv(frame, ROOT / "private/PHASE1_DERIVED_SCORES.csv.gz", "gzip")
    return frame


def session_scores(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    attack = frame[frame.kind.eq("ATTACK")]
    return attack.groupby(["family", "member", "session_id"], as_index=False)[column].max()


def analyze(frame: pd.DataFrame) -> dict:
    conditions = {"MIRABEL_MARGIN_PERCENTILE": "mirabel_gumbel_margin",
                  "CLEAN_V1_RAW_LOO": "risk_union2", "DV_LOO_V2": "DV_LOO_V2"}
    calibration = frame[frame.split.eq("CALIBRATION")]
    holdout = frame[frame.split.eq("HOLDOUT")]
    threshold_rows=[]; fpr_rows=[]; tpr_rows=[]; auc_rows=[]
    thresholds={}
    for condition,column in conditions.items():
        for target in TARGET_FPRS:
            tau, fp, actual = strict_threshold(calibration[column], target)
            thresholds[(condition,target)] = tau
            threshold_rows.append({"condition":condition,"target_fpr":target,"threshold":tau,
                                   "calibration_n":len(calibration),"calibration_fp":fp,"calibration_fpr":actual})
            for domain in ("ALL",)+DOMAINS:
                cell=holdout if domain=="ALL" else holdout[holdout.domain.eq(domain)]
                count=int((cell[column]>tau).sum())
                fpr_rows.append({"condition":condition,"target_fpr":target,"domain":domain,
                                 "holdout_n":len(cell),"false_positives":count,"actual_fpr":count/len(cell)})
            sessions=session_scores(frame,column)
            for family in FAMILIES:
                for population,member in (("MEMBER",1),("NONMEMBER",0),("ALL",None)):
                    cell=sessions[sessions.family.eq(family)]
                    if member is not None: cell=cell[cell.member.eq(member)]
                    detected=(cell[column]>tau).astype(int)
                    low,high=bootstrap_rate(detected)
                    tpr_rows.append({"condition":condition,"family":family,"population":population,
                                     "target_fpr":target,"threshold":tau,"sessions":len(cell),
                                     "detected":int(detected.sum()),"tpr":float(detected.mean()),
                                     "bootstrap_ci_low":low,"bootstrap_ci_high":high})
        sessions=session_scores(frame,column)
        benign_scores=holdout[column].to_numpy(float)
        for family in FAMILIES:
            attacks=sessions[sessions.family.eq(family)][column].to_numpy(float)
            point,low,high=bootstrap_auc(attacks,benign_scores)
            auc_rows.append({"condition":condition,"family":family,"positive":"ALL_ATTACK_SESSIONS",
                             "negative":"BENIGN_HOLDOUT_QUERIES","roc_auc":point,
                             "bootstrap_ci_low":low,"bootstrap_ci_high":high})
    binary=[]
    for domain in ("ALL",)+DOMAINS:
        cell=holdout if domain=="ALL" else holdout[holdout.domain.eq(domain)]
        binary.append({"condition":"ORIGINAL_MIRABEL_BINARY_RHO_0.05","domain":domain,
                       "holdout_n":len(cell),"alarms":int(cell.original_mirabel_alarm.sum()),
                       "actual_fpr":float(cell.original_mirabel_alarm.mean())})
    original_session=frame[frame.kind.eq("ATTACK")].groupby(
        ["family","member","session_id"],as_index=False).original_mirabel_alarm.max()
    for family in FAMILIES:
        for population,member in (("MEMBER",1),("NONMEMBER",0),("ALL",None)):
            cell=original_session[original_session.family.eq(family)]
            if member is not None:cell=cell[cell.member.eq(member)]
            binary.append({"condition":"ORIGINAL_MIRABEL_BINARY_RHO_0.05","family":family,
                           "population":population,"sessions":len(cell),"alarms":int(cell.original_mirabel_alarm.sum()),
                           "tpr":float(cell.original_mirabel_alarm.mean())})
    thresholds_df=pd.DataFrame(threshold_rows); fpr_df=pd.DataFrame(fpr_rows); tpr_df=pd.DataFrame(tpr_rows)
    atomic_csv(thresholds_df,ROOT/"tables/PHASE1_THRESHOLDS.csv")
    atomic_csv(fpr_df,ROOT/"tables/PHASE1_BENIGN_FPR.csv")
    atomic_csv(tpr_df,ROOT/"tables/PHASE1_ATTACK_TPR.csv")
    atomic_csv(pd.DataFrame(auc_rows),ROOT/"tables/PHASE1_DETECTOR_AUC.csv")
    atomic_csv(pd.DataFrame(binary),ROOT/"tables/ORIGINAL_MIRABEL_BINARY.csv")
    def tpr(condition,family,target=.03):
        row=tpr_df[(tpr_df.condition.eq(condition))&(tpr_df.family.eq(family))&
                   (tpr_df.population.eq("MEMBER"))&np.isclose(tpr_df.target_fpr,target)]
        return float(row.iloc[0].tpr)
    overall=float(fpr_df[(fpr_df.condition.eq("DV_LOO_V2"))&(fpr_df.domain.eq("ALL"))&
                         np.isclose(fpr_df.target_fpr,.03)].iloc[0].actual_fpr)
    dv_menta=tpr("DV_LOO_V2","MEntA"); mir_menta=tpr("MIRABEL_MARGIN_PERCENTILE","MEntA")
    checks={"overall_holdout_fpr3_le_0_04":overall<=.04,
            "menta_tpr3_ge_0_50":dv_menta>=.50,
            "menta_improvement_over_mirabel_ge_0_10":dv_menta-mir_menta>=.10-1e-12,
            "s2_degradation_le_0_05":tpr("DV_LOO_V2","S²-MIA")>=tpr("MIRABEL_MARGIN_PERCENTILE","S²-MIA")-.05-1e-12,
            "mba_degradation_le_0_05":tpr("DV_LOO_V2","MBA")>=tpr("MIRABEL_MARGIN_PERCENTILE","MBA")-.05-1e-12}
    verdict="DV_LOO_V2_PHASE1_PASS" if all(checks.values()) else "DV_LOO_V2_PHASE1_REJECTED"
    result={"campaign":"DV_LOO_V2_PHASE1","verdict":verdict,"development_only":True,
            "primary_population":"MEMBER_ATTACK_SESSIONS","primary_fpr":.03,
            "checks":checks,"metrics":{"holdout_fpr3":overall,"MEntA":{"DV_LOO_V2":dv_menta,
            "MIRABEL_MARGIN_PERCENTILE":mir_menta,"CLEAN_V1_RAW_LOO":tpr("CLEAN_V1_RAW_LOO","MEntA")},
            "S²-MIA":{"DV_LOO_V2":tpr("DV_LOO_V2","S²-MIA"),"MIRABEL_MARGIN_PERCENTILE":tpr("MIRABEL_MARGIN_PERCENTILE","S²-MIA")},
            "MBA":{"DV_LOO_V2":tpr("DV_LOO_V2","MBA"),"MIRABEL_MARGIN_PERCENTILE":tpr("MIRABEL_MARGIN_PERCENTILE","MBA")}},
            "phase2_opened":False,"protected_generation_count":0}
    atomic_json(ROOT/"FINAL_RESULT.json",result)
    lines=["# DV-LOO V2 Phase 1 결과","",f"- Verdict: **{verdict}**",f"- Holdout FPR@3%: **{overall:.4f}**",
           f"- MEntA TPR@3%: DV-LOO **{dv_menta:.3f}**, Mirabel margin **{mir_menta:.3f}**, CLEAN raw LOO **{tpr('CLEAN_V1_RAW_LOO','MEntA'):.3f}**",
           f"- S²-MIA TPR@3%: DV-LOO **{tpr('DV_LOO_V2','S²-MIA'):.3f}**, Mirabel **{tpr('MIRABEL_MARGIN_PERCENTILE','S²-MIA'):.3f}**",
           f"- MBA TPR@3%: DV-LOO **{tpr('DV_LOO_V2','MBA'):.3f}**, Mirabel **{tpr('MIRABEL_MARGIN_PERCENTILE','MBA'):.3f}**","",
           "Phase 1은 detector development screen이며 post-generation privacy 결과가 아니다. Phase 2는 자동으로 열지 않았다."]
    atomic_text(ROOT/"reports/PHASE1_REPORT_KO.md","\n".join(lines)+"\n")
    checkpoint(verdict, final_result_sha256=sha256_file(ROOT/"FINAL_RESULT.json"), phase2_opened=False)
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--device",default="cuda");args=parser.parse_args()
    if args.device!="cuda":raise RuntimeError("precommitted reconstruction device is cuda")
    scores,_=verify();mirabel=reconstruct_mirabel(scores);frame=add_risks(scores,mirabel);analyze(frame)


if __name__ == "__main__":
    main()
