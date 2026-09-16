#!/usr/bin/env python3
"""Assemble already-verified Final V2 results into paper assets (CPU-only)."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("/home/traffic_3/workspace/workspace/SH/RAG_MIA_DEFENSE_CLEAN_20260911")
EXP = ROOT / "experiments" / "FINAL_V2_REMAINING_VALIDATION_AND_PAPER_FREEZE_V1"
FREEZE = ROOT / "experiments" / "FINAL_LC_V2_FINAL_FREEZE_V1"
VIS = ROOT / "experiments" / "FINAL_V2_PAPER_VISUAL_AND_QUALITATIVE_SIDECAR_V1"
FINQA = ROOT / "experiments" / "FINAL_V2_FINQA_VALIDATION_ADDENDUM_20260916"
GEOM = ROOT / "experiments" / "LOCAL_RETRIEVAL_GEOMETRY_FPR_MATCHED_V1"
S1 = ROOT / "experiments" / "S1_G_CONTRIBUTION_AUDIT_V1"
FUSION = ROOT / "experiments" / "S1_G4_FUSION_DIAG_CPU_V1"
NONMEMBER = ROOT / "experiments" / "NONMEMBER_INTERVENTION_COST_CPU_V1"
ROBUST = ROOT / "experiments" / "FINAL_V2_ROBUST_THRESHOLD_CONTAMINATION_V1"


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(value)
    tmp.replace(path)


def atomic_json(path: Path, value) -> None:
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def write_bundle(name: str, frame: pd.DataFrame) -> None:
    table = EXP / "tables" / f"{name}.csv"
    frame.to_csv(table, index=False)
    atomic_text(EXP / "paper" / f"{name}.md", frame.to_markdown(index=False) + "\n")
    latex = frame.to_latex(index=False, escape=True, float_format=lambda x: f"{x:.4f}")
    atomic_text(EXP / "paper" / f"{name}.tex", latex)


def copy_table(name: str, source: Path) -> pd.DataFrame:
    frame = pd.read_csv(source)
    write_bundle(name, frame)
    return frame


def collect_ablation() -> pd.DataFrame:
    rows = []
    fpr = pd.read_csv(GEOM / "tables" / "FPR_MATCHED_TPR_ALL.csv")
    for _, row in fpr.iterrows():
        for attack in ("MEntA", "MBA", "RAG-MIA", "S2-MIA", "DCMI-Std-Q2", "Core5_macro", "IA"):
            column = attack + "_TPR"
            if column in row:
                rows.append({"section": "observation_range", "variant": row["method"], "environment": "Core",
                             "requested_fpr": row["requested_fpr"], "actual_fpr": row["actual_fpr"],
                             "attack_or_metric": attack, "value": row[column], "source": str(GEOM / "tables" / "FPR_MATCHED_TPR_ALL.csv")})
    contrib = pd.read_csv(S1 / "tables" / "S1_G_CONTRIBUTION_METRICS.csv")
    contrib["method"] = contrib["candidate"]; contrib["cohort"] = "Core"; contrib["metric"] = "Core5_macro_TPR"; contrib["value"] = contrib["Core5_macro_TPR"]
    for _, row in contrib.iterrows():
        rows.append({"section": "statistic", "variant": row.get("method"), "environment": row.get("cohort", "Core"),
                     "requested_fpr": row.get("requested_fpr"), "actual_fpr": row.get("actual_fpr"),
                     "attack_or_metric": row.get("metric", row.get("attack", "reported")),
                     "value": row.get("value", row.get("tpr", row.get("auc"))), "source": str(S1 / "tables" / "S1_G_CONTRIBUTION_METRICS.csv")})
    fusion = pd.read_csv(FUSION / "tables" / "FUSION_TPR_FPR025.csv")
    fusion["detector"] = fusion["method"]; fusion["environment"] = fusion["setting"]; fusion["actual_fpr"] = fusion["benign_eval_actual_fpr"]; fusion["tpr"] = fusion["attack_query_tpr"]
    for _, row in fusion.iterrows():
        rows.append({"section": "fusion", "variant": row.get("detector", row.get("method")),
                     "environment": row.get("environment"), "requested_fpr": 0.025, "actual_fpr": row.get("actual_fpr"),
                     "attack_or_metric": row.get("attack", "reported"), "value": row.get("tpr"),
                     "source": str(FUSION / "tables" / "FUSION_TPR_FPR025.csv")})
    contamination = pd.read_csv(ROBUST / "tables" / "TABLE_ROBUST_THRESHOLD_CONTAMINATION.csv")
    contamination["environment"] = contamination["setting"]; contamination["contamination_fraction_total"] = contamination["contamination_fraction_requested"]; contamination["member_tpr_macro"] = contamination["pooled_member_tpr"]
    for _, row in contamination.iterrows():
        rows.append({"section": "calibration_contamination", "variant": f"{row['detector']}::{row['estimator']}",
                     "environment": row["environment"], "requested_fpr": 0.025,
                     "actual_fpr": row["locked_benign_fpr"], "attack_or_metric": f"contamination={row['contamination_fraction_total']}",
                     "value": row["member_tpr_macro"], "source": str(ROBUST / "tables" / "TABLE_ROBUST_THRESHOLD_CONTAMINATION.csv")})
    decomposition = pd.read_csv(NONMEMBER / "tables" / "MATCHED_MEMBER_NONMEMBER_TPR.csv")
    decomposition["detector"] = decomposition["method"]; decomposition["environment"] = decomposition["setting"]
    for _, row in decomposition.iterrows():
        for metric in ("member_tpr", "nonmember_tpr"):
            if metric in row:
                rows.append({"section": "member_nonmember", "variant": row.get("detector"),
                             "environment": row.get("environment"), "requested_fpr": 0.025,
                             "actual_fpr": row.get("actual_fpr"), "attack_or_metric": f"{row.get('attack')}::{metric}",
                             "value": row[metric], "source": str(NONMEMBER / "tables" / "MATCHED_MEMBER_NONMEMBER_TPR.csv")})
    frame = pd.DataFrame(rows)
    return frame


def method_figure() -> None:
    plt.rcParams.update({"font.size": 10, "font.family": "DejaVu Sans"})
    fig, ax = plt.subplots(figsize=(11, 4.3))
    ax.axis("off")
    boxes = [(0.03, "Query"), (0.19, "Top-4\nretrieval"), (0.38, "G4 = s1 − mean(s1…s4)\nbenign-only 97.5th percentile"),
             (0.68, "Alarm?"), (0.82, "Hide rank-1\nsource"), (0.94, "Generate\nonce")]
    widths = [0.10, 0.13, 0.25, 0.10, 0.13, 0.10]
    for (x, label), width in zip(boxes, widths):
        ax.add_patch(plt.Rectangle((x, .42), width, .25, facecolor="#E8F1FA", edgecolor="#24557A", linewidth=1.5))
        ax.text(x + width / 2, .545, label, ha="center", va="center", weight="bold" if "G4" in label else None)
    for index in range(len(boxes) - 1):
        x0 = boxes[index][0] + widths[index]
        x1 = boxes[index + 1][0]
        ax.annotate("", xy=(x1, .545), xytext=(x0, .545), arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "#444"})
    ax.text(.38, .22, "MIRABEL: corpus-wide extreme-value statistic", color="#B54B4B", ha="center")
    ax.text(.38, .12, "Final V2: local Top-4 concentration; no attack labels or classifier", color="#24557A", ha="center")
    ax.set_xlim(0, 1.06); ax.set_ylim(0, 1)
    fig.tight_layout()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(EXP / "figures" / f"FIG1_METHOD_OVERVIEW.{ext}", dpi=600 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def existing_figures() -> None:
    mapping = {
        "fig3_low_fpr_roc": "FIG2_LOW_FPR_DETECTION",
        "fig5b_qwen_core5_e2e_privacy": "FIG4_E2E_PRIVACY",
        "fig7_qwen_llama_privacy_transfer": "FIG5_GENERATOR_TRANSFER",
        "fig8_retriever_transfer": "FIG5_RETRIEVER_TRANSFER",
        "fig9_gold_utility_fp_damage": "FIG6_BENIGN_UTILITY",
    }
    for old, new in mapping.items():
        for ext in ("png", "pdf"):
            source = VIS / "figures" / f"{old}.{ext}"
            if source.is_file():
                shutil.copy2(source, EXP / "figures" / f"{new}.{ext}")
    for old, new in (("FIG_MEMBER_NONMEMBER_DECOMPOSITION", "FIG3_MEMBER_NONMEMBER_DECOMPOSITION"),
                     ("FIG_CORE_FINQA_GEOMETRY", "FIG7_FINQA_TRANSFER"),
                     ("FIG_CALIBRATION_CONTAMINATION", "FIG10_CALIBRATION_LIMITATION")):
        source_root = FINQA if "CONTAMINATION" not in old else FINQA
        for ext in ("png", "pdf"):
            source = source_root / "figures" / f"{old}.{ext}"
            if source.is_file():
                shutil.copy2(source, EXP / "figures" / f"{new}.{ext}")
    for old, new in (("FIG_ROBUST_THRESHOLD_LOCKED_FPR", "FIG10_ROBUST_CALIBRATION_FPR"),
                     ("FIG_ROBUST_THRESHOLD_MEMBER_TPR", "FIG10_ROBUST_CALIBRATION_TPR")):
        for ext in ("png", "pdf"):
            source = ROBUST / "figures" / f"{old}.{ext}"
            if source.is_file():
                shutil.copy2(source, EXP / "figures" / f"{new}.{ext}")


def claim_audit() -> None:
    rows = [
        ("V2 universally outperforms MIRABEL.", "UNSUPPORTED", "FinQA DCMI/MBA/RAG-MIA cells do not uniformly beat MIRABEL."),
        ("V2 improves average Core5 low-FPR detection.", "SUPPORTED", "Frozen Core5 matched-FPR table."),
        ("V2 reduces E2E membership leakage across Core5.", "SUPPORTED", "Frozen Qwen Core5 native scorer table."),
        ("V2 transfers to FinQA without adaptation.", "UNSUPPORTED", "Numeric zero-shot TPR degrades."),
        ("V2 transfers with benign-only recalibration.", "SUPPORTED", "FinQA final verdict and Core5 E2E table."),
        ("Numeric threshold is domain invariant.", "UNSUPPORTED", "Core and FinQA calibrated thresholds/scales differ."),
        ("V2 is retriever agnostic.", "PARTIAL", "BGE/GTE/MPNet measured, with weaker absolute MPNet performance."),
        ("V2 is generator independent.", "PARTIAL", "Qwen and one Llama transfer only."),
        ("Missed low-G queries are naturally non-leaky.", "UNSUPPORTED", "Only 1/10 environment-attack correlations supported it."),
        ("V2 is hallucination-free.", "UNSUPPORTED", "NLI is proxy; false-positive subset has measurable damage."),
        ("V2 has low overall benign damage.", "SUPPORTED", "Low aggregate intervention, with FP-subset caveat."),
        ("False-positive users can suffer substantial damage.", "SUPPORTED", "Frozen conditional FP utility audit."),
        ("Top-k concentration is a novel mathematical statistic.", "UNSUPPORTED", "The statistic family predates this application."),
        ("Local retrieval concentration for low-FPR RAG-MIA detect-and-hide was not found in audited prior work.", "PARTIAL", "Use only with explicit related-work search boundary."),
    ]
    lines = ["# Final claim audit", "", "| Claim | Status | Evidence boundary |", "|---|---|---|"]
    for claim, status, evidence in rows:
        lines.append(f"| {claim} | **{status}** | {evidence} |")
    atomic_text(EXP / "paper" / "FINAL_CLAIM_AUDIT.md", "\n".join(lines) + "\n")


def paper_docs() -> None:
    limitations = """# Final limitations

- IA-Std-Q15-API1 remains the hardest standardized stress test and is not paper-exact IA.
- Hard-benign false-positive risk requires the separately precommitted constructed benchmark.
- Calibration is sensitive to contamination; MAD is a robustness diagnostic, not a universally fixed numeric constant.
- The numeric threshold is not domain invariant; FinQA requires benign-only recalibration.
- FinQA member TPR is materially below Core member TPR.
- MPNet transfer has weaker absolute performance.
- Rank-1 hide can leave a target at ranks 2–4 under decoy/rank shift.
- No formal differential-privacy guarantee is provided.
- No universal MIA guarantee is supported.
- NLI factuality scores are proxies and do not establish hallucination-free behavior.
"""
    atomic_text(EXP / "paper" / "FINAL_LIMITATIONS.md", limitations)
    atomic_text(EXP / "paper" / "FINAL_METHOD_SPEC.md", """# Final V2 / G4 / LRC-4 method

For Top-4 normalized retrieval similarities \(s_1\ge s_2\ge s_3\ge s_4\),

\[G_4(q)=s_1-\frac{s_1+s_2+s_3+s_4}{4}.\]

The alarm rule is strict `G4 > tau`, where `tau` is estimated only from benign calibration queries at the precommitted 2.5% operating budget. On alarm, the current-turn rank-1 source is removed, remaining context is deterministically backfilled, and the frozen generator runs once. No attack label, classifier, family-specific threshold, session state, or attack-specific route is used.
""")
    atomic_text(EXP / "paper" / "FINAL_RELATED_WORK_POSITIONING.md", """# Related-work positioning

The defensible contribution is not a new concentration statistic. It is the use of local Top-4 retrieval concentration as a benign-calibrated, low-FPR signal coupled to a simple detect-and-hide RAG membership-privacy action, evaluated with native attack scorers and end-to-end answer generation. MIRABEL is the corpus-wide extreme-value baseline; V2 removes the full-corpus fit at inference but inherits the practical limitation that false-positive rank-1 removal can damage a legitimate answer.
""")
    atomic_text(EXP / "paper" / "METHOD_NAME_CANDIDATES.md", """# Method name candidates

The frozen mapping remains `Final V2 = G4 = LRC-4`. Candidate submission names, not automatically adopted:

1. Local Retrieval Concentration Guard (LRC-Guard)
2. Benign-Calibrated Local Concentration Defense (BC-LCD)
3. Top-4 Retrieval Concentration Hide (TRC-Hide)
""")
    atomic_text(EXP / "paper" / "FINAL_PAPER_OUTLINE_KO.md", """# 논문 개요

1. 문제: stealth RAG-MIA에서 정상 질의와 공격 질의가 겹쳐 low-FPR detection이 어렵다.
2. 관찰: full-corpus 극값 대신 Top-4 국소 집중도에 강한 member tail이 존재한다.
3. 방법: benign-only 2.5% 보정, G4 경보, rank-1 hide, 1회 생성.
4. 평가: Core5 detection/E2E, Llama, retriever transfer, FinQA benign-recalibrated transfer.
5. 실사용 감사: benign utility, FP subset, hard-benign, adaptive/decoy, contamination.
6. 한계: IA, domain scale shift, rank shift, no DP/universal guarantee.
""")
    atomic_text(EXP / "paper" / "FINAL_PPT_OUTLINE_KO.md", """# 발표 개요

1. 왜 기존 탐지기는 오탐이 문제인가
2. MIRABEL과 Final V2의 차이
3. G4 수식과 benign-only operating point
4. Core5 low-FPR detection
5. 실제 생성 후 개인정보 누출
6. Qwen/Llama 및 retriever 전이
7. FinQA: 숫자 threshold 실패와 정상 데이터 보정 성공
8. 정상 답변 손상과 false-positive subset
9. hard-benign/adaptive stress 결과
10. 주장 가능한 범위와 한계
""")


def main() -> None:
    for directory in ("tables", "paper", "figures", "reports"):
        (EXP / directory).mkdir(parents=True, exist_ok=True)
    table1 = copy_table("TABLE1_CORE5_MATCHED_FPR_DETECTION", FREEZE / "tables" / "core5_detection.csv")
    table2 = copy_table("TABLE2_CORE5_QWEN_E2E_PRIVACY", FREEZE / "tables" / "core5_e2e_privacy.csv")
    table3 = copy_table("TABLE3_LLAMA_GENERATOR_TRANSFER", VIS / "tables" / "fig7_qwen_llama_privacy_transfer.csv")
    table4 = copy_table("TABLE4_RETRIEVER_TRANSFER", FREEZE / "transfer" / "tables" / "retriever_transfer_detection.csv")
    table5 = copy_table("TABLE5_FINQA_UNTOUCHED_DOMAIN_E2E", FINQA / "tables" / "TABLE_FINQA_CORE5_E2E.csv")
    gold = pd.read_csv(FREEZE / "tables" / "gold_qa_utility.csv")
    factual = pd.read_csv(FREEZE / "audits" / "FACTUALITY_SUMMARY.csv")
    gold.insert(0, "source_section", "gold_qa")
    factual.insert(0, "source_section", "factuality_proxy")
    table6 = pd.concat([gold, factual], ignore_index=True, sort=False)
    write_bundle("TABLE6_BENIGN_UTILITY_FACTUALITY", table6)
    ablation = collect_ablation()
    write_bundle("TABLE9_FINAL_ABLATION", ablation)
    geometry = pd.read_csv(FINQA / "tables" / "TABLE_CORE_FINQA_GEOMETRY.csv")
    write_bundle("TABLE_CORPUS_GEOMETRY", geometry)
    method_figure()
    existing_figures()
    claim_audit()
    paper_docs()
    source_files = [FREEZE / "tables" / "core5_detection.csv", FREEZE / "tables" / "core5_e2e_privacy.csv",
                    VIS / "tables" / "fig7_qwen_llama_privacy_transfer.csv",
                    FREEZE / "transfer" / "tables" / "retriever_transfer_detection.csv",
                    FINQA / "tables" / "TABLE_FINQA_CORE5_E2E.csv", GEOM / "tables" / "FPR_MATCHED_TPR_ALL.csv",
                    ROBUST / "tables" / "TABLE_ROBUST_THRESHOLD_CONTAMINATION.csv"]
    atomic_json(EXP / "audits" / "EXISTING_RESULT_ASSEMBLY_LINEAGE.json",
                {"completed_utc": utc(), "sources": [{"path": str(path), "sha256": sha(path)} for path in source_files],
                 "recomputed_model_scores": False, "method_modified": False, "hard_benign_pending": True,
                 "adaptive_pending": True, "document_document_geometry": "MISSING_FINQA_DOCUMENT_EMBEDDING_CACHE"})
    status = json.loads((EXP / "STATUS.json").read_text())
    status.update({"cpu_existing_result_assembly": "COMPLETE", "updated_utc": utc(),
                   "stage": "CPU_EXISTING_RESULTS_ASSEMBLED_GPU_VALIDATION_PENDING"})
    atomic_json(EXP / "STATUS.json", status)
    print(json.dumps({"status": status["stage"], "tables": 7, "figures": len(list((EXP / 'figures').glob('*')))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
