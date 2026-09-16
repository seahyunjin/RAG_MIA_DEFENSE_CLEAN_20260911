#!/usr/bin/env python3
"""Build deterministic, slide-ready tables and figures from confirmed results.

This builder intentionally excludes the known lineage-mismatched Topi/Mixed grid,
the AUC=0.68 diagnostic, the 67.9%/14.5% figures, and the old
"Core TopiOCQA s1=0.708" diagnostic.
"""

from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent
RELEASE = PACKAGE.parent
REPO = RELEASE.parents[1]
TABLES = PACKAGE / "tables"
TABLE_PNG = PACKAGE / "figures" / "table_png"
PLOTS_PNG = PACKAGE / "figures" / "plots_png"
PLOTS_PDF = PACKAGE / "figures" / "plots_pdf"


TABLE_DATA: dict[str, list[dict[str, object]]] = {
    "SLIDE09_CORE5_MATCHED_FPR_DETECTION.csv": [
        {"Attack": "MEntA", "MIRABEL_TPR_pct": 84.86, "V2_TPR_pct": 94.92, "Delta_pct_point": 10.06},
        {"Attack": "MBA", "MIRABEL_TPR_pct": 98.80, "V2_TPR_pct": 99.90, "Delta_pct_point": 1.10},
        {"Attack": "RAG-MIA", "MIRABEL_TPR_pct": 88.80, "V2_TPR_pct": 98.70, "Delta_pct_point": 9.90},
        {"Attack": "S2-MIA", "MIRABEL_TPR_pct": 98.00, "V2_TPR_pct": 99.87, "Delta_pct_point": 1.88},
        {"Attack": "DCMI-Q2", "MIRABEL_TPR_pct": 72.85, "V2_TPR_pct": 87.00, "Delta_pct_point": 14.15},
        {"Attack": "Macro mean", "MIRABEL_TPR_pct": 88.66, "V2_TPR_pct": 96.08, "Delta_pct_point": 7.42},
    ],
    "SLIDE10_CORE5_QWEN_E2E_PRIVACY.csv": [
        {"Attack": "MEntA", "No_Defense_EAUC": 0.978261, "MIRABEL_EAUC": 0.601906, "V2_EAUC": 0.541943},
        {"Attack": "MBA", "No_Defense_EAUC": 0.909827, "MIRABEL_EAUC": 0.500262, "V2_EAUC": 0.504586},
        {"Attack": "RAG-MIA", "No_Defense_EAUC": 0.981000, "MIRABEL_EAUC": 0.554000, "V2_EAUC": 0.507000},
        {"Attack": "S2-MIA", "No_Defense_EAUC": 0.685857, "MIRABEL_EAUC": 0.505632, "V2_EAUC": 0.503755},
        {"Attack": "DCMI-Q2", "No_Defense_EAUC": 0.979513, "MIRABEL_EAUC": 0.518514, "V2_EAUC": 0.501980},
        {"Attack": "Macro mean", "No_Defense_EAUC": 0.906892, "MIRABEL_EAUC": 0.536063, "V2_EAUC": 0.511852},
    ],
    "SLIDE11_GOLD_QA_UTILITY.csv": [
        {"Method": "No Defense", "Intervention_pct": 0.0, "Gold_F1": 0.208945, "Gold_F1_retention_pct": 100.00, "Answer_preservation_F1": 1.000000, "New_refusal_pct": 0.0, "Contradiction_introduced_pct": 0.0},
        {"Method": "MIRABEL", "Intervention_pct": 5.1, "Gold_F1": 0.206115, "Gold_F1_retention_pct": 98.65, "Answer_preservation_F1": 0.981391, "New_refusal_pct": 0.3, "Contradiction_introduced_pct": 0.2},
        {"Method": "Final V2", "Intervention_pct": 3.9, "Gold_F1": 0.203186, "Gold_F1_retention_pct": 97.24, "Answer_preservation_F1": 0.980548, "New_refusal_pct": 0.7, "Contradiction_introduced_pct": 0.4},
    ],
    "SLIDE12_K_ABLATION_MATCHED_FPR.csv": [
        {"Method": "G2", "Actual_FPR_pct": 2.50, "Core5_macro_TPR_pct": 98.625, "IA_TPR_pct": 48.982},
        {"Method": "G4", "Actual_FPR_pct": 2.50, "Core5_macro_TPR_pct": 99.096, "IA_TPR_pct": 53.463},
        {"Method": "G64", "Actual_FPR_pct": 2.50, "Core5_macro_TPR_pct": 99.342, "IA_TPR_pct": 63.504},
        {"Method": "G3000", "Actual_FPR_pct": 2.50, "Core5_macro_TPR_pct": 99.530, "IA_TPR_pct": 75.676},
        {"Method": "MIRABEL", "Actual_FPR_pct": 2.50, "Core5_macro_TPR_pct": 98.228, "IA_TPR_pct": 45.362},
    ],
    "SLIDE13_MEMBER_NONMEMBER_INTERVENTION.csv": [
        {"Detector": "s1", "Member_TPR_pct": 96.69, "Nonmember_intervention_pct": 12.51, "Intervention_precision_pct": 88.34},
        {"Detector": "V2 (G4)", "Member_TPR_pct": 94.60, "Nonmember_intervention_pct": 2.92, "Intervention_precision_pct": 96.75},
        {"Detector": "MIRABEL", "Member_TPR_pct": 85.30, "Nonmember_intervention_pct": 1.56, "Intervention_precision_pct": 97.91},
    ],
    "APPENDIX_FINQA_HARD_TO_CORE_CROSS_CORPUS.csv": [
        {"Detector": "V2 (G4)", "Hard_locked_FPR_pct": 1.89, "Core5_member_TPR_pct": 95.53, "Core5_nonmember_intervention_pct": 3.64, "Intervention_precision_pct": 96.12},
        {"Detector": "MIRABEL", "Hard_locked_FPR_pct": 2.34, "Core5_member_TPR_pct": 81.99, "Core5_nonmember_intervention_pct": 1.13, "Intervention_precision_pct": 98.32},
    ],
    "SLIDE15_FINQA_HARD_TYPE_FPR.csv": [
        {"Type": "Yes/No", "V2_FPR_pct": 2.67, "MIRABEL_FPR_pct": 4.00},
        {"Type": "Exact-Fact", "V2_FPR_pct": 1.96, "MIRABEL_FPR_pct": 0.65},
        {"Type": "Deep-study", "V2_FPR_pct": 3.80, "MIRABEL_FPR_pct": 2.53},
        {"Type": "Reask / paraphrase", "V2_FPR_pct": 2.60, "MIRABEL_FPR_pct": 3.25},
    ],
    "SLIDE14_FINQA_E2E_PRIVACY.csv": [
        {"Attack": "DCMI", "No_Defense_EAUC": 0.557442, "MIRABEL_EAUC": 0.524001, "V2_EAUC": 0.529501},
        {"Attack": "MBA", "No_Defense_EAUC": 0.632368, "MIRABEL_EAUC": 0.541713, "V2_EAUC": 0.541803},
        {"Attack": "MEntA", "No_Defense_EAUC": 0.763614, "MIRABEL_EAUC": 0.692686, "V2_EAUC": 0.647771},
        {"Attack": "RAG-MIA", "No_Defense_EAUC": 0.574500, "MIRABEL_EAUC": 0.530500, "V2_EAUC": 0.538500},
        {"Attack": "S2-MIA", "No_Defense_EAUC": 0.540000, "MIRABEL_EAUC": 0.523750, "V2_EAUC": 0.502500},
        {"Attack": "Macro mean", "No_Defense_EAUC": 0.613585, "MIRABEL_EAUC": 0.562530, "V2_EAUC": 0.552015},
        {"Attack": "Worst", "No_Defense_EAUC": 0.763614, "MIRABEL_EAUC": 0.692686, "V2_EAUC": 0.647771},
    ],
    "SLIDE15_IA_LIMITATION.csv": [
        {"Q1_protocol": "First-position Q1", "No_Defense_EAUC": 0.873548, "MIRABEL_EAUC": 0.729330, "V2_EAUC": 0.671065, "Gate_0p65": "FAIL"},
        {"Q1_protocol": "Fixed-position Q1", "No_Defense_EAUC": 0.832929, "MIRABEL_EAUC": 0.770500, "V2_EAUC": 0.714581, "Gate_0p65": "FAIL"},
    ],
    "APPENDIX_LLAMA_TRANSFER.csv": [
        {"Attack": "MEntA", "Qwen_V2_EAUC": 0.541943, "Llama_V2_EAUC": 0.534837},
        {"Attack": "MBA", "Qwen_V2_EAUC": 0.504586, "Llama_V2_EAUC": 0.500428},
        {"Attack": "RAG-MIA", "Qwen_V2_EAUC": 0.507000, "Llama_V2_EAUC": 0.506500},
        {"Attack": "S2-MIA", "Qwen_V2_EAUC": 0.503755, "Llama_V2_EAUC": 0.507509},
        {"Attack": "DCMI-Q2", "Qwen_V2_EAUC": 0.501980, "Llama_V2_EAUC": 0.500994},
        {"Attack": "Macro mean", "Qwen_V2_EAUC": 0.511852, "Llama_V2_EAUC": 0.510054},
    ],
    "APPENDIX_RETRIEVER_TRANSFER.csv": [
        {"Retriever": "bge-m3", "MIRABEL_Core5_macro_TPR_pct": 89.28, "V2_Core5_macro_TPR_pct": 96.06, "V2_min_attack_TPR_pct": 86.95, "V2_max_attack_TPR_pct": 99.90},
        {"Retriever": "gte-base", "MIRABEL_Core5_macro_TPR_pct": 81.21, "V2_Core5_macro_TPR_pct": 87.98, "V2_min_attack_TPR_pct": 78.70, "V2_max_attack_TPR_pct": 99.00},
        {"Retriever": "mpnet-base", "MIRABEL_Core5_macro_TPR_pct": 26.72, "V2_Core5_macro_TPR_pct": 50.16, "V2_min_attack_TPR_pct": 23.30, "V2_max_attack_TPR_pct": 65.80},
    ],
    "APPENDIX_CALIBRATION_CONTAMINATION_10PCT.csv": [
        {"Domain": "Core", "Detector": "V2", "Clean_TPR_pct": 94.60, "Percentile_10pct_TPR_pct": 44.34, "MAD_10pct_TPR_pct": 93.21, "Percentile_locked_FPR_pct": 0.00, "MAD_locked_FPR_pct": 1.90},
        {"Domain": "Core", "Detector": "MIRABEL", "Clean_TPR_pct": 85.30, "Percentile_10pct_TPR_pct": 49.12, "MAD_10pct_TPR_pct": 81.24, "Percentile_locked_FPR_pct": 0.10, "MAD_locked_FPR_pct": 1.80},
        {"Domain": "FinQA", "Detector": "V2", "Clean_TPR_pct": 54.50, "Percentile_10pct_TPR_pct": 28.07, "MAD_10pct_TPR_pct": 51.51, "Percentile_locked_FPR_pct": 0.50, "MAD_locked_FPR_pct": 2.00},
        {"Domain": "FinQA", "Detector": "MIRABEL", "Clean_TPR_pct": 51.33, "Percentile_10pct_TPR_pct": 28.52, "MAD_10pct_TPR_pct": 42.23, "Percentile_locked_FPR_pct": 0.10, "MAD_locked_FPR_pct": 1.00},
    ],
    "APPENDIX_LATENCY.csv": [
        {"Method": "MIRABEL", "Corpus_n": 3000, "Mean_ms_per_query": 0.471009, "Median_ms": 0.470834, "P95_ms": 0.471538, "Persistent_input_MiB": 11.718750},
        {"Method": "V2 (G4)", "Corpus_n": 3000, "Mean_ms_per_query": 0.000532, "Median_ms": 0.000531, "P95_ms": 0.000544, "Persistent_input_MiB": 0.015259},
        {"Method": "s1", "Corpus_n": 3000, "Mean_ms_per_query": 0.000146, "Median_ms": 0.000146, "P95_ms": 0.000149, "Persistent_input_MiB": 0.003815},
    ],
    "APPENDIX_TARGET_SURVIVAL_FAILURES.csv": [
        {"Attack": "MEntA", "Member_queries": 5000, "Alarm_rank2to4_target_survives_n": 1, "Miss_rank1_target_survives_n": 244, "Miss_rank1_pct": 4.88},
        {"Attack": "MBA", "Member_queries": 1000, "Alarm_rank2to4_target_survives_n": 0, "Miss_rank1_target_survives_n": 1, "Miss_rank1_pct": 0.10},
        {"Attack": "RAG-MIA", "Member_queries": 1000, "Alarm_rank2to4_target_survives_n": 0, "Miss_rank1_target_survives_n": 13, "Miss_rank1_pct": 1.30},
        {"Attack": "S2-MIA", "Member_queries": 799, "Alarm_rank2to4_target_survives_n": 0, "Miss_rank1_target_survives_n": 1, "Miss_rank1_pct": 0.13},
        {"Attack": "DCMI-Q2", "Member_queries": 2000, "Alarm_rank2to4_target_survives_n": 0, "Miss_rank1_target_survives_n": 236, "Miss_rank1_pct": 11.80},
    ],
}




EXTRA_SUMMARY_TABLES: dict[str, list[dict[str, object]]] = {
    "SLIDE08_RUNTIME_COST.csv": TABLE_DATA["APPENDIX_LATENCY.csv"],
    "APPENDIX_IDK_SIDECHANNEL.csv": [
        {"Attack_or_scope": "MEntA", "IDK_EAUC": 0.501},
        {"Attack_or_scope": "DCMI-Q2", "IDK_EAUC": 0.510},
        {"Attack_or_scope": "MBA", "IDK_EAUC": 0.519},
        {"Attack_or_scope": "RAG-MIA", "IDK_EAUC": 0.500},
        {"Attack_or_scope": "S2-MIA", "IDK_EAUC": 0.510},
        {"Attack_or_scope": "Pooled No Defense", "IDK_EAUC": 0.994},
        {"Attack_or_scope": "Pooled MIRABEL", "IDK_EAUC": 0.563},
        {"Attack_or_scope": "Pooled Final V2", "IDK_EAUC": 0.501},
    ],
    "APPENDIX_ASSEMBLY_VALIDATION.csv": [
        {"Metric": "V2 alarm total", "Evaluated_n": 8015, "Result": "8015"},
        {"Metric": "Selected source != rank-1", "Evaluated_n": 8015, "Result": "0"},
        {"Metric": "Multi-detector same hidden source", "Evaluated_n": 7645, "Result": "100%"},
        {"Metric": "Same hidden source, answer-hash mismatch", "Evaluated_n": 7645, "Result": "0"},
        {"Metric": "Missing near threshold", "Evaluated_n": 10, "Result": "10"},
    ],
    "APPENDIX_FINQA_THRESHOLD_TRANSFER.csv": [
        {"Score": "s1", "TPR_drop_range_pct_point": "20-36"},
        {"Score": "G4", "TPR_drop_range_pct_point": "16-28"},
        {"Score": "MIRABEL", "TPR_drop_range_pct_point": "15-51"},
    ],
    "APPENDIX_FUSION_DIAGNOSTIC.csv": [
        {"Candidate": "z-sum", "Uniformly_better": "False", "Verdict": "FUSION_NOT_UNIFORMLY_BETTER"},
        {"Candidate": "alpha=0.25/0.5/0.75", "Uniformly_better": "False", "Verdict": "FUSION_NOT_UNIFORMLY_BETTER"},
        {"Candidate": "2D AND", "Uniformly_better": "False", "Verdict": "FUSION_NOT_UNIFORMLY_BETTER"},
    ],
}

SOURCE_FILES = [
    "releases/final-v2-validation-2026-09-17/results/core/TABLE1_CORE5_MATCHED_FPR_DETECTION.csv",
    "releases/final-v2-validation-2026-09-17/results/core/TABLE2_CORE5_QWEN_E2E_PRIVACY.csv",
    "releases/final-v2-validation-2026-09-17/results/core/TABLE3_LLAMA_GENERATOR_TRANSFER.csv",
    "releases/final-v2-validation-2026-09-17/results/core/TABLE4_RETRIEVER_TRANSFER.csv",
    "releases/final-v2-validation-2026-09-17/results/core/TABLE6_BENIGN_UTILITY_FACTUALITY.csv",
    "experiments/NONMEMBER_INTERVENTION_COST_CPU_V1/tables/MATCHED_INTERVENTION_EFFICIENCY.csv",
    "experiments/HARD_BENIGN_RECALIBRATION_CORE5_CPU_V1/tables/TABLE_THRESHOLDS_AND_LOCKED_FPR.csv",
    "experiments/HARD_BENIGN_RECALIBRATION_CORE5_CPU_V1/tables/TABLE_CORE5_MEMBER_NONMEMBER_PRECISION.csv",
    "experiments/HARD_BENIGN_RECALIBRATION_CORE5_CPU_V1/tables/TABLE_HARD_BENIGN_TYPE_FPR.csv",
    "experiments/HARD_BENIGN_RECALIBRATION_CORE5_CPU_V1/tables/TABLE_HARD_VS_TITLE475.csv",
    "experiments/LOCAL_RETRIEVAL_GEOMETRY_FPR_MATCHED_V1/tables/FPR_MATCHED_TPR_2p5PCT.csv",
    "experiments/FINAL_V2_FINQA_VALIDATION_ADDENDUM_20260916/tables/TABLE_FINQA_CORE5_E2E.csv",
    "experiments/FINAL_V2_ROBUST_THRESHOLD_CONTAMINATION_V1/tables/TABLE_ROBUST_THRESHOLD_CONTAMINATION.csv",
    "experiments/DETECTOR_LATENCY_CPU_V1/tables/DETECTOR_LATENCY.csv",
    "experiments/DETECTOR_LATENCY_CPU_V1/tables/MIRABEL_EXTRAPOLATION.csv",
    "experiments/FINAL_LC_IA_RESOLUTION_V1/fpr3_sensitivity/IA_Q1_RESULT.json",
    "experiments/V2_TARGET_SURVIVAL_RANK_AUDIT_V1/tables/V2_ALARM_TARGET_RANK_CROSSTAB.csv",
    "experiments/V2_TARGET_SURVIVAL_RANK_AUDIT_V1/tables/MEMBER_TARGET_RANK_DISTRIBUTION.csv",
]


FIGURE_SOURCES = {
    "FIG01_METHOD_OVERVIEW": "experiments/FINAL_V2_REMAINING_VALIDATION_AND_PAPER_FREEZE_V1/figures/FIG1_METHOD_OVERVIEW",
    "FIG02_LOW_FPR_DETECTION": "experiments/FINAL_V2_REMAINING_VALIDATION_AND_PAPER_FREEZE_V1/figures/FIG2_LOW_FPR_DETECTION",
    "FIG03_MEMBER_NONMEMBER": "experiments/FINAL_V2_REMAINING_VALIDATION_AND_PAPER_FREEZE_V1/figures/FIG3_MEMBER_NONMEMBER_DECOMPOSITION",
    "FIG04_QWEN_E2E_PRIVACY": "experiments/FINAL_V2_REMAINING_VALIDATION_AND_PAPER_FREEZE_V1/figures/FIG4_E2E_PRIVACY",
    "FIG05_GENERATOR_TRANSFER": "experiments/FINAL_V2_REMAINING_VALIDATION_AND_PAPER_FREEZE_V1/figures/FIG5_GENERATOR_TRANSFER",
    "FIG06_RETRIEVER_TRANSFER": "experiments/FINAL_V2_REMAINING_VALIDATION_AND_PAPER_FREEZE_V1/figures/FIG5_RETRIEVER_TRANSFER",
    "FIG07_BENIGN_UTILITY": "experiments/FINAL_V2_REMAINING_VALIDATION_AND_PAPER_FREEZE_V1/figures/FIG6_BENIGN_UTILITY",
    "FIG08_FINQA_TRANSFER": "experiments/FINAL_V2_REMAINING_VALIDATION_AND_PAPER_FREEZE_V1/figures/FIG7_FINQA_TRANSFER",
    "FIG09_CALIBRATION_CONTAMINATION": "experiments/FINAL_V2_FINQA_VALIDATION_ADDENDUM_20260916/figures/FIG_CALIBRATION_CONTAMINATION",
    "FIG10_CALIBRATION_LIMITATION": "experiments/FINAL_V2_REMAINING_VALIDATION_AND_PAPER_FREEZE_V1/figures/FIG10_CALIBRATION_LIMITATION",
    "FIG11_ROBUST_CALIBRATION_FPR": "experiments/FINAL_V2_REMAINING_VALIDATION_AND_PAPER_FREEZE_V1/figures/FIG10_ROBUST_CALIBRATION_FPR",
    "FIG12_ROBUST_CALIBRATION_TPR": "experiments/FINAL_V2_REMAINING_VALIDATION_AND_PAPER_FREEZE_V1/figures/FIG10_ROBUST_CALIBRATION_TPR",
    "FIG13_LOW_FPR_ROC_0_TO_5": "experiments/LOCAL_RETRIEVAL_GEOMETRY_FPR_MATCHED_V1/figures/ROC_FPR_0_TO_5PCT",
    "FIG14_G4_SCORE_HISTOGRAM": "experiments/LOCAL_RETRIEVAL_GEOMETRY_FPR_MATCHED_V1/figures/G4_BENIGN_CORE5_IA_HISTOGRAM",
    "FIG15_DETECTOR_LATENCY": "experiments/DETECTOR_LATENCY_CPU_V1/figures/detector_latency_logscale",
    "FIG16_RAW_SCORE_DISTRIBUTIONS": "experiments/FINAL_V2_PAPER_VISUAL_AND_QUALITATIVE_SIDECAR_V1/figures/fig1_raw_detector_score_distributions",
    "FIG17_BENIGN_PERCENTILE_TAIL": "experiments/FINAL_V2_PAPER_VISUAL_AND_QUALITATIVE_SIDECAR_V1/figures/fig2_benign_percentile_tail",
    "FIG18_SAME_QUERY_REORDERING": "experiments/FINAL_V2_PAPER_VISUAL_AND_QUALITATIVE_SIDECAR_V1/figures/fig4_same_query_reordering",
    "FIG19_GAINED_VS_LOST_TP": "experiments/FINAL_V2_PAPER_VISUAL_AND_QUALITATIVE_SIDECAR_V1/figures/fig5_gained_vs_lost_tp",
    "FIG20_LLAMA_SCORE_HISTOGRAMS": "experiments/FINAL_V2_PAPER_VISUAL_AND_QUALITATIVE_SIDECAR_V1/figures/fig6_llama_member_nonmember_histograms",
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def display_value(value: object) -> str:
    if isinstance(value, float):
        if abs(value) >= 10:
            return f"{value:.2f}"
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def render_table(csv_path: Path) -> None:
    with csv_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    columns = list(rows[0])
    values = [[display_value(float(r[c])) if _is_float(r[c]) else r[c] for c in columns] for r in rows]
    width = max(11.5, min(22.0, 2.25 * len(columns)))
    height = max(2.3, 0.52 * len(rows) + 1.25)
    fig, ax = plt.subplots(figsize=(width, height))
    ax.axis("off")
    table = ax.table(cellText=values, colLabels=columns, cellLoc="center", colLoc="center", loc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.35)
    for (row, _), cell in table.get_celld().items():
        if row == 0:
            cell.set_facecolor("#17365D")
            cell.set_text_props(color="white", weight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#EAF0F8")
        cell.set_edgecolor("#AAB7C4")
    fig.tight_layout(pad=0.5)
    out = TABLE_PNG / f"{csv_path.stem}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _is_float(value: str) -> bool:
    try:
        float(value)
        return True
    except ValueError:
        return False


def plot_k_ablation() -> None:
    rows = TABLE_DATA["SLIDE12_K_ABLATION_MATCHED_FPR.csv"][:-1]
    xs = [2, 4, 64, 3000]
    core = [float(r["Core5_macro_TPR_pct"]) for r in rows]
    ia = [float(r["IA_TPR_pct"]) for r in rows]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.plot(xs, core, marker="o", linewidth=2.3, label="Core5 macro TPR")
    ax.plot(xs, ia, marker="s", linewidth=2.3, label="IA TPR")
    ax.axhline(98.228, color="#777777", linestyle="--", linewidth=1.2, label="MIRABEL Core5")
    ax.axhline(45.362, color="#AA7777", linestyle=":", linewidth=1.2, label="MIRABEL IA")
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs, ["2", "4", "64", "3000"])
    ax.set_xlabel("Observation range k")
    ax.set_ylabel("TPR at matched 2.5% FPR (%)")
    ax.set_ylim(40, 102)
    ax.grid(alpha=0.25)
    ax.legend(ncol=2, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOTS_PNG / "FIG_K_ABLATION_MATCHED_FPR.png", dpi=300, bbox_inches="tight")
    fig.savefig(PLOTS_PDF / "FIG_K_ABLATION_MATCHED_FPR.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_hard_benign() -> None:
    rows = TABLE_DATA["APPENDIX_FINQA_HARD_TO_CORE_CROSS_CORPUS.csv"]
    labels = [str(r["Detector"]) for r in rows]
    x = np.arange(len(labels))
    width = 0.24
    fig, ax = plt.subplots(figsize=(8.2, 4.7))
    ax.bar(x - width, [float(r["Hard_locked_FPR_pct"]) for r in rows], width, label="D-hard(E) locked FPR")
    ax.bar(x, [float(r["Core5_nonmember_intervention_pct"]) for r in rows], width, label="Core5 nonmember (cross-corpus)")
    ax.bar(x + width, [float(r["Core5_member_TPR_pct"]) for r in rows], width, label="Core5 member TPR (cross-corpus)")
    ax.set_xticks(x, labels)
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncol=3, fontsize=8, loc="upper center")
    fig.tight_layout()
    fig.savefig(PLOTS_PNG / "FIG_FINQA_HARD_BENIGN_RECALIBRATION.png", dpi=300, bbox_inches="tight")
    fig.savefig(PLOTS_PDF / "FIG_FINQA_HARD_BENIGN_RECALIBRATION.pdf", bbox_inches="tight")
    plt.close(fig)


def copy_figures() -> None:
    for name, stem in FIGURE_SOURCES.items():
        for suffix, target_dir in ((".png", PLOTS_PNG), (".pdf", PLOTS_PDF)):
            source = REPO / f"{stem}{suffix}"
            if source.exists():
                shutil.copy2(source, target_dir / f"{name}{suffix}")


def build_source_manifest() -> None:
    rows = []
    for relative in SOURCE_FILES:
        path = REPO / relative
        if not path.exists():
            rows.append({"path": relative, "sha256": "MISSING", "bytes": "MISSING"})
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({"path": relative, "sha256": digest, "bytes": path.stat().st_size})
    write_csv(PACKAGE / "SOURCE_MANIFEST_SHA256.csv", rows)


def main() -> None:
    for directory in (TABLES, TABLE_PNG, PLOTS_PNG, PLOTS_PDF):
        directory.mkdir(parents=True, exist_ok=True)
    for name, rows in {**TABLE_DATA, **EXTRA_SUMMARY_TABLES}.items():
        path = TABLES / name
        write_csv(path, rows)
        render_table(path)
    plot_k_ablation()
    plot_hard_benign()
    copy_figures()
    build_source_manifest()


if __name__ == "__main__":
    main()
