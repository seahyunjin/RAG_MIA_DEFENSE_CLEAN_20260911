#!/usr/bin/env python3
# filepath: /fred/oz396/lbao/MIA/DCMI/evaluate.py
"""
DCMI Evaluation Script
Evaluates membership inference using the DCMI black-box method from the paper.

For each target sample x:
  - frag(q)  = Yes/No from original query
  - frag(q') = Yes/No from perturbed query (included in the same answers file)
  - frag_calibrated(q) = frag(q) - frag(q')
  - I(x) = 1 if frag_calibrated(q) == 1  → MEMBER
         = 0 if frag_calibrated(q) in {0, -1} → NON-MEMBER

The answers file contains both original and perturbed queries:
  - Original query ID:  "doc123"
  - Perturbed query ID: "doc123_perturbed"

Normalization is performed on raw_answer (re-derived here) rather than
trusting the pre-stored rag_answer field.
"""

import os
import sys
import json
import argparse
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import roc_auc_score, roc_curve
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.process_json import read_jsonl, write_json


# ─────────────────────────────────────────────
# Normalization
# ─────────────────────────────────────────────

def normalize_answer(raw_answer: str) -> str:
    """
    Re-derive Yes/No from raw_answer using word-boundary matching
    to avoid false matches inside words like 'significant', 'notable', 'known'.

    Priority:
      1. Explicit "don't know" / uncertainty phrases → Unknown
      2. Word-boundary \byes\b / \bno\b matching
      3. If both present → trust whichever appears first
      4. Fallback → No
    """
    a = raw_answer.lower().strip()

    # Remove model-specific tokens
    a = a.replace("<|eot_id|>", "").replace("<|end_of_text|>", "").strip()

    # Uncertainty phrases → Unknown
    dont_know = [
        "i don't know", "i do not know", "don't know", "do not know",
        "i'm not sure", "i am not sure", "not sure",
        "cannot find", "can't find", "unable to find",
        "no information", "no context",
    ]
    for pat in dont_know:
        if pat in a:
            return "Unknown"

    has_yes = bool(re.search(r'\byes\b', a))
    has_no  = bool(re.search(r'\bno\b',  a))

    if has_yes and not has_no:
        return "Yes"
    elif has_no and not has_yes:
        return "No"
    elif has_yes and has_no:
        # Both present — trust whichever comes first
        pos_yes = re.search(r'\byes\b', a).start()
        pos_no  = re.search(r'\bno\b',  a).start()
        return "Yes" if pos_yes < pos_no else "No"
    else:
        return "No"  # default fallback


def calibrated_score(frag_orig: str, frag_pert: str) -> int:
    """
    frag_calibrated(q) = frag(q) - frag(q')   (equation 12 in the paper)

    Using Yes=1, No=0:
      original=Yes, perturbed=No  →  1 - 0 = 1  → MEMBER
      original=Yes, perturbed=Yes →  1 - 1 = 0  → NON-MEMBER
      original=No,  perturbed=*   → ≤ 0          → NON-MEMBER
    """
    score_orig = 1 if frag_orig == "Yes" else 0
    score_pert = 1 if frag_pert == "Yes" else 0
    return score_orig - score_pert          # ∈ {-1, 0, 1}


def extract_model_name(path: str) -> str:
    """Extract model tag from answers path (filename or parent directory)."""
    basename = os.path.basename(path)
    m = re.search(r'answers_([^./]+)\.jsonl', basename)
    if m:
        return m.group(1)
    # Pipeline layout: .../answers/topk3/<model-tag>/answers.jsonl
    if basename == "answers.jsonl":
        return Path(path).parent.name
    return "unknown-model"


# ─────────────────────────────────────────────
# Core evaluation
# ─────────────────────────────────────────────

def build_answer_map(jsonl_path: str) -> dict:
    """
    Returns {query_id: {rag_answer, raw_answer, derived_answer,
                        membership_label, is_perturbed, original_id}}

    derived_answer = normalize_answer(raw_answer) — always re-derived here,
    ignoring the pre-stored rag_answer field which may be wrong due to
    word-boundary bugs in generate_output.py.
    """
    mapping = {}
    skipped = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"  WARNING: Skipping malformed line {i}: {e}")
                skipped += 1
                continue

            qid = item.get("_id")
            if not qid:
                continue

            raw_answer = item.get("raw_answer") or item.get("rag_answer") or ""
            rag_answer = item.get("rag_answer") or ""

            # Re-derive answer from raw using correct word-boundary normalization
            derived_answer = normalize_answer(raw_answer)

            mapping[qid] = {
                "raw_answer":       raw_answer,
                "rag_answer":       rag_answer,      # stored value (may be wrong)
                "derived_answer":   derived_answer,  # re-derived from raw (used for eval)
                "membership_label": item.get("membership_label"),
                "is_perturbed":     item.get("is_perturbed", False),
                "original_id":      item.get("original_id"),
            }

    if skipped > 0:
        print(f"  WARNING: Skipped {skipped} malformed lines in {jsonl_path}")

    return mapping


def evaluate_dcmi(
    answers_path: str,
    corpus_member_path: str,
    corpus_nonmember_path: str,
    output_dir: str,
    model_name: str = "unknown",
):
    """
    Evaluate DCMI from a single answers file that contains both
    original (doc_id) and perturbed (doc_id_perturbed) query answers.
    Uses raw_answer re-derived via normalize_answer() for evaluation.
    """
    print(f"\n{'='*60}")
    print("DCMI Membership Inference Evaluation")
    print(f"  Answers  : {answers_path}")
    print(f"  Output   : {output_dir}")
    print(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)

    # ── load answers ──────────────────────────────────────────
    answer_map = build_answer_map(answers_path)
    print(f"Loaded {len(answer_map)} total answers")

    # Separate original and perturbed answers
    original_answers  = {}
    perturbed_answers = {}

    for qid, info in answer_map.items():
        if info["is_perturbed"] or str(qid).endswith("_perturbed"):
            doc_id = info["original_id"] if info.get("original_id") else str(qid).replace("_perturbed", "")
            perturbed_answers[doc_id] = info
        else:
            original_answers[qid] = info

    print(f"  Original answers : {len(original_answers)}")
    print(f"  Perturbed answers: {len(perturbed_answers)}")

    common_doc_ids = set(original_answers.keys()) & set(perturbed_answers.keys())
    print(f"  Paired documents : {len(common_doc_ids)}")

    if not common_doc_ids:
        raise ValueError("No paired original/perturbed answers found!")

    # ── load corpus membership ────────────────────────────────
    member_ids    = {d["_id"] for d in read_jsonl(corpus_member_path)}
    nonmember_ids = {d["_id"] for d in read_jsonl(corpus_nonmember_path)}

    # ── compute calibrated scores ─────────────────────────────
    rows = []
    answer_stats = {
        "orig_stored":    {"Yes": 0, "No": 0, "Unknown": 0},
        "orig_derived":   {"Yes": 0, "No": 0, "Unknown": 0},
        "pert_stored":    {"Yes": 0, "No": 0, "Unknown": 0},
        "pert_derived":   {"Yes": 0, "No": 0, "Unknown": 0},
    }

    mismatch_count = 0

    for doc_id in sorted(common_doc_ids):
        # Ground-truth membership
        if doc_id in member_ids:
            membership = "member"
        elif doc_id in nonmember_ids:
            membership = "nonmember"
        else:
            label = original_answers[doc_id].get("membership_label")
            membership = "member" if label == 1 else "nonmember" if label == 0 else "unknown"

        orig_info = original_answers[doc_id]
        pert_info = perturbed_answers[doc_id]

        # Stored (old) answers
        ans_orig_stored = orig_info["rag_answer"]
        ans_pert_stored = pert_info["rag_answer"]

        # Re-derived answers from raw
        ans_orig = orig_info["derived_answer"]
        ans_pert = pert_info["derived_answer"]

        # Track mismatches between stored and derived
        if ans_orig_stored != ans_orig or ans_pert_stored != ans_pert:
            mismatch_count += 1

        # Update stats
        answer_stats["orig_stored"][ans_orig_stored] = answer_stats["orig_stored"].get(ans_orig_stored, 0) + 1
        answer_stats["orig_derived"][ans_orig]        = answer_stats["orig_derived"].get(ans_orig, 0) + 1
        answer_stats["pert_stored"][ans_pert_stored]  = answer_stats["pert_stored"].get(ans_pert_stored, 0) + 1
        answer_stats["pert_derived"][ans_pert]        = answer_stats["pert_derived"].get(ans_pert, 0) + 1

        cal      = calibrated_score(ans_orig, ans_pert)
        hard_pred = 1 if cal == 1 else 0

        rows.append({
            "doc_id":                doc_id,
            "membership":            membership,
            "raw_answer_original":   orig_info["raw_answer"],
            "raw_answer_perturbed":  pert_info["raw_answer"],
            "stored_answer_original":  ans_orig_stored,
            "stored_answer_perturbed": ans_pert_stored,
            "derived_answer_original": ans_orig,
            "derived_answer_perturbed": ans_pert,
            "calibrated_score":      cal,
            "predicted_member":      hard_pred,
        })

    df = pd.DataFrame(rows)

    print(f"\n  Answers re-derived from raw_answer (mismatches with stored: {mismatch_count})")

    # ── metrics ───────────────────────────────────────────────
    all_ids = member_ids | nonmember_ids
    df = df[df["doc_id"].isin(all_ids)]

    if len(df) == 0:
        print("ERROR: No documents matched corpus member/nonmember IDs!")
        return None

    df["label"] = df["doc_id"].apply(lambda x: 1 if x in member_ids else 0)

    y_true   = df["label"].values
    y_scores = df["calibrated_score"].values.astype(float)
    y_pred   = df["predicted_member"].values

    if len(np.unique(y_true)) < 2:
        print("WARNING: Only one class present in ground truth. AUC undefined.")
        auc = 0.5
    else:
        auc = roc_auc_score(y_true, y_scores)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy  = (tp + tn) / len(df) if len(df) > 0 else 0.0

    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    tpr_val = recall

    if len(np.unique(y_true)) >= 2:
        fpr_curve, tpr_curve, _ = roc_curve(y_true, y_scores)
    else:
        fpr_curve = np.array([0, 1])
        tpr_curve = np.array([0, 1])

    def tpr_at(target_fpr):
        idx = np.where(fpr_curve <= target_fpr)[0]
        if len(idx) == 0:
            return 0.0
        return float(tpr_curve[idx[np.argmax(tpr_curve[idx])]])

    tpr_at_fpr = {
        "tpr@fpr=0.005": tpr_at(0.005),
        "tpr@fpr=0.01":  tpr_at(0.01),
        "tpr@fpr=0.05":  tpr_at(0.05),
    }

    # ── print results ─────────────────────────────────────────
    print(f"\nAnswer statistics (stored vs re-derived from raw):")
    print(f"  Original  stored  : {answer_stats['orig_stored']}")
    print(f"  Original  derived : {answer_stats['orig_derived']}")
    print(f"  Perturbed stored  : {answer_stats['pert_stored']}")
    print(f"  Perturbed derived : {answer_stats['pert_derived']}")

    cal_dist = df["calibrated_score"].value_counts().sort_index().to_dict()
    print(f"  Calibrated score distribution: {cal_dist}")

    print(f"\n{'='*60}")
    print("DCMI Membership Inference Results")
    print(f"{'='*60}")
    print(f"  Documents evaluated : {len(df)}")
    print(f"  Members (GT)        : {int(y_true.sum())}")
    print(f"  Non-members (GT)    : {int((1-y_true).sum())}")
    print(f"\n  AUC-ROC  : {auc:.4f}")
    for k, v in tpr_at_fpr.items():
        print(f"  {k}: {v:.4f}")
    print(f"\n  Precision: {precision:.4f}")
    print(f"  Recall   : {recall:.4f}")
    print(f"  F1       : {f1:.4f}")
    print(f"  Accuracy : {accuracy:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    TP={tp}  FP={fp}")
    print(f"    FN={fn}  TN={tn}")

    # ── save CSV ──────────────────────────────────────────────
    csv_path = os.path.join(output_dir, f"dcmi_scores_{model_name}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nScores CSV : {csv_path}")

    # ── save JSON ─────────────────────────────────────────────
    results = {
        "model": model_name,
        "evaluation": {
            "n_documents":   len(df),
            "n_members":     int(y_true.sum()),
            "n_nonmembers":  int((1 - y_true).sum()),
            "answer_mismatches_stored_vs_derived": mismatch_count,
        },
        "metrics": {
            "auc_roc":   auc,
            "precision": precision,
            "recall":    recall,
            "f1":        f1,
            "accuracy":  accuracy,
            "tpr":       tpr_val,
            "fpr":       fpr_val,
            **tpr_at_fpr,
        },
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "answer_stats": answer_stats,
        "calibrated_score_distribution": {int(k): int(v) for k, v in cal_dist.items()},
        "method": (
            "DCMI black-box: I(x) = 1 iff frag_calibrated(q) = frag(q) - frag(q') == 1. "
            "Answers re-derived from raw_answer using word-boundary normalization."
        ),
    }
    json_path = os.path.join(output_dir, f"dcmi_evaluation_{model_name}.json")
    write_json(results, json_path)
    print(f"Results JSON: {json_path}")

    # ── ROC curve ─────────────────────────────────────────────
    plt.figure(figsize=(7, 6))
    plt.plot(fpr_curve, tpr_curve, linewidth=2, label=f"AUC = {auc:.4f}")
    plt.plot([0, 1], [0, 1], "k--", label="Random (AUC = 0.5)")
    plt.scatter([fpr_val], [tpr_val], color="red", zorder=5,
                label=f"Threshold (FPR={fpr_val:.3f}, TPR={tpr_val:.3f})")
    plt.xlim([0, 1]); plt.ylim([0, 1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title(f"DCMI ROC – {model_name}")
    plt.legend(loc="lower right"); plt.grid(True)
    roc_path = os.path.join(output_dir, f"dcmi_roc_{model_name}.png")
    plt.savefig(roc_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"ROC curve  : {roc_path}")

    return results


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DCMI membership inference (black-box calibrated method)"
    )
    parser.add_argument(
        "--answers_file",
        type=str,
        required=True,
        help="Path to RAG answers file (contains both original and perturbed query answers)",
    )
    parser.add_argument(
        "--corpus_member",
        type=str,
        default="data/BeIR_nfcorpus/corpus_member.jsonl",
    )
    parser.add_argument(
        "--corpus_nonmember",
        type=str,
        default="data/BeIR_nfcorpus/corpus_nonmember.jsonl",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (auto-derived if not set)",
    )

    args = parser.parse_args()

    model_name = extract_model_name(args.answers_file)

    if args.output_dir is None:
        p = Path(args.answers_file)
        topk             = p.parent.name
        answers_dir_name = p.parent.parent.name
        droot            = p.parent.parent.parent

        if answers_dir_name == "answers":
            eval_dir_name = "evaluation_dcmi"
        else:
            defense_suffix = answers_dir_name.replace("answers_", "")
            eval_dir_name  = f"evaluation_dcmi_{defense_suffix}"

        args.output_dir = str(droot / eval_dir_name / topk / model_name)

    evaluate_dcmi(
        answers_path          = args.answers_file,
        corpus_member_path    = args.corpus_member,
        corpus_nonmember_path = args.corpus_nonmember,
        output_dir            = args.output_dir,
        model_name            = model_name,
    )

    print("\n" + "="*60)
    print("DCMI evaluation completed.")
    print("="*60)


if __name__ == "__main__":
    main()