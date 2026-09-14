#!/usr/bin/env python3
"""Paper-only explanation of frozen Mirabel and QLL low-FPR score tails.

This stage consumes the completed Exp212 post-transfer diagnostic.  It does not
fit a detector, choose a threshold, generate an answer, or alter the defense.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT = Path("/home/traffic_3/workspace/workspace/SH/LoRA-mirabel-Decter")
CAMPAIGN = PROJECT / "final_validation_stateless_qll_source_hide_20260901"
ROOT = CAMPAIGN / "stage_03b_low_fpr_mechanism"
EXP212 = PROJECT / "exp212_retriever_transfer_stateless_qll_20260831"
QUERY_SCORES = EXP212 / "private/LOW_FPR_QUERY_SCORES.private.csv.gz"
THRESHOLDS = EXP212 / "tables/LOW_FPR_THRESHOLDS.csv"
COMPARISON = EXP212 / "tables/TPR_AT_LOW_FPR_MIRABEL_VS_STATELESS_QLL.csv"
LOW_FPR_RESULT = EXP212 / "LOW_FPR_COMPARISON_RESULT.json"
PRECOMMIT = ROOT / "configs/PRECOMMIT.json"
RETRIEVERS = ("BGE", "GTE")
ATTACKS = ("DCMI", "MEntA", "RAG-MIA", "RAGLeak", "BudgetLeak-Z", "MBA", "S²-MIA")
DETECTORS = {
    "Original Mirabel": ("mirabel_margin", "MIRABEL Gumbel margin"),
    "Stateless QLL Source Hide": ("stateless_qll_score", "QLL dominance"),
}
FPR_COLORS = {0.01: "#111111", 0.03: "#D97706", 0.05: "#DC2626"}
ATTACK_COLOR = "#2563EB"
BENIGN_COLOR = "#9CA3AF"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
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
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                                 default=lambda item: item.item() if hasattr(item, "item") else str(item)) + "\n")


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".csv", dir=path.parent)
    os.close(fd)
    try:
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def checkpoint(stage: str, **details: object) -> None:
    payload = {"stage": stage, "updated_utc": now(), "pid": os.getpid(), **details}
    atomic_json(ROOT / "checkpoints" / f"{stage}.json", payload)
    atomic_json(ROOT / "HEARTBEAT.json", payload)
    atomic_text(ROOT / "STATUS.md", "\n".join([
        "# Stage 3B — Low-FPR mechanism explanation", "", f"- Stage: **{stage}**",
        f"- Updated UTC: `{payload['updated_utc']}`", f"- PID: `{payload['pid']}`",
        *[f"- {key}: `{value}`" for key, value in details.items()], ""]) )


def validate_precommit() -> dict:
    config = json.loads(PRECOMMIT.read_text())
    inputs = {
        "query_scores": QUERY_SCORES,
        "thresholds": THRESHOLDS,
        "comparison": COMPARISON,
        "BGE_qll_cases": EXP212 / "private/BGE_QLL_CASES.private.csv.gz",
        "GTE_qll_cases": EXP212 / "private/GTE_QLL_CASES.private.csv.gz",
        "low_fpr_result": LOW_FPR_RESULT,
    }
    if config["model_or_threshold_selection_allowed"]:
        raise RuntimeError("mechanism stage must not select a model or threshold")
    rows = []
    for key, path in inputs.items():
        if not path.exists():
            raise FileNotFoundError(path)
        actual = sha256_file(path)
        expected = config["input_hashes"][key]
        if actual != expected:
            raise RuntimeError(f"frozen mechanism input drift: {key}")
        rows.append({"input": key, "path": str(path), "sha256": actual, "access": "READ_ONLY"})
    atomic_csv(pd.DataFrame(rows), ROOT / "audits/FROZEN_INPUTS.csv")
    checkpoint("PREFLIGHT_COMPLETE", frozen_inputs=len(rows), new_thresholds=0, new_defenses=0)
    return config


def session_scores(query: pd.DataFrame) -> pd.DataFrame:
    """Use the already precommitted any-turn maximum for each detector."""
    keys = ["retriever", "kind", "attack_family", "session_id", "member"]
    output = query.groupby(keys, as_index=False).agg(
        turns=("turn", "nunique"),
        mirabel_margin=("mirabel_margin", "max"),
        stateless_qll_score=("stateless_qll_score", "max"),
    )
    return output


def smooth_density(values: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    hist, _ = np.histogram(values, bins=edges, density=True)
    radius = 5
    grid = np.arange(-radius, radius + 1, dtype=float)
    kernel = np.exp(-0.5 * (grid / 1.5) ** 2)
    kernel /= kernel.sum()
    density = np.convolve(hist, kernel, mode="same")
    centers = (edges[:-1] + edges[1:]) / 2
    return centers, density


def overlap_coefficient(benign: np.ndarray, attack: np.ndarray, edges: np.ndarray) -> float:
    left, _ = np.histogram(benign, bins=edges, density=True)
    right, _ = np.histogram(attack, bins=edges, density=True)
    return float(np.sum(np.minimum(left, right) * np.diff(edges)))


def score_range(values: np.ndarray, thresholds: dict[float, float]) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    low, high = np.quantile(finite, [0.002, 0.998])
    low = min(float(low), min(thresholds.values()))
    high = max(float(high), max(thresholds.values()))
    span = max(high - low, 1e-6)
    return low - 0.04 * span, high + 0.04 * span


def threshold_map(thresholds: pd.DataFrame, retriever: str, detector: str) -> dict[float, float]:
    cell = thresholds[(thresholds.retriever.eq(retriever)) &
                      (thresholds.unit.eq("session_any_turn")) &
                      (thresholds.detector.eq(detector))]
    result = {float(row.target_fpr): float(row.threshold) for row in cell.itertuples(index=False)}
    if set(result) != {0.01, 0.03, 0.05}:
        raise RuntimeError(f"missing frozen threshold: {retriever}/{detector}")
    return result


def draw_panel(ax, benign: np.ndarray, attack: np.ndarray, edges: np.ndarray,
               thresholds: dict[float, float], family: str, tprs: dict[float, float],
               xlabel: str, localization: tuple[float, float] | None = None) -> None:
    bx, by = smooth_density(benign, edges)
    ax.fill_between(bx, by, color=BENIGN_COLOR, alpha=.35, label="Benign")
    ax.plot(bx, by, color="#4B5563", linewidth=1.2)
    axx, ayy = smooth_density(attack, edges)
    ax.fill_between(axx, ayy, color=ATTACK_COLOR, alpha=.16)
    ax.plot(axx, ayy, color=ATTACK_COLOR, linewidth=1.6, label=family)
    for fpr in (0.01, 0.03, 0.05):
        ax.axvline(thresholds[fpr], color=FPR_COLORS[fpr], linewidth=1.0,
                   linestyle={0.01: "-", 0.03: "--", 0.05: ":"}[fpr])
    annotation = "TPR@1/3/5% = " + "/".join(f"{100*tprs[fpr]:.1f}" for fpr in (0.01, 0.03, 0.05)) + "%"
    if localization is not None:
        annotation += f"\nTarget Top-4 / QLL Hit@1 = {100*localization[0]:.1f}/{100*localization[1]:.1f}%"
    ax.text(.985, .94, annotation, transform=ax.transAxes, ha="right", va="top", fontsize=7.2,
            bbox={"boxstyle": "round,pad=.25", "facecolor": "white", "alpha": .86, "edgecolor": "#D1D5DB"})
    ax.set_title(family, loc="left", fontsize=9.5, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=8)
    ax.set_ylabel("Density", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.grid(axis="y", alpha=.15)


def standalone_figure(retriever: str, detector: str, score_col: str, label: str,
                      sessions: pd.DataFrame, thresholds: dict[float, float],
                      comparison: pd.DataFrame, mechanism: pd.DataFrame) -> None:
    cell = sessions[sessions.retriever.eq(retriever)]
    benign = cell[cell.kind.eq("BENIGN")][score_col].to_numpy(float)
    values = cell[score_col].to_numpy(float)
    low, high = score_range(values, thresholds)
    edges = np.linspace(low, high, 100)
    fig, axes = plt.subplots(4, 2, figsize=(10.4, 12.0), constrained_layout=True)
    for ax, family in zip(axes.flat, ATTACKS):
        attack = cell[(cell.kind.ne("BENIGN")) & cell.attack_family.eq(family)][score_col].to_numpy(float)
        metric = comparison[(comparison.retriever.eq(retriever)) &
                            (comparison.unit.eq("session_any_turn")) &
                            (comparison.family.eq(family)) &
                            (comparison.membership_scope.eq("ALL"))]
        prefix = "mirabel" if detector == "Original Mirabel" else "stateless_qll"
        tprs = {float(row.target_fpr): float(getattr(row, f"tpr_{prefix}")) for row in metric.itertuples(index=False)}
        loc_row = mechanism[(mechanism.retriever.eq(retriever)) & mechanism.attack_family.eq(family)].iloc[0]
        localization = (float(loc_row.target_top4_retrieval_rate_member),
                        float(loc_row.qll_top1_target_alignment_member)) if detector != "Original Mirabel" else None
        draw_panel(ax, benign, attack, edges, thresholds, family, tprs, label, localization)
    axes.flat[-1].axis("off")
    handles = [plt.Line2D([0], [0], color="#4B5563", lw=2, label="Benign"),
               plt.Line2D([0], [0], color=ATTACK_COLOR, lw=2, label="Attack"),
               *[plt.Line2D([0], [0], color=FPR_COLORS[fpr], lw=1.5,
                            linestyle={0.01: "-", 0.03: "--", 0.05: ":"}[fpr],
                            label=f"Benign {int(100*fpr)}% FPR threshold") for fpr in (0.01, 0.03, 0.05)]]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=8)
    fig.suptitle(f"{retriever}: {label} — benign vs. attack-family distributions", fontsize=14, fontweight="bold")
    stem = f"{retriever}_{'MIRABEL' if detector == 'Original Mirabel' else 'QLL'}_SCORE_DISTRIBUTIONS"
    fig.savefig(ROOT / f"figures/{stem}.pdf", bbox_inches="tight")
    fig.savefig(ROOT / f"figures/{stem}.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def comparison_figure(retriever: str, sessions: pd.DataFrame, thresholds_frame: pd.DataFrame,
                      comparison: pd.DataFrame, mechanism: pd.DataFrame) -> None:
    fig, axes = plt.subplots(len(ATTACKS), 2, figsize=(12.2, 16.8), constrained_layout=True)
    cell = sessions[sessions.retriever.eq(retriever)]
    for column, (detector, (score_col, label)) in enumerate(DETECTORS.items()):
        thresholds = threshold_map(thresholds_frame, retriever, detector)
        low, high = score_range(cell[score_col].to_numpy(float), thresholds)
        edges = np.linspace(low, high, 100)
        benign = cell[cell.kind.eq("BENIGN")][score_col].to_numpy(float)
        prefix = "mirabel" if detector == "Original Mirabel" else "stateless_qll"
        for row_index, family in enumerate(ATTACKS):
            attack = cell[(cell.kind.ne("BENIGN")) & cell.attack_family.eq(family)][score_col].to_numpy(float)
            metric = comparison[(comparison.retriever.eq(retriever)) &
                                (comparison.unit.eq("session_any_turn")) &
                                (comparison.family.eq(family)) &
                                (comparison.membership_scope.eq("ALL"))]
            tprs = {float(row.target_fpr): float(getattr(row, f"tpr_{prefix}")) for row in metric.itertuples(index=False)}
            loc_row = mechanism[(mechanism.retriever.eq(retriever)) & mechanism.attack_family.eq(family)].iloc[0]
            localization = (float(loc_row.target_top4_retrieval_rate_member),
                            float(loc_row.qll_top1_target_alignment_member)) if column == 1 else None
            draw_panel(axes[row_index, column], benign, attack, edges, thresholds, family, tprs, label, localization)
            if row_index == 0:
                axes[row_index, column].text(.01, 1.18, label, transform=axes[row_index, column].transAxes,
                                             fontsize=12, fontweight="bold", ha="left")
    fig.suptitle(f"{retriever}: MIRABEL margin vs. QLL dominance at matched benign FPR", fontsize=15, fontweight="bold")
    fig.savefig(ROOT / f"figures/{retriever}_MIRABEL_VS_QLL_LOW_FPR.pdf", bbox_inches="tight")
    fig.savefig(ROOT / f"figures/{retriever}_MIRABEL_VS_QLL_LOW_FPR.png", dpi=320, bbox_inches="tight")
    plt.close(fig)


def build_tables(query: pd.DataFrame, sessions: pd.DataFrame, thresholds: pd.DataFrame,
                 comparison: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    long_rows = []
    mechanism_rows = []
    for retriever in RETRIEVERS:
        session_cell = sessions[sessions.retriever.eq(retriever)]
        benign = session_cell[session_cell.kind.eq("BENIGN")]
        qll_cases = pd.read_csv(EXP212 / f"private/{retriever}_QLL_CASES.private.csv.gz",
                                keep_default_na=False, low_memory=False,
                                dtype={"case_id": str, "session_id": str, "target_document_id": str,
                                       "qll_top1_source": str})
        members = qll_cases[(qll_cases.kind.ne("BENIGN")) & qll_cases.member.eq(1)].copy()
        members["target_top4"] = pd.to_numeric(members.target_rank).between(1, 4)
        members["qll_hit1"] = members.qll_top1_source.astype(str).eq(members.target_document_id.astype(str))
        for detector, (score_col, _) in DETECTORS.items():
            for row in benign.itertuples(index=False):
                long_rows.append({"retriever": retriever, "detector": detector, "cohort": "BENIGN",
                                  "session_id": str(row.session_id), "member": -1,
                                  "score": float(getattr(row, score_col))})
            for family in ATTACKS:
                attack = session_cell[(session_cell.kind.ne("BENIGN")) & session_cell.attack_family.eq(family)]
                for row in attack.itertuples(index=False):
                    long_rows.append({"retriever": retriever, "detector": detector, "cohort": family,
                                      "session_id": str(row.session_id), "member": int(row.member),
                                      "score": float(getattr(row, score_col))})
        for family in ATTACKS:
            family_sessions = session_cell[(session_cell.kind.ne("BENIGN")) & session_cell.attack_family.eq(family)]
            family_queries = query[(query.retriever.eq(retriever)) & query.kind.ne("BENIGN") &
                                   query.attack_family.eq(family)]
            loc = members[members.attack_family.eq(family)]
            row = {"retriever": retriever, "attack_family": family,
                   "attack_sessions": len(family_sessions), "attack_query_turns": len(family_queries),
                   "member_query_turns_for_localization": len(loc),
                   "target_top4_retrieval_rate_member": float(loc.target_top4.mean()) if len(loc) else math.nan,
                   "qll_top1_target_alignment_member": float(loc.qll_hit1.mean()) if len(loc) else math.nan}
            for detector, (score_col, _) in DETECTORS.items():
                threshold_values = threshold_map(thresholds, retriever, detector)
                benign_values = benign[score_col].to_numpy(float)
                attack_values = family_sessions[score_col].to_numpy(float)
                low, high = score_range(np.concatenate([benign_values, attack_values]), threshold_values)
                edges = np.linspace(low, high, 160)
                prefix = "mirabel" if detector == "Original Mirabel" else "qll"
                row[f"{prefix}_overlap_coefficient"] = overlap_coefficient(benign_values, attack_values, edges)
                row[f"{prefix}_attack_q95_minus_benign_q95"] = float(
                    np.quantile(attack_values, .95) - np.quantile(benign_values, .95))
                for fpr, threshold in threshold_values.items():
                    row[f"{prefix}_threshold_fpr_{int(100*fpr):02d}"] = threshold
                    row[f"{prefix}_tpr_fpr_{int(100*fpr):02d}"] = float(np.mean(attack_values > threshold))
            mechanism_rows.append(row)
    distribution = pd.DataFrame(long_rows)
    mechanism = pd.DataFrame(mechanism_rows)
    # Exact cross-check against the already published matched-FPR table.
    for row in mechanism.itertuples(index=False):
        reference = comparison[(comparison.retriever.eq(row.retriever)) &
                               comparison.unit.eq("session_any_turn") &
                               comparison.family.eq(row.attack_family) &
                               comparison.membership_scope.eq("ALL")]
        for metric_prefix, reference_prefix in (("mirabel", "mirabel"), ("qll", "stateless_qll")):
            for fpr in (0.01, 0.03, 0.05):
                observed = getattr(row, f"{metric_prefix}_tpr_fpr_{int(100*fpr):02d}")
                expected = float(reference.loc[reference.target_fpr.eq(fpr), f"tpr_{reference_prefix}"].iloc[0])
                if not np.isclose(observed, expected, atol=1e-12):
                    raise RuntimeError(f"TPR cross-check failed: {row.retriever}/{row.attack_family}/{metric_prefix}/{fpr}")
    atomic_csv(distribution, ROOT / "tables/SCORE_DISTRIBUTIONS_SOURCE.csv")
    atomic_csv(mechanism, ROOT / "tables/ATTACK_LOCALIZATION_AND_LOW_FPR_TPR.csv")
    return distribution, mechanism


def write_report(mechanism: pd.DataFrame) -> None:
    bge = mechanism[mechanism.retriever.eq("BGE")].set_index("attack_family")
    weak = bge[(bge.target_top4_retrieval_rate_member >= .50) & (bge.qll_tpr_fpr_03 < .25)].index.tolist()
    right_tail = bge.sort_values("qll_tpr_fpr_03", ascending=False).head(3).index.tolist()
    overlap = bge.loc[[name for name in ("S²-MIA", "MBA", "BudgetLeak-Z") if name in bge.index],
                      ["qll_overlap_coefficient", "mirabel_overlap_coefficient"]]
    lines = ["# Stage 3B — QLL dominance low-FPR mechanism", "",
             "이 분석은 동결 score와 정상 이용자 1,000개에서 이미 정해진 1%/3%/5% FPR threshold만 재사용한 설명용 분석이다. 새 defense·threshold·score 조합은 만들지 않았다.", "",
             "## BGE 핵심표", "", bge.reset_index().to_markdown(index=False, floatfmt=".4f"), "",
             "## 관찰", "",
             f"- Target이 Top-4에 자주 있어도 QLL dominance 3% threshold를 잘 넘지 못한 family: **{', '.join(weak) if weak else '없음'}**.",
             f"- BGE QLL 3% FPR right-tail TPR 상위 family: **{', '.join(right_tail)}**.",
             "- S²-MIA/MBA/BudgetLeak의 benign-overlap 계수(1에 가까울수록 많이 겹침):", "",
             overlap.to_markdown(floatfmt=".4f"), "",
             "## 해석 제한", "",
             "- Target localization과 dominance alarm은 다른 사건이다. QLL이 target source를 1위로 골라도 softmax dominance가 정상 상위 꼬리 threshold를 넘지 않을 수 있다.",
             "- TPR은 공격 트래픽 탐지율이며 생성 답변의 membership E-AUC가 아니다.",
             "- 이 결과를 이용해 threshold, detector, defense를 다시 선택하지 않는다.",
             "- GTE 결과와 기존 deep-rank BudgetLeak 실패를 함께 보존한다.", ""]
    atomic_text(ROOT / "reports/LOW_FPR_MECHANISM_REPORT_KO.md", "\n".join(lines))


def main() -> None:
    for name in ("tables", "figures", "reports", "audits", "checkpoints", "private", "logs"):
        (ROOT / name).mkdir(parents=True, exist_ok=True)
    config = validate_precommit()
    query = pd.read_csv(QUERY_SCORES, keep_default_na=False, low_memory=False,
                        dtype={"case_id": str, "session_id": str})
    thresholds = pd.read_csv(THRESHOLDS)
    comparison = pd.read_csv(COMPARISON)
    sessions = session_scores(query)
    counts = sessions.groupby(["retriever", "kind"]).size().to_dict()
    for retriever in RETRIEVERS:
        if counts.get((retriever, "BENIGN")) != 1000:
            raise RuntimeError(f"{retriever} benign session identity failure")
        attack_count = int((sessions.retriever.eq(retriever) & sessions.kind.ne("BENIGN")).sum())
        if attack_count != 3900:
            raise RuntimeError(f"{retriever} attack session identity failure: {attack_count}")
    distribution, mechanism = build_tables(query, sessions, thresholds, comparison)
    checkpoint("TABLES_COMPLETE", distribution_rows=len(distribution), mechanism_rows=len(mechanism))
    for retriever in RETRIEVERS:
        for detector, (score_col, label) in DETECTORS.items():
            standalone_figure(retriever, detector, score_col, label, sessions,
                              threshold_map(thresholds, retriever, detector), comparison, mechanism)
        comparison_figure(retriever, sessions, thresholds, comparison, mechanism)
        checkpoint("FIGURE_RETRIEVER_COMPLETE", retriever=retriever, png=3, pdf=3)
    write_report(mechanism)
    outputs = sorted([*ROOT.glob("figures/*.png"), *ROOT.glob("figures/*.pdf"), *ROOT.glob("tables/*.csv")])
    manifest = pd.DataFrame([{"path": str(path), "bytes": path.stat().st_size,
                              "sha256": sha256_file(path)} for path in outputs])
    atomic_csv(manifest, ROOT / "audits/OUTPUT_MANIFEST.csv")
    result = {"verdict": "LOW_FPR_MECHANISM_EXPLANATION_COMPLETE", "retrievers": list(RETRIEVERS),
              "benign_sessions_per_retriever": 1000, "attack_sessions_per_retriever": 3900,
              "attack_families": list(ATTACKS), "new_defenses": 0, "new_thresholds": 0,
              "threshold_source": "frozen benign-only Exp212 1/3/5% FPR operating points",
              "model_or_threshold_selection_used": False, "figures_png": 6, "figures_pdf": 6,
              "source_tables": 2, "precommit_sha256": sha256_file(PRECOMMIT),
              "completed_utc": now()}
    atomic_json(ROOT / "FINAL_RESULT.json", result)
    checkpoint("COMPLETE", verdict=result["verdict"], outputs=len(outputs))


if __name__ == "__main__":
    main()
