#!/usr/bin/env python3
"""Independent verifier for the immutable final freeze/release package."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
OUT = HERE / "38_final_freeze_release"
GITHUB = ROOT / "release_github_dualguard"
ANON = ROOT / "release_submission_anonymous"


def cr(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree_sha(path: Path) -> str:
    h = hashlib.sha256()
    for item in sorted(x for x in path.rglob("*") if x.is_file() and "__pycache__" not in x.parts):
        h.update(str(item.relative_to(path)).encode()); h.update(b"\0"); h.update(bytes.fromhex(sha(item)))
    return h.hexdigest()


def main() -> None:
    frozen = json.loads((OUT / "00_FINAL_METHOD_FREEZE.json").read_text())
    verdict = json.loads((OUT / "FINAL_VERDICT.json").read_text())
    query = json.loads((OUT / "QUERY_AUDIT_SUMMARY.json").read_text())
    output = json.loads((OUT / "OUTPUT_AUDIT_SUMMARY.json").read_text())
    smoke = json.loads((OUT / "RELEASE_PACKAGE_SMOKE_TEST.json").read_text())
    attack = cr(OUT / "FINAL_ATTACK_PERFORMANCE_TABLE.csv")
    single = cr(OUT / "TABLE_SINGLE_QUERY_ATTACKS.csv")
    session = cr(OUT / "TABLE_SESSION_ATTACKS_Q1_Q5.csv")
    utility = cr(OUT / "FINAL_CORRECTED_BENIGN_UTILITY.csv")
    ordinary = cr(OUT / "CSV_BENIGN_ORDINARY_QUERIES_AND_OUTPUTS.csv")
    conversational = cr(OUT / "CSV_BENIGN_CONVERSATIONAL_QUERIES_AND_OUTPUTS.csv")
    deep = cr(OUT / "CSV_TRUE_DEEP_DIVE_QUERIES_AND_OUTPUTS.csv")
    ia = cr(OUT / "CSV_IA_ATTACK_QUERIES_AND_OUTPUTS.csv")
    menta = cr(OUT / "CSV_MENTA_SESSION_ATTACK_QUERIES_AND_OUTPUTS.csv")
    report = (OUT / "FINAL_FREEZE_AND_RELEASE_REPORT_KO.md").read_text()
    qissues = (OUT / "CSV_QUERY_AUDIT_ISSUES.csv").read_text().strip()

    session_map = {(x["Attack"], x["Method"]): x for x in session}
    core_attacks = {x["attack"] for x in single}
    required_figures = [
        "FIG_IA_EXACT_ATTACK_SCORE_HIST", "FIG_IA_MEMBER_VS_BENIGN_INFLUENCE_HIST",
        "FIG_IA_MEMBER_VS_DEEP_INFLUENCE_HIST", "FIG_IA_Q1_Q5_TRAJECTORY",
        "FIG_MENTA_Q1_Q5_TRAJECTORY",
    ]
    forbidden = ("/home/", "traffic_3", "api_key", "access_token", "github.com/")
    scan = []
    for package in (GITHUB, ANON):
        for path in package.rglob("*"):
            if path.is_file():
                text = path.read_text(encoding="utf-8", errors="ignore").casefold()
                for needle in forbidden:
                    if needle.casefold() in text:
                        scan.append({"package": package.name, "file": str(path.relative_to(package)), "needle": needle})

    checks = {
        "method_status_exact": frozen["status"] == "METHOD_FROZEN_CLD_OLD_CFCLIP",
        "tau_exact": float(frozen["tau"]) == 0.0901547372341156,
        "gamma_exact": float(frozen["gamma"]) == 9.5659109592,
        "gold_engine_serial": frozen["gold_engine"] == "SERIAL_KVCACHE",
        "immutable": frozen["immutable"] is True,
        "domain_transfer_forbidden": "domain transfer" in frozen["forbidden_current_run"],
        "six_attacks_present": {"RAG-MIA", "S²-MIA", "MBA", "DCMI-Std-Q2", "IA", "MEntA"}.issubset({x["attack"] for x in attack}),
        "single_query_core4_present": {"RAG-MIA", "S²-MIA", "MBA", "DCMI-Std-Q2"} == core_attacks,
        "IA_exact_trajectory_present": all(("IA", method) in session_map for method in ("NO_DEFENSE", "CLD_ONLY", "EXACT_OLD_DUAL")),
        "MEntA_primary_trajectory_present": all(("MEntA", method) in session_map for method in ("NO_DEFENSE", "CLD_ONLY", "OLD_DUAL")),
        "corrected_utility_4x3": len(utility) == 12,
        "ordinary_50": len(ordinary) == 50,
        "conversational_200": len(conversational) == 200,
        "deep_154": len(deep) == 154,
        "IA_920": len(ia) == 920,
        "MEntA_1000": len(menta) == 1000,
        "deep_sorted": [(x["session_id"], int(x["turn"])) for x in deep] == sorted((x["session_id"], int(x["turn"])) for x in deep),
        "query_audit_pass": query["status"] == "QUERY_AUDIT_PASS" and query["blocking_issue_count"] == 0,
        "query_issue_file_empty": qissues == "",
        "output_audit_pass": output["status"] == "OUTPUT_AUDIT_PASS" and output["blocking_issue_count"] == 0,
        "diagnostics_disclosed": output["diagnostic_count"] > 0 and output["diagnostic_breakdown"].get("KNOWN_MAX_TOKEN_CAP") == 39,
        "all_figures_png_pdf": all((OUT / f"{name}.png").stat().st_size > 1000 and (OUT / f"{name}.pdf").stat().st_size > 1000 for name in required_figures),
        "github_package_present": GITHUB.is_dir(),
        "anonymous_package_present": ANON.is_dir(),
        "no_sensitive_path_hits": not scan,
        "smoke_pass": smoke["status"] == "RELEASE_PACKAGE_SMOKE_PASS" and all(x["returncode"] == 0 for x in smoke["commands"]),
        "github_tree_hash": tree_sha(GITHUB) == smoke["github"]["tree_sha256"],
        "anonymous_tree_hash": tree_sha(ANON) == smoke["anonymous"]["tree_sha256"],
        "fast_not_in_public_package": not any("fast" in str(x.relative_to(GITHUB)).casefold() for x in GITHUB.rglob("*") if x.is_file()),
        "verdict_ready": verdict["verdict"] == "FINAL_FROZEN_MODEL_PACKAGE_READY",
        "method_unmodified": verdict["method_modified"] is False,
        "no_domain_transfer_run": verdict["domain_transfer_run"] is False,
        "final_line_exact": report.rstrip().endswith("FINAL_RECOMMENDATION: package ready"),
    }
    result = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "checks": checks,
        "mismatches": [key for key, value in checks.items() if not value],
        "sensitive_scan_hits": scan,
        "recomputed": {
            "attack_rows": len(attack), "utility_rows": len(utility),
            "query_rows": len(ordinary)+len(conversational)+len(deep)+len(ia)+len(menta),
            "github_tree_sha256": tree_sha(GITHUB), "anonymous_tree_sha256": tree_sha(ANON),
        },
    }
    (OUT / "INDEPENDENT_VERIFICATION.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, sort_keys=True) + "\n")

    # Refresh the manifest after the verifier is present.
    files = sorted(x for x in OUT.iterdir() if x.is_file() and x.name != "FINAL_ARTIFACT_MANIFEST.md")
    lines = ["# Final Artifact Manifest", "", "## Validation outputs", ""]
    lines += [f"- `{x.name}` — SHA256 `{sha(x)}`" for x in files]
    lines += ["", "## GitHub package", "", f"- `{GITHUB.relative_to(ROOT)}` — tree SHA256 `{tree_sha(GITHUB)}`",
              "", "## Anonymous submission package", "", f"- `{ANON.relative_to(ROOT)}` — tree SHA256 `{tree_sha(ANON)}`",
              "", "## Verification", "", f"- `INDEPENDENT_VERIFICATION.json` — `{result['status']}`", ""]
    (OUT / "FINAL_ARTIFACT_MANIFEST.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
