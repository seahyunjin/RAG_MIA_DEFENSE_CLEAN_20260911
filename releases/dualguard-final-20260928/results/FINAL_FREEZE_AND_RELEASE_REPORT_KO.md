# Final Freeze and Release Report

## 1. Frozen model

`METHOD_FROZEN_CLD_OLD_CFCLIP`: Retriever → CLD/G4 → old CF-Clip → Frozen Generator.

## 2. Frozen parameters

tau `0.0901547372341156`, gamma `9.5659109592`, gold engine `SERIAL_KVCACHE`.

## 3. Single-query table

See `TABLE_SINGLE_QUERY_ATTACKS.csv` and `FINAL_ATTACK_PERFORMANCE_TABLE.md`.

## 4. Session table

| Attack | Method | Q1 | Q2 | Q3 | Q4 | Q5 | Q5-Q1 | lineage |
|---|---|---|---|---|---|---|---|---|
| IA | CFCLIP_ONLY | 0.612890 | 0.651288 | 0.648807 | 0.695475 | 0.714674 | 0.101784 | SUPPLEMENTARY_BATCHED_FOUR_ARM |
| IA | CLD_ONLY | 0.668301 | 0.729206 | 0.766482 | 0.828922 | 0.845818 | 0.177517 | SERIAL_KVCACHE_EXACT |
| IA | EXACT_OLD_DUAL | 0.612890 | 0.646267 | 0.644908 | 0.690749 | 0.709712 | 0.096822 | SERIAL_KVCACHE_EXACT |
| IA | FAST_OLD_DUAL | 0.612890 | 0.646267 | 0.644908 | 0.690749 | 0.709712 | 0.096822 | SERIAL_KVCACHE_EXACT |
| IA | NO_DEFENSE | 0.679525 | 0.746810 | 0.784204 | 0.847767 | 0.864721 | 0.185196 | SERIAL_KVCACHE_EXACT |
| MEntA | CLD_ONLY | 0.653700 | 0.670900 | 0.715450 | 0.713550 | 0.704900 | 0.051200 | FRESH_ZERO_OVERLAP_PRIMARY |
| MEntA | NO_DEFENSE | 0.686500 | 0.737200 | 0.781450 | 0.789000 | 0.784450 | 0.097950 | FRESH_ZERO_OVERLAP_PRIMARY |
| MEntA | OLD_DUAL | 0.660350 | 0.673250 | 0.708550 | 0.705900 | 0.715550 | 0.055200 | FRESH_ZERO_OVERLAP_PRIMARY |

## 5. IA histogram

`FIG_IA_EXACT_ATTACK_SCORE_HIST.png` uses validated SERIAL_KVCACHE exact rows.

## 6. IA/benign histogram

Both influence figures are mechanism diagnostics, not detector ROC curves.

## 7. IA trajectory

`FIG_IA_Q1_Q5_TRAJECTORY.png`.

## 8. MEntA trajectory

`FIG_MENTA_Q1_Q5_TRAJECTORY.png`; residual leakage is not hidden.

## 9. Corrected benign table

See `FINAL_CORRECTED_BENIGN_UTILITY.csv`.

## 10. Deep-dive table

Gold-provenance deep-dive rows: `154`.

## 11. Query/output CSV inventory

Ordinary `50`, conversational `200`, deep `154`, IA `920`, MEntA `1000`.

## 12. Query sanity audit

`QUERY_AUDIT_PASS`; blocking `0`, diagnostic duplicates `{'exact_duplicate_rows': 1232, 'expected_world_mirror_duplicates': 460, 'near_duplicate_pairs_jaccard_ge_0_90': 57}`.

## 13. Qualitative examples

`QUERY_OUTPUT_AUDIT_SAMPLES.md` uses fixed SHA256 selection and complete sessions/segments.

## 14. GitHub package

{"path": "release_github_dualguard", "tree_sha256": "65ce9c1eaded76cc8d4d033ab70ecf995a2655f29515508e579c03d58979348d"}

## 15. Anonymous submission package

{"path": "release_submission_anonymous", "tree_sha256": "fbae402bb233a368722f33d38e97f0c02c32f5f536b3e734bb1404a70736e120"}

## 16. Smoke test

`RELEASE_PACKAGE_SMOKE_PASS`.

## 17. Limitations

Fast is not production-ready; CF-only session supplements do not replace exact/fresh primary lineages; no domain transfer, transfer experiment, adaptive attack, baseline rerun, or method tuning was performed.

## 18. Final verdict

`FINAL_FROZEN_MODEL_PACKAGE_READY`

FINAL_RECOMMENDATION: package ready
