# IA-Std-Q15-v4-TwoStage Final Summary

1. v3 mathematical failure: generated=606, valid=505, invalid=101
2. v4 preflight validity: 58/100 (58.0%)
3. v4 full validity: NOT RUN
4. member/nonmember validity gap: 0.0400
5. IA Final LC TPR @2.5%: NOT RUN
6. MIRABEL TPR @2.5%: NOT RUN
7. IA E2E No Defense: NOT RUN
8. IA E2E MIRABEL: NOT RUN
9. IA E2E Final LC: NOT RUN
10. Final scientific status: `STANDARDIZED_ATTACK_UNAVAILABLE`

## Failure categories

- `STAGE1:DUPLICATE_QUESTION`: 9
- `STAGE1:INVALID_JSON`: 2
- `STAGE2:ID_SEQUENCE_MISMATCH`: 1
- `STAGE2:INVALID_JSON`: 1
- `STAGE2:INVALID_JUDGMENT`: 29

The frozen format preflight failed before any detector/privacy result was calculated.
No full-2,000 generation, retry, prompt/parser revision, attack-semantic change, or model change was performed.
Core5 matched-FPR detection, Core5 matched-budget E2E, and benign-only Gold recalibration were not rerun or modified.

The original GPU process then raised a report-only `NoneType.get` exception after writing the correct terminal heartbeat and preflight result. This file is a CPU-only reporting repair from those frozen artifacts.
