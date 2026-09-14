# Final LC IA Audit and DB Churn CPU Sidecar

- Updated UTC: `2026-09-13T09:45:48.731051+00:00`
- IA terminal: `True`
- GPU used: `False`
- Answer generation: `0`
- Final LC changes: `0`

## IA STREAMING VALIDITY

- Phase: PREFLIGHT
- Questions/judgments read: 1013/0
- Complete valid sessions: 0/0
- Known invalid sessions: 35
- Gate still mathematically possible: False

## IA NEAR-DUPLICATE DIAGNOSTIC

- Suspicious sessions: 100
- Diagnostic only; it does not reject sessions or alter generation.

## DB CHURN

- Index verdict: `DB_CHURN_GPU_EMBEDDING_REQUIRED`
- V10/V25/V50 scoring was not fabricated.
- Missing frozen embeddings by version: V0=0, V10=223, V25=556, V50=1113

## CALIBRATION SIZE SENSITIVITY

- Verdict: `CALIBRATION_SIZE_SENSITIVITY_INPUT_INSUFFICIENT`
- N=500 frozen result retained; N=100/200/300 were not approximated.

## COST / CPU

- Elapsed: 855.6s
- Peak RSS: 175.0 MiB
