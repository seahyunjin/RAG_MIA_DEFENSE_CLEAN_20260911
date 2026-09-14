# DualTail Member Exposure Replication V2 — Phase 1

- 판정: `DUALTAIL_MEMBER_EXPOSURE_NOT_REPLICATED`
- 기존 V1 판정: `BC_DUALTAIL_V1_NO_TPR_GAIN` (변경 없음)

| Attack | MIRABEL member TPR@3% | DualTail | Delta |
|---|---:|---:|---:|
| MEntA | 0.908 | 0.933 | +0.025 |
| MBA | 0.990 | 1.000 | +0.010 |
| RAG-MIA | 0.940 | 0.991 | +0.051 |
| Macro | 0.946 | 0.975 | +0.029 |

## Gate
- menta_delta_at_least_5pp: `FAIL`
- menta_cluster_bootstrap_ci_low_above_zero: `PASS`
- mba_noninferior_within_5pp: `PASS`
- rag_mia_noninferior_within_5pp: `PASS`
- core3_macro_member_tpr_improved: `PASS`

## Recovery
- recovered queries/targets: 11/9
- lost queries/targets: 17/13
- largest target fraction: 0.182

Phase 2 generation allowed: `False`
