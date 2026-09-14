# LC-MIRABEL Large Score-only Report

- 판정: `LC_MIRABEL_DETECTION_NOT_SUPPORTED`
- Positive: member-target attack query
- Negative: 동일 benign holdout 1,000 query

| Attack | MIRABEL TPR@3% | LC TPR@3% | Delta | Bootstrap 95% CI |
|---|---:|---:|---:|---:|
| MEntA | 0.8648 | 0.8939 | +0.0291 | [0.0231, 0.0353] |
| MBA | 0.9930 | 0.9966 | +0.0036 | [0.0006, 0.0076] |
| RAG-MIA | 0.9170 | 0.9603 | +0.0433 | [0.0304, 0.0560] |
| **Macro** | **0.9249** | **0.9503** | **+0.0253** | — |

## Gate
- menta_delta_at_least_5pp: `FAIL`
- menta_bootstrap_ci_low_above_zero: `PASS`
- mba_noninferior_within_2pp: `PASS`
- rag_mia_noninferior_within_2pp: `PASS`
- core3_macro_delta_at_least_2pp: `PASS`

실패 시 명세대로 Qwen E2E와 추가 local-feature 탐색을 실행하지 않는다.
