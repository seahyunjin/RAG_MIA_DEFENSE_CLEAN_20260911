# Phase B — Training-free Sparse Exposure

- Screen 판정: `SPARSE_EXPOSURE_SCREEN_FAILED`
- 현재 최종 판정: `TRAINING_FREE_HANDCRAFTED_DETECTOR_SEARCH_CLOSED`
- 이 100/100 cohort는 development data이다.

| Attack | MIRABEL TPR@3% | Sparse TPR@3% | Delta |
|---|---:|---:|---:|
| MEntA | 0.908 | 0.895 | -0.013 |
| MBA | 0.990 | 0.982 | -0.008 |
| RAG-MIA | 0.940 | 0.897 | -0.042 |
| Macro | 0.946 | 0.925 | -0.021 |

## Gate
- menta_delta_at_least_5pp: `FAIL`
- menta_cluster_bootstrap_ci_low_above_zero: `FAIL`
- mba_noninferior_within_5pp: `PASS`
- rag_mia_noninferior_within_5pp: `PASS`
- core3_macro_member_tpr_improved: `FAIL`

- MEntA bootstrap 95% CI: [-0.0252, -0.0028]
- gain concentration warning: `None`

Screen 실패 시 명세대로 generation과 추가 handcrafted 탐색을 실행하지 않는다.
