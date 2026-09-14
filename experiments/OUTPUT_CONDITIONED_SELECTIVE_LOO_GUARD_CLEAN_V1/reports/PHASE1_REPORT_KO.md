# CLEAN_V1 Phase 1 결과

- Verdict: **LOO_LOCATOR_LIMITED**
- New retrieval/query scores: **780**
- Protected-answer generation: **0**
- Legacy numeric artifact mixed: **NO**

## Oracle mechanism

| Attack | AUC | 95% CI | Member mean | Nonmember mean | Effect 95% CI | TPR@3% |
|---|---:|---:|---:|---:|---:|---:|
| MEntA | 1.000 | [1.000, 1.000] | 3.14767 | 0.00000 | [2.70861, 3.76544] | 0.800 |
| S²-MIA | 1.000 | [1.000, 1.000] | 0.68353 | 0.00000 | [0.39006, 1.03289] | 0.100 |
| MBA | 0.900 | [0.775, 1.000] | 0.41740 | 0.00000 | [0.31070, 0.51690] | 0.000 |

## Locator comparison at benign-calibrated 3% FPR

| Attack | Oracle | MIRABEL | QLL | UNION-2 |
|---|---:|---:|---:|---:|
| MEntA | 0.800 | 0.250 | 0.300 | 0.300 |
| S²-MIA | 0.100 | 0.000 | 0.000 | 0.000 |
| MBA | 0.000 | 0.000 | 0.000 | 0.000 |

## Gate

- Oracle gate: **True**
- Locator gate evaluated: **True**
- Locator gate pass: **False**
- Phase 2 was not started automatically by this script.

Oracle의 target-not-retrieved row는 overall 분석에서 risk=0으로 기록했다. target이 Top-4에 있는 subset은 별도 retrieval/분포 표로 보존했으며, nonmember target은 corpus 밖이므로 그 subset의 membership AUC는 일반적으로 정의되지 않는다.
