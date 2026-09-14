# DV-LOO V2 Phase 1 최종 보고서

## 최종 판정

`DV_LOO_V2_PHASE1_REJECTED`

DV-LOO V2는 3% 목표 FPR에서 MEntA member-session TPR 70%를 얻었지만, 동일한 정상 calibration으로 보정한 Original Mirabel margin도 70%였다. 따라서 필수 조건인 Original Mirabel 대비 `+10%p` 개선을 충족하지 못했다. LOO 후보는 여기서 종료하며 Phase 2 보호 답변 생성은 실행하지 않았다.

## 재현성과 입력

- 전체 query: 780개
- 정상 query: 500개 — calibration 250개, locked holdout 250개
- 공격: 120세션, 280개 query
- 각 공격: member 20세션 + nonmember 20세션
- MEntA: 세션당 Q5, S²-MIA/MBA: 세션당 Q1
- full-corpus Mirabel 재구성: 3개 도메인 각각 1,000문서
- Mirabel top-1 ID: 780/780 일치
- top-1 similarity 최대 절대 오차: 0.0
- 공격 표본을 이용한 score/threshold 보정: 0개
- 보호 답변 생성: 0개

## Member attack-session TPR

각 수치는 20개 member session 중 하나 이상의 query가 threshold를 넘은 세션의 비율이다.

| 방법 | 공격 | TPR@1% | TPR@3% | TPR@5% |
|---|---:|---:|---:|---:|
| Original Mirabel margin | MEntA | 60% | 70% | 75% |
| Original Mirabel margin | S²-MIA | 75% | 80% | 80% |
| Original Mirabel margin | MBA | 75% | 75% | 85% |
| CLEAN_V1 raw LOO | MEntA | 20% | 30% | 45% |
| CLEAN_V1 raw LOO | S²-MIA | 0% | 0% | 0% |
| CLEAN_V1 raw LOO | MBA | 0% | 0% | 0% |
| DV-LOO V2 | MEntA | 40% | 70% | 75% |
| DV-LOO V2 | S²-MIA | 70% | 75% | 75% |
| DV-LOO V2 | MBA | 75% | 75% | 75% |

3% 지점 bootstrap 95% CI는 MEntA DV-LOO `[0.50, 0.90]`, S²-MIA `[0.55, 0.95]`, MBA `[0.55, 0.90]`이다. 세 공격 모두 세션이 20개라 불확실성이 크다.

## Locked benign holdout FPR

| 방법 | 목표 FPR | 전체 | NFCorpus | SciDocs | TREC-COVID |
|---|---:|---:|---:|---:|---:|
| Original Mirabel margin | 1% | 0.8% | 1.77% | 0% | 0% |
| Original Mirabel margin | 3% | 0.8% | 1.77% | 0% | 0% |
| Original Mirabel margin | 5% | 2.8% | 4.42% | 1.79% | 0% |
| CLEAN_V1 raw LOO | 1% | 0% | 0% | 0% | 0% |
| CLEAN_V1 raw LOO | 3% | 0.8% | 1.77% | 0% | 0% |
| CLEAN_V1 raw LOO | 5% | 2.0% | 3.54% | 0.89% | 0% |
| DV-LOO V2 | 1% | 0% | 0% | 0% | 0% |
| DV-LOO V2 | 3% | 0.8% | 1.77% | 0% | 0% |
| DV-LOO V2 | 5% | 1.2% | 2.65% | 0% | 0% |

Calibration 250개와 strict `>` 정책 때문에 가능한 FPR이 이산적이다. 따라서 표의 1%/3%/5%는 목표 operating point이고 실제 locked-holdout FPR은 별도로 표시했다.

## Detector ROC-AUC

Pooled attack session을 positive, locked benign holdout query를 negative로 계산했다.

| 방법 | MEntA | S²-MIA | MBA |
|---|---:|---:|---:|
| Original Mirabel margin | 0.778 [0.695, 0.855] | 0.623 [0.495, 0.741] | 0.658 [0.543, 0.767] |
| CLEAN_V1 raw LOO | 0.817 [0.752, 0.877] | 0.619 [0.535, 0.700] | 0.538 [0.446, 0.628] |
| DV-LOO V2 | 0.803 [0.727, 0.873] | 0.702 [0.598, 0.801] | 0.658 [0.549, 0.761] |

## 고정 rho Original Mirabel 참고 결과

논문의 원래 이진 규칙 `rho=0.05`, 즉 `s_max > tau_q`를 그대로 적용하면 정상 holdout FPR은 전체 25.2%였다. 도메인별로 NFCorpus 31.86%, SciDocs 22.32%, TREC-COVID 8.0%였다. 이는 동일 FPR 비교용 결과가 아니다. 동일 FPR 비교에서는 연속 margin을 정상 calibration 데이터로 보정했다.

## Gate

| 조건 | 결과 |
|---|---:|
| DV-LOO holdout FPR@3% ≤ 4% | PASS — 0.8% |
| MEntA TPR@3% ≥ 50% | PASS — 70% |
| MEntA가 Original Mirabel보다 ≥10%p 높음 | **FAIL — 0%p** |
| S²-MIA 저하 ≤5%p | PASS — 정확히 -5%p |
| MBA 저하 ≤5%p | PASS — 0%p |

## 해석

DV-LOO의 성능 대부분은 Original Mirabel direct margin이 제공했다. LOO branch를 max로 결합해도 3% FPR에서 MEntA를 추가 검출하지 못했고, S²-MIA는 5%p 낮아졌다. 따라서 이 개발 cohort에서 Output-LOO를 추가하는 것이 low-FPR detection을 개선한다는 근거는 없다.

이 결과는 생성 후 개인정보 보호 성능을 평가한 결과가 아니다. Phase 1 실패 규칙에 따라 segmenter, NLI, protected generation, fallback 및 Phase 2 specification을 만들지 않았다.
