# BC DualTail Detector V1 Strict — Phase 1 결과

- 최종 판정: `BC_DUALTAIL_V1_NO_TPR_GAIN`
- 성격: development score-only screening; 논문 최종 증거가 아님
- PRECOMMIT SHA-256: `a9fe49e945128efb196e6e5bb337bc836db812dec3337b0b4144fad8a294b2ff`
- 정상 Q5 session GT: `SESSION_LEVEL_BENIGN_FPR_NOT_SUPPORTED`

## 동일 query-level FPR 비교

| Attack | MIRABEL TPR@3% | DualTail TPR@3% | Delta |
|---|---:|---:|---:|
| MEntA | 0.430 | 0.478 | +0.048 |
| MBA | 0.500 | 0.500 | +0.000 |
| RAG-MIA | 0.475 | 0.500 | +0.025 |
| Macro | 0.468 | 0.493 | +0.024 |

위 TPR은 locked benign holdout과 attack query의 empirical ROC를 3%에서 선형 보간한 ranking diagnostic이다. deployment threshold 결과가 아니다.

## Screening gate

- menta_gain_at_least_5pp: `FAIL`
- mba_noninferior_within_5pp: `PASS`
- rag_mia_noninferior_within_5pp: `PASS`
- core3_macro_tpr_improved: `PASS`

## MEntA miss recovery (matched 3% binary case audit)

- MIRABEL miss query: 114
- DualTail 신규 회복 query: 9
- 신규 회복 session: 3
- DualTail이 잃은 MIRABEL TP query: 0
- 한 session이 만든 최대 신규 회복 query: 4

## 해석

Phase 1 실패 시 명세에 따라 generation을 실행하지 않고 retriever-score handcrafted fusion 탐색을 종료한다. 통과 시에만 동결된 MIRABEL Top-1 hide와 Qwen으로 Phase 2를 진행한다.
