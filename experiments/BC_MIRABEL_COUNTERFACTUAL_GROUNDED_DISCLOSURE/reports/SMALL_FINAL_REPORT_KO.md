# BC-MIRABEL + Counterfactual Grounded Disclosure Small E2E 최종 보고

## 최종 판정

`BC_CGD_MENTA_FAILED`

BC-CGD는 정상 이용자 개입률을 낮추고 S²-MIA와 MBA 누출을 줄였지만, 필수 MEntA privacy gate와 외부 side-channel gate를 통과하지 못했다. 사전등록 규칙에 따라 Full Core6와 gold-answer QA confirmatory evaluation은 열지 않았다. 이 결과를 성공 모델이나 paper candidate로 부르면 안 된다.

## 재현 조건

- Preflight: 23/23 PASS
- 최종 precommit SHA-256: `80a69b1ff1e5721d815305e54d87a55809c3a6503dcbae70ec3c8f67555220bb`
- BC-MIRABEL primary threshold: `0.1116663235201894`, strict `>`
- 정상 calibration: 250개, 공격 query 사용 0개
- Small evaluation: 정상 holdout 250개 + 공격 query 280개
- MEntA: member/nonmember 각 20세션, 세션당 Q5
- S²-MIA: member/nonmember 각 20세션, Q1
- MBA: 원 40세션 중 source-inconsistent 3세션을 합성 복구하지 않아 37세션만 scorer-valid
- 새 제거-source 답변: 187개
- 유료 API 호출: 0

## BC-MIRABEL 정상 오탐

| Nominal FPR | Threshold | Calibration FPR | Holdout FPR |
|---:|---:|---:|---:|
| 1% | 0.126402 | 0.8% | 0.8% |
| 3% (primary) | 0.111666 | 2.8% | 0.8% |
| 5% | 0.093006 | 4.8% | 2.8% |

Primary holdout false positive는 2/250이며 둘 다 NFCorpus에서 발생했다. NFCorpus 2/113(1.77%), SciDocs 0/112, TREC-COVID 0/25이다. Small 표본이므로 이 값만으로 새 도메인 일반화를 주장할 수 없다.

## 실제 생성 후 privacy

낮을수록 좋은 `E-AUC=max(AUC,1-AUC)`이다.

| 조건 | MEntA | S²-MIA | MBA* |
|---|---:|---:|---:|
| No Defense | 0.9763 | 0.7600 | 0.9342 |
| Original MIRABEL | 0.7588 | 0.6500 | 0.6272 |
| BC-MIRABEL simple hide | 0.8613 | 0.6125 | 0.6579 |
| **BC-CGD** | **0.8850** | **0.5700** | **0.5322** |

\* MBA는 member 19/nonmember 18의 동일 valid cohort 비교다. 3개 원 세션의 mask index가 비연속이어서 네 조건 모두 제외됐으며 가짜 점수를 넣지 않았다.

MEntA hard gate는 세 항목 모두 실패했다.

- 요구 `BC-CGD <= 0.75`: 실제 `0.8850`
- 요구 BC-MIRABEL보다 최소 0.05 개선: 실제로는 `0.0238` 악화
- 요구 Original MIRABEL보다 개선: 실제로 `0.1263` 악화

반대로 direct attack gate는 통과했다. BC-CGD는 S²-MIA에서 Original보다 0.0800, MBA에서 0.0950 낮았다. 그러나 MEntA 필수 gate 실패가 전체 후보를 종료한다.

## 왜 MEntA가 실패했는가

| 조건 | Query alarm | Session any-alarm | Target Retrieval@4 | Effective protection/query |
|---|---:|---:|---:|---:|
| Original MIRABEL | 41.5% | 67.5% | 45.5% | 32.5% |
| BC-MIRABEL | 20.0% | 35.0% | 45.5% | 19.5% |
| BC-CGD | 20.0% | 35.0% | 45.5% | 20.0% |

Benign-only threshold 보정은 정상 오탐을 크게 낮췄지만 MEntA query의 80%, MEntA session의 65%가 safe path로 그대로 통과했다. 위험으로 판정된 query에서 BC-CGD locator Hit@1은 100%였으므로 주 병목은 locator가 아니라 gate coverage와 최종 disclosure다.

BC-CGD의 73개 risk query에서 A0 unit은 259개였다.

- Counterfactual answer에서도 stable: 38개
- 남은 source에 grounded: 11개
- 두 조건을 동시에 만족: **0개**
- 최종 fallback `I don't know.`: **73/73 risk query**

즉 이번 후보는 실제로 안정된 근거 claim을 선택해 공개하지 못하고 모든 risk path를 refusal로 축약했다. 이 때문에 MEntA member/nonmember 차이가 충분히 사라지지 않았고, output disclosure 가설도 지지되지 않았다.

## 정상 답변 보존성

Small에는 gold answer provenance가 없으므로 아래는 QA 정답률이 아니라 A0 답변 보존성이다.

| 조건 | 정상 개입 | 답변 변경 | Answer Preservation Token-F1 | Exact Match | New refusal | Groundedness retention |
|---|---:|---:|---:|---:|---:|---:|
| No Defense | 0.0% | 0.0% | 1.0000 | 1.0000 | 0.0% | 1.0000 |
| Original MIRABEL | 25.2% | 15.2% | 0.8757 | 0.8480 | 8.4% | 0.8369 |
| BC-MIRABEL | 0.8% | 0.8% | 0.9920 | 0.9920 | 0.8% | 1.0000 |
| **BC-CGD** | **0.8%** | **0.8%** | **0.9920** | **0.9920** | **0.8%** | **1.0000** |

전체 평균 hard utility gate는 모두 통과했다. 하지만 반드시 intervention subset을 따로 봐야 한다.

- BC-CGD false-positive 정상 query: 2개
- 두 답변 모두 `I don't know.`로 변경
- subset Answer Preservation Token-F1: 0
- subset Exact Match: 0
- subset new refusal: 100%

따라서 평균 traffic 부담은 작아졌지만, 실제 오탐된 정상 query의 손상은 심각하다. 또한 자동 NLI의 No-Defense 절대 groundedness가 0.0597로 매우 낮아 `retention=1.0`만으로 factuality나 hallucination-free를 주장하면 안 된다.

## 외부 side-channel

사전등록된 답변 관측 feature 여섯 개를 각각 단독 사용했다. BC-CGD의 최악값은 다음과 같다.

- Worst E-AUC: **0.7988** > 0.65 — FAIL
- 원인: MEntA session의 total/mean answer length
- S²-MIA exact-IDK indicator: 0.6750 — 역시 0.65 초과
- MBA worst: 0.5013

Internal alarm/intervention flag는 약속대로 외부 hard gate에서 제외하고 diagnostic으로만 보존했다.

## Gate 요약

| Gate | 결과 |
|---|---|
| MEntA privacy | **FAIL** |
| S²/MBA robustness | PASS |
| 정상 전체 평균 utility | PASS |
| 외부 side-channel | **FAIL** |
| Full Core6 open | NO |
| Gold-answer QA >=1,000 open | NO |

## 허용되는 결론

- 정상 데이터만 이용한 BC-MIRABEL 보정은 이 holdout에서 Original MIRABEL의 높은 정상 개입률을 25.2%에서 0.8%로 낮췄다.
- BC-CGD final answers는 S²-MIA와 valid MBA cohort에서 Original MIRABEL보다 낮은 E-AUC를 보였다.
- 그러나 필수 MEntA와 외부 answer-length side-channel을 해결하지 못했고, risk path가 전부 refusal로 붕괴했다.
- 따라서 BC-CGD는 최종 방어 모델이 아니며 실패 원인을 보여주는 ablation/mechanism result로만 보존한다.
