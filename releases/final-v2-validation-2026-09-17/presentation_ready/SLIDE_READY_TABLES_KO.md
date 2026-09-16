# 슬라이드 복사·붙여넣기용 성능표

## 슬라이드 9 — 같은 오탐률에서 탐지

정상 benign 1,000개에서 실제 FPR 2.5%를 동일하게 맞춘 query-level TPR이다.

| 공격 | MIRABEL | Final V2 | 차이 |
|---|---:|---:|---:|
| MEntA | 84.86% | **94.92%** | +10.06%p |
| MBA | 98.80% | **99.90%** | +1.10%p |
| RAG-MIA | 88.80% | **98.70%** | +9.90%p |
| S²-MIA | 98.00% | **99.87%** | +1.87%p |
| DCMI-Q2 | 72.85% | **87.00%** | +14.15%p |
| **평균** | **88.66%** | **96.08%** | **+7.42%p** |

결론: Final V2는 동일한 정상 오탐 예산에서 다섯 공격 모두 MIRABEL보다 높은 탐지율을 보였다.

## 슬라이드 10 — 실제 생성 후 개인정보 누출

E-AUC는 0.5에 가까울수록 공격자의 member/nonmember 구분이 어렵다.

| 공격 | No Defense | MIRABEL | Final V2 |
|---|---:|---:|---:|
| MEntA | 0.978 | 0.602 | **0.542** |
| MBA | 0.910 | **0.500** | 0.505 |
| RAG-MIA | 0.981 | 0.554 | **0.507** |
| S²-MIA | 0.686 | 0.506 | **0.504** |
| DCMI-Q2 | 0.980 | 0.519 | **0.502** |
| **평균** | **0.907** | **0.536** | **0.512** |

결론: 탐지율 상승이 실제 생성 후 privacy 개선으로 이어졌으며, Final V2 평균 E-AUC는 chance에 더 가까웠다.

## 슬라이드 11 — 정상 QA 유틸리티

| 방법 | 개입률 | Gold F1 | Gold F1 보존 | 답변 보존 Token-F1 | 새 거부 | 새 contradiction |
|---|---:|---:|---:|---:|---:|---:|
| No Defense | 0.0% | 0.2089 | 100.00% | 1.0000 | 0.0% | 0.0% |
| MIRABEL | 5.1% | 0.2061 | 98.65% | 0.9814 | 0.3% | 0.2% |
| Final V2 | **3.9%** | 0.2032 | 97.24% | **0.9805** | 0.7% | 0.4% |

결론: 전체 정상 QA 손상은 제한적이지만, 실제 개입된 39개에서는 답변 보존 F1 0.501, Gold-F1 변화 -0.1477, 새 거부 17.95%로 손상이 집중됐다.

## 슬라이드 12 — 관측 범위 k

L1은 기하 진단이며 최종 전이 성능표가 아니다. 모든 행은 locked benign에서 실제 FPR 2.5%를 맞췄다.

| 방법 | Core5 TPR | IA TPR |
|---|---:|---:|
| G2 | 98.62% | 48.98% |
| G4 | 99.10% | 53.46% |
| G64 | 99.34% | 63.50% |
| G3000 | 99.53% | 75.68% |
| MIRABEL | 98.23% | 45.36% |

결론: Core5는 k=2부터 거의 포화하지만 IA는 넓은 코퍼스 관측이 필요해 Top-4 국소 점수의 blind spot으로 남는다.

## 슬라이드 13 — 합산 TPR이 오도하는 이유

| 탐지기 | Member TPR | Nonmember 개입 | 개입 precision |
|---|---:|---:|---:|
| s1 | **96.69%** | 12.51% | 88.34% |
| Final V2 | 94.60% | **2.92%** | 96.75% |
| MIRABEL | 85.30% | **1.56%** | 97.91% |

결론: s1의 높은 합산 TPR은 nonmember까지 넓게 경보한 결과이며, G4는 member 탐지와 불필요한 개입의 균형이 더 좋다.

### Hard-benign 재보정

| 탐지기 | E-lock FPR | Core5 member TPR | Core5 nonmember 개입 | 개입 precision |
|---|---:|---:|---:|---:|
| Final V2 | **1.89%** | **95.53%** | 3.64% | 96.12% |
| MIRABEL | 2.34% | 81.99% | **1.13%** | **98.32%** |

결론: 다양한 어려운 정상 질문으로 보정해도 V2의 member TPR 우위는 유지됐지만 nonmember 개입은 MIRABEL보다 높았다.

## 슬라이드 14 — FinQA 도메인 전이

새 도메인의 정상 질의만 사용해 threshold를 재보정했다.

| 공격 | No Defense | MIRABEL | Final V2 |
|---|---:|---:|---:|
| DCMI | 0.557 | **0.524** | 0.530 |
| MBA | 0.632 | **0.542** | 0.542 |
| MEntA | 0.764 | 0.693 | **0.648** |
| RAG-MIA | 0.575 | **0.531** | 0.539 |
| S²-MIA | 0.540 | 0.524 | **0.503** |
| **평균** | **0.614** | **0.563** | **0.552** |
| **최악** | **0.764** | **0.693** | **0.648** |

결론: Final V2는 FinQA 평균·최악 privacy를 개선했지만 공격별 완전 우위는 아니며, 숫자 threshold가 아닌 benign-only 보정 절차가 전이됐다.

## 슬라이드 15 — 현재 한계

| IA Q1 | No Defense | MIRABEL | Final V2 | 0.65 gate |
|---|---:|---:|---:|---:|
| First-position | 0.874 | 0.729 | 0.671 | 실패 |
| Fixed-position | 0.833 | 0.771 | 0.715 | 실패 |

- IA target이 Top-4 밖인 비율은 약 22.1%이고, 국소 retrieval concentration이 약하다.
- MPNet의 V2 Core5 공격별 TPR 범위는 23.3~65.8%로 절대 성능이 낮다.
- percentile calibration은 10% 오염에서 Core V2 TPR이 94.60%→44.34%로 붕괴한다.
- 현재 결과는 general candidate를 지지하지만 universal defense를 확정하지 않는다.


## 부록 — 생성기와 retriever 전이

- Llama Final V2 평균 E-AUC 0.510, 공격별 0.500~0.535.
- BGE-m3 V2 Core5 macro TPR 96.06%, GTE-base 87.98%, MPNet-base 50.16%.
- 세 retriever 모두 같은 평가 내에서 V2가 MIRABEL보다 높았지만, MPNet 절대 성능이 낮아 retriever-agnostic 주장은 금지한다.

## 부록 — IDK 외부 부가채널

| 공격 | V2 IDK E-AUC |
|---|---:|
| MEntA | 0.501 |
| DCMI-Q2 | 0.510 |
| MBA | 0.519 |
| RAG-MIA | 0.500 |
| S²-MIA | 0.510 |

Pooled IDK 신호는 No Defense 0.994 → MIRABEL 0.563 → Final V2 0.501로 감소했다.

## 부록 — 지연시간

| 방법 | D=3,000 평균 ms/query | p95 ms | 입력 메모리 |
|---|---:|---:|---:|
| MIRABEL | 0.471009 | 0.471538 | 11.719 MiB |
| Final V2 G4 | 0.000532 | 0.000544 | 0.015 MiB |
| s1 | 0.000146 | 0.000149 | 0.004 MiB |

MIRABEL 선형 외삽은 D=1M에서 약 161.96ms와 3.81GiB다. 이 값은 전수 계산 가정의 CPU microbenchmark이며 전체 RAG latency가 아니다.

## 부록 — 보정 오염 10%

| 도메인 | 방법 | clean TPR | percentile TPR | MAD TPR |
|---|---|---:|---:|---:|
| Core | V2 | 94.60% | 44.34% | 93.21% |
| Core | MIRABEL | 85.30% | 49.12% | 81.24% |
| FinQA | V2 | 54.50% | 28.07% | 51.51% |
| FinQA | MIRABEL | 51.33% | 28.52% | 42.23% |

결론: empirical percentile은 오염을 정상처럼 받아들여 경보가 거의 사라지는 반면, 환경별로 고정한 MAD 계열은 더 강건했다. MAD c는 환경별이므로 universal constant로 주장하지 않는다.

## 부록 — 조립·실패 진단

- V2 alarm 8,015건에서 selected source와 rank-1 불일치 0건.
- 두 개 이상 detector가 hide한 7,645건에서 hidden source 100% 일치, answer hash 불일치 0건.
- 결측 10건은 threshold 근방이며 별도 보고한다.
- Core5에서 경보했지만 target이 rank2~4에 살아남은 decoy exposure는 MEntA 1건뿐이었다.
- 주 미탐은 rank-1 target에서 발생했다: DCMI 11.8%, MEntA 4.88%.
- z합, α가중합, 2D AND는 Core/FinQA/title-like benign에서 일관된 우위를 만들지 못했다.
