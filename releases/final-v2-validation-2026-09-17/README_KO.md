# Final V2 검증 릴리스 (2026-09-17)

> 발표용 최신 확정 표·PNG/PDF는 [presentation_ready/](presentation_ready/README_KO.md)에 정리했다. 이 폴더는 계보 불일치 진단을 제외하고, Gold-F1 보존율과 answer-preservation을 구분해 표기한다.

이 폴더는 RAG 멤버십 추론 방어의 동결 후보인 **Top-4 Local Retrieval Concentration Defense**(내부명 Final V2/G4/LRC-4)의 재현 코드, 핵심 수치, 그림과 해석을 모은 발표·논문용 릴리스다.

## 방법

검색된 상위 4개 문서의 정규화 유사도를 `s1 >= s2 >= s3 >= s4`라고 할 때 다음 점수를 사용한다.

`G4(q) = s1 - mean(s1, s2, s3, s4)`

정상 질의만으로 임계값을 정해 목표 정상 오탐률을 2.5%로 맞춘다. `G4 > threshold`이면 현재 질의의 rank-1 문서를 제거하고 결정적으로 다음 문서를 보충한 뒤 생성기를 한 번 실행한다. 공격 family label, 학습 분류기, LoRA, 세션 상태와 공격별 임계값은 사용하지 않는다.

## 핵심 결과

### 1. 동일 정상 FPR 2.5%에서 Core5 query-level TPR

| Attack | MIRABEL | Final V2 | Delta |
|---|---:|---:|---:|
| MEntA Q5 | 84.86% | **94.92%** | +10.06%p |
| MBA | 98.80% | **99.90%** | +1.10%p |
| RAG-MIA | 88.80% | **98.70%** | +9.90%p |
| S²-MIA | 98.00% | **99.87%** | +1.88%p |
| DCMI-Std-Q2 | 72.85% | **87.00%** | +14.15%p |
| Macro mean | 88.66% | **96.08%** | +7.42%p |

이 표는 정상 오탐률을 동일하게 맞춘 apples-to-apples 비교다. 따라서 “FPR이 더 낮다”가 아니라 “같은 FPR에서 TPR이 높다”로 해석한다.

### 2. Qwen 실제 생성 후 공격자 성능

0.5는 무작위 추측에 해당하며 낮을수록 개인정보 보호가 강하다. 각 공격의 동결 native scorer를 사용했다.

| Attack | No Defense | MIRABEL | Final V2 |
|---|---:|---:|---:|
| MEntA | 0.978 | 0.602 | **0.542** |
| MBA | 0.910 | **0.500** | 0.505 |
| RAG-MIA | 0.981 | 0.554 | **0.507** |
| S²-MIA | 0.686 | 0.506 | **0.504** |
| DCMI-Std-Q2 | 0.980 | 0.519 | **0.502** |
| Macro mean | 0.907 | 0.536 | **0.512** |
| Worst attack | 0.981 | 0.602 | **0.542** |

MBA에서는 MIRABEL이 미세하게 낮지만 두 방법 모두 사실상 chance 수준이다. Final V2는 Core5 평균과 최악값에서 가장 낮다.

### 3. Generator transfer

| Generator | No Defense mean | Final V2 mean | Final V2 worst |
|---|---:|---:|---:|
| Qwen | 0.907 | **0.512** | 0.542 |
| Llama | 0.890 | **0.510** | 0.535 |

동일 방어 규칙은 Qwen과 Llama에서 유사한 결과를 냈다. 생성기 두 개만 평가했으므로 generator-universal 주장은 하지 않는다.

### 4. FinQA 정상-only 재보정 전이

| Condition | Mean attacker performance | Worst attack |
|---|---:|---:|
| No Defense | 0.614 | 0.764 |
| MIRABEL | 0.563 | 0.693 |
| Final V2 | **0.552** | **0.648** |

Final V2는 평균과 최악값을 개선했지만 공격별로 모두 MIRABEL보다 좋지는 않았다. 숫자 threshold 자체는 도메인 불변이 아니며, 새 도메인에서는 정상 질의만 이용한 재보정이 필요하다.

### 5. Gold 정상 QA

| Method | Intervention | Gold F1 | F1 retention | New refusal |
|---|---:|---:|---:|---:|
| No Defense | 0.0% | 0.2089 | 100% | 0.0% |
| MIRABEL | 5.1% | 0.2061 | 98.6% | 0.3% |
| Final V2 | **3.9%** | 0.2032 | **97.2%** | 0.7% |

전체 개입률은 낮지만, Final V2가 실제로 개입한 정상 질의의 조건부 utility 변화는 약 -14.8%p였다. 전체 평균만으로 정상 사용자 피해가 없다고 주장할 수 없다.

## Constructed Hard-Benign V1

Detector score를 보기 전에 FinQA 정상 문서와 legitimate information need만으로 만든 독립 constructed benchmark다. 외부 표준 benchmark가 아니다. 2,150개를 목표로 했으며 중복 padding 없이 유효한 2,113개를 동결했다.

Natural subset의 절대 Gold-F1은 No Defense부터 0.0071로 매우 낮아 reference/자유생성 불일치의 영향을 받는다. 따라서 이 constructed benchmark의 주 utility 근거는 절대 Gold-F1이 아니라 paired answer-preservation, new refusal과 NLI proxy다.

| Metric | MIRABEL | Final V2 |
|---|---:|---:|
| Hard-benign macro intervention/FPR | **2.46%** | 3.41% |
| Answer-preservation F1 | **98.48%** | 97.41% |
| New-refusal rate | **1.18%** | 2.22% |
| Discrete-query intervention | **1.63%** | 3.26% |

Final V2의 약점은 Exact-Fact FPR 3.78%, Yes/No FPR 5.59%, Deep-study FPR 3.67%였다. 변경된 90개 질의에 대한 NLI proxy에서는 unsupported-sentence rate가 평균 0.518에서 0.815로 증가했다. 이 값은 hallucination ground truth가 아니며, 동결된 50개 수동 감사 template은 `REQUIRES_HUMAN_REVIEW`로 남겨 두었다.

### 해석

- Core5 저오탐 탐지와 end-to-end privacy는 강하다.
- Qwen/Llama 전이는 지지된다.
- FinQA에서는 정상-only calibration protocol이 필요하다.
- BGE/GTE에서는 강하지만 MPNet 절대 성능이 약하므로 retriever-agnostic 주장은 불가하다.
- 어려운 정상 질문에서는 MIRABEL보다 더 많은 개입·거부·답변 변형이 발생한다.
- 따라서 현재 결과는 **유망한 stateless low-FPR defense candidate**를 지지하지만, universal 또는 hallucination-free defense를 확정하지 않는다.

## 폴더

- `code/`: 최종 검증과 hard-benign 평가 코드
- `configs/`: 동결 방법·lineage·local-only 실행 정책
- `results/core/`: Core5, generator/retriever transfer, FinQA, utility, ablation CSV
- `results/hard-benign/`: constructed hard-benign FPR, utility, HIR, factuality proxy
- `data/`: 동결 query bank와 retrieval score/decision artifact; 원문 문서·전체 생성 답변 제외
- `figures/`: 발표·논문용 핵심 그림
- `docs/`: 방법, 주장 감사, 한계와 발표 개요

## 재현 경계

코드는 원 저장소의 동결 input artifact와 로컬 모델 snapshot을 기대한다. 모델 가중치, 원시 생성 cache, API key와 개인정보 가능성이 있는 전체 답변은 이 릴리스에 포함하지 않았다. 모든 마지막 검증은 무료 로컬 모델만 사용했고 유료 API 호출은 없었다.
