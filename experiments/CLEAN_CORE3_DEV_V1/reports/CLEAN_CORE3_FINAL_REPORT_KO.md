# CLEAN CORE3 개발 진단 최종 보고서

## 판정

`CLEAN_CORE3_DEV_V1_DIAGNOSTIC_COMPLETE`

이 결과는 **MEntA·MBA·RAG-MIA 개발 공격 3종**의 병목 지도이다. S²-MIA·DCMI·IA는 포함하지 않았고, general/universal 방어 성공을 뜻하지 않는다.

## 정상 질의 오탐

| Operating point | Calibration target | Locked holdout FPR |
|---|---:|---:|
| ORIGINAL | 5.0% | 33.6% (168/500) |
| BC_Q99 | 1.0% | 1.2% (6/500) |
| BC_Q97 | 3.0% | 3.8% (19/500) |
| BC_Q95 | 5.0% | 6.4% (32/500) |

## End-to-end 개인정보 누출

아래는 각 공격 논문의 복구된 원 점수로 계산한 native ROC-AUC이다. 0.5에 가까울수록 member/nonmember 구분이 어렵다.

| Attack | No Defense | Original MIRABEL | BC-MIRABEL q97 | BC valid N | BC bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| MEntA | 0.999 | 0.506 | 0.603 | 40/40 | [0.442, 0.750] |
| MBA | 0.904 | 0.583 | 0.583 | **15/40** | [0.361, 0.778] |
| RAG-MIA | 1.000 | 0.500 | 0.500 | 40/40 | [0.500, 0.500] |

MBA는 exact output parser가 정의되지 않은 응답을 임의의 0점으로 채우지 않고 fail-close했다. 따라서 MBA의 AUC는 valid subset 기준이며, 25/40개가 invalid인 BC 결과를 완전한 공격 방어 성공으로 해석하면 안 된다.

## Detect → Locate → Protect (BC q97)

| Attack | Member Retrieval@4 | Alarm | Locator Hit@1 | EPO |
|---|---:|---:|---:|---:|
| MEntA | 100.0% | 43.5% | 100.0% | 87.0% |
| MBA | 100.0% | 50.0% | 100.0% | 100.0% |
| RAG-MIA | 100.0% | 50.0% | 100.0% | 100.0% |

## 병목

| Attack | Label | 수치 근거 |
|---|---|---|
| MEntA | MIXED | Retrieval@4=1.000, alarm|retrieved=0.870, locator=1.000, AUC=0.603 |
| MBA | MIXED | Retrieval@4=1.000, alarm|retrieved=1.000, locator=1.000, AUC=0.583 |
| RAG-MIA | MIXED | Retrieval@4=1.000, alarm|retrieved=1.000, locator=1.000, AUC=0.500 |

## BC q97 정상 오탐 피해

- Locked holdout에서 오탐: **19/500 (3.8%)**
- 오탐 subset 답변 보존 Token-F1 진단: **0.590**
- 새 refusal: **5/19**
- 실제 답변 변경: **10/19**; 변경된 10개만의 보존 Token-F1은 **0.220**
- gold answer가 없으므로 정확도/utility라고 부르지 않았다.

## 과학적 결론

- 동결된 계층 규칙상 세 공격은 모두 `MIXED`다. 그러나 원인은 동일하지 않다: MEntA는 BC가 member 질의의 13%를 놓치고 5개 모두 제거한 member session이 65%뿐인 coverage 문제가 남는다. RAG-MIA는 hard hide만으로 AUC 0.5가 됐다. MBA는 parser-valid coverage 부족으로 확정할 수 없다.
- Original MIRABEL은 이 새 corpus의 정상 holdout에서 높은 오탐을 재현했다. BC 보정은 경보 빈도를 크게 줄였지만, 최종 privacy는 반드시 공격별 native AUC와 함께 해석해야 한다.
- simple top-1 hide가 충분하다고 확실히 말할 수 있는 것은 이 개발 표본의 RAG-MIA다. MEntA의 점 추정치는 0.65 이하지만 CI 상한이 0.750이고, MBA는 valid N가 부족하다.
- BC q97은 전체 정상 요청의 개입률은 3.8%로 낮췄지만, 실제 오탐 19개 안에서는 새 refusal 26.3%와 큰 답변 변화가 나타났다. 따라서 traffic 오탐 문제는 완화했으나 정상 답변 손상 문제까지 해결한 것은 아니다.

## 다음 방향

**한 가지 권고: detector를 더 복잡하게 만들기보다, BC q97 operating point는 유지한 채 오탐 시 정상 답변 손상을 줄이는 protection action을 다음 연구 대상으로 삼는다.** 새 구조는 이 campaign에서 자동 구현하지 않는다.

## 재현성

- Precommit SHA-256: `379f9f073b16c6f624a593420ce7bdc894a2898d337129a269c3164e9fec5373`
- Retrieval cache SHA-256: `56f20d4105ea9c4c9ef22dc6767d336687522ff70c7a43ac7bb73294f8c92ff6`
- Generated answers SHA-256: `90a06d14a6f00c643ab9c949b96907fee91ad7b0447f2a710ccfdf615b89e0ac`
- 세 공격은 development set이며 unseen confirmation으로 주장하지 않는다.
