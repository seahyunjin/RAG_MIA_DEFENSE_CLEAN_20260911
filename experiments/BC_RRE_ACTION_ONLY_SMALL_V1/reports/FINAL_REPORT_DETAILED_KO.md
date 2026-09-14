# BC-RRE action-only small 최종 상세 보고서

## 최종 판정

`BC_RRE_BENIGN_DAMAGE_FAILED`

RRE는 세 공격의 개인정보 비열등성 기준을 모두 통과했지만, BC-q97이 오탐한 정상 질의의 답변 손상을 줄이지 못했다. 사전 고정한 hard-stop에 따라 RRE 계열 탐색을 종료했으며 100/100 확대 실험은 열지 않았다.

## Part A — MBA parser 감사

기존 scorer는 5개 mask를 모두 출력하지 않은 답변 전체를 unscorable로 제외했다. 그러나 MBA 논문의 원 점수는 `정확히 복원한 mask 수 / M`이다. 따라서 새 scorer는 정확한 `[Mask_i]: answer` 줄만 읽고, 빠졌거나 형식이 틀린 mask는 고치지 않고 오답으로 계산한다. fuzzy regex, 의미 유사도, 추정값은 사용하지 않았다.

- 기존 BC 유효: 15/40
- 기존 BC invalid 25건 분류: `PARSER_IMPLEMENTATION_BUG` 25, `MODEL_OUTPUT_INVALID` 0, `PROTOCOL_MISMATCH` 0
- 출력 형식 완전 준수: BC 14/40 (형식 준수율은 별도 진단이며 score 포함 여부를 결정하지 않음)
- 수정 후 원 점수 산출: 세 조건 모두 40/40
- 기존 답변 재사용, 새 생성: 0건
- scorer unit test: 6/6 PASS

| 조건 | 기존 valid | 수정 후 valid | Member 평균 | Nonmember 평균 | Native ROC-AUC |
|---|---:|---:|---:|---:|---:|
| No Defense | 25/40 | 40/40 | 0.600 | 0.010 | 0.894 |
| Original MIRABEL | 15/40 | 40/40 | 0.020 | 0.010 | 0.525 |
| BC-MIRABEL q97 Simple Hide | 15/40 | 40/40 | 0.020 | 0.010 | 0.525 |

## Part B — reference pool 감사

현재 workspace에는 기존에 정의된 독립 reference pool이 없었다. 결과를 보기 전에 공개 BEIR SciFact corpus 전체를 쓰는 규칙을 고정하고 archive를 받은 뒤 오염을 검사했다. overlap을 발견한 뒤 문서를 제거하거나 다른 salt/corpus를 재탐색하지 않았다.

- Reference: BEIR SciFact 전체 5,183문서
- Protected DB: 3,000문서
- ID overlap: protected/member/nonmember 모두 0
- NFKC·lowercase·공백 정규화 text duplicate: protected/member/nonmember 모두 0
- BGE-M3 reference Top-20 후 첫 문서 선택
- Replacement availability: 146/146 (100%)
- Exact/near-duplicate diagnostic: 0/146, 0/146

## Part C — front-end 동일성

Simple Hide와 RRE는 BC-q97 detector 및 MIRABEL locator가 완전히 동일하다.

- q97 risk IDs: 146/146 동일 (공격 127, 정상 오탐 19)
- safe IDs: 634/634 동일
- 제거 source ID: 146/146 동일
- replacement는 제거된 정확한 slot에 삽입: 146/146
- 다른 세 source의 순서와 Top-4 source 수 유지: 146/146
- safe 질의 새 생성: 0
- RRE 새 생성: 146

## Part D — 정상 오탐 답변 손상

| Action | 답변 변경 | 새 refusal | Answer-preservation Token-F1 | 변경 답변만 F1 | 평균 답변 길이(단어) |
|---|---:|---:|---:|---:|---:|
| Simple Hide | 10/19 | 5/19 | 0.590 | 0.220 | 15.37 |
| RRE | 11/19 | 6/19 | 0.504 | 0.143 | 17.16 |

RRE는 보존 F1을 `-0.086` 낮췄고, 답변 변경과 새 refusal을 각각 1건 늘렸다. A0는 gold answer가 아니므로 위 값은 QA utility/정확도가 아니라 **ANSWER_PRESERVATION_DIAGNOSTIC**이다.

## Part E — 공격별 개인정보 누출

각 공격에서 복구된 논문 원 점수로 계산한 native ROC-AUC이다. E-AUC/symmetric AUC를 primary로 쓰지 않았다.

| 공격 | No Defense | Simple Hide | RRE | RRE bootstrap 95% CI | RRE 비열등성 |
|---|---:|---:|---:|---:|---|
| MEntA Q5 | 0.999 | 0.603 | 0.636 | [0.476, 0.781] | PASS (`+0.034 <= +0.05`) |
| MBA Q1 | 0.894 | 0.525 | 0.500 | [0.425, 0.575] | PASS |
| RAG-MIA Q1 | 1.000 | 0.500 | 0.500 | [0.500, 0.500] | PASS |

표본은 attack별 member 20/nonmember 20의 development screen이다. 특히 MEntA CI가 넓으므로 최종 논문 성능이나 generalization으로 해석할 수 없다.

## Part F — 외부 side-channel 진단

MEntA에서 최종 답변으로 관측 가능한 단독 feature만 평가했다. 이 값은 secondary diagnostic이며 primary privacy metric이 아니다.

| 조건 | 길이 E-AUC | 문장 수 E-AUC | refusal/IDK 수 E-AUC | empty E-AUC |
|---|---:|---:|---:|---:|
| No Defense | 0.905 | 0.886 | 1.000 | 0.500 |
| Simple Hide | 0.613 | 0.591 | 0.659 | 0.500 |
| RRE | 0.618 | 0.601 | 0.683 | 0.500 |

RRE의 refusal/IDK 패턴 E-AUC 0.683은 Simple Hide 0.659보다 악화했다. 이는 hard gate가 아니라 추가 위험 신호다.

## Part G — 사전 고정 hard gate

### 정상 오탐 손상

- Preservation F1가 Simple 0.590보다 +0.15 개선 또는 0.75 이상: **FAIL**
- 새 refusal이 5/19보다 감소: **FAIL** (6/19)
- 답변 변경 수가 증가하지 않음: **FAIL** (11 > 10)

### 개인정보 비열등성

- MEntA가 Simple보다 +0.05 초과 악화하지 않음: **PASS**
- RAG-MIA가 0.500보다 +0.05 초과 악화하지 않음: **PASS**
- MBA가 Simple보다 +0.05 초과 악화하지 않음: **PASS**

## 과학적 결론

관련성이 높은 독립 reference 문서를 빈 slot에 채우는 것만으로는 false-positive 답변 손상이 복구되지 않았다. 오히려 네 문서 water-filling으로 기존 세 문서의 visible token 몫이 줄고, 새 reference evidence가 답변을 바꾸거나 refusal을 늘릴 수 있다. 개인정보 비열등성은 유지됐지만 이 실험의 주목적인 정상 답변 손상 완화에는 실패했다.

명세에 따라 reference Top-k, replacement 수, blending, weighted context, query rewriting, QLL/LOO, threshold를 추가 탐색하지 않는다.

## 재현성

- Main precommit SHA-256: `5d216ebb844679a1007cfb853ad9082560b48833b12fddc1598a8c806172187d`
- RRE answers SHA-256: `bc41bf3c0159f9961ad75fd8e2a332acedda4396414eb128bb141292da49610d`
- Final integrity audit: `FINAL_AUDIT_PASS`
- Parent CLEAN_CORE3 artifact hashes: 모두 불변
- Scoring 시작 전 import-path 오류는 `PRECOMMIT_IMPLEMENTATION_ADDENDUM_01`에 기록했으며 scorer 수식·답변·detector에는 변화가 없다.
