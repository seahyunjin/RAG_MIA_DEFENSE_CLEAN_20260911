# BC-RRE action-only small 결과

## 최종 판정

`BC_RRE_BENIGN_DAMAGE_FAILED`

이 실험은 동일한 BC-q97 경보와 MIRABEL locator에서 Simple Hide와 1-for-1 reference replacement만 비교한 development screen이다.

## MBA parser 감사

- 기존 all-or-nothing parser를 논문의 fixed-M reconstruction score로 수정했다.
- fuzzy 복구 없이 정확한 mask line만 채점하고, 빠진 mask는 오답으로 계산했다.
- 세 조건 모두 40/40 점수화; 새 생성 0건.

## 정상 오탐 19건 답변 손상

| Action | Changed | New refusal | Preservation F1 | Changed-only F1 |
|---|---:|---:|---:|---:|
| Simple Hide | 10/19 | 5/19 | 0.590 | 0.220 |
| RRE | 11/19 | 6/19 | 0.504 | 0.143 |

A0는 gold가 아니므로 위 값은 utility가 아니라 answer-preservation diagnostic이다.

## 공격별 원 점수 ROC-AUC

| Attack | No Defense | Simple Hide | RRE | RRE 95% CI |
|---|---:|---:|---:|---:|
| MEntA | 0.999 | 0.603 | 0.636 | [0.476, 0.781] |
| MBA | 0.894 | 0.525 | 0.500 | [0.425, 0.575] |
| RAG-MIA | 1.000 | 0.500 | 0.500 | [0.500, 0.500] |

## Hard gate

- Benign: `{'preservation_improved': False, 'new_refusal_reduced': False, 'answer_change_not_increased': False}`
- Privacy: `{'MEntA_noninferior': True, 'RAG_MIA_noninferior': True, 'MBA_noninferior': True}`

## 해석

PASS여도 RRE가 MEntA를 해결했다는 뜻이 아니다. 20/20 development target에서 동일 탐지 coverage를 유지하면서 정상 오탐 답변 손상을 줄였는지만 판정한다.

## 재현성

- Precommit SHA-256: `5d216ebb844679a1007cfb853ad9082560b48833b12fddc1598a8c806172187d`
- RRE answers SHA-256: `bc41bf3c0159f9961ad75fd8e2a332acedda4396414eb128bb141292da49610d`
