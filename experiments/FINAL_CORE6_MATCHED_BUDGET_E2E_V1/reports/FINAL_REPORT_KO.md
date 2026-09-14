# FINAL_CORE6_MATCHED_BUDGET_E2E_V1

- 최종 판정: `CORE6_PROTOCOL_INCOMPLETE`
- 새 공격 쿼리/답변 생성: `0 / 0`
- Final LC 및 이전 생성·검색 artifact: `UNCHANGED`

## DCMI STATUS

`DCMI_SPEC_UNDERDETERMINED`

- 논문 black-box 조건은 6% antonym perturbation과 `Yes=1, No=0`의 원본-교란 응답 차이를 정의한다.
- 공식 저장소 `ef961a1f23d09c62c5fbf69d798e325fb0362e55`의 `perturb.py`는 3%를 사용한다.
- 고정 decoding 존재: temperature=False, top_p=False, seed=False.
- 공식 scorer는 실제 입력 대신 placeholder 배열을 요구한다.
- 따라서 임의 설정으로 paper-exact 숫자를 만들지 않았다.

## IA STATUS

`IA_PAPER_EXACT_UNAVAILABLE` (`IA_REPOSITORY_VARIANT`는 stress-test로만 가능)

- 논문: Q30, GPT-4o 질문, GPT-4o-mini GT, `lambda=5` UNK penalty.
- 공개 현재 bundle: 모든 선택 target이 Q15이며 GT field가 없다.
- Git 전체 이력: result blob 51개, 최대 Q15, Q30 row 0개.
- 공식 selected cohort와 현재 동결 2,000-target cohort의 동일 label ID overlap은 175개뿐이다.
- 따라서 Q15/plain-accuracy 결과를 논문 IA로 승격하지 않았다.

## CORE6 PROTOCOL TABLE

| Attack | 상태 | Main Core6 |
|---|---|---:|
| MEntA | `PAPER_PROTOCOL_READY` | YES |
| MBA | `PAPER_PROTOCOL_READY` | YES |
| RAG-MIA | `PAPER_FAITHFUL_REIMPLEMENTATION` | YES |
| S²-MIA | `PAPER_FAITHFUL_REIMPLEMENTATION` | YES |
| DCMI | `DCMI_SPEC_UNDERDETERMINED` | NO |
| IA | `IA_PAPER_EXACT_UNAVAILABLE` | NO |

## CORE6 DETECTION SAME-FPR

실행하지 않았다. DCMI와 IA가 모두 main-table eligible이 아니므로 Core6 query manifest 자체를 만들지 않았다.

## CORE6 MATCHED-BUDGET E2E

실행하지 않았다. 누락 protocol을 임의로 채우지 않았고, 따라서 same-budget MIRABEL 대 Final LC Core6 수치는 없다.

## MBA MISSINGNESS

누락·malformed mask를 오답으로 계산하고 표본을 하나도 제외하지 않도록 기존 동결 답변을 재계산했다.

| Condition | N | AUC | E-AUC (secondary) | incomplete/malformed | member | nonmember |
|---|---:|---:|---:|---:|---:|---:|
| NO_DEFENSE | 2000 | 0.916822 | 0.916822 | 634 | 67 | 567 |
| ORIGINAL_MIRABEL_SIMPLE_HIDE | 2000 | 0.499380 | 0.500620 | 1220 | 618 | 602 |
| GLOBAL_BC_SIMPLE_HIDE | 2000 | 0.496617 | 0.503383 | 1191 | 616 | 575 |
| FINAL_LC_OUTER_SIMPLE_HIDE | 2000 | 0.496976 | 0.503024 | 1197 | 617 | 580 |

기존 valid-only MBA AUC와 직접 섞으면 안 된다. 위 값은 `N=2,000`, 제외 0의 수정된 primary audit다.

## CORE6 VERDICT

`CORE6_PROTOCOL_INCOMPLETE`

동결 Core4 결과만 보존했다. Core6 broad protection, Gold QA utility pass, universal defense 주장은 모두 금지한다.

## 다음에 필요한 원본 자산

1. DCMI 6% 교란 원본 묶음과 perturbation model revision/decoding.
2. IA Q30 + GPT-4o-mini GT 원본 묶음과 검증 가능한 member/nonmember provenance.
