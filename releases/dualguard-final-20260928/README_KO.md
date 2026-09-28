# DualGuard 최종 동결 릴리스 — 2026-09-28

## 최종 상태

- 판정: `FINAL_FROZEN_MODEL_PACKAGE_READY`
- 방법: `CLD/G4 + old CF-Clip`
- G4 threshold: `0.0901547372341156`
- CF-Clip gamma: `9.5659109592`
- Gold engine: `SERIAL_KVCACHE`
- 방법/threshold 변경: 없음
- domain-transfer 실행: 없음

## 폴더 구성

- `results/`: 공격 성능표, 정상 utility 표, IA/MEntA 그림, 모든 query/output CSV, 감사 및 최종 보고서
- `public_package/`: 공개 GitHub용 최소 실행 패키지
- `anonymous_submission_package/`: 익명 투고용 재현 패키지
- `provenance/`: 최종 bundle 생성 및 독립 검증 코드

## 핵심 결과

| Attack | No defense | Frozen dual | Reduction |
|---|---:|---:|---:|
| IA Q5 | 0.864721 | 0.709712 | 0.155009 |
| MEntA Q5 | 0.784450 | 0.715550 | 0.068900 |

Single-query 결과와 confidence interval은 `results/FINAL_ATTACK_PERFORMANCE_TABLE.csv`를 사용한다.

## Query/output CSV

- ordinary benign: 50 rows
- conversational benign: 200 rows
- true same-document deep-dive: 154 rows
- IA session attack: 920 rows
- MEntA session attack: 1,000 rows

각 CSV에는 query와 No Defense/CLD/Dual 생성 답변 및 해당되는 retrieval·G4·CF diagnostic이 포함된다.

## 검증

- 독립 검증: `results/INDEPENDENT_VERIFICATION.json`
- clean-directory smoke test: `results/RELEASE_PACKAGE_SMOKE_TEST.json`
- query audit blocking issue: 0
- output audit blocking issue: 0
- 공개/익명 패키지 credential 및 로컬 절대경로 노출: 0

## 제한사항

- Fast engine은 production gold가 아니며 공개 실행 경로에 포함하지 않았다.
- DCMI 결과는 `DCMI-Std-Q2`이며 paper-exact protocol로 과장하지 않는다.
- CF-only IA/MEntA 행은 supplementary lineage이며 exact/fresh primary 결과를 대체하지 않는다.
- domain transfer, 새 adaptive attack, 새 method tuning은 이 릴리스에서 수행하지 않았다.

최종 상세 보고서는 `results/FINAL_FREEZE_AND_RELEASE_REPORT_KO.md`를 참조한다.

FINAL_RECOMMENDATION: package ready
