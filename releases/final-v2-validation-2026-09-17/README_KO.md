# Final V2 검증 릴리스 (계보 정정본, 2026-09-17)

> 발표 구성은 [`presentation_ready/PRESENTATION_22_SLIDES_KO.md`](presentation_ready/PRESENTATION_22_SLIDES_KO.md), 핵심 표는 [`presentation_ready/PRESENTATION_TABLES.md`](presentation_ready/PRESENTATION_TABLES.md), 실제 입출력은 [`presentation_ready/QUERY_ANSWER_EXAMPLES_FOR_SLIDES_KO.md`](presentation_ready/QUERY_ANSWER_EXAMPLES_FOR_SLIDES_KO.md)에 있다.

## 방법

검색 상위 4개 정규화 유사도 `s1>=s2>=s3>=s4`에서 `G4=s1-mean(s1,s2,s3,s4)`를 계산한다. 정상 질의만으로 low-FPR 임계를 보정하고, `G4>tau`이면 rank-1 문서를 숨긴 뒤 결정적 backfill로 한 번 생성한다. 공격 label, 학습 분류기, LoRA, 세션 상태, 공격별 임계값은 쓰지 않는다.

## Retrieval lineage

| 계보 | 검색 DB | 정상 셋 | 용도 |
|---|---|---|---|
| L1 | BEIR Core DB 3,000 (`7e68ce0d29fb…`) | A/B/C/C-Sci | Core5 탐지·기하 분석 |
| L2 | TopiOCQA utility DB (`4fa9fc359d95…`) | Gold QA | 유틸리티 전용; L1 공격 점수와 직접 조인 금지 |
| D | FinQA DB (`0a3883915a69…`) | D-cal/D-lock | FinQA 공격·정상 전이 |
| D-hard(E) | 같은 FinQA DB | E-cal/E-lock/E-full | FinQA 어려운 정상 질의; Core hard-benign이 아님 |

`E-cal→FinQA`는 **FinQA 동일 코퍼스 매칭 비교**다. `E-cal→BEIR Core5`는 **cross-corpus threshold transfer**이며 matched 비교가 아니다. Core 코퍼스 기반 사전등록 hard-benign은 아직 남은 작업이다.

## 발표에 사용 가능한 핵심 결과

- L1에서 동일 C(BEIR) benign 실제 FPR 2.5%: Core5 TPR MIRABEL 88.66%, Final V2 96.08%.
- Qwen Core5 E-AUC 평균: No Defense 0.907, MIRABEL 0.536, Final V2 0.512.
- Llama Core5 E-AUC 평균: Final V2 0.510.
- L2 Gold QA: Final V2 F1 보존 97.2%, 개입률 3.9%; 개입 subset 손상은 별도 표기.
- D FinQA: E-AUC 평균 MIRABEL 0.563, Final V2 0.552; worst 0.693 vs 0.648.
- D-hard(E) 동일 코퍼스 보정은 `tables/FINQA_HARD_BENIGN_MATCHED.csv`.
- D-hard(E) 임계를 L1 Core5에 적용한 진단은 `tables/CROSS_CORPUS_THRESHOLD_TRANSFER.csv`; matched 결과로 인용 금지.

## L1 재실행(3a–3c)

- A/B를 L1에 재투영한 B의 s1 중앙값: 0.435949.
- 동결 tau를 B-on-L1에 적용한 FPR: G4 0%, MIRABEL M 0%.
- Core member vs B-on-L1 AUC: s1 0.999557, G4 0.997474, M 0.993058.
- 상세 격자: `tables/SIGNAL_CALIBRATION_GRID_L1.csv`.

## Quarantine

L1 Core member 점수와 L2 TopiOCQA 점수를 섞은 네 진단은 `quarantine/`으로 격리했다: (1) 구 격자 A/B/Mixed, (2) MIRABEL AUC 0.6795, (3) 67.9%/14.5% FPR, (4) Core TopiOCQA s1 0.708. 발표·논문·모델 선택에 사용하지 않는다.

## 한계

IA Q1은 Final V2 E-AUC 0.671/0.715로 gate를 통과하지 못했다. MPNet 전이도 약하다. D-hard(E)는 FinQA 계보이므로 BEIR Core hard-benign 근거가 아니다. 따라서 현재 주장은 universal defense가 아니라, 평가한 Core5와 두 생성기에서 강한 stateless low-FPR candidate다.

## 폴더

- `tables/`: 계보 정정 표
- `audits/`: DB·ID·해시 감사와 CSV lineage sidecar
- `quarantine/`: 무효 L1×L2 진단
- `results/finqa-hard-benign/`: FinQA D-hard answer-level 결과
- `presentation_ready/`: 22장 구성, 핵심 표·그림, 실제 query/answer
- `code/lineage_guard.py`: 다른 retrieval DB hash 조인 시 예외
