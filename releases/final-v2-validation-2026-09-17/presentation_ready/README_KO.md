# Final V2 발표용 계보 정정 패키지

이 폴더에는 발표 본문에 필요한 소수의 표·그림과 실제 공격 입력/생성 답변만 정리했다.

## 한 줄 결론

Final V2는 C(BEIR) benign의 실제 FPR을 2.5%로 맞춘 L1 Core5에서 MIRABEL보다 높은 평균 TPR(96.08% vs 88.66%)과 더 낮은 Qwen E2E 평균 E-AUC(0.512 vs 0.536)를 보였다. FinQA에서는 정상 질의만으로 재보정할 때 평균 E-AUC 0.552, 최악 0.648이었다. IA Q1, MPNet, 개입 정상 subset 손상이 남아 universal·hallucination-free 주장은 하지 않는다.

## 반드시 분리할 계보

- **L1:** BEIR Core DB 3,000. Core5 탐지·기하.
- **L2:** TopiOCQA 자체 DB. Gold QA utility 전용.
- **D:** FinQA 공격·정상.
- **D-hard(E):** FinQA DB에서 검색한 어려운 정상 2,113개. Core hard-benign이 아니다.

`E-cal→FinQA`는 FinQA 동일 코퍼스 비교다. `E-cal→Core5`는 cross-corpus threshold transfer이며 matched 성능으로 말하지 않는다.

## 발표 시작점

1. `PRESENTATION_22_SLIDES_KO.md` — 정확히 22장 구성
2. `PRESENTATION_TABLES.md` — 본문에 넣을 7개 표만 선별
3. `FIGURE_INDEX_KO.md` — 본문 그림 9개
4. `PPT_ATTACK_QUERY_ANSWER_EXAMPLES_KO.md` — 6공격 축약 사례
5. `QUERY_ANSWER_EXAMPLES_FOR_SLIDES_KO.md` — 동결 원문 사례
6. `figures/plots_png/FIG21_SIX_ATTACK_G4_HISTOGRAM.png` — IA 포함 6공격 분포

## 중요 정정

- Gold F1 보존율은 97.24%; 98.05%는 answer-preservation Token-F1이다.
- D-hard(E) 재보정의 E-lock FPR은 V2 1.89%, MIRABEL 2.34%다.
- 이를 L1 Core5에 적용한 95.53%/81.99%는 cross-corpus transfer이며 동일 코퍼스 matched 비교가 아니다.
- 0.6795 AUC, 67.9%/14.5% FPR, Core TopiOCQA s1=0.708, 구 A/B/Mixed 격자는 L1×L2 혼합으로 `quarantine/`에 있다.

## 주장 금지

- 모든 공격·retriever·도메인의 universal defense
- IA 해결
- 숫자 임계 자체의 zero-shot 전이
- 정상 답변 손상·환각이 없음
