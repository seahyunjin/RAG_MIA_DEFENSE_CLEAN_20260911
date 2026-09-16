# Final V2 발표용 확정 결과 패키지

이 폴더는 발표 슬라이드에 바로 붙일 수 있도록 확정된 계보의 성능표와 피규어만 정리한 패키지다.

## 한 줄 결론

Final V2는 정상 오탐률을 2.5%로 동일하게 맞춘 Core5에서 MIRABEL보다 높은 평균 탐지율(96.08% vs 88.66%)과 더 낮은 Qwen E2E 평균 E-AUC(0.512 vs 0.536)를 보였다. FinQA에서도 정상 질의만으로 재보정하면 평균 E-AUC 0.552, 최악 0.648을 달성했다. 다만 IA Q1, MPNet의 낮은 절대 탐지율, 개입된 정상 질의에 집중되는 utility 손상이 남아 있으므로 universal·hallucination-free 주장은 하지 않는다.

## 계보

- Core/BGE: 동결 benign 1,000에서 실제 FPR 2.5%를 맞춘 Core5 비교.
- E hard-benign: 사전등록 2,113개를 calibration 1,000 / locked 1,113으로 고정 분할. 공격 샘플 없이 방법별 threshold를 보정.
- D FinQA: 독립 도메인에서 benign-only 재보정 후 E2E 평가.
- L1 local geometry: Core 3,000문서 DB에 동결 query embedding을 투영한 CPU 진단. 최종 전이 증거가 아니라 observation-range mechanism 분석이다.
- Score-independent: 생성기 전이, 지연시간, 답변 조립 및 target-rank 감사.

## 중요 정정

- Gold F1 0.2089 → 0.2032의 보존율은 97.24%다.
- 98.05%는 Gold F1 보존율이 아니라 Answer Preservation Token-F1이다.
- 개입된 정상 39개의 Answer Preservation Token-F1은 약 0.501이고, Gold-F1 변화는 평균 -0.1477, 새 거부는 17.95%다.
- hard-benign V2 1.89% / MIRABEL 2.34%는 E-calibration 1,000에서 각각 재보정한 뒤 E-lock 1,113에서 측정한 결과다. 기존 3.41% / 2.46%는 다른 동결 threshold를 그대로 적용한 결과이므로 같은 표에서 섞지 않는다.

## 제외한 수치

다음은 계보 불일치가 확인되어 이번 패키지에 넣지 않았다.

- Topi/Mixed calibration grid
- AUC 0.68
- 67.9% / 14.5%
- Core TopiOCQA s1 = 0.708
- TopiOCQA Gold retrieval은 Core 3,000문서 DB와 문서 ID가 겹치지 않아 L1 결과와 직접 비교하지 않는다.

## 폴더

- tables/: 슬라이드별 최소 성능표 CSV
- source_tables/: 표의 근거가 된 소형 원본 CSV/JSON 복사본
- figures/table_png/: 표를 300dpi PNG로 렌더링한 파일
- figures/plots_png/: 발표용 고해상도 PNG
- figures/plots_pdf/: 논문·벡터 편집용 PDF
- SLIDE_READY_TABLES_KO.md: 복사·붙여넣기용 표와 한 줄 결론
- FIGURE_INDEX_KO.md: 그림별 권장 슬라이드와 주의사항
- SOURCE_MANIFEST_SHA256.csv: 원본 결과 파일 경로·SHA-256
- tools/build_slide_assets.py: 동일 패키지를 다시 생성하는 빌더

## 발표 주장 범위

가능한 주장:

- 같은 정상 FPR에서 Core5 탐지율 우위
- Qwen/Llama 두 생성기에서 E2E privacy 개선
- FinQA에서 benign-only calibration protocol 전이
- hard-benign 재보정에서 V2의 높은 member TPR과 낮은 locked FPR
- 코퍼스 전수 점수가 필요한 MIRABEL보다 Top-4 점수만 쓰는 V2의 낮은 추가 지연

금지할 주장:

- 모든 공격·retriever·도메인에 보편적으로 동작
- IA 해결
- 숫자 threshold 자체의 zero-shot 도메인 전이
- 정상 답변 손상이나 환각이 없음
