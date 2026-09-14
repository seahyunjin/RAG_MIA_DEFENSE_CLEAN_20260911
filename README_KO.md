# RAG MIA 방어 연구 정리본

이 디렉터리는 2026-09-11 기준 대용량 실험 artifact를 정리하기 전에 만든 코드 전용 보존본이다.

## 남긴 것

- `EXPERIMENT_ROADMAP_ALL_20260911_KO.md`: 290개 실험 디렉터리의 판정·핵심 지표·최종 JSON SHA-256
- `code/qll_final_model`: 동결 QLL source-hide 코드
- `code/qll_release`: QLL 방어와 주요 baseline/evaluation 코드
- `code/paper_evaluation`: Mirabel/QLL 동일 조건 및 core-6 평가 코드
- `code/ad_mirabel_core`: AD-Mirabel 핵심 Python 모듈
- `code/menta_official`: MEntA 공식 공격 및 Mirabel 구현
- `data/menta_q5_sessions`: 3개 BEIR 데이터셋의 MEntA Q5 세션 입력
- `papers`: MEntA, Mirabel, DCMI, IA-MIA, MBA 원 논문 PDF
- `CODE_MANIFEST_SHA256.txt`: 보존 파일 무결성 목록

## 남기지 않은 것

- 생성 답변 및 대용량 raw/private 데이터(MEntA Q5 재현 입력만 예외)
- model checkpoint와 response cache
- SQLite cache
- CSV/JSON 실험 결과 원본(핵심 수치와 최종 JSON 해시는 로드맵에 기록)
- PNG/PDF/SVG/JPG figure
- PPT 렌더 이미지와 중간 발표 산출물
- Python/pytest cache

따라서 이 폴더는 **코드와 실험 의사결정의 최소 보존본**이다. 삭제된 데이터와 생성 답변을 재현하려면 원 데이터셋과 로컬 모델을 다시 준비해야 한다.

## 검증

- 전체 보존 Python 코드 syntax compile: PASS
- 동결 QLL final model unit tests: 6 passed
- 보존 코드에는 figure 파일이 없어야 한다.

새 실험은 로드맵의 `2026-09-11 이후 고정 연구 목표`를 따른다. 핵심은 8/8 universal 주장보다 MEntA·S²-MIA·MBA를 포함한 stealth 공격의 실제 생성 후 누출 완화, 낮은 정상 이용자 손상, 환각 증가 없음이다.
