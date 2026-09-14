# 2026-09-11 정리 결과

## 영구 삭제 완료

- `LoRA-mirabel-Decter`: 약 12 GB
- `AD-test-LLM3`: 약 16 GB
- `AD-test-LLM2`: 618 MB
- 기존 `LLM_pdf`의 발표자료, 그림, 압축본, CSV, 대본 등 비논문 자료: 약 167 MB

위 삭제 항목은 복구용 휴지통으로 이동하지 않고 영구 삭제했다.

## 보존 완료

- 실험 290개 로드맵과 판정/해시
- 최소 재현 코드
- MEntA Q5 세션 입력 3개 데이터셋
- 관련 논문 5편: MEntA, Mirabel, DCMI, IA-MIA, MBA
- `LLM_pdf`에는 위 논문 5편만 남김

## 디스크 점유 진단

`/home/traffic_3/workspace`에서 현재 사용자에게 보이는 파일 합계는 약 188 GB이다.

- `workspace/SH`: 124 GB
  - `ReNode-L`: 71 GB
    - `xxltrafficdata`: 70 GB
  - `UNIFIED_TRAFFIC_BENCHMARK_20260727`: 42 GB
- `.cache`: 39 GB
  - Hugging Face cache: 35 GB
- `miniconda3`: 25 GB
  - conda environments: 23 GB
- `workspace/AD-test-LLM`: 1.4 GB

파일시스템 전체 사용량 2.3 TB와 위 188 GB의 차이는 같은 볼륨의 다른 경로/사용자 파일 등으로, `/home/traffic_3/workspace` 하위만 계산한 값과 직접 일치하지 않는다.

## 주의

ReNode-L과 교통 벤치마크는 현재 RAG-MIA 연구와 별개로 보이지만 다른 연구 자산일 수 있어 삭제하지 않았다. Hugging Face 모델 캐시도 향후 Qwen/Llama/BGE/MPNet 재현에 필요할 수 있어 삭제하지 않았다.
