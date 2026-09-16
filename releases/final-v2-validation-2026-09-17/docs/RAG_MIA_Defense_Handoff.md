# RAG MIA Defense Handoff

## 1. 현재 후보
Final V2 = Top-4 local concentration `G4=s1-mean(s1..s4)` + benign-only threshold + rank-1 hide + deterministic backfill.

## 5. 결과 계보

### 5.4 k-ablation
A/B benign query embedding은 Core DB 3,000에 다시 투영한 L1 점수다. L2 Gold retrieval 점수와 혼합하지 않는다.

### 5.6 제목형 475
C-Sci 결과는 **사후 추출, 독립 재현 미완**이다. 보조 진단으로만 사용한다.

### 5.8 FinQA 진단
구 문서의 `Core TopiOCQA s1=0.708` 행은 **L2, 무효**다. L1 Core 공격 점수와 L2 utility DB 점수를 비교했으므로 quarantine했다. FinQA member G4 감소 진단 자체는 D 계보로 유지한다.

### 5.10 FinQA hard benign(E) 분리

1. `E-cal -> FinQA 공격`: D-hard(E)와 D가 같은 FinQA DB를 쓰는 동일 코퍼스 매칭 비교. `tables/FINQA_HARD_BENIGN_MATCHED.csv`.
2. `E-cal -> BEIR Core5`: FinQA hard-benign 임계를 L1 공격에 옮긴 cross-corpus threshold transfer. `tables/CROSS_CORPUS_THRESHOLD_TRANSFER.csv`. matched 비교로 부르지 않는다.
3. `E-cal -> E-lock` FPR 1.89%(V2), 2.34%(MIRABEL)는 두 표에 참고로 유지한다.

### 5.13 L1 재실행 격자
A-on-L1, E-cal(D-hard), Mixed를 재계산했다. E-cal 행의 공격 열은 FinQA만 포함하며 Core5 열은 비웠다. 유효 표는 `tables/SIGNAL_CALIBRATION_GRID_L1.csv`다.

### 5.14 계보 감사
L1=`7e68ce0d29fb6be76160461725fa89573313173280bfd0b04b4b95d4891f4d4c`, L2=`4fa9fc359d953064c9fd58d631cf3ebec6e8b988f397943fa3ec3394b83be2a9`, D/D-hard=`0a3883915a6923f2ec252a592184e7944f8ea159a81cc8191e9598fd8cb15975`. E retrieval의 Top-4 8,452개는 L1 ID 0개, FinQA prefix 8,452개로 D-hard 재분류했다. 다른 DB hash를 조인하면 `code/lineage_guard.py`가 예외를 낸다.

## 8. 남은 작업

- **Core 코퍼스 기반 사전등록 hard benign 구축**
- IA blind spot 해결 없이 universal claim 금지
- MPNet 전이 보강
- 개입 정상 subset의 수동 factuality 감사
