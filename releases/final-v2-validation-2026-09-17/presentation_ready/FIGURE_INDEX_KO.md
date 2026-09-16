# 피규어 인덱스

PNG는 슬라이드 삽입용, PDF는 논문·벡터 편집용이다.

| 파일 | 권장 위치 | 핵심 메시지 |
|---|---|---|
| FIG01_METHOD_OVERVIEW | 방법론 | Top-4 G4 점수, benign-only 보정, rank-1 hide |
| FIG02_LOW_FPR_DETECTION | 슬라이드 9 | 같은 저오탐에서 V2 탐지율 우위 |
| FIG03_MEMBER_NONMEMBER | 슬라이드 13 | member TPR과 nonmember 개입을 분리 |
| FIG04_QWEN_E2E_PRIVACY | 슬라이드 10 | 실제 생성 후 Core5 E-AUC |
| FIG05_GENERATOR_TRANSFER | 부록 | Qwen/Llama 전이 |
| FIG06_RETRIEVER_TRANSFER | 부록/한계 | BGE·GTE 우위, MPNet 약점 |
| FIG07_BENIGN_UTILITY | 슬라이드 11 | 전체 정상 utility와 개입 subset 손상 |
| FIG08_FINQA_TRANSFER | 슬라이드 14 | 독립 도메인 benign-only 재보정 |
| FIG09_CALIBRATION_CONTAMINATION | 부록 | percentile 오염 취약성과 MAD 민감도 |
| FIG10_CALIBRATION_LIMITATION | 한계 | threshold 자체는 도메인 불변이 아님 |
| FIG11/12_ROBUST_CALIBRATION | 부록 | 오염 시 FPR/TPR 변화 |
| FIG13_LOW_FPR_ROC_0_TO_5 | 부록 | low-FPR 구간 ROC |
| FIG14_G4_SCORE_HISTOGRAM | 메커니즘 | benign/Core5/IA의 G4 중첩 |
| FIG15_DETECTOR_LATENCY | 비용 | MIRABEL 전수 계산과 G4 상수 연산 |
| FIG16_RAW_SCORE_DISTRIBUTIONS | 메커니즘 | 공격별 raw-score 분포 |
| FIG17_BENIGN_PERCENTILE_TAIL | 메커니즘 | benign tail에서 공격 생존 질량 |
| FIG18_SAME_QUERY_REORDERING | 부록 | V1→V2 동일 질의 순위 변화 |
| FIG19_GAINED_VS_LOST_TP | 부록 | V2가 얻고 잃은 true positive |
| FIG20_LLAMA_SCORE_HISTOGRAMS | 부록 | Llama member/nonmember scorer 분포 |
| FIG_K_ABLATION_MATCHED_FPR | 슬라이드 12 | k 증가 시 Core5 포화와 IA 증가 |
| FIG_HARD_BENIGN_RECALIBRATION | 슬라이드 13 | hard-benign FPR·member/nonmember 개입 |

주의: FIG09와 FIG10은 원본 artifact에서 동일 그래프를 재사용한다. 슬라이드에는 하나만 사용하면 된다.
