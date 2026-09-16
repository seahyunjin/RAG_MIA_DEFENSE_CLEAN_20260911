# 발표에 사용할 최종 표

발표 본문에는 아래 7개만 사용한다. 나머지는 부록이다.

| 슬라이드 | 파일 | 핵심 메시지 | 계보 |
|---:|---|---|---|
| 10 | `tables/SLIDE09_CORE5_MATCHED_FPR_DETECTION.csv` | 같은 C(BEIR) FPR에서 V2 TPR 우위 | L1 |
| 11 | `tables/SLIDE13_MEMBER_NONMEMBER_INTERVENTION.csv` | s1의 TPR은 nonmember 개입 비용이 큼 | L1 |
| 12 | `tables/SLIDE10_CORE5_QWEN_E2E_PRIVACY.csv` | 생성 후 E-AUC 감소 | L1 |
| 14 | `tables/SLIDE11_GOLD_QA_UTILITY.csv` | 전체와 개입 subset utility 분리 | L2 |
| 15 | `../tables/FINQA_HARD_BENIGN_MATCHED.csv` | D와 D-hard(E) 동일 FinQA 계보 | D/D-hard |
| 16 | `tables/SLIDE14_FINQA_E2E_PRIVACY.csv` | 정상-only 재보정 절차의 도메인 전이 | D |
| 17/21 | `tables/SLIDE12_K_ABLATION_MATCHED_FPR.csv`, `tables/SLIDE15_IA_LIMITATION.csv` | IA 한계 | L1 |

## 발표에서 제외하거나 부록으로만 둘 것

- `quarantine/` 전체
- `CROSS_CORPUS_THRESHOLD_TRANSFER.csv`: matched 성능이 아니라 보조 전이 진단
- C-Sci 제목형 475: 사후 추출, 독립 재현 미완
- 보정 오염·fusion·전체 latency sweep: 질문이 나올 때만
