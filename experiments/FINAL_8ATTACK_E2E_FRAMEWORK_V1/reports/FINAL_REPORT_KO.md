# FINAL_8ATTACK_E2E_FRAMEWORK_V1

- 판정: `FINAL_8ATTACK_PARTIAL_4_SUPPORTED_4_UNAVAILABLE`

## 생성 후 개인정보 누출

| Attack | No Defense | Original MIRABEL | Global BC | Final LC+outer | Native metric |
|---|---:|---:|---:|---:|---|
| MEntA | 0.9783 | 0.5178 | 0.5944 | 0.5854 | ROC-AUC |
| MBA | 0.9282 | 0.5025 | 0.5037 | 0.5027 | ROC-AUC |
| RAG-MIA | 0.9810 | 0.5015 | 0.5415 | 0.5230 | ROC-AUC |
| S²-MIA | 0.6834 | 0.5006 | 0.5044 | 0.5063 | S2-MIA-T balanced accuracy |
| DCMI | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | 원 프로토콜 입력/scorer 미복구 |
| IA | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | 원 프로토콜 입력/scorer 미복구 |
| RAGLeak | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | 원 프로토콜 입력/scorer 미복구 |
| BudgetLeak | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | UNAVAILABLE | 원 프로토콜 입력/scorer 미복구 |

## 해석 제한

- E-AUC는 보조 적응형 공격 진단일 뿐 원 논문 primary metric이 아니다.
- 정상 답변 보존율은 gold QA correctness가 아니다.
- 4개 프로토콜이 unavailable이므로 8공격 또는 universal 방어 주장을 하지 않는다.
