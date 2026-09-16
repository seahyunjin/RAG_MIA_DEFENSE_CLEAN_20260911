# 발표용 22장 구성안

> 대상: RAG/MIA를 처음 접하는 학부 연구생·석사 연구자·교수. 내부 실험명 대신 질문과 실험 목적을 제목으로 쓴다.

## 1. 검색 결과의 쏠림으로 RAG 멤버십 누출을 줄일 수 있는가

연구 질문·발표자·날짜만 둔다.

- 권장 자료: `—`
- 발표 한 문장: 연구 질문·발표자·날짜만 둔다..

## 2. 문제는 탐지 이후에도 남는다

정상 질문과 은닉 공격 질문이 닮으면 낮은 오탐과 높은 탐지를 동시에 얻기 어렵고, 통과한 공격은 생성 답변에서 누출된다.

- 권장 자료: `개념도`
- 발표 한 문장: 정상 질문과 은닉 공격 질문이 닮으면 낮은 오탐과 높은 탐지를 동시에 얻기 어렵고, 통과한 공격은 생성 답변에서 누출된다..

## 3. RAG 멤버십 추론이란

공격자는 질의와 답변을 이용해 특정 문서가 검색 DB에 포함됐는지 추정한다. E-AUC 0.5는 무작위 수준이다.

- 권장 자료: `간단한 흐름도`
- 발표 한 문장: 공격자는 질의와 답변을 이용해 특정 문서가 검색 DB에 포함됐는지 추정한다.

## 4. MIRABEL의 장점과 실사용 병목

전수 유사도 분포로 이상 질의를 찾지만, 정상 오탐과 전체 코퍼스 계산 비용이 남는다. 같은 정상 FPR에서 비교해야 한다.

- 권장 자료: `MIRABEL vs 제안법 비교`
- 발표 한 문장: 전수 유사도 분포로 이상 질의를 찾지만, 정상 오탐과 전체 코퍼스 계산 비용이 남는다.

## 5. 연구 목표

공격 종류를 입력받지 않고 정상 질의만으로 low-FPR를 보정하며, 실제 생성 후 누출과 정상 답변 손상을 함께 줄인다.

- 권장 자료: `기여 3개`
- 발표 한 문장: 공격 종류를 입력받지 않고 정상 질의만으로 low-FPR를 보정하며, 실제 생성 후 누출과 정상 답변 손상을 함께 줄인다..

## 6. 제안 방법은 Top-4만 본다

G4가 임계를 넘으면 가장 가까운 문서 하나를 숨기고 다음 문서를 채운 뒤 한 번만 생성한다.

- 권장 자료: `FIG01_METHOD_OVERVIEW`
- 발표 한 문장: G4가 임계를 넘으면 가장 가까운 문서 하나를 숨기고 다음 문서를 채운 뒤 한 번만 생성한다..

## 7. 점수와 보정

G4=s1-mean(s1..s4). 공격 표본 없이 정상 calibration의 상위 꼬리로 임계를 정한다.

- 권장 자료: `수식`
- 발표 한 문장: G4=s1-mean(s1..s4).

## 8. 네 계보를 섞지 않았다

L1=BEIR Core, L2=TopiOCQA utility, D=FinQA, D-hard(E)=FinQA 어려운 정상. L1×L2 혼합 진단은 격리했다.

- 권장 자료: `계보 표`
- 발표 한 문장: L1=BEIR Core, L2=TopiOCQA utility, D=FinQA, D-hard(E)=FinQA 어려운 정상.

## 9. 여섯 공격은 서로 다르다

MEntA Q5, MBA, RAG-MIA, S²-MIA, DCMI-Q2, IA-Q1. 공격별 원 scorer를 유지했다.

- 권장 자료: `프로토콜 표`
- 발표 한 문장: MEntA Q5, MBA, RAG-MIA, S²-MIA, DCMI-Q2, IA-Q1.

## 10. 같은 오탐률에서 더 많이 잡았다

Core5에서 MIRABEL 88.66%, Final V2 96.08%.

- 권장 자료: `SLIDE09_CORE5_MATCHED_FPR_DETECTION`
- 발표 한 문장: Core5에서 MIRABEL 88.66%, Final V2 96.08%..

## 11. 불필요한 개입도 함께 봐야 한다

V2 member TPR 94.60%, nonmember 개입 2.92%; s1은 높은 TPR 대신 12.51%를 개입했다.

- 권장 자료: `SLIDE13_MEMBER_NONMEMBER_INTERVENTION`
- 발표 한 문장: V2 member TPR 94.60%, nonmember 개입 2.92%; s1은 높은 TPR 대신 12.51%를 개입했다..

## 12. 탐지가 실제 누출 감소로 이어졌다

Qwen Core5 평균 E-AUC: 0.907→0.536→0.512 (No Defense→MIRABEL→V2).

- 권장 자료: `SLIDE10_CORE5_QWEN_E2E_PRIVACY`
- 발표 한 문장: Qwen Core5 평균 E-AUC: 0.907→0.536→0.512 (No Defense→MIRABEL→V2)..

## 13. 생성기를 바꿔도 유지됐다

Llama V2 평균 0.510, 공격별 0.500~0.535. 두 생성기 범위의 전이만 주장한다.

- 권장 자료: `FIG05_GENERATOR_TRANSFER`
- 발표 한 문장: Llama V2 평균 0.510, 공격별 0.500~0.535.

## 14. 정상 답변 손상은 평균과 조건부가 다르다

Gold F1 보존 97.2%, 개입 3.9%. 그러나 개입 39개 subset의 손상은 집중됐다.

- 권장 자료: `FIG07_BENIGN_UTILITY`
- 발표 한 문장: Gold F1 보존 97.2%, 개입 3.9%.

## 15. FinQA의 어려운 정상 질문은 별도 계보다

D-hard(E)는 FinQA 동일 코퍼스 calibration이며 Core hard-benign이 아니다. cross-corpus transfer 수치는 본 결과와 분리한다.

- 권장 자료: `FINQA_HARD_BENIGN_MATCHED`
- 발표 한 문장: D-hard(E)는 FinQA 동일 코퍼스 calibration이며 Core hard-benign이 아니다.

## 16. 새 도메인에서는 절차가 전이됐다

FinQA 평균 E-AUC MIRABEL 0.563, V2 0.552; worst 0.693 vs 0.648. 숫자 임계는 재사용하지 않는다.

- 권장 자료: `SLIDE14_FINQA_E2E_PRIVACY`
- 발표 한 문장: FinQA 평균 E-AUC MIRABEL 0.563, V2 0.552; worst 0.693 vs 0.648.

## 17. 넓게 볼수록 IA만 좋아졌다

Core5는 k=2부터 포화하지만 IA TPR은 G4 53.46%→G3000 75.68%.

- 권장 자료: `SLIDE12_K_ABLATION_MATCHED_FPR`
- 발표 한 문장: Core5는 k=2부터 포화하지만 IA TPR은 G4 53.46%→G3000 75.68%..

## 18. 여섯 공격의 점수 분포

같은 benign 분포와 같은 동결 임계에서 IA의 꼬리 분리가 약함을 직접 보인다.

- 권장 자료: `FIG21_SIX_ATTACK_G4_HISTOGRAM`
- 발표 한 문장: 같은 benign 분포와 같은 동결 임계에서 IA의 꼬리 분리가 약함을 직접 보인다..

## 19. 실제 질문과 답변에서 무엇이 바뀌나

MEntA와 RAG-MIA 사례: 원 입력, 무방어 답변, V2 답변, native score를 원문 그대로 제시한다.

- 권장 자료: `PPT_ATTACK_QUERY_ANSWER_EXAMPLES_KO`
- 발표 한 문장: MEntA와 RAG-MIA 사례: 원 입력, 무방어 답변, V2 답변, native score를 원문 그대로 제시한다..

## 20. 국소 계산은 가볍다

D=3,000 CPU microbenchmark에서 MIRABEL 0.471ms, G4 0.000532ms. 검색 이후 점수 계산만 비교한 값이다.

- 권장 자료: `FIG15_DETECTOR_LATENCY`
- 발표 한 문장: D=3,000 CPU microbenchmark에서 MIRABEL 0.471ms, G4 0.000532ms.

## 21. 아직 해결하지 못한 것

IA gate 실패, MPNet 약함, 개입 정상 subset 손상, Core 사전등록 hard-benign 부재. universal·hallucination-free 주장 금지.

- 권장 자료: `SLIDE15_IA_LIMITATION`
- 발표 한 문장: IA gate 실패, MPNet 약함, 개입 정상 subset 손상, Core 사전등록 hard-benign 부재.

## 22. 결론

평가한 Core5와 두 생성기에서 V2는 같은 오탐 예산의 탐지와 생성 후 privacy를 개선했다. 다음은 IA·retriever·Core hard-benign 확인이다.

- 권장 자료: `핵심 수치 3개`
- 발표 한 문장: 평가한 Core5와 두 생성기에서 V2는 같은 오탐 예산의 탐지와 생성 후 privacy를 개선했다.

