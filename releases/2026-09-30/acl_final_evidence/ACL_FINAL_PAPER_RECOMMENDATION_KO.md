# ACL/EMNLP 최종 투고 권고

## 1. 현재 evidence로 ACL Main이 충분한가?

현재 상태는 **ACL Main에 제출은 가능하지만, 강한 accept 정당화에는 아직 부족한 borderline evidence**다. 가장 방어 가능한 선택은 ACL Findings/EMNLP Findings까지 함께 염두에 둔 정직한 RIF 논문이다. 이유는 prospective Core4와 exact IA complementarity가 강한 반면, paper-exact IA, full-RIF transfer, 동일-cohort strong baseline, 독립 confirmatory single-query test가 비어 있기 때문이다.

## 2. 가장 강한 contribution 3개

1. **두 표면의 단순한 training-free 결합**: CLD는 retrieval rank-1 dominance를, old CF-Clip은 생성 token의 counterfactual document influence를 제어한다.
2. **공격별 routing 없는 폭넓은 실증**: prospective NFCorpus single-query 공격 네 종류에서 누출 지표가 감소하며, exact IA Q5에서는 CLD 0.8458 대비 RIF 0.7097의 추가 이득이 paired comparison으로 확인된다.
3. **정직한 composition boundary**: stateless RIF의 Q30 잔여 누출을 숨기지 않고, 별도 optional B=2 policy가 노출을 제한하지만 Q3+ utility를 희생함을 정량화한다.

## 3. reviewer가 공격할 약점 5개

1. IA가 paper-exact가 아니며 Q30은 local surrogate다.
2. MEntA fresh primary에서 RIF가 CLD보다 Q5 AUROC가 0.01065 높아 CF의 보편적 complementarity가 없다.
3. B=2는 Q3+ refusal policy이므로 defense novelty보다 access-control baseline으로 보일 수 있고 Sybil reset에 취약하다.
4. full RIF의 retriever/generator transfer와 exact strong-baseline comparison이 없다.
5. exact RIF가 Vanilla 대비 약 3.95배 느리고, single-query 주 결과가 DEV 규모다.

## 4. submission 전에 반드시 필요한 추가 실험

- **Paper-exact IA reproduction**: 공식 summary+Q30, scorer, Top-k, generator/decoding을 고정한 No Defense/CLD/CF/RIF paired test.
- **동일-lineage independent Core4 confirmatory**: 새 target, 동일 four-arm, 2,000+ paired bootstrap; 특히 S² 불확실성을 해소.
- **동일-cohort MIRABEL comparison**: 동일 benign calibration/FPR, 동일 attacks, 동일 utility/runtime assumptions. 다른 baseline은 정확한 port가 없으면 assumptions table 유지.
- **Full-RIF transfer 최소 1개**: 다른 generator에서 gamma zero-shot exact token path 또는 다른 retriever에서 frozen tau의 실패를 정식 negative result로 확정. 성공을 위해 retune하지 않는다.
- **End-to-end artifact integrity check**: 공개 코드가 frozen prompts/context policy와 일치하는지 재현 스크립트와 hash로 검증.

## 5. 있으면 좋은 실험

- 사용자 연구 또는 blinded human factuality 평가.
- 더 긴 answer/runtime profile과 다양한 GPU hardware 측정.
- 인증·credential rotation을 포함한 B=2 운영 보안 분석.
- adaptive paraphrase를 새로 생성한 prospective test. 현재 것은 저장 후보/출력 기반 offline stress test다.

## 6. B=2 배치 권고

**Main method가 아니라 optional safeguard**로 둔다. 본문 limitation/long-horizon section에 핵심 수치 한 표를 넣고, 상세 adaptive/Sybil/utility 표는 appendix로 보낸다. RIF novelty와 분리하고 `authenticated high-security deployment option`으로만 표현한다.

## 7. 최종 포지셔닝

권장 제목 개념: **RIF: Retrieval and Generation Influence Control for Membership Privacy in RAG**.

핵심 문장: "Retrieval filtering alone does not fully address membership leakage expressed through generated responses. RIF combines query-local retrieval-dominance suppression with counterfactual generation-influence clipping without retraining or attack-specific classifiers."

장기 세션 문장: "Repeated interrogation can accumulate residual leakage beyond stateless per-query defenses. An optional authenticated fixed evidence horizon bounds evidence-bearing calls at an explicit loss of open-ended deep-reading utility."

최종 판정: `ACL_MAIN_BORDERLINE__FINDINGS_READY_AFTER_REQUIRED_VALIDATION`
