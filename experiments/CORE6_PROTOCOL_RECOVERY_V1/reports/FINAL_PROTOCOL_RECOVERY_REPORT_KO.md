# CORE6_PROTOCOL_RECOVERY_V1 최종 보고서

## SUMMARY VERDICT

최종 판정은 **`CORE_N_PROTOCOL_PARTIAL (N=3)`**이다.

논문 정의, 쿼리 생성 규칙, 원 점수 방향과 집계, 실패 닫힘(fail-close) 시험까지 현재 확정 가능한 공격은 **MEntA, MBA, RAG-MIA 3개**다. S²-MIA와 DCMI는 논문만으로 하나의 완전한 실행 명세를 고정할 수 없는 항목이 남았고, IA는 논문의 30-query/불변 GT 묶음을 복구하지 못했다. 따라서 현재 결과를 `Core6 paper-exact` 또는 `Core5 + IA-pending`이라고 부르면 안 된다.

이번 캠페인에서는 명세대로 방어 모델, 검색, 타깃 선정, member/nonmember 분할, 공격 쿼리 생성, 답변 생성을 전혀 실행하지 않았다.

## PAPER VS REPO DISCREPANCIES

세부 표는 `audits/PAPER_REPO_DISCREPANCIES.csv`에 있다.

- MEntA: 논문은 `summary || question`을 최종 질의로 정의한다. 보존 코드의 검색 단계는 이 결합문을 임베딩하지만 답변 생성 단계는 원 질문만 사용한다. 복구 어댑터는 논문 정의대로 요약과 질문을 모두 보존하고 결합한다.
- S²-MIA: 논문 그림과 설명은 생성 답변을 나머지 절반과 비교한다고 서술하지만, Eq. 4는 전체 target sample과의 BLEU를 정의한다. 또한 T형과 M형이 모두 원 공격으로 제시된다.
- MBA: 논문은 GPT2-XL 난이도, 철자/fragment 필터, `M∈{5,10,15,20}` 선택을 사용하지만 로컬 기본값은 `important` masking과 고정 M=5다. 이 fallback은 금지했다.
- DCMI: 논문 black-box 고정값은 6%인데 공식 저장소 perturb 코드는 3%다. 저장소의 scorer 진입점에는 placeholder 배열도 있다.
- IA: 논문은 30 questions와 λ=5 UNK penalty를 사용한다. 공식 저장소 데이터는 문서당 15 questions이고 scorer는 plain accuracy다.
- RAG-MIA: 공식 frozen scorer 저장소를 찾지 못했다. 논문 prompt #2와 명시된 Yes/No 규칙만 재구현했다.

## MEntA

판정: **`MENTA_PROTOCOL_READY` / `PROTOCOL_ACCEPTED`**

- GPT-4.1-nano가 문서의 서로 다른 부분을 덮는 구체 질문 5개를 생성한다.
- IA의 topic-focused description prompt로 만든 짧은 요약을 각 질문 앞에 붙인다.
- 질문 순서는 q1→q5로 고정한다.
- 답변을 atomic claims로 나눈 뒤, 고정한 DeBERTa NLI와 claim extractor를 사용한다.
- 질의별 신호는 `I_entail - I_idk`, 문서 점수는 5개 신호의 평균이다.
- NLI revision: `tasksource/deberta-base-long-nli@04dcf11f844b07bc57015169fca2b7d6df8299d5`
- Claim extractor revision: `Babelscape/t5-base-summarization-claim-extractor@94775fb1c8dc2c3ef1bfec413f9f961e6ba5a1c8`

주의: 논문은 hosted model의 bit-exact revision과 전체 decoding을 완전히 명시하지 않는다. 질문 temperature 0.7과 요약 temperature 0.3은 보존 실행 패키지 기본값이다. 향후 생성 결과 자체를 manifest로 동결해야 한다.

## S²-MIA

판정: **`S2_SPEC_UNDERDETERMINED` / `PROTOCOL_REVISE`**

확정된 부분은 단일 질의, target 분할, BLEU와 generation perplexity의 방향성이다. 그러나 다음 두 항목을 임의로 선택하면 paper-exact가 아니다.

1. BLEU reference가 전체 target인가, query를 제외한 remaining text인가.
2. 최종 공격을 reference-set threshold 방식 S²-MIA-T로 고정할지, classifier 방식 S²-MIA-M으로 고정할지.

따라서 cosine primary는 제거했지만 현재 scorer는 **component reimplementation**일 뿐 최종 benchmark scorer가 아니다.

## MBA

판정: **`MBA_PROTOCOL_READY` / `PROTOCOL_ACCEPTED`**

- Proxy LM: `openai-community/gpt2-xl@15ea56dee5df4983c59b2538573817e1667135e2`
- Spelling model: `oliverguhr/spelling-correction-english-base@0e3958355a09d2816ed2701fdc2f4471d46c320e`
- target을 M개 동일 길이 subtext로 나누고 각 구간에서 유효 단어 중 난이도가 가장 높은 하나를 고른다.
- stopword·구두점·fragment·철자 오류·인접 mask 제약을 적용한다.
- M은 논문 후보 `{5,10,15,20}` 중 reference/train split 기준으로 고정해야 한다.
- 점수는 `정확히 복원한 mask 수 / M`; malformed/missing/duplicate index는 보정하지 않고 실패시킨다.

현재 어댑터는 고정 revision에서 계산된 proxy difficulty record를 입력으로 요구한다. Proxy model 실행은 substrate 이후 단계이며 이번 캠페인에서는 수행하지 않았다.

## DCMI

판정: **`DCMI_SPEC_UNDERDETERMINED` / `PROTOCOL_REVISE`**

확정된 black-box 핵심은 원문과 6% antonym perturbation을 같은 Yes/No template로 질의하고, `Yes=1, No=0`으로 바꾼 뒤 `f(q)-f(q′)`를 계산하는 것이다. 다만 논문은 perturbation을 만드는 third-party LLM의 정확한 revision과 decoding을 고정하지 않았고, 공식 repo는 3%를 사용한다. 따라서 pair builder와 differential scorer는 보존했지만 완전한 paper protocol로 승인하지 않았다.

## IA

판정: **`IA_PAPER_EXACT_UNAVAILABLE` / `PROTOCOL_UNAVAILABLE`**

논문 식은 복구했다.

`score = (1/n) Σ [I(response=GT) - 5·I(response=UNK)]`, `n=30`.

그러나 workspace/archive, 공식 저장소, 저장소 이력에서 동일 target에 대응하는 원 30-query + GPT-4o-mini GT 불변 묶음을 찾지 못했다. 공식 저장소의 15-query/plain-accuracy 버전은 별도 `IA_REPOSITORY_VARIANT` stress test로만 사용할 수 있고 원 논문 IA라고 부르면 안 된다. 현재 생성 함수는 의도적으로 실패 닫힘 처리한다.

## RAG-MIA

판정: **`RAGMIA_PROTOCOL_READY` / `PROTOCOL_ACCEPTED`**

- 논문 black-box prompt #2를 사용한다.
- `Yes → member`, `No → nonmember`, Yes/No가 모두 없는 출력은 논문대로 nonmember다.
- Yes와 No가 모두 있는 출력은 논문에 정의가 없어 임의 분류하지 않고 실패시킨다.
- 원 보고 항목은 black-box TPR/FPR과 ROC-AUC다.

## METRIC PROVENANCE

공격별 논문 출처, 수식, 함수, polarity, scorer SHA-256, 원 metric은 `audits/METRIC_PROVENANCE.csv`에 고정했다. **E-AUC/symmetric AUC는 이 캠페인의 공식 primary metric이 아니며 `INTERNAL_DIAGNOSTIC_ONLY`다.**

## UNIT TEST RESULTS

- 총 20개 실행, 20개 통과
- failure 0, error 0, skip 0
- 포함 범위: deterministic parsing, query count, member/nonmember label, scorer polarity, aggregation, malformed input fail-close
- Python compile check: PASS
- Campaign manifest checksum: PASS

기계 판독 결과와 전체 로그는 각각 `audits/UNIT_TEST_RESULTS.json`, `audits/UNIT_TEST_RESULTS.txt`에 있다.

## INDEPENDENT REVIEW

요청된 Claude Code MCP/Fable 도구는 현재 런타임에 없어 외부 독립 검증을 수행하지 못했다. 이를 숨기지 않고 모든 행을 `external_independent_review=NOT_AVAILABLE`로 표시했다. Codex의 paper↔code 교차 감사 결과는 `codex_review`이며 독립 검증으로 부르지 않는다. 후속 검토 체크리스트는 `reports/INDEPENDENT_REVIEW.md`에 있다.

## FINAL CORE SET

- Development attacks ready: **MEntA, MBA**
- Development attacks blocked: **S²-MIA**
- Confirmation attacks ready: **RAG-MIA**
- Confirmation attacks blocked: **DCMI, IA**
- IA status: **IA paper-exact unavailable; repository variant only if separately labeled**

즉 지금 즉시 사용 가능한 것은 Core3다. 그러나 사전 지정된 development/confirmation 분리를 유지하려면 S²와 DCMI가 빠진 현재 구성으로는 다음 defense 개발 캠페인을 시작하면 안 된다.

## MISSING ASSETS

1. S²-MIA 저자 확인 또는 frozen official scorer: BLEU reference와 T/M branch를 확정할 자료.
2. DCMI 저자 frozen perturbation artifact 또는 정확한 model revision/decoding.
3. IA 동일-target 30-query + GPT-4o-mini GT 원본 bundle과 그 hash.
4. MEntA hosted query/summary generation outputs의 향후 immutable manifest.
5. 외부 독립 protocol review.

## NEXT STEP

**`CLEAN_SHARED_CORE6_SUBSTRATE` 구축은 아직 불가하다.** 먼저 S²/DCMI의 모호성을 저자 자료로 해소하고 IA 원 bundle을 확보하거나, 사용자가 명시적으로 `Core3` 또는 정확히 이름 붙인 repo variant를 사용할 공격 세트로 승인해야 한다. 그 전에는 target selection, protected DB, member/nonmember split, query generation을 시작하지 않는다.
