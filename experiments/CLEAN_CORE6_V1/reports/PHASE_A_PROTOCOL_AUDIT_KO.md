# CLEAN_CORE6_V1 — Phase A Protocol Audit

## 최종 판정

`IA_PROTOCOL_UNAVAILABLE`

`CORE6_PHASE_A_FAIL_CLOSED`

Phase B 이후 작업은 시작하지 않았다. 즉, 공통 target 40개 선정, member/nonmember 배정, protected DB 구축, query 생성, retrieval, 답변 생성 및 공격 점수 계산은 모두 0건이다. 기존 답변이나 공격 점수도 재사용하지 않았다.

중단 이유는 성능 실패가 아니라 **원 공격 프로토콜의 재현성 실패**다. IA의 기존 immutable ground-truth가 현재 저장소와 전체 workspace에 없고, 원 논문의 30-query/GPT-4o 계보를 현재 자료로 정확히 복원할 수 없다. 이 상태에서 새 GT를 만들면 사용자가 금지한 임의 IA GT 생성이 된다.

## 핵심 결과

| 공격 | 논문상 원형 | 현재 재현 가능성 | 판정 |
|---|---|---|---|
| MEntA | 5개 자연 질의, atomic-claim NLI, entailment-hit − abstention | 보존된 재현 패키지로 protocol/scorer 확인 | `PROTOCOL_AND_SCORER_AVAILABLE` |
| IA | 원 논문 기본 30개 yes/no 질의, GPT-4o 생성, GPT-4o-mini GT, lambda=5 | immutable GT 없음; 공식 저장소 bundle은 문서당 15개이고, 현재 로컬 재현은 GPT-4.1-nano 및 기본 5개 평가 | `IA_PROTOCOL_UNAVAILABLE` |
| DCMI | 원본/6% perturbation 2-query pair, Yes 차이 점수 | 수식은 확인; 공식 repo `MIA.py`는 placeholder array이고 `perturb.py`는 3% | `SPECIFICATION_AVAILABLE_IMPLEMENTATION_INCOMPLETE` |
| S²-MIA | target을 둘로 나눠 1회 질의; BLEU와 generation perplexity | 현재 로컬 evaluator는 sentence-embedding cosine을 primary score로 사용 | `SPECIFICATION_AVAILABLE_SCORER_MISMATCH` |
| MBA | proxy LM의 difficult-word rank로 mask 구성, mask recovery accuracy | 현재 파이프라인은 기본 5-mask + `important` strategy로 원 논문과 다름 | `SPECIFICATION_AVAILABLE_QUERY_GENERATOR_MISMATCH` |
| RAG-MIA | target sample 포함 Yes/No prompt; black-box discrete score | 논문 protocol은 확인; 보존 패키지에 공식 구현 없음, legacy parser만 존재 | `PAPER_PROTOCOL_AVAILABLE_OFFICIAL_SCORER_MISSING` |

전체 열은 [`CORE6_PROTOCOL_AUDIT.csv`](../audits/CORE6_PROTOCOL_AUDIT.csv)에 기록했다.

## IA 강제 중단 감사

### A. 기존 immutable GT의 exact recovery

실패했다.

- 기대했던 `IA_ORIGINAL_GROUND_TRUTH.private.csv.gz` 및 동등한 immutable per-query GT를 전체 `/home/traffic_3/workspace`에서 찾지 못했다.
- 보존된 MEntA reproduction에는 GT 생성 코드만 있고 과거 실행의 불변 GT는 없다.
- 새로 GPT 계열 모델을 호출해 만든 답은 **복구**가 아니라 **새로운 표본 생성**이다.

### B. official protocol의 clean reproduction

현재 자료만으로는 exact reproduction으로 인정할 수 없다.

1. IA 논문은 GPT-4o로 문서당 30개 질문을 생성하고 GPT-4o-mini로 GT를 만든다.
2. 감사한 공식 GitHub commit `a53e8f...`의 포함 데이터는 NFCorpus 3,633문서와 TREC-COVID 500문서 모두 문서당 15개 질문이다.
3. 공식 코드가 읽도록 지정한 파일명은 `target_data_with_questions.json`이지만 실제 포함 파일명은 `clean_data_with_questions.json`이다.
4. 공식 scorer는 단순 response/GT accuracy를 계산하며, 논문 식의 `lambda=5` UNK penalty가 구현돼 있지 않다.
5. 로컬 MEntA reproduction은 30-query pool을 만들지만 query model이 GPT-4.1-nano이고 evaluation 기본값이 5이며 scorer 기본 penalty는 1이다. 원 IA와 동일하지 않다.
6. hosted GPT 모델의 정확한 revision/output 및 원래 stochastic run을 content hash로 되살릴 자료가 없다.

따라서 IA를 포함한 공정한 Core6 결과를 지금 생성하면 “동일 substrate + original protocol/scorer”라는 캠페인 전제를 위반한다.

## 공격별 세부 감사

### MEntA

- target 조건: 서로 다른 5개 정보 질의를 만들 수 있는 충분한 본문.
- session: target당 query index 0–4.
- scorer: 답변을 atomic claim으로 분해한 뒤 target document가 claim을 entail하는지를 DeBERTa long-NLI로 검사한다. query별 entailment hit에서 IDK/abstention을 빼고 5개 query 평균을 사용한다.
- metric: ROC-AUC, accuracy, TPR@low FPR.
- 제한: GPT-4.1-nano는 hosted model이므로 query manifest를 생성 직후 hash/freeze해야 한다.

### IA

- target 조건: 30개의 다양한 yes/no 질문을 만들 수 있고 nonmember near-duplicate가 제거된 문서.
- session: 원 논문 기본 30-query document session.
- scorer:

  `IA(D) = (1/n) * sum(1[r_i = g_i] - 5 * 1[r_i = UNK])`

- immutable requirement: exact query pool, selection/order, GPT-4o-mini GT, missing/UNK policy.
- 현재 위 요건을 충족하는 artifact가 없다.

### DCMI

- target당 original query와 6% perturbed query의 2개 pair.
- black-box score는 `f_rag(q) - f_rag(q')`이며 Yes를 1, 그 외를 0으로 둔다.
- 논문 수식과 로컬 reproduction은 확인했지만, 공식 repo의 scorer 입력은 placeholder이고 perturbation code는 3%라서 exact executable lineage는 아니다.

### S²-MIA

- target의 앞부분을 query로, 나머지를 reconstruction reference로 사용한다.
- 원 논문은 BLEU와 생성 perplexity를 사용하며 threshold variant(S²-MIA-T)와 learned variant(S²-MIA-M)를 구분한다.
- 현재 로컬 파일은 BLEU/PPL도 계산하지만 membership ROC-AUC의 primary score로 sentence-embedding cosine을 사용한다. 그대로 쓰면 원 scorer 재평가가 아니다.

### MBA

- 원 논문은 proxy LM으로 예측이 어려운 단어를 고르고, mask reconstruction accuracy를 membership score로 사용한다.
- 원 실험은 `M in {5,10,15,20}`을 비교한다.
- 현재 로컬 pipeline 기본은 `NUM_MASKS=5`이고 CLI 기본 masking strategy는 `important`다. paper-native difficult-word construction으로 고정돼 있지 않다.

### RAG-MIA

- black-box best prompt는 target sample이 context에 나타나는지 Yes/No로 묻는 단일 질의다.
- 논문은 5개 prompt 조건을 비교한다. black-box는 discrete TPR/FPR와 ROC-AUC를, gray-box는 token probability를 사용한다.
- 보존된 `menta_official` 패키지에는 RAG-MIA 구현이 없고 `exp44_native.py`의 자체 parser만 있다. 명확한 Yes/No가 없는 출력 처리까지 포함한 official frozen scorer provenance가 없다.

## 로컬 증거와 해시

기계 판독 가능한 전체 목록은 [`LOCAL_ASSET_INVENTORY.csv`](../audits/LOCAL_ASSET_INVENTORY.csv)에 있다.
외부 공식 저장소의 commit, key-file hash, 포함 query 수 감사 결과는 [`OFFICIAL_REPOSITORY_AUDIT.json`](../audits/OFFICIAL_REPOSITORY_AUDIT.json)에 고정했다.

주요 scorer SHA-256:

- MEntA: `830f57407c09d9a932a5e4eb5b7ee0ac653d2065ecff4c10241620dea90438b9`
- IA reproduction: `4255d599cbb6efc6273f507cc5377a3b7ca9b8208ba66feb9a2e31a1f0c02d69`
- DCMI reproduction: `91bcafbd64b8b5fcfb373b06b79d787ef509b4ca361d7d885b1b9d787e6487aa`
- S² reproduction: `fe373049d977cdec30daed2013ff495bd5ad61174fd889bbf2721bcaee75fb43`
- MBA reproduction: `0523e798013a35247b6ae5583b03c15347f5f45dfa3e8b835ac31623a253104d`
- legacy native helper: `7534e6d844d5558c5ff768209ca1c6b5aa46622c46f7cfc4ae2228b12e6a6215`

독립 감사한 repository commits:

- IA: `a53e8fda4f1492f204d968d7afb53ce8287c193c`
- DCMI: `ef961a1f23d09c62c5fbf69d798e325fb0362e55`

## Phase B를 열기 위한 정확한 조건

최소한 다음 자료가 필요하다.

1. IA의 공유 target용 immutable query/GT bundle 또는 저자 제공 version-pinned exact reproduction bundle.
2. IA 30-query pool, question selection/order, `lambda=5`와 UNK/missing-output 처리의 단일 frozen 구현.
3. S²-MIA의 T 또는 M variant를 사전 선택하고 BLEU/perplexity 원 scorer를 고정한 구현.
4. MBA difficult-word mask 생성, proxy model/tokenizer revision 및 M-selection 규칙을 고정한 구현.
5. RAG-MIA prompt ID, exact serialization, Yes/No parser 및 ambiguous-output policy를 고정한 구현.
6. DCMI의 6% perturbation provenance와 pairwise black-box scorer를 묶은 실행 가능한 구현.

그 전에는 shared target pool을 선택하면 안 된다. target을 먼저 선택했다가 누락 protocol에 맞춰 교체하면 selection bias가 생기기 때문이다.

## 과학적 결론

현재 과거 Core6 수치를 한 표에 합치는 것은 공정한 benchmark가 아니다. 공격마다 membership DB, target cohort, query budget, scorer 계보가 달랐고 일부 scorer는 원 논문과 달랐다.

따라서 이번 올바른 결과는 **성능 숫자**가 아니라 **fail-closed**다. IA를 임의로 메우지 않았고, 잘못된 “Core6 비교표”가 만들어지는 것을 차단했다.

## 다음 한 단계

새 방어를 만들지 말고, 먼저 **Core5 + IA-pending**으로 명세를 바꾸는 것이 허용되는지 결정해야 한다. 허용한다면 DCMI/S²/MBA/RAG-MIA의 frozen protocol gap부터 메운 뒤 공통 eligibility rule을 precommit한다. `Core6` 명칭을 유지하려면 IA 저자 artifact 또는 exact immutable GT bundle 확보가 선행되어야 한다.
