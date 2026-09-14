# LOO mechanism lineage recovery audit

감사일: 2026-09-11  
범위: 기존 2,000개 공격 query, 1,000개 정상 query, LOO/QLL/MIRABEL 요약값의 원본 lineage 복구  
실행 정책: fail-closed, read-only audit

## 최종 판정

- 기존 실험: `LEGACY_PHASE1_LINEAGE_INCOMPLETE`
- 독립 clean rebuild 준비도: `CLEAN_REBUILD_SPEC_INSUFFICIENT`
- `LINEAGE_RECOVERED`: **아님**
- Clean Rebuild Phase-1: **시작하지 않음**
- GPU generation/retrieval: **0건**

기존 LOO 값과 새 retrieval 결과를 섞지 않았다. 기존 수치는 historical diagnostic으로만 보존한다.

## 복구된 항목

1. 공격 query 단위 legacy 결과 2,000행
   - 6개 공격: DCMI 160, IA 1,200, MBA 80, MEntA 400, RAG-MIA 80, S²-MIA 80 query
   - 각 family는 40 member/40 nonmember session이다.
   - `L_full`, `L_minus_1..4`, `A_1..4`, Answer-LOO `argmax_rank/source_id`, MIRABEL margin, QLL dominance, query hash가 2,000/2,000행에 남아 있다.
2. 정상 query 단위 legacy 결과 1,000행
   - 위 LOO/score 요약값과 query hash는 남아 있다.
   - 그러나 원 query text 및 per-query Top-4 source ID는 없다.
3. 공격 query text 2,000행
   - `QUERY_RESPONSE_REVIEW.csv.gz`에 보존되어 있다.
4. MEntA subset exact Top-4 source ID/order
   - 20 member + 20 nonmember session, 5 turns/session = 200 unique query.
   - B90/B95 두 조건 400 cache row에서 동일 case의 source order 충돌은 0건이다.
   - source ID/order map SHA-256: `f4785165d4548dc52d8d28d7824c80a5460bf6f7f798cb7e68198c4abb6cacc7`
   - legacy Answer-LOO argmax source가 recovered Top-4 안에 존재: 200/200.
   - argmax rank와 recovered source 위치 일치: 200/200.
   - target이 실제 Top-4에 있던 96건에서 target rank 일치: 96/96.
   - target이 Top-4에 없던 104건은 저장된 `target_in_top4=False`와 일치한다.
5. checkpoint identity
   - Qwen2.5-3B-Instruct revision `aa8e72537993ba99e69dfaafa59ed015b17504d1`이 존재한다.
   - BGE-M3 revision `5617a9f61b028005a4858fdac845db406aefb181`이 존재한다.
   - generator config/seed 등은 기존 precommit에 남아 있다.

## 복구되지 않은 필수 항목

1. 전체 old input bundle
   - 삭제된 `GPU_INPUTS.jsonl.gz`의 기록된 SHA-256은 `1b63d166c2b091893d59649d65ea175227c2df4e9b8896f2eeb9c0a8092cde24`이나 실제 파일은 없다.
2. AD-test-LLM3의 exact frozen 1,000-document corpus ID와 순서
   - 3개 domain의 원본 corpus 및 그 hash manifest가 없다.
   - 남아 있는 AD-test-LLM NFCorpus member corpus는 1,816행인 다른 lineage이므로 대체하지 않았다.
3. 전체 per-query exact Top-4 source IDs/order
   - MEntA 200 query 외 S²-MIA, MBA, 나머지 공격 및 정상 1,000 query에서 복구되지 않았다.
4. exact QLL selected source document ID
   - QLL dominance scalar는 남아 있지만 선택 source ID는 없다.
5. clean Phase-1 정상 cohort construction
   - 정상 500개의 원 query text와 결정적 source rule이 완전하지 않다.
6. frozen retrieval cache
   - BGE `*.canonical_v2.jsonl.gz` cache는 없다.

## 검색 범위

- 현재 SH repository와 sibling experiment directories
- compact backup의 code/result manifests
- CSV/CSV.GZ/JSON/JSONL/SQLite report 및 cache
- `/home/traffic_3/workspace/workspace` 전체의 `GPU_INPUTS`, canonical retrieval, Top-4, QLL-source 후보
- 2026-09-07~11 Codex session 기록의 old path 및 archive/copy 언급
- 로컬 git repository, shell history, archive 목록, tmux/log 및 삭제 파일 open handle

실제 파일로 복구된 것만 위 결과에 포함했다. 로그에 path/hash만 남은 항목은 복구로 인정하지 않았다.

## Clean rebuild를 시작하지 않은 이유

공식 MEntA 코드에는 seed 42, 1,000 member/1,000 nonmember, TF-IDF near-duplicate 제거 같은 split procedure가 남아 있다. 그러나 이것만으로 당시 substrate를 결정적으로 재구성할 수는 없다.

- Hugging Face dataset revision과 당시 입력 파일 hash/order가 고정되어 있지 않다.
- 삭제된 AD-test-LLM3 exact corpus와 동일함을 검증할 reference hash가 없다.
- Phase-1 benign 500 query의 원문 및 결정적 구성 규칙이 완전하지 않다.
- 전체 attack/normal Top-4와 QLL locator source를 다시 계산해도 old lineage와 동일하다고 증명할 수 없다.

따라서 새 corpus를 임의 구성하면 clean re-evaluation조차 사전 정의된 동일 protocol이라고 보장할 수 없다. 최신 지시의 중단 조건에 따라 `CLEAN_REBUILD_SPEC_INSUFFICIENT`로 종료했다.

## 연구 결과 사용 규칙

- 기존 2,000/1,000 LOO 값: historical diagnostic으로만 사용.
- MEntA 200 query Top-4: partial lineage recovery evidence로만 사용.
- 새 retrieval과 legacy LOO를 같은 표·통계·detector에 결합 금지.
- `재현(reproduction)` 표현 금지.
- 다음 실행은 exact old bundle을 복구하거나, 별도 version-pinned corpus/query manifest를 먼저 동결한 신규 clean re-evaluation이어야 한다.

## 재개에 필요한 최소 입력

1. AD-test-LLM3의 domain별 exact 1,000-doc corpus 파일 또는 ordered ID manifest + SHA-256.
2. 정상 Phase-1 query 원문과 source/label construction manifest.
3. per-query Top-4 source IDs/order와 QLL selected source ID를 포함한 old input bundle, 또는 이를 처음부터 새로 만들 수 있는 version-pinned protocol.
4. 공격 session/query manifest와 target inclusion/exclusion rule의 hash-pinned 사본.

이 네 항목이 확보되기 전에는 Phase-1 Oracle/MIRABEL/QLL/UNION-2 TPR을 새로 보고하지 않는다.
