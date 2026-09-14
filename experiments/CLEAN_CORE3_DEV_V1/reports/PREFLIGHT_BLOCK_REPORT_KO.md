# CLEAN_CORE3_DEV_V1 사전검증 보고서

## PRECHECK

- Protocol source verification: **PHASE1_PROTOCOL_ARTIFACT_VERIFICATION_PASS**
- 전체 실행 준비 상태: **CLEAN_CORE3_PREFLIGHT_READY**
- Recovery manifest SHA-256: `c7ade1c1ba6665b4cd36e4c8f84f2183780b3baa2e994fcd3cb24e263f064e44`
- READY 공격: MEntA / MBA / RAG-MIA
- 제외 공격: S²-MIA / DCMI / IA
- Unit tests: 20/20, PASS

## 실행 중단 사유

- 없음

필수 protocol runtime asset이 모두 확인됐다. 기존 질의, `important` masking, 다른 무료 모델로 대체하면 명세 위반이므로 실행하지 않았다.

## 보존된 원본 입력

BEIR NFCorpus/SciDocs/TREC-COVID의 corpus/query hash는 이전에 고정된 public-source hash와 모두 일치한다. 이 검증은 raw input 가용성 확인일 뿐, target selection이나 새 DB 구성을 수행한 것이 아니다.

## 수행하지 않은 작업

- COMMON_ELIGIBLE_POOL 및 shared 40 targets 선택
- member/nonmember assignment 및 protected DB 구성
- 공격 질의 생성
- benign 500/500 split
- retrieval/MIRABEL/BC threshold
- 세 조건 답변 생성 및 paper-faithful scoring

따라서 현재는 PRECOMMIT SHA, benign FPR, EPO, post-generation privacy, bottleneck map을 보고할 수 없다. 값이나 대체 결과를 만들지 않았다.
