# Independent review status

- Requested reviewer: Claude Code MCP / Fable-5.1, read-only, maximum effort
- Availability in this runtime: **NOT AVAILABLE**
- External independent review performed: **No**
- No result in this campaign is labeled as independently verified.

Codex performed a first-party cross-audit against the paper PDFs/text, the two located official repositories, and the preserved local MEntA package. That audit is recorded as `codex_review` in `audits/FINAL_PROTOCOL_TABLE.csv`; it is not a substitute for independent review.

An external reviewer should inspect, at minimum:

1. the MEntA paper-versus-preserved-code summary-prepending difference;
2. the S²-MIA full-target versus remaining-text BLEU inconsistency and choice between S²-MIA-T/M;
3. MBA difficult-word preprocessing and the frozen proxy-model revisions;
4. DCMI paper 6% versus repository 3%, plus the unidentified perturbation-model revision;
5. IA 30-query/lambda=5 paper protocol versus the 15-query/plain-accuracy repository artifact;
6. RAG-MIA handling of malformed answers containing both “Yes” and “No”.
