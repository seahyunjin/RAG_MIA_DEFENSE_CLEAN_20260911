# ACL Final Claim Audit

| Claim | Verdict | Evidence boundary |
|---|---|---|
| Training-free | SUPPORT | Frozen inference-time CLD and CF-Clip; no learned classifier or attack labels. |
| Dual-surface architecture | SUPPORT | Retrieval dominance and counterfactual generation influence are distinct operations. |
| Single-query mitigation | SUPPORT | Prospective NFCorpus four-arm results reduce MBA 0.8676→0.5204, RAG-MIA 0.9600→0.5000, S² balanced accuracy 0.6800→0.5600, and DCMI-Std-Q2 differential AUC 0.9700→0.5000. S² CI includes zero. |
| CF-Clip incremental IA benefit | SUPPORT | Exact paired Q5 CLD−RIF = 0.136106, 95% CI [0.072779, 0.195476]. IA is the evaluated local protocol, not paper-exact IA. |
| MEntA repeated-query mitigation | PARTIAL | Fresh primary No Defense 0.78445→RIF 0.71555 with paired reduction CI above zero, but RIF is 0.01065 worse than CLD at Q5; CLD−RIF CI [-0.061600, 0.039400]. Separate fresh NFCorpus B=2 cohort is stronger but cannot be merged. |
| Long-horizon IA mitigation | PARTIAL | Local Q30 surrogate: 0.9746→0.8900 under stateless RIF; leakage remains high. Optional B=2 reaches 0.6830. |
| Open-ended session defense | UNSUPPORTED | Stateless residual leakage and account-reset/Sybil bypass remain. |
| Benign utility preservation | PARTIAL | RIF Token-F1 loss is 3.83% ordinary, 2.17% conversational, and 1.42% true deep-dive, but deep-dive NLI support drops about 3.20pp. B=2 has severe Q3+ cost. |
| Model/retriever generalization | PARTIAL | CLD component transfers to one Llama family; zero-shot retriever operating point is unstable. Full RIF transfer is not evaluated. |
| Universal defense | UNSUPPORTED | Evaluated attacks/settings only; DCMI and IA are standardized/local variants. |

## Allowed claims

- "RIF is a training-free inference-time composition of query-local retrieval-dominance suppression and counterfactual generation-influence clipping."
- "On the evaluated NFCorpus single-query attacks, RIF reduces the reported membership leakage metrics while preserving task-level benign utility within the disclosed cohorts."
- "In the evaluated IA Q1–Q5 local protocol, CF-Clip provides additional privacy gain beyond CLD on paired sessions."
- "Stateless RIF attenuates but does not eliminate long-horizon composition leakage."
- "An optional authenticated B=2 evidence horizon bounds evidence-bearing calls at an explicit Q3+ utility cost."

## Forbidden claims

- "RIF is a universal RAG privacy defense."
- "RIF eliminates membership leakage or solves session MIA."
- "The reported IA result is paper-exact IA."
- "Full RIF is retriever- or generator-independent."
- "B=2 preserves open-ended conversation or deep-reading utility."
- "S² improvement is statistically significant."
