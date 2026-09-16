# Final claim audit

| Claim | Status | Evidence boundary |
|---|---|---|
| V2 universally outperforms MIRABEL. | **UNSUPPORTED** | FinQA DCMI/MBA/RAG-MIA cells do not uniformly beat MIRABEL. |
| V2 improves average Core5 low-FPR detection. | **SUPPORTED** | Frozen Core5 matched-FPR table. |
| V2 reduces E2E membership leakage across Core5. | **SUPPORTED** | Frozen Qwen Core5 native scorer table. |
| V2 transfers to FinQA without adaptation. | **UNSUPPORTED** | Numeric zero-shot TPR degrades. |
| V2 transfers with benign-only recalibration. | **SUPPORTED** | FinQA final verdict and Core5 E2E table. |
| Numeric threshold is domain invariant. | **UNSUPPORTED** | Core and FinQA calibrated thresholds/scales differ. |
| V2 is retriever agnostic. | **PARTIAL** | BGE/GTE/MPNet measured, with weaker absolute MPNet performance. |
| V2 is generator independent. | **PARTIAL** | Qwen and one Llama transfer only. |
| Missed low-G queries are naturally non-leaky. | **UNSUPPORTED** | Only 1/10 environment-attack correlations supported it. |
| V2 is hallucination-free. | **UNSUPPORTED** | NLI is proxy; false-positive subset has measurable damage. |
| V2 has low overall benign damage. | **SUPPORTED** | Low aggregate intervention, with FP-subset caveat. |
| False-positive users can suffer substantial damage. | **SUPPORTED** | Frozen conditional FP utility audit. |
| Top-k concentration is a novel mathematical statistic. | **UNSUPPORTED** | The statistic family predates this application. |
| Local retrieval concentration for low-FPR RAG-MIA detect-and-hide was not found in audited prior work. | **PARTIAL** | Use only with explicit related-work search boundary. |
