# Phase A2 — DB churn E2E causal decomposition

- Verdict: `CHURN_EXPOSURE_DISAPPEARS`
- V0→V50 Retrieval@4 drop: `0.3540`
- V0→V50 TPR|EXPOSED drop: `-0.0056`
- No-Defense distance-to-chance reduction: `0.1611`

| DB | Retrieval@4 | TPR | TPR given exposed | No-defense distance | Final-LC distance |
|---|---:|---:|---:|---:|---:|
| V0 | 0.9990 | 0.9164 | 0.9172 | 0.4117 | 0.0232 |
| V10 | 0.9400 | 0.8634 | 0.9190 | 0.3894 | 0.0241 |
| V25 | 0.7830 | 0.7188 | 0.9156 | 0.3145 | 0.0194 |
| V50 | 0.6450 | 0.5974 | 0.9228 | 0.2506 | 0.0206 |
