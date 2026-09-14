# Phase B — FP-safe closed-book fallback

- Verdict: `FP_FALLBACK_UTILITY_FAILED`
- FP F1: `0.2184` → `0.2148`
- FP refusal: `0.2222` → `0.2222`

| Attack | Simple Hide | Fallback | Delta | Gate |
|---|---:|---:|---:|---|
| MEntA | 0.5580 | 0.7432 | +0.1852 | FAIL |
| MBA | 0.4897 | 0.5324 | +0.0427 | FAIL |
| RAG-MIA | 0.5275 | 0.5275 | +0.0000 | PASS |
| S²-MIA | 0.4950 | 0.6425 | +0.1475 | FAIL |
| DCMI-Std-Q2 | 0.5150 | 0.5225 | +0.0075 | PASS |
