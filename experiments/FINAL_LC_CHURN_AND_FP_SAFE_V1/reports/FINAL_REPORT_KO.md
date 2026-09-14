# Final LC DB Churn and FP-Safe Validation — Corrected Core5 Scope

- IA: `IA_STD_Q15_V6_FAILED_FINAL` / `STANDARDIZED_ATTACK_UNAVAILABLE`
- Initial churn aggregation: `INVALID_S2_SCOPE_INCLUSION` (preserved under history/)
- Correction: S² uses only frozen S2_EVALUATION 1,598 rows (799 member / 799 nonmember).
- Detector, embeddings, retrieval, thresholds and the other four attack IDs were unchanged.

| DB | Original MIRABEL FPR | Final LC strict FPR | Final LC refresh FPR | MEntA TPR | MBA TPR | RAG-MIA TPR | S² TPR | DCMI TPR | Core5 mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| V0 | 30.500% | 2.500% | 2.500% | 0.8840 | 0.9960 | 0.9520 | 0.9962 | 0.7845 | 0.9225 |
| V10 | 31.000% | 2.500% | 2.300% | 0.8264 | 0.9340 | 0.8900 | 0.9337 | 0.7370 | 0.8642 |
| V25 | 30.600% | 2.300% | 1.700% | 0.7066 | 0.7990 | 0.7530 | 0.8023 | 0.6255 | 0.7373 |
| V50 | 29.800% | 2.600% | 2.500% | 0.5548 | 0.6290 | 0.5990 | 0.6370 | 0.4945 | 0.5829 |

- Churn verdict: `DB_CHURN_RECALIBRATION_FAILED`
- Strict verdict: `DB_CHURN_STRICT_TRANSFER_FAILED`
- FP-safe verdict: `FP_SAFE_REDISTRIBUTION_NOT_APPLICABLE`
- Existing packing already water-fills remaining Top-3; candidate generation/privacy screen were not opened.
- The benign holdout calibrates and reports empirical threshold exceedance under the inherited protocol; it is not an untouched FPR test.
- Training/gradient/attack-calibration/new Qwen generations: 0 / 0 / 0 / 0

## 12-line summary

1. IA final: IA_STD_Q15_V6_FAILED_FINAL / STANDARDIZED_ATTACK_UNAVAILABLE
2. Missing embedding union: 1,847 unique documents
3. V10 strict/refresh FPR: 2.500% / 2.300%
4. V25 strict/refresh FPR: 2.300% / 1.700%
5. V50 strict/refresh FPR: 2.600% / 2.500%
6. V50 refresh Core5 mean TPR loss: 33.968%
7. Retraining steps: 0
8. Packing: remaining Top-3 water-fill to 2,048 tokens
9. Simple-Hide FP F1: ~0.218 (prior frozen)
10. FP candidate F1: N/A
11. Privacy screen: N/A
12. Next: churn failure is detector sensitivity, not an FPR calibration failure.
