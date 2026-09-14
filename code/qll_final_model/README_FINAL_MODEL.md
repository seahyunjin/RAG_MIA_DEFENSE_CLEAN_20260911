# Stateless QLL Source Hide — Final Model

## Exact model

Retriever Top-4 → QLL → dominance → benign-calibrated threshold → optional one-source hide → one generation.

QLL is the length-normalized mean query-token log probability conditioned on each source. Dominance is the maximum softmax probability over the four QLL values. Calibration uses the 95th percentile of benign dominance only; attack examples are not accepted.

The layer runs between retriever and generator. Final context policy: **EQUAL_REDISTRIBUTION**.

No Sticky, C1, Mirabel runtime, GlobalCap64, Prefix64, learned attack classifier, attack-specific threshold, or rank-specific rule is present.
