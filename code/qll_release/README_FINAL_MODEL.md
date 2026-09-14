# Stateless QLL Source Hide — Final Code Archive

Frozen runtime: **Stateless QLL Source Hide + Equal Redistribution**.

1. Retrieve Top-4 sources.
2. Compute length-normalized query log-likelihood for every source.
3. Softmax the four QLL values and take dominance `d = max(p_i)`.
4. Compare with a benign-only 95th-percentile threshold using strict `>`.
5. If triggered, hide the current QLL top-1 source.
6. Equal-redistribute the 2,048-token context budget over surviving sources.
7. Generate exactly once; keep no session state.

Reference threshold: `0.5300846414247485`. Attack examples used for calibration: `0`.

The `experiment_history/` directory is historical evidence only. Failed C1, Sticky,
Cap64, SE-Mirabel and No-Redistribution branches are not imported by `final_model/`.
