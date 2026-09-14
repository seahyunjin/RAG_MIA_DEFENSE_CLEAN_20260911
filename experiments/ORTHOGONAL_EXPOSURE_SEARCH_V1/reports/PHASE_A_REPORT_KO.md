# Phase A — Existing-signal Complementarity Audit

- 판정: `A_NO_MEANINGFUL_COMPLEMENT`
- QLL: exact recalculation, mechanism audit only (non-black-box)
- Answer-LOO: `SIGNAL_UNAVAILABLE` (fresh A0/ablation provenance 없음)

| Signal | MEntA unique recovery | MEntA net recovery | Runtime |
|---|---:|---:|---|
| DualTail | 0.022 | -0.012 | STRICT_BLACKBOX |
| QLL | 0.000 | -0.884 | ANALYSIS_ONLY_NON_BLACKBOX |

Phase A에서는 fusion/model selection을 하지 않았다. 명세대로 Phase B로 이동한다.
