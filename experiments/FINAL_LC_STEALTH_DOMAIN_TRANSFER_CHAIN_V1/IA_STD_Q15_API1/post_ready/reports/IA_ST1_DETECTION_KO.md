# IA-Std-Q15-ST1 matched-FPR detection

- Verdict: `IA_STEALTH_DETECTION_PASS`
- Valid sessions: `1941`
- Primary actual benign FPR: MIRABEL `0.0250`, Final LC `0.0250`

| Nominal benign FPR | MIRABEL TPR | Final LC TPR | Delta |
|---:|---:|---:|---:|
| 1.0% | 0.0837 | 0.0771 | -0.0066 |
| 2.5% | 0.1504 | 0.1667 | +0.0163 |
| 3.0% | 0.1661 | 0.1667 | +0.0006 |
| 5.0% | 0.2348 | 0.2548 | +0.0200 |

- Session-cluster bootstrap delta 95% CI: `[0.0125, 0.0202]`
- Claim boundary: IA-ST1 is a standardized stealth stress test, not Original/paper-exact IA.
