# Final limitations

- IA-Std-Q15-API1 remains the hardest standardized stress test and is not paper-exact IA.
- Hard-benign false-positive risk requires the separately precommitted constructed benchmark.
- Calibration is sensitive to contamination; MAD is a robustness diagnostic, not a universally fixed numeric constant.
- The numeric threshold is not domain invariant; FinQA requires benign-only recalibration.
- FinQA member TPR is materially below Core member TPR.
- MPNet transfer has weaker absolute performance.
- Rank-1 hide can leave a target at ranks 2–4 under decoy/rank shift.
- No formal differential-privacy guarantee is provided.
- No universal MIA guarantee is supported.
- NLI factuality scores are proxies and do not establish hallucination-free behavior.
