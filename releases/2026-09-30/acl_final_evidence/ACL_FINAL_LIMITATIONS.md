# ACL Final Limitations

1. The exact IA Q1–Q5 evidence uses a local question-only ConvFinQA protocol; the Q30 result is explicitly a local NFCorpus surrogate, not paper-exact IA.
2. The strongest single-query results are a prospective balanced DEV, not a held-out multi-seed confirmatory test. S²'s paired reduction CI crosses zero.
3. MEntA component evidence is mixed: fresh primary RIF improves over No Defense but is slightly worse than CLD at Q5, so CF-Clip complementarity is IA-specific in current evidence.
4. Stateless RIF leaves substantial long-horizon IA leakage (Q30 AUROC 0.8900). B=2 reduces it by refusing protected-evidence use after Q2 and therefore causes severe later-turn utility loss.
5. B=2 requires durable authenticated identity. Full credential reset is an acknowledged Sybil bypass; conversation-ID reset alone is not.
6. Exact RIF runtime is about 3.95× Vanilla per query (0.440092 vs 0.111453 seconds) and 8.34 vs 29.46 tokens/s on the audited hardware.
7. Full RIF zero-shot transfer is missing. Frozen-tau CLD is unstable across GTE/MPNet, and CF-Clip has not been exactly evaluated on another generator family.
8. Strong baselines are not available on an identical end-to-end cohort. MIRABEL's matched-FPR IA audit has a protocol mismatch; PAD, EPD, and DP-Fusion require nontrivial incompatible ports or extra resources.
9. DCMI-Std-Q2 is standardized rather than paper-exact DCMI. Results must retain the qualifier.
10. Some four-arm IA/MEntA evidence is supplementary/exploratory batched lineage. It is not merged with exact/fresh primary rows.
11. Utility relies on stored automatic metrics and limited numeric-reference coverage; no broad human preference or numeric accuracy claim is warranted.
12. Adaptive robustness supports within-account candidate reordering under B=2, but full account reset restores IA leakage to AUROC 0.8900.
