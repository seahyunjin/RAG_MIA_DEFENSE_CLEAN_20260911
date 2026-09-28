# Final attack performance

| threat_scope | attack | method | cohort | N | AUROC_or_leakage | paired_reduction_NoDefense_minus_method | Dual_minus_CLD | lineage |
|---|---|---|---|---|---|---|---|---|
| single-query | MBA | NO_DEFENSE | prospective balanced DEV | 100 | 0.867600 |  |  | FOUR_ARM_PROSPECTIVE |
| single-query | RAG-MIA | NO_DEFENSE | prospective balanced DEV | 100 | 0.960000 |  |  | FOUR_ARM_PROSPECTIVE |
| single-query | S²-MIA | NO_DEFENSE | prospective balanced DEV | 100 | 0.680000 |  |  | FOUR_ARM_PROSPECTIVE |
| single-query | DCMI-Std-Q2 | NO_DEFENSE | prospective balanced DEV | 100 | 0.970000 |  |  | FOUR_ARM_PROSPECTIVE |
| single-query | MBA | CLD_ONLY | prospective balanced DEV | 100 | 0.530400 |  |  | FOUR_ARM_PROSPECTIVE |
| single-query | RAG-MIA | CLD_ONLY | prospective balanced DEV | 100 | 0.500000 |  |  | FOUR_ARM_PROSPECTIVE |
| single-query | S²-MIA | CLD_ONLY | prospective balanced DEV | 100 | 0.570000 |  |  | FOUR_ARM_PROSPECTIVE |
| single-query | DCMI-Std-Q2 | CLD_ONLY | prospective balanced DEV | 100 | 0.510000 |  |  | FOUR_ARM_PROSPECTIVE |
| single-query | MBA | CFCLIP_ONLY | prospective balanced DEV | 100 | 0.870600 |  |  | FOUR_ARM_PROSPECTIVE |
| single-query | RAG-MIA | CFCLIP_ONLY | prospective balanced DEV | 100 | 0.720000 |  |  | FOUR_ARM_PROSPECTIVE |
| single-query | S²-MIA | CFCLIP_ONLY | prospective balanced DEV | 100 | 0.690000 |  |  | FOUR_ARM_PROSPECTIVE |
| single-query | DCMI-Std-Q2 | CFCLIP_ONLY | prospective balanced DEV | 100 | 0.930000 |  |  | FOUR_ARM_PROSPECTIVE |
| single-query | MBA | CLD_CFCLIP | prospective balanced DEV | 100 | 0.520400 | 0.347200 | -0.010000 | FOUR_ARM_PROSPECTIVE |
| single-query | RAG-MIA | CLD_CFCLIP | prospective balanced DEV | 100 | 0.500000 | 0.460000 | 0.000000 | FOUR_ARM_PROSPECTIVE |
| single-query | S²-MIA | CLD_CFCLIP | prospective balanced DEV | 100 | 0.560000 | 0.120000 | -0.010000 | FOUR_ARM_PROSPECTIVE |
| single-query | DCMI-Std-Q2 | CLD_CFCLIP | prospective balanced DEV | 100 | 0.500000 | 0.470000 | -0.010000 | FOUR_ARM_PROSPECTIVE |
| session Q1-Q5 | IA | NO_DEFENSE | 92 paired target sessions | 184 | 0.864721 |  |  | SERIAL_KVCACHE_EXACT |
| session Q1-Q5 | IA | CLD_ONLY | 92 paired target sessions | 184 | 0.845818 |  |  | SERIAL_KVCACHE_EXACT |
| session Q1-Q5 | IA | CLD_CFCLIP | 92 paired target sessions | 184 | 0.709712 | 0.15500945179584125 | -0.136106 | SERIAL_KVCACHE_EXACT |
| session Q1-Q5 | IA | CFCLIP_ONLY | 92 paired target sessions | 184 | 0.714674 |  |  | SUPPLEMENTARY_BATCHED_FOUR_ARM_NOT_FINAL_PLOT_SOURCE |
| session Q1-Q5 | MEntA | NO_DEFENSE | fresh zero-overlap 100+100 sessions | 200 | 0.784450 |  |  | FRESH_ZERO_OVERLAP_PRIMARY |
| session Q1-Q5 | MEntA | CLD_ONLY | fresh zero-overlap 100+100 sessions | 200 | 0.704900 |  |  | FRESH_ZERO_OVERLAP_PRIMARY |
| session Q1-Q5 | MEntA | CLD_CFCLIP | fresh zero-overlap 100+100 sessions | 200 | 0.715550 | 0.068900 | 0.010650 | FRESH_ZERO_OVERLAP_PRIMARY |
| session Q1-Q5 | MEntA | CFCLIP_ONLY | separate balanced four-arm DEV | 200 | 0.739350 |  |  | SUPPLEMENTARY_NOT_PRIMARY_MENTA_COHORT |
