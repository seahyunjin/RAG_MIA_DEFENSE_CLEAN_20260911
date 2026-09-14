# Final LC CPU-only Analysis Sidecar

- Completed UTC: `2026-09-13T08:10:33.009545+00:00`
- Precommit SHA-256: `e78db2824c3de2bbbcedcbc07aa6c626ba8796fb6506656d8525fbdc7f28192c`
- IA GPU generation was not stopped, signalled, or modified.
- No GPU/model/retrieval/generation/training was used by this sidecar.

## CORE5 STATISTICS

| Attack | MIRABEL TPR | Final LC TPR | Delta (95% cluster CI) | Recovered/lost queries | Recovered/lost units |
|---|---:|---:|---:|---:|---:|
| MEntA | 0.8486 | 0.8840 | +0.0354 [+0.0288, +0.0420] | 189/12 | 31/0 |
| MBA | 0.9880 | 0.9960 | +0.0080 [+0.0030, +0.0140] | 8/0 | 8/0 |
| RAG-MIA | 0.8880 | 0.9520 | +0.0640 [+0.0480, +0.0800] | 68/4 | 68/4 |
| S²-MIA | 0.9800 | 0.9962 | +0.0163 [+0.0088, +0.0250] | 13/0 | 13/0 |
| DCMI-Std-Q2 | 0.7285 | 0.7845 | +0.0560 [+0.0455, +0.0670] | 117/5 | 27/0 |

## CORE5 E2E CI

Native metrics are primary. E-AUC is retained only as a secondary diagnostic where defined.

| Attack | Condition | Native point | 95% CI | Delta vs No Defense | Delta vs matched MIRABEL |
|---|---|---:|---:|---:|---:|
| MEntA | NO_DEFENSE | 0.9783 | [0.9707, 0.9851] | +0.0000 | +0.3764 |
| MEntA | MIRABEL_MATCHED_2_5 | 0.6019 | [0.5822, 0.6212] | -0.3764 | +0.0000 |
| MEntA | FINAL_LC_MATCHED_2_5 | 0.5854 | [0.5650, 0.6061] | -0.3929 | -0.0165 |
| MBA | NO_DEFENSE | 0.9098 | [0.8967, 0.9214] | +0.0000 | +0.4101 |
| MBA | MIRABEL_MATCHED_2_5 | 0.4997 | [0.4884, 0.5114] | -0.4101 | +0.0000 |
| MBA | FINAL_LC_MATCHED_2_5 | 0.4980 | [0.4869, 0.5094] | -0.4119 | -0.0018 |
| RAG-MIA | NO_DEFENSE | 0.9810 | [0.9745, 0.9865] | +0.0000 | +0.4270 |
| RAG-MIA | MIRABEL_MATCHED_2_5 | 0.5540 | [0.5440, 0.5640] | -0.4270 | +0.0000 |
| RAG-MIA | FINAL_LC_MATCHED_2_5 | 0.5230 | [0.5165, 0.5305] | -0.4580 | -0.0310 |
| S²-MIA | NO_DEFENSE | 0.6859 | [0.6633, 0.7071] | +0.0000 | +0.1802 |
| S²-MIA | MIRABEL_MATCHED_2_5 | 0.5056 | [0.4862, 0.5244] | -0.1802 | +0.0000 |
| S²-MIA | FINAL_LC_MATCHED_2_5 | 0.5063 | [0.4894, 0.5250] | -0.1796 | +0.0006 |
| DCMI-Std-Q2 | NO_DEFENSE | 0.9795 | [0.9730, 0.9855] | +0.0000 | +0.4610 |
| DCMI-Std-Q2 | MIRABEL_MATCHED_2_5 | 0.5185 | [0.5105, 0.5270] | -0.4610 | +0.0000 |
| DCMI-Std-Q2 | FINAL_LC_MATCHED_2_5 | 0.5060 | [0.4995, 0.5125] | -0.4735 | -0.0125 |

## DOMAIN FPR

| Method | Domain | FPR | Wilson 95% CI |
|---|---|---:|---:|
| Original MIRABEL | ALL | 0.3050 | [0.2773, 0.3342] |
| Original MIRABEL | nfcorpus | 0.3220 | [0.2825, 0.3642] |
| Original MIRABEL | scidocs | 0.2800 | [0.2415, 0.3220] |
| Original MIRABEL | trec-covid | 0.4400 | [0.2667, 0.6293] |
| Global BC | ALL | 0.0300 | [0.0211, 0.0425] |
| Global BC | nfcorpus | 0.0240 | [0.0138, 0.0415] |
| Global BC | scidocs | 0.0379 | [0.0241, 0.0591] |
| Global BC | trec-covid | 0.0000 | [0.0000, 0.1332] |
| Final LC | ALL | 0.0250 | [0.0170, 0.0366] |
| Final LC | nfcorpus | 0.0160 | [0.0081, 0.0313] |
| Final LC | scidocs | 0.0358 | [0.0225, 0.0566] |
| Final LC | trec-covid | 0.0000 | [0.0000, 0.1332] |
| Original MIRABEL | WORST_DOMAIN | 0.4400 | [0.2667, 0.6293] |
| Global BC | WORST_DOMAIN | 0.0379 | [0.0241, 0.0591] |
| Final LC | WORST_DOMAIN | 0.0358 | [0.0225, 0.0566] |

## ABLATION

Low-FPR curves for Raw MIRABEL, global benign calibration, semantic-local LC, Final LC, and random-neighbor LC are saved as CSV. Final LC and LC have identical same-FPR ranking because both use R_LC; the outer threshold only defines the fixed deployment point.

Random-neighbor semantic recovery check:

| Attack | Semantic TPR | Random TPR | Semantic-only recovered |
|---|---:|---:|---:|
| MEntA | 0.8840 | 0.8124 | 378 |
| RAG-MIA | 0.9520 | 0.8430 | 110 |

Optional k sensitivity was not run: S² query embeddings are not frozen, and recomputing/model-loading was forbidden. The current k=200 was not reselected.

## GOLD STRICT VS REFRESH

| Condition | Intervention | Gold F1 | Delta F1 | Refusal | New refusal | Empty | Mean words |
|---|---:|---:|---:|---:|---:|---:|---:|
| No Defense | 0.000 | 0.2089 | +0.0000 | 0.435 | 0.000 | 0.000 | 17.4 |
| Final LC strict transfer | 0.692 | 0.1779 | -0.0310 | 0.477 | 0.058 | 0.000 | 17.1 |
| Final LC benign refresh | 0.027 | 0.2079 | -0.0010 | 0.434 | 0.001 | 0.000 | 17.6 |

## FP 27 DAMAGE AUDIT

All 27 refreshed false-positive interventions are in `tables/FP_27_CASE_DAMAGE_AUDIT.csv`, including answers, rank, qrels, score damage, refusal, and exact lexical evidence flags.

## FP HARM MECHANISM

| Class | N | Mean Delta Gold F1 |
|---|---:|---:|
| UNIQUE_EVIDENCE_REMOVED | 0 | +nan |
| REDUNDANT_EVIDENCE_AVAILABLE | 11 | -0.0248 |
| REFUSAL_DRIVEN | 6 | -0.0488 |
| UNRESOLVED | 10 | -0.0482 |

This is an exact normalized-string diagnostic, not semantic evidence or NLI.

## DB CHURN MANIFEST

| Version | Removed/added | Member audit | Nonmember present | Manifest hash |
|---|---:|---:|---:|---|
| V0 | 0/0 | True | 0 | `7053a12fad418757cd4226c001633a7c558ad5b1e26e2bd76a57cbb8fed2f463` |
| V10 | 223/223 | True | 0 | `4008381323a40829fa43b47cd874836011608c7f622e58736549e7ec509921db` |
| V25 | 556/556 | True | 0 | `c2077f540a652d34e79da7bfad7b9d1035051bce12f715e98cac2a85cb8b3ae8` |
| V50 | 1113/1113 | True | 0 | `922efe10dbf648359cdfd7cf9fbad43d42cc67bc1958ea6d50ce9c57d0de4e38` |

Only deterministic manifests were created; no index or embeddings were built.

## NEW DOMAIN READINESS

No locally complete untouched-domain substrate is READY. PubMedQA and HotpotQA remain candidates pending immutable corpus construction and formal overlap hashes; TopiOCQA/QuAC/QReCC are not untouched.

## RAGLEAK/BUDGETLEAK STATUS

| Attack | Status | Missing critical artifact |
|---|---|---|
| RAGLeak | UNAVAILABLE | exact crops, prompt, scorer implementation, immutable target pairing |
| BudgetLeak | UNAVAILABLE | author scorer and paired QA/reference bundle on current Core5 target universe |

Neither unavailable protocol is mixed with Final LC Core5 results.

## COST/LATENCY

See `tables/COST_LATENCY.csv`. Per-query detector timing is reported only for the instrumented combined BGE encoding + retrieval + MIRABEL/LC scoring path; kNN and MIRABEL subcomponents were not separately timed.

## Scientific boundary

This sidecar confirms frozen Core5 statistics and diagnoses calibration/generalization preparation. It does not add IA, RAGLeak, BudgetLeak, a new domain result, or a new detector. Those claims remain pending their protocol/input gates.
