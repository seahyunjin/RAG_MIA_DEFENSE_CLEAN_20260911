# Constructed Hard-Benign V1 최종 보고

이 평가는 외부 표준 benchmark가 아니라, detector score를 보기 전에 FinQA 정상 질의/문서로 만든 독립 constructed benchmark다.

## False-positive rate

| Subgroup | Method | N | Alarm | FPR | Wilson 95% CI |
|---|---:|---:|---:|---:|---:|
| Natural | MIRABEL | 499 | 14 | 0.0281 | [0.0168, 0.0465] |
| Natural | FINAL_V2 | 499 | 15 | 0.0301 | [0.0183, 0.0490] |
| Deep-study | MIRABEL | 300 | 6 | 0.0200 | [0.0092, 0.0429] |
| Deep-study | FINAL_V2 | 300 | 11 | 0.0367 | [0.0206, 0.0645] |
| Adaptive clarification | MIRABEL | 300 | 10 | 0.0333 | [0.0182, 0.0603] |
| Adaptive clarification | FINAL_V2 | 300 | 8 | 0.0267 | [0.0136, 0.0517] |
| Reask / paraphrase | MIRABEL | 293 | 13 | 0.0444 | [0.0261, 0.0744] |
| Reask / paraphrase | FINAL_V2 | 293 | 13 | 0.0444 | [0.0261, 0.0744] |
| Exact-Fact | MIRABEL | 291 | 2 | 0.0069 | [0.0019, 0.0247] |
| Exact-Fact | FINAL_V2 | 291 | 11 | 0.0378 | [0.0212, 0.0664] |
| Yes/No | MIRABEL | 143 | 5 | 0.0350 | [0.0150, 0.0792] |
| Yes/No | FINAL_V2 | 143 | 8 | 0.0559 | [0.0286, 0.1065] |
| True/False | MIRABEL | 149 | 1 | 0.0067 | [0.0012, 0.0370] |
| True/False | FINAL_V2 | 149 | 4 | 0.0268 | [0.0105, 0.0670] |
| MCQ | MIRABEL | 138 | 1 | 0.0072 | [0.0013, 0.0399] |
| MCQ | FINAL_V2 | 138 | 2 | 0.0145 | [0.0040, 0.0513] |
| Discrete Macro | MIRABEL | 430 | 7 | 0.0163 | [0.0079, 0.0332] |
| Discrete Macro | FINAL_V2 | 430 | 14 | 0.0326 | [0.0195, 0.0539] |
| Hard-Benign Macro | MIRABEL | 2113 | 52 | 0.0246 | [0.0188, 0.0321] |
| Hard-Benign Macro | FINAL_V2 | 2113 | 72 | 0.0341 | [0.0271, 0.0427] |

## Answer utility

| Subgroup | Method | N | Preservation F1 | New refusal | Gold N | Gold F1 |
|---|---:|---:|---:|---:|---:|---:|
| Natural | NO_DEFENSE | 499 | 1.0000 | 0.0000 | 499 | 0.0071 |
| Natural | MIRABEL | 499 | 0.9843 | 0.0100 | 499 | 0.0070 |
| Natural | FINAL_V2 | 499 | 0.9776 | 0.0160 | 499 | 0.0066 |
| Deep-study | NO_DEFENSE | 300 | 1.0000 | 0.0000 | 0 |  |
| Deep-study | MIRABEL | 300 | 0.9878 | 0.0100 | 0 |  |
| Deep-study | FINAL_V2 | 300 | 0.9690 | 0.0300 | 0 |  |
| Adaptive clarification | NO_DEFENSE | 300 | 1.0000 | 0.0000 | 0 |  |
| Adaptive clarification | MIRABEL | 300 | 0.9796 | 0.0167 | 0 |  |
| Adaptive clarification | FINAL_V2 | 300 | 0.9783 | 0.0167 | 0 |  |
| Reask / paraphrase | NO_DEFENSE | 293 | 1.0000 | 0.0000 | 0 |  |
| Reask / paraphrase | MIRABEL | 293 | 0.9718 | 0.0239 | 0 |  |
| Reask / paraphrase | FINAL_V2 | 293 | 0.9681 | 0.0307 | 0 |  |
| Exact-Fact | NO_DEFENSE | 291 | 1.0000 | 0.0000 | 0 |  |
| Exact-Fact | MIRABEL | 291 | 0.9931 | 0.0069 | 0 |  |
| Exact-Fact | FINAL_V2 | 291 | 0.9751 | 0.0206 | 0 |  |
| Yes/No | NO_DEFENSE | 143 | 1.0000 | 0.0000 | 0 |  |
| Yes/No | MIRABEL | 143 | 0.9774 | 0.0140 | 0 |  |
| Yes/No | FINAL_V2 | 143 | 0.9566 | 0.0420 | 0 |  |
| True/False | NO_DEFENSE | 149 | 1.0000 | 0.0000 | 0 |  |
| True/False | MIRABEL | 149 | 0.9933 | 0.0067 | 0 |  |
| True/False | FINAL_V2 | 149 | 0.9803 | 0.0134 | 0 |  |
| MCQ | NO_DEFENSE | 138 | 1.0000 | 0.0000 | 0 |  |
| MCQ | MIRABEL | 138 | 1.0000 | 0.0000 | 0 |  |
| MCQ | FINAL_V2 | 138 | 0.9857 | 0.0145 | 0 |  |
| Discrete Macro | NO_DEFENSE | 430 | 1.0000 | 0.0000 | 0 |  |
| Discrete Macro | MIRABEL | 430 | 0.9902 | 0.0070 | 0 |  |
| Discrete Macro | FINAL_V2 | 430 | 0.9741 | 0.0233 | 0 |  |
| Hard-Benign Macro | NO_DEFENSE | 2113 | 1.0000 | 0.0000 | 499 | 0.0071 |
| Hard-Benign Macro | MIRABEL | 2113 | 0.9848 | 0.0118 | 499 | 0.0070 |
| Hard-Benign Macro | FINAL_V2 | 2113 | 0.9741 | 0.0222 | 499 | 0.0066 |

## Factuality boundary

- Changed/intervened answer NLI proxy rows: 90
- NLI는 hallucination ground truth로 해석하지 않았다.
- deterministic 50-case human audit template의 label은 작성하지 않고 `REQUIRES_HUMAN_REVIEW`로 남겼다.
- 생성 subgroup에는 공인 gold label이 없으므로 answer preservation만 보고한다.
