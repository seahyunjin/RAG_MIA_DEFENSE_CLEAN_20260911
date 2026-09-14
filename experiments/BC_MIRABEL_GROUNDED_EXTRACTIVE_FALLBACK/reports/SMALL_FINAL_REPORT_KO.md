# BC-GEF Small 결과

- Verdict: **BC_GEF_SMALL_REJECTED**
- Risk queries: **73**
- Extractive empty cases: **0**
- Exact extractive provenance coverage: **100.00%**

## Original/frozen attack metrics (primary)

| condition        | family   |   sessions |   member_n |   nonmember_n |   native_auc |   native_auc_ci_low |   native_auc_ci_high |   member_mean |   nonmember_mean |
|:-----------------|:---------|-----------:|-----------:|--------------:|-------------:|--------------------:|---------------------:|--------------:|-----------------:|
| BC_CGD           | MBA      |         37 |         19 |            18 |       0.5322 |              0.4211 |               0.6389 |        0.1158 |           0.0278 |
| BC_CGD           | MEntA    |         40 |         20 |            20 |       0.8850 |              0.7612 |               0.9788 |        0.4100 |          -0.0200 |
| BC_CGD           | S²-MIA   |         40 |         20 |            20 |       0.4300 |              0.2525 |               0.6200 |        0.5369 |           0.5867 |
| BC_GEF           | MBA      |         37 |         19 |            18 |       0.5322 |              0.4211 |               0.6389 |        0.1158 |           0.0278 |
| BC_GEF           | MEntA    |         40 |         20 |            20 |       0.8850 |              0.7625 |               0.9775 |        0.4300 |          -0.0200 |
| BC_GEF           | S²-MIA   |         40 |         20 |            20 |       0.8075 |              0.6500 |               0.9450 |        0.7506 |           0.5867 |
| BC_MIRABEL       | MBA      |         37 |         19 |            18 |       0.6579 |              0.5219 |               0.7895 |        0.1684 |           0.0278 |
| BC_MIRABEL       | MEntA    |         40 |         20 |            20 |       0.8613 |              0.7150 |               0.9775 |        0.4100 |          -0.0200 |
| BC_MIRABEL       | S²-MIA   |         40 |         20 |            20 |       0.6125 |              0.4300 |               0.7900 |        0.6399 |           0.5867 |
| ORIGINAL_MIRABEL | MBA      |         37 |         19 |            18 |       0.6272 |              0.4942 |               0.7632 |        0.1053 |           0.0278 |
| ORIGINAL_MIRABEL | MEntA    |         40 |         20 |            20 |       0.7588 |              0.6137 |               0.8875 |        0.2100 |          -0.0400 |
| ORIGINAL_MIRABEL | S²-MIA   |         40 |         20 |            20 |       0.6500 |              0.4700 |               0.8225 |        0.6355 |           0.5710 |

## Gate

| check                                 | pass   |
|:--------------------------------------|:-------|
| menta_gain_over_bc_ge_0_05            | False  |
| menta_better_than_original            | False  |
| menta_native_auc_le_0_75              | False  |
| s2_not_worse_than_original_plus_0_05  | False  |
| mba_not_worse_than_original_plus_0_05 | True   |
| one_direct_equal_or_better            | True   |

## Benign answer-preservation diagnostic

| condition        |   queries |   intervention_rate |   safe_path_exact_preservation |   answer_preservation_token_f1 |   answer_preservation_exact_match |   refusal_rate |   new_refusal |   empty_answer_rate |   mean_answer_chars |
|:-----------------|----------:|--------------------:|-------------------------------:|-------------------------------:|----------------------------------:|---------------:|--------------:|--------------------:|--------------------:|
| BC_CGD           |       250 |              0.0080 |                         1.0000 |                         0.9920 |                            0.9920 |         0.6880 |        0.0080 |              0.0000 |            142.3000 |
| BC_GEF           |       250 |              0.0080 |                         1.0000 |                         0.9929 |                            0.9920 |         0.6800 |        0.0000 |              0.0000 |            144.2640 |
| BC_MIRABEL       |       250 |              0.0080 |                         1.0000 |                         0.9920 |                            0.9920 |         0.6880 |        0.0080 |              0.0000 |            142.3000 |
| NO_DEFENSE       |       250 |              0.0000 |                         1.0000 |                         1.0000 |                            1.0000 |         0.6800 |        0.0000 |              0.0000 |            143.9320 |
| ORIGINAL_MIRABEL |       250 |              0.2520 |                         1.0000 |                         0.8757 |                            0.8480 |         0.7640 |        0.0840 |              0.0000 |            108.9200 |

이 benign 지표는 A0 답변 보존성이지 gold QA 정답률이 아니다. Risk-path 문장은 남은 검색 문서에서 직접 추출되지만 `hallucination-free`라고 주장하지 않는다.
