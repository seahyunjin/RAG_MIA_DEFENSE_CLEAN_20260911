# Corrected benign utility

| protocol | subset | method | n | Token_F1 | relative_F1_change_vs_NoDefense | NLI_support | support_delta_vs_NoDefense | unsupported | contradiction | refusal | answerability | NovelFactRecall | numeric_accuracy | detail_accuracy |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| REFERENCE_HISTORY | Ordinary natural benign | No Defense | 50 | 0.237671 | 0.000000 | 0.720000 | 0.000000 | 0.280000 | 0.043333 | 0.000000 | 1.000000 | 0.455395 | 1.000000 | 0.444226 |
| REFERENCE_HISTORY | Ordinary natural benign | CLD | 50 | 0.233772 | -0.016408 | 0.720000 | 0.000000 | 0.280000 | 0.043333 | 0.000000 | 1.000000 | 0.449681 | 1.000000 | 0.436726 |
| REFERENCE_HISTORY | Ordinary natural benign | Frozen Dual | 50 | 0.228575 | -0.038274 | 0.715476 | -0.004524 | 0.284524 | 0.025000 | 0.000000 | 1.000000 | 0.422290 | 0.500000 | 0.411690 |
| REFERENCE_HISTORY | Reference-history conversational benign | No Defense | 200 | 0.213330 | 0.000000 | 0.528966 | 0.000000 | 0.471034 | 0.064759 | 0.170000 | 0.830000 | 0.241039 | 0.127778 | 0.302761 |
| REFERENCE_HISTORY | Reference-history conversational benign | CLD | 200 | 0.210257 | -0.014404 | 0.513990 | -0.014976 | 0.486010 | 0.070202 | 0.175000 | 0.825000 | 0.234726 | 0.127778 | 0.298261 |
| REFERENCE_HISTORY | Reference-history conversational benign | Frozen Dual | 200 | 0.208696 | -0.021723 | 0.498333 | -0.030633 | 0.501667 | 0.063636 | 0.175000 | 0.825000 | 0.232915 | 0.127778 | 0.294731 |
| REFERENCE_HISTORY | True same-document deep-dive | No Defense | 154 | 0.208913 | 0.000000 | 0.563104 | 0.000000 | 0.436896 | 0.061705 | 0.149351 | 0.850649 | 0.287668 | 0.141667 | 0.337692 |
| REFERENCE_HISTORY | True same-document deep-dive | CLD | 154 | 0.208879 | -0.000164 | 0.553974 | -0.009130 | 0.446026 | 0.068590 | 0.155844 | 0.844156 | 0.289484 | 0.141667 | 0.339515 |
| REFERENCE_HISTORY | True same-document deep-dive | Frozen Dual | 154 | 0.205956 | -0.014152 | 0.531081 | -0.032024 | 0.468919 | 0.059615 | 0.155844 | 0.844156 | 0.279992 | 0.141667 | 0.328925 |
| CLOSED_LOOP | Closed-loop conversational benign | No Defense | 200 | 0.203348 | 0.000000 | 0.469223 | 0.000000 | 0.530777 | 0.087121 | 0.120000 | 0.880000 | 0.224836 | 0.122222 | 0.280998 |
| CLOSED_LOOP | Closed-loop conversational benign | CLD | 200 | 0.202317 | -0.005067 | 0.465238 | -0.003985 | 0.534762 | 0.085238 | 0.125000 | 0.875000 | 0.224240 | 0.122222 | 0.280796 |
| CLOSED_LOOP | Closed-loop conversational benign | Frozen Dual | 200 | 0.206993 | 0.017929 | 0.460985 | -0.008239 | 0.539015 | 0.076231 | 0.120000 | 0.880000 | 0.242667 | 0.155556 | 0.297210 |
