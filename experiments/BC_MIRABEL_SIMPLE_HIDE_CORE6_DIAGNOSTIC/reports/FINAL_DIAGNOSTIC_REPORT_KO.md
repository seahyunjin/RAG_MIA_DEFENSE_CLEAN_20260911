# BC-MIRABEL Simple-Hide Core6 진단

- Verdict: **PARTIAL_DIAGNOSTIC_ONLY_NEW_CORE6_SUBSTRATE_REQUIRED**

## 핵심 preflight 결과

| family   | status                          | reason                                                                                                      |
|:---------|:--------------------------------|:------------------------------------------------------------------------------------------------------------|
| DCMI     | PROTOCOL_OR_LINEAGE_UNAVAILABLE | PROTOCOL_OR_LINEAGE_UNAVAILABLE: frozen CLEAN corpus does not realize this family's member/nonmember labels |
| IA       | PROTOCOL_OR_LINEAGE_UNAVAILABLE | PROTOCOL_OR_LINEAGE_UNAVAILABLE: frozen CLEAN corpus does not realize this family's member/nonmember labels |
| MBA      | AVAILABLE                       | CLEAN membership labels match actual corpus inclusion/exclusion                                             |
| MEntA    | AVAILABLE                       | CLEAN membership labels match actual corpus inclusion/exclusion                                             |
| RAG-MIA  | PROTOCOL_OR_LINEAGE_UNAVAILABLE | PROTOCOL_OR_LINEAGE_UNAVAILABLE: frozen CLEAN corpus does not realize this family's member/nonmember labels |
| S²-MIA   | AVAILABLE                       | CLEAN membership labels match actual corpus inclusion/exclusion                                             |

동결 CLEAN DB에서 DCMI/IA/RAG-MIA의 member/nonmember label이 실제 문서 포함 여부와 일치하지 않아 세 공격은 fail-close했다. 이 상태에서 생성하면 membership 공격이 아니라 잘못된 label을 평가하게 된다.

## 유효한 동결 결과

| family   | status                          |   native_auc |   effective_auc_diagnostic |   member_retrieval_at4 |   member_alarm_rate |   member_effective_opportunity | classification                   |
|:---------|:--------------------------------|-------------:|---------------------------:|-----------------------:|--------------------:|-------------------------------:|:---------------------------------|
| MEntA    | VALID_FROZEN_RESULT             |       0.8613 |                     0.8613 |                 0.9100 |              0.4000 |                         0.3900 | DETECTION_LIMITED                |
| IA       | PROTOCOL_OR_LINEAGE_UNAVAILABLE |     nan      |                   nan      |               nan      |            nan      |                       nan      | UNAVAILABLE_MEMBERSHIP_SUBSTRATE |
| DCMI     | PROTOCOL_OR_LINEAGE_UNAVAILABLE |     nan      |                   nan      |               nan      |            nan      |                       nan      | UNAVAILABLE_MEMBERSHIP_SUBSTRATE |
| S²-MIA   | VALID_FROZEN_RESULT             |       0.6125 |                     0.6125 |                 1.0000 |              0.8000 |                         0.8000 | ALREADY_CONTROLLED               |
| MBA      | VALID_FROZEN_RESULT             |       0.6579 |                     0.6579 |                 0.9000 |              0.7500 |                         0.7500 | DETECTION_LIMITED                |
| RAG-MIA  | PROTOCOL_OR_LINEAGE_UNAVAILABLE |     nan      |                   nan      |               nan      |            nan      |                       nan      | UNAVAILABLE_MEMBERSHIP_SUBSTRATE |

## Benign FPR

| operating_point   |   threshold | domain          |   n |   actual_holdout_fpr |   false_positives |
|:------------------|------------:|:----------------|----:|---------------------:|------------------:|
| ORIGINAL          |      0.0000 | ALL             | 250 |               0.2520 |                63 |
| ORIGINAL          |      0.0000 | BeIR_nfcorpus   | 113 |               0.3186 |                36 |
| ORIGINAL          |      0.0000 | BeIR_scidocs    | 112 |               0.2232 |                25 |
| ORIGINAL          |      0.0000 | BeIR_trec-covid |  25 |               0.0800 |                 2 |
| q99               |      0.1264 | ALL             | 250 |               0.0080 |                 2 |
| q99               |      0.1264 | BeIR_nfcorpus   | 113 |               0.0177 |                 2 |
| q99               |      0.1264 | BeIR_scidocs    | 112 |               0.0000 |                 0 |
| q99               |      0.1264 | BeIR_trec-covid |  25 |               0.0000 |                 0 |
| q97               |      0.1117 | ALL             | 250 |               0.0080 |                 2 |
| q97               |      0.1117 | BeIR_nfcorpus   | 113 |               0.0177 |                 2 |
| q97               |      0.1117 | BeIR_scidocs    | 112 |               0.0000 |                 0 |
| q97               |      0.1117 | BeIR_trec-covid |  25 |               0.0000 |                 0 |
| q95               |      0.0930 | ALL             | 250 |               0.0280 |                 7 |
| q95               |      0.0930 | BeIR_nfcorpus   | 113 |               0.0442 |                 5 |
| q95               |      0.0930 | BeIR_scidocs    | 112 |               0.0179 |                 2 |
| q95               |      0.0930 | BeIR_trec-covid |  25 |               0.0000 |                 0 |

## 결론

새 방어를 만들기 전에 6개 공격의 member target은 포함하고 nonmember target은 제외한 새로운 CLEAN Core6 corpus를 결과 확인 전에 한 번 동결해야 한다. 이후 동일 BC-MIRABEL simple-hide로 여섯 공격을 평가해야 방향을 정할 수 있다.
