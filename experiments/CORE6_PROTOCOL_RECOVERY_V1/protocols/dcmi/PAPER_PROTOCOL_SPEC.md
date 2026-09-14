# DCMI — PAPER PROTOCOL SPEC

Source: https://arxiv.org/abs/2509.06026

## 논문

DCMI: A Differential Calibration Membership Inference Attack Against Retrieval-Augmented Generation

## 학회/연도

CCS 2025 / arXiv:2509.06026

## 공격 목표

isolate membership sensitivity using an original and minimally perturbed query pair

## Member

original target sample is in the retrieval database

## Nonmember

original target sample is outside the retrieval database

## Target 요건

text with perturbable adjectives/adverbs

## Query 수

2

## Query 생성

original sample and antonym-perturbed sample in the same Yes/No template; black-box setting uses 6% perturbation

## 생성 모델

paper says a third-party LLM but does not identify/freeze it; official repo uses gpt-4o

## Prompt

paper Appendix D perturbation prompt and 'Is this ... right?' query prompt

## Decoding

not specified for perturbation LLM

## Session

ordered original then perturbed pair

## Retrieval 가정

black-box access to final Yes/No answers

## Membership score

f_rag(q)-f_rag(q'), with Yes=1 and No=0

## Aggregation

one pairwise difference per target

## Threshold

black-box discrete decision; paper reports threshold/reference variations by adversary setting

## 원 평가 metric

ROC-AUC, accuracy, precision, recall, F1

## Official repository

https://github.com/Xinyu140203/RAG_MIA

## Commit

ef961a1f23d09c62c5fbf69d798e325fb0362e55

## Paper PDF SHA-256

ee83fcb9ec83a0a7e8d8aedbf0a87f50e351a49a896a1a144b75ec7b12d3f4ef

## Official repo alignment

NO: repository perturbation is 3% and scorer entry point contains placeholders; paper black-box condition uses 6%

## 논문↔repo 차이

official MIA.py contains placeholder arrays; perturb.py fixes 3%, while paper black-box experiments use 6%

## 구현 명칭

DCMI_PAPER_FAITHFUL_COMPONENT_REIMPLEMENTATION

## 현재 판정

DCMI_SPEC_UNDERDETERMINED

## 미결정 사항

paper does not freeze the perturbation LLM revision or decoding configuration

## Metric provenance rule

이 공격의 원 metric만 공식 결과로 사용한다. E-AUC/symmetric AUC는 INTERNAL_DIAGNOSTIC_ONLY이다.
