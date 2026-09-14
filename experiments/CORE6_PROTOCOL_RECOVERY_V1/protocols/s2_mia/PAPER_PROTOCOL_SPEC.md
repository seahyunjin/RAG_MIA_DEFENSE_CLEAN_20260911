# S²-MIA — PAPER PROTOCOL SPEC

Source: https://arxiv.org/abs/2406.19234

## 논문

Generating Is Believing: Membership Inference Attacks against Retrieval-Augmented Generation

## 학회/연도

arXiv 2024

## 공격 목표

infer membership from generated-text semantic overlap and generation perplexity

## Member

target sample is in the retrieval database

## Nonmember

target sample is outside the retrieval database

## Target 요건

structured question/answer or explicit query/remaining-text pair

## Query 수

1

## Query 생성

use target question/first part as the single query

## 생성 모델

None

## Prompt

paper prompt reproduced in query_generator.py

## Decoding

target RAG configuration; token log probabilities required for perplexity

## Session

one query per target

## Retrieval 가정

paper uses Contriever/DPR, top-k=5

## Membership score

features=(BLEU, generation perplexity); T variant applies learned thresholds; M variant trains a classifier

## Aggregation

single-query feature vector

## Threshold

greedy reference-set calibration for S²-MIA-T

## 원 평가 metric

ROC-AUC, PR-AUC

## Official repository

not located

## Commit

None

## Paper PDF SHA-256

d41e78fc61d5b5ac49f880f8ebffa2ba1434d60b69de6e81b86f0dd85bd458bc

## Official repo alignment

NO_OFFICIAL_REPO_LOCATED

## 논문↔repo 차이

local MEntA evaluator makes embedding cosine the primary score; forbidden for paper-faithful use

## 구현 명칭

S2_PAPER_FAITHFUL_COMPONENT_REIMPLEMENTATION

## 현재 판정

S2_SPEC_UNDERDETERMINED

## 미결정 사항

paper text alternates between BLEU(target sample, output) and comparison with the remaining half; it also presents both T and M as original attacks

## Metric provenance rule

이 공격의 원 metric만 공식 결과로 사용한다. E-AUC/symmetric AUC는 INTERNAL_DIAGNOSTIC_ONLY이다.
