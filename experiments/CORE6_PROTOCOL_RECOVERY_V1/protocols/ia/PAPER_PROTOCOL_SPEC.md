# IA — PAPER PROTOCOL SPEC

Source: https://arxiv.org/abs/2502.00306

## 논문

Riddle Me This! Stealthy Membership Inference for Retrieval-Augmented Generation

## 학회/연도

ACM CCS 2025

## 공격 목표

infer membership from accuracy on many natural, document-specific yes/no questions

## Member

target document is in the RAG datastore

## Nonmember

target and TF-IDF>=0.95 near duplicates are absent

## Target 요건

supports 30 diverse yes/no questions and a short target description

## Query 수

30

## Query 생성

paper: GPT-4o generates 30 questions and a document description; questions may be ranked/selected

## 생성 모델

gpt-4o; exact historical revision/output unavailable

## Prompt

paper appendix prompt exists, but immutable generated bundle for shared targets does not

## Decoding

paper/repository lineage is not sufficient for bit-exact recovery

## Session

30-query document session in the paper

## Retrieval 가정

black-box RAG; paper main setup k=3

## Membership score

(1/n) sum(I(response=GT)-5*I(response=UNK))

## Aggregation

mean across 30 queries

## Threshold

evaluated by ROC/low-FPR operating points

## 원 평가 metric

ROC-AUC, accuracy, TPR@1% FPR

## Official repository

https://github.com/ali7naseh/RAG_MIA

## Commit

a53e8fda4f1492f204d968d7afb53ce8287c193c

## Paper PDF SHA-256

3454915b3651d2e90d23f1a27d28bf9807bdbc0a5701079861b91506da9f0791

## Official repo alignment

NO: paper default is 30 questions and lambda=5; repository bundle contains 15 questions per document and scorer computes plain accuracy

## 논문↔repo 차이

paper default is 30 queries and lambda=5; repository bundles 15/query doc data and its scorer calculates plain accuracy

## 구현 명칭

FAIL_CLOSED_SPEC_AND_SCORER_FORMULA_ONLY

## 현재 판정

IA_PAPER_EXACT_UNAVAILABLE

## Metric provenance rule

이 공격의 원 metric만 공식 결과로 사용한다. E-AUC/symmetric AUC는 INTERNAL_DIAGNOSTIC_ONLY이다.
