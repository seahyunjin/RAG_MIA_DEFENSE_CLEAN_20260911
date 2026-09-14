# MEntA — PAPER PROTOCOL SPEC

Source: https://arxiv.org/abs/2605.24312

## 논문

Five Queries Are Enough: Query-Efficient and Surrogate-Free Membership Inference Attacks on RAG via Entailment

## 학회/연도

USENIX Security 2026

## 공격 목표

infer target-document membership from entailment-supported facts in five natural RAG answers

## Member

target document is in the retrieval corpus

## Nonmember

target document is absent from the retrieval corpus

## Target 요건

enough distinct facts for five document-specific natural questions

## Query 수

5

## Query 생성

GPT-4.1-nano creates five highly specific questions plus a topic-focused summary; the paper query is summary || question

## 생성 모델

gpt-4.1-nano (hosted revision not content-addressable)

## Prompt

implemented in query_generator.py and preserved MEntA/generate_queries.py

## Decoding

{"question_temperature": 0.7, "summary_temperature": 0.3, "max_question_tokens": 1500, "provenance": "preserved reproduction defaults; the paper does not fully report hosted decoding parameters"}

## Session

ordered q1..q5, mean document-level aggregation

## Retrieval 가정

black-box RAG; paper evaluation uses top-k=3

## Membership score

mean_q(I_entail(q) - I_idk(q))

## Aggregation

mean over exactly five queries

## Threshold

chosen on evaluation/reference scores when classification is required

## 원 평가 metric

ROC-AUC, accuracy, TPR@low-FPR

## Official repository

preserved MEntA reproduction package (local copy; Git commit metadata absent)

## Commit

None

## Paper PDF SHA-256

047d52b3abb9dbc5b224305bce316f194b6ad0e933fd9122108c50601ef39728

## Official repo alignment

PARTIAL: the paper defines summary||question; preserved code augments retrieval encoding but passes the unaugmented question to answer generation

## 논문↔repo 차이

The scorer formula agrees. The paper defines summary||question, while preserved retrieve.py uses the summary for retrieval encoding and generate_rag_output.py sends query['text']; the recovered adapter follows the paper and records both components. Hosted query-model revision must be frozen by output manifest.

## 구현 명칭

PRESERVED_REPRODUCTION_ADAPTER

## 현재 판정

MENTA_PROTOCOL_READY

## Metric provenance rule

이 공격의 원 metric만 공식 결과로 사용한다. E-AUC/symmetric AUC는 INTERNAL_DIAGNOSTIC_ONLY이다.
