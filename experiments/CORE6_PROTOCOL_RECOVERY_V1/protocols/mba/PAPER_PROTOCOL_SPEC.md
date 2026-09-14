# MBA — PAPER PROTOCOL SPEC

Source: https://arxiv.org/abs/2410.20142

## 논문

Mask-based Membership Inference Attacks for Retrieval-Augmented Generation

## 학회/연도

The Web Conference 2025

## 공격 목표

infer membership from recovery accuracy of difficult words masked from the target

## Member

target document is in the retrieval database

## Nonmember

target document is outside the retrieval database

## Target 요건

enough valid, non-adjacent, non-stopword targets across M equal subtexts

## Query 수

1

## Query 생성

rank word difficulty with GPT2-XL, select one maximum-rank eligible word per equal subtext, integrate indexed masks

## 생성 모델

openai-community/gpt2-xl@15ea56dee5df4983c59b2538573817e1667135e2

## Prompt

indexed [Mask_i] reconstruction prompt

## Decoding

target RAG generation; exact mask parser fails closed

## Session

one masked query

## Retrieval 가정

paper evaluates top-k=10

## Membership score

number of correctly reconstructed masks / M

## Aggregation

single-query reconstruction accuracy

## Threshold

gamma in (0,1], selected for best F1 on reference/training split

## 원 평가 metric

ROC-AUC, accuracy, precision, recall, F1, retrieval recall

## Official repository

not located as a standalone author attack repository

## Commit

None

## Paper PDF SHA-256

11f415cbb80cf617e0a417368f411aee07e8f0ecb89ae44d819ead9a772acc87

## Official repo alignment

NO_STANDALONE_OFFICIAL_REPO_LOCATED

## 논문↔repo 차이

preserved local pipeline defaults to 5 masks and important-word masking; final adapter forbids that fallback

## 구현 명칭

MBA_PAPER_FAITHFUL_REIMPLEMENTATION

## 현재 판정

MBA_PROTOCOL_READY

## Metric provenance rule

이 공격의 원 metric만 공식 결과로 사용한다. E-AUC/symmetric AUC는 INTERNAL_DIAGNOSTIC_ONLY이다.
