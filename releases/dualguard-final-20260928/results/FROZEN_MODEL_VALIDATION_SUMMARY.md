# Frozen Model Validation Summary

## 1. Frozen method equation

`G4=s1-(s1+s2+s3+s4)/4`; `L_i(v)=[log p_F(v)-log p_-i(v)]_+`; `z'(v)=z_F(v)-[max_i L_i(v)-gamma]_+`.

## 2. Parameters

- tau: `0.0901547372341156`
- gamma: `9.5659109592`
- Gold engine: `SERIAL_KVCACHE`

## 3. Single-query attacks

Rows: `16`; see `TABLE_SINGLE_QUERY_ATTACKS.csv`.

## 4. IA

Exact SERIAL_KVCACHE Q1–Q5 trajectory and histogram reproduced.

## 5. MEntA

Fresh zero-overlap Q1–Q5 trajectory retained; residual leakage is reported.

## 6. Corrected benign

Rows: `12`; only corrected protocols are final.

## 7. True deep-dive

Queries: `154` from gold document provenance.

## 8. Query audit

`QUERY_AUDIT_PASS`; blocking issues `0`.

## 9. Output audit

`OUTPUT_AUDIT_PASS`; blocking issues `0`; diagnostics `4674`.

## 10. Figures

IA exact histogram, IA/benign mechanism diagnostics, IA trajectory, and MEntA trajectory are included as PNG/PDF.

## 11. Limitations

CF-only IA/MEntA rows are supplementary where an exact/fresh four-arm lineage was unavailable. Fast is not production gold. DCMI-Std-Q2 is standardized rather than paper-exact. No domain transfer was run.

FROZEN_MODEL_ARTIFACTS_VALIDATED
