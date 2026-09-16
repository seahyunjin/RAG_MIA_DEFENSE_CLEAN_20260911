# Final V2 validation release — lineage-corrected

Final V2 uses `G4 = s1 - mean(s1..s4)`, benign-only calibration, and rank-1 hide with deterministic backfill.

## Retrieval lineage

- **L1:** 3,000-document BEIR Core DB; Core5 detection and geometry.
- **L2:** TopiOCQA utility DB; utility only. Never join L2 scores to L1 attack scores.
- **D:** FinQA attack/benign DB.
- **D-hard(E):** hard benign queries retrieved against the FinQA DB. This is not a Core hard-benign bank.

`E-cal -> FinQA` is a matched within-FinQA comparison. `E-cal -> BEIR Core5` is a cross-corpus threshold-transfer diagnostic, not a matched comparison. Four historical L1xL2 diagnostics are isolated under `quarantine/`. See `README_KO.md` and `presentation_ready/` for validated tables, the 22-slide plan, and frozen input/output cases.
