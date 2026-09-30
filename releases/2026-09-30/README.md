# 2026-09-30 Frozen Defense Release

This dated release keeps the two deployable policies separate:

1. `defense_1_rif/`: the main paper method, RIF = frozen CLD/G4 + old CF-Clip.
2. `defense_2_rif_fixed_horizon_b2/`: an optional account-bound composition safeguard that calls RIF only for the first two protected-evidence requests.

The folders are intentionally independent. The second package accepts a RIF
callable through dependency injection and does not duplicate or alter the RIF
equations.

`acl_final_evidence/` contains the requested submission audit. Rows from
different cohorts or generation lineages carry explicit lineage labels and are
never silently merged.

