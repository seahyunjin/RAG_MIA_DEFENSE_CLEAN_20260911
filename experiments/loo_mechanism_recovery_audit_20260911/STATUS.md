# Recovery audit status

- Status: `RECOVERY_AUDIT_DONE`
- Legacy decision: `LEGACY_PHASE1_LINEAGE_INCOMPLETE`
- Clean rebuild decision: `CLEAN_REBUILD_SPEC_INSUFFICIENT`
- Phase-1: `NOT_STARTED`
- Retrieval/model forward/generation performed: `NO`
- Legacy/new lineage mixed: `NO`
- Historical artifacts modified: `NO`

Reason: exact old frozen corpus, complete per-query Top-4 IDs, and exact QLL selected source IDs were not recovered. A MEntA-only 200-query Top-4 subset was recovered and validated, but it is insufficient for the requested benign/MEntA/S²-MIA/MBA Phase-1.
