# CORE6 Protocol Recovery V1

This campaign reconstructs attack protocols only. It does **not** run a defense, retrieval, target selection, membership split, attack generation, or protected-answer generation.

## Reproduce the audit

```bash
python scripts/run_unit_tests.py
python scripts/build_protocol_metadata.py
python -m compileall -q protocols scripts
sha256sum -c CORE6_PROTOCOL_MANIFEST.sha256
```

## Current boundary

- Ready: MEntA, MBA, RAG-MIA
- Requires protocol resolution: S²-MIA, DCMI
- Paper-exact unavailable: IA
- Verdict: `CORE_N_PROTOCOL_PARTIAL` with `N=3`

Do not build the shared benchmark substrate until the attack set is explicitly chosen and the unresolved protocols are either recovered or clearly labeled as variants.
