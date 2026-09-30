# Defense 2: RIF + Fixed Evidence Horizon B=2

This is an optional authenticated deployment safeguard, not the main RIF
method and not an attack detector.

- The ledger is keyed by an authenticated account/credential, not a
  client-supplied conversation ID.
- Requests 1 and 2 call the separately deployed frozen RIF implementation.
- Requests 3 and later do not access protected retrieval and return the frozen
  deterministic abstention `I don't know.`
- The fixed horizon is `B=2`; there is no adaptive budget or attack-specific
  threshold.

The SQLite implementation uses an HMAC of the credential and an atomic
`BEGIN IMMEDIATE` transaction. The caller must run `serve` before any protected
retrieval so denied requests cannot touch private evidence.

Full account/credential reset is a Sybil limitation. The policy does not claim
to preserve open-ended deep-reading: the frozen evaluation shows a large Q3+
utility cost.

```python
from fixed_horizon import FixedEvidenceHorizon

guard = FixedEvidenceHorizon("ledger.sqlite3", server_secret=b"rotate-me")
result = guard.serve(account_credential, lambda: frozen_rif_request())
```

Run tests:

```bash
python -m unittest discover -s tests -v
```

