# DualGuard (frozen research artifact)

## Method overview

DualGuard is a training-free, attack-family-routing-free composition over retrieval and generation surfaces. It does not provide a formal differential-privacy guarantee.

CLD/G4: `G4 = s1 - (s1+s2+s3+s4)/4`, alarm when `G4 > 0.0901547372341156`; the rank-1 source is withheld.

Old CF-Clip: `L_i(v)=[log p_F(v)-log p_-i(v)]_+`, `L(v)=max_i L_i(v)`, and `z'(v)=z_F(v)-[L(v)-9.5659109592]_+`.

Expected inputs are four ranked retrieval scores/documents and, for exact CF-Clip, one FULL plus one leave-one-parent-out prompt per retained parent. The output is the CLD decision, optional removed source, and a frozen-generator answer.

The exact implementation is `SERIAL_KVCACHE`. The speculative Fast implementation is not production-ready and is not included. Runtime is substantially higher than vanilla greedy decoding because every output token evaluates all DROP branches.

## Quick check

```bash
python inference.py --toy
python -m tests.smoke_test
python evaluate.py --dry-run
```

## Claims and limitations

The frozen evaluation supports IA mitigation, MEntA mitigation, and corrected benign/deep-reading aggregate utility preservation. It does not establish a universal defense, elimination of MEntA, zero false positives, production-fast inference, or formal DP.

## Citation

Citation placeholder for anonymous review/public release.
