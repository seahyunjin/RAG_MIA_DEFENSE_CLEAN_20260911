#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from ia_v6_constraints import (AllowedSequenceTrieLogitsProcessor,
                               DuplicateSequenceLogitsProcessor,
                               serialize_session)


def finite(scores, token):
    return bool(torch.isfinite(scores[0, token]))


def main():
    checks = []
    base = torch.zeros((1, 20))

    # 1. Exact previous-sequence completion is blocked.
    out = DuplicateSequenceLogitsProcessor(2, [[[3, 4, 5]]])(torch.tensor([[8, 9, 3, 4]]), base.clone())
    checks.append(not finite(out, 5))

    # 2. A non-duplicate continuation is unchanged.
    out = DuplicateSequenceLogitsProcessor(2, [[[3, 4, 5]]])(torch.tensor([[8, 9, 3, 7]]), base.clone())
    checks.append(torch.equal(out, base))

    # 3. Prefix-only overlap remains allowed until exact completion is imminent.
    out = DuplicateSequenceLogitsProcessor(2, [[[3, 4, 5]]])(torch.tensor([[8, 9, 3]]), base.clone())
    checks.append(torch.equal(out, base))

    # 4. A different final token is allowed.
    out = DuplicateSequenceLogitsProcessor(2, [[[3, 4, 5]]])(torch.tensor([[8, 9, 3, 4]]), base.clone())
    checks.append(finite(out, 6))

    # 5. Multiple previous sequences are handled together.
    out = DuplicateSequenceLogitsProcessor(2, [[[3, 4, 5], [3, 4, 6]]])(torch.tensor([[8, 9, 3, 4]]), base.clone())
    checks.append(not finite(out, 5) and not finite(out, 6))

    # 6. EOS is unchanged unless it is itself the forbidden completion token.
    out = DuplicateSequenceLogitsProcessor(2, [[[3, 4, 5]]])(torch.tensor([[8, 9, 3, 4]]), base.clone())
    checks.append(finite(out, 2))

    # 7. The categorical trie permits exact labels only, then EOS.
    trie = AllowedSequenceTrieLogitsProcessor(2, [[10], [11, 12], [13]], eos_token_id=2)
    step0 = trie(torch.tensor([[8, 9]]), base.clone())
    step1 = trie(torch.tensor([[8, 9, 11]]), base.clone())
    complete = trie(torch.tensor([[8, 9, 10]]), base.clone())
    checks.append({i for i in range(20) if finite(step0, i)} == {10, 11, 13}
                  and {i for i in range(20) if finite(step1, i)} == {12}
                  and {i for i in range(20) if finite(complete, i)} == {2})

    # 8. Host serialization is deterministic and schema-valid.
    one = serialize_session("doc-1", [f"Q{i}?" for i in range(1, 16)])
    two = serialize_session("doc-1", [f"Q{i}?" for i in range(1, 16)])
    parsed = json.loads(one)
    checks.append(one == two and parsed["target_id"] == "doc-1"
                  and [x["id"] for x in parsed["queries"]] == list(range(1, 16)))

    if not all(checks):
        failed = [index + 1 for index, value in enumerate(checks) if not value]
        raise SystemExit(f"FAIL checks={failed}")
    print(f"PASS {len(checks)}/{len(checks)}")


if __name__ == "__main__":
    main()
