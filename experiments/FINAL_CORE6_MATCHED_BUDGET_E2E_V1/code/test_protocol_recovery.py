#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
EXP = HERE.parent
ROOT = EXP.parents[1]
SPEC = importlib.util.spec_from_file_location("protocol_recovery", HERE / "run_protocol_recovery.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC and SPEC.loader
SPEC.loader.exec_module(MODULE)


def test_auc_with_ties() -> None:
    assert MODULE.rank_auc([0, 0, 1, 1], [0, 0, 1, 1]) == 1.0
    assert MODULE.rank_auc([0, 1], [0, 0]) == 0.5


def test_mba_missing_is_incorrect() -> None:
    parsed, malformed, duplicate = MODULE.parse_mba_exact_lines("[Mask_1]: alpha\nI don't know\n[Mask_1]: beta")
    assert parsed == {"Mask_1": "alpha"}
    assert malformed == 1
    assert duplicate == 1
    assert "Mask_2" not in parsed


def test_precommit_integrity() -> None:
    path = EXP / "configs" / "CORE6_PROTOCOL_RECOVERY_PRECOMMIT.json"
    expected = (EXP / "configs" / "CORE6_PROTOCOL_RECOVERY_PRECOMMIT.sha256").read_text().split()[0]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_frozen_manifest_integrity() -> None:
    previous = ROOT / "experiments" / "FINAL_8ATTACK_E2E_FRAMEWORK_V1" / "configs"
    manifest = previous / "FINAL_DEFENSE_MANIFEST.json"
    expected = (previous / "FINAL_DEFENSE_MANIFEST.sha256").read_text().split()[0]
    assert hashlib.sha256(manifest.read_bytes()).hexdigest() == expected


def test_fail_closed_result() -> None:
    result = json.loads((EXP / "RESULT.json").read_text())
    assert result["verdict"] == "CORE6_PROTOCOL_INCOMPLETE"
    assert result["dcmi_status"] == "DCMI_SPEC_UNDERDETERMINED"
    assert result["ia_status"] == "IA_PAPER_EXACT_UNAVAILABLE"
    assert result["new_attack_queries_generated"] == 0
    assert result["new_rag_answers_generated"] == 0
    assert result["gold_qa"] == "NOT_RUN_CORE6_GATE_CLOSED"


def main() -> None:
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"PASS {test.__name__}")
    print(f"PASS total={len(tests)}")


if __name__ == "__main__":
    main()
