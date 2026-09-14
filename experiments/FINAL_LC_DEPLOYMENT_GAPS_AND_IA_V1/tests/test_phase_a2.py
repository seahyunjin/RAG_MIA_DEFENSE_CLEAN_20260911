from __future__ import annotations

import importlib.util
from pathlib import Path


CODE = Path(__file__).resolve().parents[1] / "code"


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, CODE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


def test_selection_is_deterministic_and_membership_sensitive():
    module = load("build_phase_a2_substrate")
    a = module.selection_key("MEntA", "member", "x")
    assert a == module.selection_key("MEntA", "member", "x")
    assert a != module.selection_key("MEntA", "nonmember", "x")


def test_waterfill_preserves_budget_and_caps():
    module = load("run_phase_a2_generation")
    assert module.waterfill([100, 100, 100], 2048) == [100, 100, 100]
    result = module.waterfill([5000, 5000, 5000, 5000], 2048)
    assert result == [512, 512, 512, 512]
    assert sum(result) == 2048


def test_frozen_yes_no_and_mask_parsers():
    module = load("run_phase_a2_scoring")
    assert module.yes_no("Yes") == "Yes"
    assert module.yes_no("I don't know.") == "UNK"
    value, bad = module.mask_accuracy("[Mask_1]: alpha\n[Mask_2]: beta",
                                      {"Mask_1": ["alpha"], "Mask_2": ["beta"]})
    assert value == 1.0 and not bad
    value, bad = module.mask_accuracy("[Mask_1]: alpha", {"Mask_1": ["alpha"], "Mask_2": ["beta"]})
    assert value == 0.0 and bad
def test_mba_repair_is_exact_id_only_and_preserves_other_rows():
    module = load("run_phase_a2_scoring_repair")
    source = [{"attack": "MBA", "query_id": f"q{i}", "query": f"question {i}",
               "membership": "member", "target_id": f"t{i}", "session_id": f"s{i}",
               "mask_answers": {"Mask_1": [f"a{i}"]}} for i in range(400)]
    rows = []
    for _ in range(4):
        for item in source:
            rows.append({key: item[key] for key in
                         ("attack", "query_id", "query", "membership", "target_id", "session_id")})
    rows.append({"attack": "MEntA", "query_id": "untouched"})
    enriched, audit = module.enrich_retrieval(rows, source)
    assert audit["unique_query_ids"] == 400
    assert audit["retrieval_rows_joined"] == 1600
    assert enriched[-1] == rows[-1]
    assert enriched[0]["mask_answers"] == {"Mask_1": ["a0"]}
