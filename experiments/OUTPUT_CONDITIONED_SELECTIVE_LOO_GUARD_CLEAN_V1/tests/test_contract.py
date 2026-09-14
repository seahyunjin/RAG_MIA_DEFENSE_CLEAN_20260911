from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import common
import prepare
import analyze_phase1


def test_campaign_identity_and_clean_source_contract():
    assert common.CAMPAIGN == "OUTPUT_CONDITIONED_SELECTIVE_LOO_GUARD_CLEAN_V1"
    text = (ROOT / "code/prepare.py").read_text()
    assert "ATTACK_SCORES" not in text
    assert "L_full" not in text
    assert 'usecols=allowed' in text
    assert "target_rank" in text and "explicitly_ignored_columns" in text


def test_small_cohort_is_deterministic_and_balanced():
    frame = prepare.source_only_attack_queries()
    assert len(frame) == 280
    assert frame.case_id.nunique() == 280
    expected = {("MEntA", 0): 20, ("MEntA", 1): 20,
                ("S²-MIA", 0): 20, ("S²-MIA", 1): 20,
                ("MBA", 0): 20, ("MBA", 1): 20}
    assert frame.groupby(["family", "member"]).session_id.nunique().to_dict() == expected
    assert frame[frame.family.eq("MEntA")].groupby("session_id").turn.apply(list).apply(lambda x: x == [1,2,3,4,5]).all()


def test_benign_split_and_no_query_overlap():
    attack = prepare.source_only_attack_queries()
    benign = prepare.select_benign(attack)
    assert len(benign) == 500
    assert benign.groupby("split").size().to_dict() == {"CALIBRATION": 250, "HOLDOUT": 250}
    assert not set(benign.query_sha256) & set(attack.query_sha256)
    assert benign.case_id.nunique() == 500
    assert benign.query_sha256.nunique() == 500
    assert not set(benign.target_id.astype(str)) & set(attack.target_id.astype(str))


def test_strict_threshold_contract():
    values = list(range(250))
    for target, maximum in [(0.01, 2), (0.03, 7), (0.05, 12)]:
        threshold, count, rate = analyze_phase1.strict_threshold(values, target)
        assert count <= maximum
        assert rate <= target
        assert sum(value > threshold for value in values) == count


def test_waterfill_and_frozen_topk():
    assert common.TOP_K == 4
    caps = common.waterfill([1000, 1000, 1000, 1000], 2048)
    assert sum(caps) == 2048
    assert len(caps) == 4


def test_phase1_never_runs_protected_generation():
    text = (ROOT / "code/run_phase1.py").read_text()
    assert "protected answer" not in text.lower()
    assert "run_generation" in text  # provisional A0 only
    analysis = (ROOT / "code/analyze_phase1.py").read_text()
    assert '"protected_generation_count": 0' in analysis
    assert '"phase2_opened": False' in analysis
