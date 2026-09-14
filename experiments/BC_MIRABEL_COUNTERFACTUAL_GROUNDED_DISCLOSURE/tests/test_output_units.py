import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from output_units import FALLBACK, assemble_kept, segment_units


def test_newline_precedes_sentence_split():
    assert segment_units("First. Second!\nThird?") == ["First.", "Second!", "Third?"]


def test_bullet_line_is_one_unit():
    assert segment_units("- First. Still the same bullet.\nPlain. Next.") == [
        "- First. Still the same bullet.", "Plain.", "Next."
    ]


def test_mba_lines_remain_ordered():
    value = "[Mask_0]: alpha\n[Mask_1]: beta"
    assert segment_units(value) == ["[Mask_0]: alpha", "[Mask_1]: beta"]


def test_assembly_never_rewrites_units():
    units = ["Exact 10.2–27.8 value.", "Second unit."]
    assert assemble_kept(units, [True, False]) == units[0]


def test_empty_keep_uses_exact_fallback():
    assert assemble_kept(["Claim."], [False]) == FALLBACK


def test_blank_answer_has_no_units():
    assert segment_units(" \n ") == []
