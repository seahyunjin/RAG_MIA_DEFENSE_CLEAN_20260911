from pathlib import Path
import importlib.util


SCRIPT = Path(__file__).parents[1] / "code/run_bc_gef.py"
SPEC = importlib.util.spec_from_file_location("bc_gef", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_newline_then_sentence_boundary():
    units = MODULE.segment_document("First sentence. Second sentence?\nThird!")
    assert [item[1] for item in units] == ["First sentence.", "Second sentence?", "Third!"]


def test_list_and_table_lines_remain_whole():
    units = MODULE.segment_document("- One. Two.\n| A | B |\n3) Three. Four.")
    assert [item[1] for item in units] == ["- One. Two.", "| A | B |", "3) Three. Four."]


def test_normalization_only_for_exact_duplicate_key():
    assert MODULE.normalize_sentence("  a   b  ") == "a b"


def test_segmenter_is_deterministic():
    text = "A. B?\n- C. D."
    assert MODULE.segment_document(text) == MODULE.segment_document(text)
