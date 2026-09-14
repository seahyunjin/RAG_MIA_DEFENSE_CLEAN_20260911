from pathlib import Path
import importlib.util


PATH = Path(__file__).parents[1] / "code/run_exp154r.py"
SPEC = importlib.util.spec_from_file_location("exp154r", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_official_source_is_available():
    assert MODULE.OFFICIAL.exists()


def test_strict_boundary_matches_canonical_contract():
    assert MODULE.base.selected_budget(0.0, 512, 64, False, "s", "d", set()) == 64
    assert MODULE.base.selected_budget(1e-12, 512, 64, False, "s", "d", set()) == 512
