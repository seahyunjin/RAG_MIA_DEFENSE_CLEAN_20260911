import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rif.cld import TAU, apply_cld, cld4


class FrozenCLDTests(unittest.TestCase):
    def test_formula(self):
        self.assertTrue(math.isclose(cld4([0.80, 0.60, 0.59, 0.58]), 0.1575))

    def test_strict_comparator(self):
        scores = [TAU * 4 / 3, 0.0, 0.0, 0.0]
        self.assertFalse(apply_cld(scores, ["d1", "d2", "d3", "d4"]).alarm)

    def test_alarm_removes_rank1_without_refill(self):
        decision = apply_cld([0.80, 0.60, 0.59, 0.58], ["d1", "d2", "d3", "d4"])
        self.assertTrue(decision.alarm)
        self.assertEqual(decision.documents, ("d2", "d3", "d4"))
        self.assertEqual(decision.removed, "d1")


if __name__ == "__main__":
    unittest.main()

