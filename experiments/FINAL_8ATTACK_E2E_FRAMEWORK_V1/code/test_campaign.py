#!/usr/bin/env python3
from __future__ import annotations

import unittest

from run_generation import alarm, waterfill


class CampaignTests(unittest.TestCase):
    def test_waterfill_preserves_budget(self):
        self.assertEqual(sum(waterfill([1000, 1000, 1000], 2048)), 2048)

    def test_waterfill_short(self):
        self.assertEqual(waterfill([2, 3], 20), [2, 3])

    def test_strict_threshold(self):
        row = {"M": 0.0, "R_GLOBAL": 3.428580914764567, "R_LC": 3.80543877128208}
        self.assertFalse(alarm(row, "ORIGINAL_MIRABEL_SIMPLE_HIDE"))
        self.assertFalse(alarm(row, "GLOBAL_BC_SIMPLE_HIDE"))
        self.assertFalse(alarm(row, "FINAL_LC_OUTER_SIMPLE_HIDE"))
        row = {"M": 1e-8, "R_GLOBAL": 3.5, "R_LC": 4.0}
        self.assertTrue(alarm(row, "ORIGINAL_MIRABEL_SIMPLE_HIDE"))
        self.assertTrue(alarm(row, "GLOBAL_BC_SIMPLE_HIDE"))
        self.assertTrue(alarm(row, "FINAL_LC_OUTER_SIMPLE_HIDE"))


if __name__ == "__main__":
    unittest.main()

