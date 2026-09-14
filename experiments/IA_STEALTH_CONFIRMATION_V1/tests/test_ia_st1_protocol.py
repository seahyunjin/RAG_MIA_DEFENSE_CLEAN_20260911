#!/usr/bin/env python3

import json
import sys
import unittest
from pathlib import Path

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))

from ia_st1_protocol import gt_request, question_request, validate_judgments, validate_questions


class ProtocolTests(unittest.TestCase):
    def test_question_request_is_strict_q15(self):
        req = question_request("A target text about alpha and beta.")
        schema = req["response_format"]["json_schema"]
        self.assertTrue(schema["strict"])
        arr = schema["schema"]["properties"]["questions"]
        self.assertEqual(arr["minItems"], 15)
        self.assertEqual(arr["maxItems"], 15)

    def test_valid_questions(self):
        questions = [f"Is fact number {i} supported by alpha?" for i in range(15)]
        out = validate_questions({"questions": questions}, "alpha is discussed separately")
        self.assertTrue(out["valid"], out)

    def test_duplicate_invalid(self):
        questions = ["Is alpha supported?"] * 15
        out = validate_questions({"questions": questions}, "alpha")
        self.assertFalse(out["valid"])
        self.assertIn("NORMALIZED_EXACT_DUPLICATE", out["reasons"])

    def test_direct_membership_invalid(self):
        questions = [f"Is item {i} a member of the database?" for i in range(15)]
        out = validate_questions({"questions": questions}, "x")
        self.assertFalse(out["valid"])
        self.assertIn("DIRECT_MEMBERSHIP_QUESTION", out["reasons"])

    def test_domain_index_and_database_are_not_membership(self):
        questions = [f"Is glycaemic index value {i} reported in the study database?" for i in range(15)]
        out = validate_questions({"questions": questions}, "A medical text")
        self.assertTrue(out["valid"], out)
        self.assertEqual(out["direct_question_slots"], [])
        self.assertTrue(out["meta_reference_slots_diagnostic"])

    def test_gt_exact_labels(self):
        self.assertTrue(validate_judgments({"judgments": ["Yes", "No", "Unknown"] * 5})["valid"])
        self.assertFalse(validate_judgments({"judgments": ["Maybe"] * 15})["valid"])

    def test_gt_request_preserves_order(self):
        questions = [f"Is Q{i}?" for i in range(1, 16)]
        req = gt_request("target", questions)
        prompt = req["messages"][1]["content"]
        self.assertLess(prompt.index("1. Is Q1?"), prompt.index("15. Is Q15?"))
        json.dumps(req)


if __name__ == "__main__":
    unittest.main()
