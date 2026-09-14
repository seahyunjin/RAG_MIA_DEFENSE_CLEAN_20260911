import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from ia_schema import parse_exact


def valid_payload():
    return {"queries": [{"id": i, "question": f"Question {i}?", "judgment": "Yes"} for i in range(1, 16)]}


class SchemaTests(unittest.TestCase):
    def test_exact(self): self.assertTrue(parse_exact(json.dumps(valid_payload())).valid)
    def test_prose_rejected(self): self.assertFalse(parse_exact("Here: " + json.dumps(valid_payload())).valid)
    def test_missing_rejected(self):
        value = valid_payload(); value["queries"].pop(); self.assertFalse(parse_exact(json.dumps(value)).valid)
    def test_id_rejected(self):
        value = valid_payload(); value["queries"][3]["id"] = 9; self.assertFalse(parse_exact(json.dumps(value)).valid)
    def test_unknown_allowed(self):
        value = valid_payload(); value["queries"][3]["judgment"] = "Unknown"; self.assertTrue(parse_exact(json.dumps(value)).valid)
    def test_lowercase_rejected(self):
        value = valid_payload(); value["queries"][3]["judgment"] = "yes"; self.assertFalse(parse_exact(json.dumps(value)).valid)
    def test_extra_key_rejected(self):
        value = valid_payload(); value["queries"][0]["extra"] = 1; self.assertFalse(parse_exact(json.dumps(value)).valid)
    def test_duplicate_question_rejected(self):
        value = valid_payload(); value["queries"][1]["question"] = value["queries"][0]["question"]
        self.assertFalse(parse_exact(json.dumps(value)).valid)


if __name__ == "__main__": unittest.main()
