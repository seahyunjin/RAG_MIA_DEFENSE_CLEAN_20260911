import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from protocols.common import ProtocolInputError
from protocols.dcmi.query_generator import PERTURBATION_RATE, build_perturbation_prompt, make_pair
from protocols.dcmi.scorer import calibrated_score, parse_binary


class DcmiProtocolTest(unittest.TestCase):
    def test_query_count_order_and_label(self):
        pair = make_pair("doc-a", "member", "The result is high.", "The result is low.")
        self.assertEqual(PERTURBATION_RATE, 0.06)
        self.assertEqual(len(pair), 2)
        self.assertEqual([x["variant"] for x in pair], ["original", "perturbed"])
        self.assertTrue(all(x["membership"] == "member" for x in pair))
        self.assertIn("ensuring the modified text remains logically correct", build_perturbation_prompt("One two three four five."))

    def test_scorer_polarity_and_aggregation(self):
        self.assertEqual(calibrated_score("Yes", "No"), 1)
        self.assertEqual(calibrated_score("No", "Yes"), -1)
        self.assertGreater(calibrated_score("Yes", "No"), calibrated_score("No", "Yes"))

    def test_malformed_fails_closed(self):
        with self.assertRaises(ProtocolInputError):
            make_pair("doc", "bad", "a", "b")
        with self.assertRaises(ProtocolInputError):
            make_pair("doc", "member", "same", "same")
        with self.assertRaises(ProtocolInputError):
            parse_binary("Yes and no")


if __name__ == "__main__":
    unittest.main()
