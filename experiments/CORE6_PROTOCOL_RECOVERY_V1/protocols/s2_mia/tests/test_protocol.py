import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from protocols.common import ProtocolInputError
from protocols.s2_mia.query_generator import make_query, make_query_from_raw_document
from protocols.s2_mia.scorer import extract_features, sentence_bleu, threshold_decision


class S2ProtocolTest(unittest.TestCase):
    def test_query_count_and_label(self):
        record = make_query("doc-a", "nonmember", "What is X?", "X is the answer.")
        self.assertEqual(record["membership"], "nonmember")
        self.assertEqual(record["query_id"].count("::q1"), 1)

    def test_feature_polarity_and_threshold(self):
        exact = sentence_bleu("alpha beta gamma delta", "alpha beta gamma delta")
        mismatch = sentence_bleu("alpha beta gamma delta", "other words entirely here")
        self.assertGreater(exact, mismatch)
        features = extract_features("alpha beta gamma delta", "alpha beta gamma delta", [-0.1, -0.2])
        self.assertEqual(threshold_decision(features, 0.9, 2.0), 1)

    def test_malformed_fails_closed(self):
        with self.assertRaises(ProtocolInputError):
            make_query("doc", "bad", "q", "r")
        with self.assertRaises(ProtocolInputError):
            make_query_from_raw_document("unstructured text")
        with self.assertRaises(ProtocolInputError):
            extract_features("ref", "candidate", [])


if __name__ == "__main__":
    unittest.main()

