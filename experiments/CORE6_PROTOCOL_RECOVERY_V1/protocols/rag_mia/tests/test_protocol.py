import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from protocols.common import ProtocolInputError
from protocols.rag_mia.query_generator import make_query
from protocols.rag_mia.scorer import black_box_membership_score


class RagMiaProtocolTest(unittest.TestCase):
    def test_query_count_and_label(self):
        query = make_query("doc-a", "nonmember", "Target sample text")
        self.assertEqual(query["prompt_id"], 2)
        self.assertEqual(query["membership"], "nonmember")
        self.assertIn("Answer with Yes or No", query["query"])

    def test_scorer_polarity_and_aggregation(self):
        self.assertEqual(black_box_membership_score("Yes"), 1)
        self.assertEqual(black_box_membership_score("No"), 0)
        self.assertEqual(black_box_membership_score("Unanswerable"), 0)
        self.assertGreater(black_box_membership_score("Yes"), black_box_membership_score("No"))

    def test_malformed_fails_closed(self):
        with self.assertRaises(ProtocolInputError):
            make_query("doc", "unknown", "sample")
        with self.assertRaises(ProtocolInputError):
            black_box_membership_score("Yes, but also no")


if __name__ == "__main__":
    unittest.main()

