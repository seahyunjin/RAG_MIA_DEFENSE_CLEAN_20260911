import hashlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from protocols.common import ProtocolInputError, ProtocolUnavailableError
from protocols.ia.query_generator import generate_queries, validate_immutable_bundle
from protocols.ia.scorer import score_session


class IaProtocolTest(unittest.TestCase):
    def test_generation_fails_closed(self):
        with self.assertRaises(ProtocolUnavailableError):
            generate_queries("document")

    def test_bundle_count_label_and_order(self):
        rows = []
        for index in range(1, 31):
            query = f"Question {index}?"
            gt = "Yes"
            rows.append(
                {
                    "target_id": "doc-a",
                    "membership": "member",
                    "query_index": index,
                    "query": query,
                    "ground_truth": gt,
                    "query_sha256": hashlib.sha256(query.encode()).hexdigest(),
                    "gt_sha256": hashlib.sha256(gt.encode()).hexdigest(),
                }
            )
        self.assertEqual(len(validate_immutable_bundle(rows, "doc-a", "member")), 30)

    def test_scorer_polarity_and_aggregation(self):
        positive = score_session(["Yes"] * 30, ["Yes"] * 30)
        abstain = score_session(["I don't know"] * 30, ["Yes"] * 30)
        self.assertEqual(positive, 1.0)
        self.assertEqual(abstain, -5.0)
        self.assertGreater(positive, abstain)

    def test_malformed_fails_closed(self):
        with self.assertRaises(ProtocolInputError):
            score_session(["Yes"], ["Yes"])
        with self.assertRaises(ProtocolInputError):
            validate_immutable_bundle([], "doc", "member")


if __name__ == "__main__":
    unittest.main()

