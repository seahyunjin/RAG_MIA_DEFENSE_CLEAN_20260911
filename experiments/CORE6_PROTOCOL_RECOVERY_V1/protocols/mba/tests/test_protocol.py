import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from protocols.common import ProtocolInputError
from protocols.mba.query_generator import make_query, select_difficult_words, tokenize_without_proxy_ranks
from protocols.mba.scorer import reconstruction_accuracy


def records():
    result = []
    for i in range(15):
        result.append(
            {
                "word": f"word{i}",
                "difficulty_rank": float(i % 3),
                "eligible": True,
                "accepted_answers": [f"word{i}"],
            }
        )
    return result


class MbaProtocolTest(unittest.TestCase):
    def test_query_count_order_and_label(self):
        selected = select_difficult_words(records(), 5)
        self.assertEqual(selected, [2, 5, 8, 11, 14])
        query = make_query("doc-a", "member", records(), 5)
        self.assertEqual(query["mask_count"], 5)
        self.assertEqual(query["membership"], "member")

    def test_scorer_polarity_and_aggregation(self):
        gt = {f"Mask_{i}": [f"answer{i}"] for i in range(1, 6)}
        perfect = "\n".join(f"[Mask_{i}]: answer{i}" for i in range(1, 6))
        wrong = "\n".join(f"[Mask_{i}]: wrong{i}" for i in range(1, 6))
        self.assertEqual(reconstruction_accuracy(perfect, gt), 1.0)
        self.assertEqual(reconstruction_accuracy(wrong, gt), 0.0)

    def test_malformed_fails_closed(self):
        with self.assertRaises(ProtocolInputError):
            tokenize_without_proxy_ranks("text")
        with self.assertRaises(ProtocolInputError):
            reconstruction_accuracy("[Mask_1]: a", {"Mask_2": ["a"]})
        with self.assertRaises(ProtocolInputError):
            select_difficult_words(records(), 7)


if __name__ == "__main__":
    unittest.main()

