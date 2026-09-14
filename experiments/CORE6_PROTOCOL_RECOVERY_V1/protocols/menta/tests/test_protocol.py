import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from protocols.common import ProtocolInputError
from protocols.menta.query_generator import (
    build_prompt,
    build_summary_prompt,
    build_system_prompt,
    make_session,
    parse_five_queries,
)
from protocols.menta.scorer import score_session


class MentaProtocolTest(unittest.TestCase):
    def test_parse_count_order_and_label(self):
        text = "\n".join(f"QUERY_{i}: What is documented fact {i}?" for i in range(1, 6))
        questions = parse_five_queries(text)
        session = make_session("doc-a", "member", "A frozen topic summary.", questions)
        self.assertEqual(len(session), 5)
        self.assertEqual([x["query_index"] for x in session], [1, 2, 3, 4, 5])
        self.assertTrue(all(x["membership"] == "member" for x in session))
        self.assertTrue(all(x["query"].startswith("A frozen topic summary.") for x in session))

    def test_frozen_prompt_contract(self):
        self.assertIn("Generate EXACTLY 5 different questions", build_prompt("Evidence text."))
        self.assertIn("QUERY_1: [first question here]", build_prompt("Evidence text."))
        self.assertIn("Always output exactly 5 questions", build_system_prompt())
        self.assertIn("topic-focused description", build_summary_prompt("Evidence text."))

    def test_scorer_polarity_and_aggregation(self):
        member_like = [{"entailed": True, "idk": False}] * 5
        nonmember_like = [{"entailed": False, "idk": True}] * 5
        self.assertEqual(score_session(member_like), 1.0)
        self.assertEqual(score_session(nonmember_like), -1.0)
        self.assertGreater(score_session(member_like), score_session(nonmember_like))

    def test_malformed_fails_closed(self):
        with self.assertRaises(ProtocolInputError):
            parse_five_queries("1. Only one question?")
        with self.assertRaises(ProtocolInputError):
            parse_five_queries("\n".join(f"QUERY_{i}: Duplicate?" for i in range(1, 6)))
        with self.assertRaises(ProtocolInputError):
            make_session("doc", "unknown", "Summary.", ["Q?"] * 5)
        with self.assertRaises(ProtocolInputError):
            make_session("doc", "member", "", [f"Question {i}?" for i in range(5)])
        with self.assertRaises(ProtocolInputError):
            score_session([{"entailed": True, "idk": True}] * 5)


if __name__ == "__main__":
    unittest.main()
