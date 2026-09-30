import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fixed_horizon import ABSTENTION, FixedEvidenceHorizon


class FixedHorizonTests(unittest.TestCase):
    def test_only_first_two_requests_call_rif(self):
        with tempfile.TemporaryDirectory() as directory:
            guard = FixedEvidenceHorizon(Path(directory) / "ledger.sqlite3", b"test-secret")
            calls = []

            def rif_call():
                calls.append(1)
                return "answer"

            rows = [guard.serve("account-A", rif_call) for _ in range(4)]
            self.assertEqual(len(calls), 2)
            self.assertEqual([row["request_index"] for row in rows], [1, 2, 3, 4])
            self.assertEqual(rows[2]["answer"], ABSTENTION)
            self.assertFalse(rows[2]["protected_retrieval_executed"])

    def test_conversation_identifier_does_not_reset_account(self):
        with tempfile.TemporaryDirectory() as directory:
            guard = FixedEvidenceHorizon(Path(directory) / "ledger.sqlite3", b"test-secret")
            guard.serve("same-account", lambda: "conversation-1")
            guard.serve("same-account", lambda: "conversation-2")
            denied = guard.serve("same-account", lambda: self.fail("must not execute"))
            self.assertTrue(denied["horizon_exhausted"])


if __name__ == "__main__":
    unittest.main()

