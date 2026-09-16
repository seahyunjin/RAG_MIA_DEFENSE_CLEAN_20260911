import unittest

from lrc4_detector import alarm, defend_retrieval, lrc4_score
from lineage_guard import RetrievalLineageMismatch, join_rows_by_query_id


class LRC4DetectorTest(unittest.TestCase):
    def test_score(self):
        self.assertAlmostEqual(lrc4_score([0.9, 0.5, 0.4, 0.2]), 0.4)

    def test_strict_threshold(self):
        scores = [0.9, 0.5, 0.4, 0.2]
        score = lrc4_score(scores)
        self.assertFalse(alarm(scores, score))
        self.assertTrue(alarm(scores, score - 1e-6))

    def test_rank1_hide_and_backfill(self):
        result = defend_retrieval(
            ["d1", "d2", "d3", "d4", "d5"],
            [0.9, 0.5, 0.4, 0.2, 0.1],
            threshold=0.3,
        )
        self.assertTrue(result.alarm)
        self.assertEqual(result.hidden_document_id, "d1")
        self.assertEqual(result.context_document_ids, ("d2", "d3", "d4", "d5"))

    def test_lineage_guard_rejects_cross_database_join(self):
        with self.assertRaises(RetrievalLineageMismatch):
            join_rows_by_query_id(
                [{"query_id": "q1", "score": 0.1}],
                [{"query_id": "q1", "score": 0.2}],
                left_lineage={"name": "core", "retrieval_db_hash": "l1-hash"},
                right_lineage={"name": "gold", "retrieval_db_hash": "l2-hash"},
            )


if __name__ == "__main__":
    unittest.main()
