import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from ia_v4_schema import parse_judgments, parse_questions


def questions():
    return '{"queries":[' + ','.join('{"id":%d,"question":"Question %d?"}' % (i, i) for i in range(1, 16)) + ']}'


def judgments():
    return '{"judgments":[' + ','.join('{"id":%d,"judgment":"Yes"}' % i for i in range(1, 16)) + ']}'


class V4SchemaTest(unittest.TestCase):
    def test_questions_valid(self): self.assertTrue(parse_questions(questions()).valid)
    def test_judgments_valid(self): self.assertTrue(parse_judgments(judgments()).valid)
    def test_no_prose(self): self.assertEqual(parse_questions('x' + questions()).reason, 'INVALID_JSON')
    def test_wrong_root(self): self.assertEqual(parse_questions('{"questions":[]}').reason, 'ROOT_SCHEMA_MISMATCH')
    def test_question_extra_key(self): self.assertEqual(parse_questions(questions().replace('"question":"Question 1?"', '"question":"Question 1?","judgment":"Yes"')).reason, 'ENTRY_SCHEMA_MISMATCH')
    def test_duplicate_question(self): self.assertEqual(parse_questions(questions().replace('Question 2?', 'Question 1?')).reason, 'DUPLICATE_QUESTION')
    def test_judgment_label_exact(self): self.assertEqual(parse_judgments(judgments().replace('"Yes"', '"yes"', 1)).reason, 'INVALID_JUDGMENT')
    def test_judgment_missing(self): self.assertEqual(parse_judgments(judgments().replace(',{"id":15,"judgment":"Yes"}', '')).reason, 'ENTRY_COUNT_NOT_15')


if __name__ == '__main__': unittest.main()
