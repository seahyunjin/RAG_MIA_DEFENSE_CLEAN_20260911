import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from ia_parser_v2 import audit_category, parse_attempt, parse_session


def inline():
    return "\n".join(f"{i}. Is fact {i} present? {'Yes' if i % 2 else 'No'}" for i in range(1, 16))


class ParserV2Tests(unittest.TestCase):
    def test_inline_exact(self):
        result=parse_attempt(inline());self.assertTrue(result.valid);self.assertEqual(len(result.items),15);self.assertEqual(result.inferred_fields,0)

    def test_split_question_answer_lines(self):
        raw="\n\n".join(f"{i}. Is fact {i} present?\nAnswer: {'Yes' if i%2 else 'No'}" for i in range(1,16))
        result=parse_attempt(raw);self.assertTrue(result.valid);self.assertEqual(result.items[1][1],"No")

    def test_explicit_question_answer_fields(self):
        raw="\n".join(f"Question {i}: Question: Is fact {i} present?\nAnswer {i}: Yes" for i in range(1,16))
        self.assertTrue(parse_attempt(raw).valid)

    def test_repeated_number_is_format_only(self):
        raw="\n".join(f"{i}. {i}. Is fact {i} present? Yes" for i in range(1,16))
        self.assertTrue(parse_attempt(raw).valid)

    def test_numeric_answer_not_inferred(self):
        rows=[f"{i}. Is fact {i} present? Yes" for i in range(1,16)];rows[4]="5. What is the value? 27"
        result=parse_attempt("\n".join(rows));self.assertFalse(result.valid);self.assertEqual(audit_category(result),"CONTENT_MISSING")

    def test_missing_record_rejected(self):
        result=parse_attempt("\n".join(inline().splitlines()[:-1]));self.assertFalse(result.valid);self.assertEqual(result.record_count,14)

    def test_nonsequential_rejected(self):
        rows=inline().splitlines();rows[-1]=rows[-1].replace("15.","14.",1);self.assertFalse(parse_attempt("\n".join(rows)).valid)

    def test_explanation_rejected(self):
        rows=inline().splitlines();rows[0]+=" because it says so";self.assertFalse(parse_attempt("\n".join(rows)).valid)

    def test_first_exact_attempt_selected(self):
        result,index=parse_session([inline(),"bad"]);self.assertTrue(result.valid);self.assertEqual(index,1)

    def test_second_frozen_attempt_selected(self):
        result,index=parse_session(["bad",inline()]);self.assertTrue(result.valid);self.assertEqual(index,2)


if __name__=="__main__":unittest.main()
