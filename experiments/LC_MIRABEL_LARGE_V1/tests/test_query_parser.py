import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "code"))
from generate_queries import parse_packet


QUESTIONS = [
    "What is the first detail?",
    "What is the second detail?",
    "What is the third detail?",
    "What is the fourth detail?",
    "What is the fifth detail?",
]


def test_parser_accepts_frozen_six_line_contract():
    text = "\n".join(
        ["SUMMARY: A concise topic summary."]
        + [f"QUERY_{index}: {question}" for index, question in enumerate(QUESTIONS, 1)]
    )
    assert parse_packet(text) == ("A concise topic summary.", QUESTIONS)


def test_parser_canonically_joins_label_value_line_breaks():
    fields = [("SUMMARY", "A concise topic summary.")] + [
        (f"QUERY_{index}", question) for index, question in enumerate(QUESTIONS, 1)
    ]
    text = "\n".join(item for label, value in fields for item in (f"{label}:", value))
    assert parse_packet(text) == ("A concise topic summary.", QUESTIONS)


def test_parser_rejects_missing_value():
    text = "SUMMARY:\nQUERY_1:\nWhat is the first detail?"
    try:
        parse_packet(text)
    except ValueError as exc:
        assert "lacks an immediately following value" in str(exc)
    else:
        raise AssertionError("missing value was accepted")
