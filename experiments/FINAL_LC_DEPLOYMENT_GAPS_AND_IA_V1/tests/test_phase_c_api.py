from ia_api_protocol import validate_judgments, validate_questions


def questions():
    return {"questions": [{"id": index, "question": f"Is fact number {index} supported?"} for index in range(1, 16)]}


def judgments():
    return {"judgments": [{"id": index, "judgment": "Yes" if index % 2 else "Unknown"} for index in range(1, 16)]}


def test_exact_structured_objects_pass():
    assert validate_questions(questions())["valid"]
    assert validate_judgments(judgments())["valid"]


def test_missing_or_reordered_id_is_structural_failure():
    value = questions()
    value["questions"][0]["id"] = 2
    result = validate_questions(value)
    assert not result["valid"]
    assert not result["structural_valid"]


def test_duplicate_question_is_content_failure_not_schema_retry():
    value = questions()
    value["questions"][1]["question"] = value["questions"][0]["question"]
    result = validate_questions(value)
    assert not result["valid"]
    assert result["structural_valid"]
    assert "NORMALIZED_EXACT_DUPLICATE" in result["reasons"]


def test_extra_field_fails_closed():
    value = judgments()
    value["judgments"][0]["explanation"] = "not allowed"
    result = validate_judgments(value)
    assert not result["valid"]
    assert not result["structural_valid"]
