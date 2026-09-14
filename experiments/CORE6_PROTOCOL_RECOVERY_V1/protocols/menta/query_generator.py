"""Validated MEntA five-query request/response adapter.

Actual hosted-model execution remains outside this module so the generated
query manifest can be frozen and hashed before benchmark use.
"""

from __future__ import annotations

import re

from protocols.common import ProtocolInputError, require_nonempty_text


QUERY_COUNT = 5
MODEL = "gpt-4.1-nano"
QUERY_TEMPERATURE = 0.7
SUMMARY_TEMPERATURE = 0.3


def build_prompt(document: str) -> str:
    document = require_nonempty_text(document, "document")
    return f"""Given the document below, generate {QUERY_COUNT} highly specific questions that can be answered by this document.

DOCUMENT:
{document}

Requirements:
- Generate EXACTLY {QUERY_COUNT} different questions
- **CRITICAL: The set of questions must cover ALL different aspects/sections of the document.**
- **DISTRIBUTION: Do not focus all questions on a single fact. If the text has a beginning, middle, and end, or multiple distinct points, ensure the {QUERY_COUNT} questions are distributed across these different parts.**
- Each question should require specific information from the document
- Focus on unique details, specific facts, or specific combinations of information
- Make each query DIFFERENT by using different phrasing, focusing on different aspects, varying question structure
- DO NOT mention "the document", "the text", "this passage", or similar references
- DO NOT add meta-preambles like "Here is a question:", "Question:", or "Query:"
- If the text uses any abbreviations or acronyms, use the same forms in your questions
- Avoid mentioning 'the study' or any references to the passage itself

Output format (IMPORTANT - follow this EXACTLY):
QUERY_1: [first question here]
QUERY_2: [second question here]
QUERY_3: [third question here]
...

Generate {QUERY_COUNT} queries now:"""


def build_system_prompt() -> str:
    return (
        "You are a helpful assistant that generates highly specific questions. "
        f"Always output exactly {QUERY_COUNT} questions in the format: "
        "QUERY_1: [question], QUERY_2: [question], etc."
    )


def build_summary_prompt(document: str) -> str:
    """Return the topic-focused description prompt reused from IA by MEntA."""

    document = require_nonempty_text(document, "document")
    return f"""Task Description:
You are tasked with generating a concise and accurate topic-focused description of a document based on its content and title (if provided). The description should:
1. Be a single, short sentence.
2. Focus only on the main topic or subject of the document, avoiding verbs and conclusions.
3. Include important keywords from the document.
4. Avoid referencing the document itself with phrases like "The document discusses," "The report highlights," or "This paper investigates."
5. Output only a short, noun-phrase-like description or topic sentence.

Examples:
- Instead of: "The report from the Düsseldorf conference highlights advancements in green energy technologies."
- Generate: "Advancements in green energy technologies and discussions at the Düsseldorf conference."
- Instead of: "The document investigates the cyclooxygenase pathway in inflammatory responses."
- Generate: "The cyclooxygenase pathway and its role in inflammatory responses."

Ensure the description is concise, focused on the main topic, and includes relevant keywords. Avoid any extra text, explanations, or labels.

Input:
Text: {document}

Output:
Provide only the one-sentence topic-focused description as the output.
"""


def parse_five_queries(text: str) -> list[str]:
    text = require_nonempty_text(text, "generated query text")
    indexed: dict[int, str] = {}
    for line in text.splitlines():
        match = re.fullmatch(r"\s*QUERY_(\d+)\s*:\s*(\S(?:.*\S)?)\s*", line, flags=re.I)
        if not match:
            if line.strip():
                raise ProtocolInputError(f"malformed MEntA query line: {line!r}")
            continue
        index = int(match.group(1))
        query = match.group(2).strip()
        if index in indexed:
            raise ProtocolInputError(f"duplicate MEntA query index: {index}")
        if not query.endswith("?"):
            raise ProtocolInputError(f"MEntA query {index} must end with a question mark")
        indexed[index] = query
    if set(indexed) != set(range(1, QUERY_COUNT + 1)):
        raise ProtocolInputError("MEntA output must contain QUERY_1 through QUERY_5 exactly once")
    queries = [indexed[index] for index in range(1, QUERY_COUNT + 1)]
    if len(set(queries)) != QUERY_COUNT:
        raise ProtocolInputError("MEntA requires five distinct questions")
    return queries


def make_session(target_id: str, membership: str, summary: str, questions: list[str]) -> list[dict]:
    from protocols.common import require_membership_label

    target_id = require_nonempty_text(target_id, "target_id")
    require_membership_label(membership)
    summary = require_nonempty_text(summary, "summary")
    if len(questions) != QUERY_COUNT or len(set(questions)) != QUERY_COUNT:
        raise ProtocolInputError("MEntA session must have five distinct ordered questions")
    return [
        {
            "session_id": f"menta::{target_id}",
            "query_id": f"menta::{target_id}::q{index}",
            "query_index": index,
            "target_id": target_id,
            "membership": membership,
            "question": require_nonempty_text(question, "question"),
            "summary": summary,
            "query": f"{summary} {require_nonempty_text(question, 'question')}",
        }
        for index, question in enumerate(questions, start=1)
    ]
