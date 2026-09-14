"""S²-MIA query construction for structured QA target samples."""

from __future__ import annotations

from protocols.common import ProtocolInputError, require_membership_label, require_nonempty_text


QUERY_COUNT = 1


def make_query(target_id: str, membership: str, query_text: str, remaining_text: str) -> dict:
    """Create the one paper query without inventing a raw-document split.

    The paper experiments use the dataset question as ``query_text`` and the
    answer/remaining part as ``remaining_text``.  Callers must supply both.
    """

    target_id = require_nonempty_text(target_id, "target_id")
    require_membership_label(membership)
    query_text = require_nonempty_text(query_text, "query_text")
    remaining_text = require_nonempty_text(remaining_text, "remaining_text")
    prompt = (
        f'Given the [Query]: "{query_text}".\n'
        "Do not include any introductory or explanatory text, use the following format for output: "
        "{[Response]: 'Provide a concise response directly addressing the [Query] by using the most relevant and matching text in the prompt.'}"
    )
    return {
        "session_id": f"s2_mia::{target_id}",
        "query_id": f"s2_mia::{target_id}::q1",
        "target_id": target_id,
        "membership": membership,
        "query": prompt,
        "query_text": query_text,
        "reference_text": remaining_text,
    }


def make_query_from_raw_document(*_args, **_kwargs):
    raise ProtocolInputError(
        "paper-experiment fidelity requires an explicit question/remaining-text pair; "
        "an arbitrary raw-document split is not silently defined"
    )

