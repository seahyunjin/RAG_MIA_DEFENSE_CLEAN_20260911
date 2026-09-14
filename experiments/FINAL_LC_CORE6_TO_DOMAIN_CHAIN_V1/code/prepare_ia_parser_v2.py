#!/usr/bin/env python3
"""Freeze the one permitted syntax-only IA parser repair before reparsing raw outputs."""
from __future__ import annotations

import subprocess

from common import EXP, TORCH_PYTHON, checkpoint, freeze_json, now, sha_file


def main():
    forbidden=[EXP/name for name in ("PHASE_A_RESULT.json","PHASE_B_RESULT.json","PHASE_C_RESULT.json","PHASE_D_RESULT.json")]
    if any(path.exists() for path in forbidden):raise RuntimeError("performance result already exists; parser repair not pre-performance")
    parser=EXP/"code"/"ia_parser_v2.py";tests=EXP/"tests"/"test_ia_parser_v2.py";database=EXP/"runtime"/"standardized_query_generation.sqlite3"
    subprocess.run([str(TORCH_PYTHON),"-m","unittest",str(tests),"-v"],check=True,cwd=EXP)
    spec={"campaign":EXP.name,"name":"IA-Std-Q15 deterministic parser v2","created_utc":now(),
        "scope":"one parser-only repair over all 2,000 frozen session records; no regeneration",
        "performance_results_opened":False,"raw_database":{"path":str(database),"sha256":sha_file(database)},
        "attempt_selection":"parse frozen attempt 1 first; use already-frozen attempt 2 only if attempt 1 is not exact; never combine attempts",
        "record_rule":"exactly numbered records 1..15, each with a nonempty question ending '?' and exactly one explicit binary Yes or No judgment",
        "allowed_format_normalization":["question/answer line break","whitespace","explicit Question/Answer field placement","exact repeated current record index"],
        "forbidden":["fuzzy matching","semantic inference","missing answer interpolation","Yes/No estimation","cross-attempt field combination","query regeneration"],
        "invalid_categories":{"FORMAT_ONLY":"previously invalid under v1 and exactly valid under v2","CONTENT_MISSING":"fewer than 15 records, missing question, or missing explicit binary judgment","OTHER":"ambiguous numbering, multiple judgments, duplicate question, unsupported extra content/layout"},
        "gate":{"valid_sessions":">=0.95","member_nonmember_validity_gap":"<=0.02 absolute","questions_per_valid_session":"exactly 15","binary_judgments_per_valid_session":"exactly 15","inferred_fields":0},
        "parser":{"path":str(parser),"sha256":sha_file(parser)},"unit_tests":{"path":str(tests),"sha256":sha_file(tests),"status":"PASS"}}
    digest=freeze_json(EXP/"configs"/"IA_STD_Q15_PARSER_V2_SPEC.json",spec)
    checkpoint("IA_STD_Q15_PARSER_V2_PRECOMMITTED",spec_sha256=digest,unit_tests="PASS",performance_results_opened=False)


if __name__=="__main__":main()
