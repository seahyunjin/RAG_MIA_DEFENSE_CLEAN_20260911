#!/usr/bin/env python3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from ia_v5_protocol import validate_label, validate_question


def main():
    checks = [
        validate_question("Is aspirin discussed?", []).valid,
        validate_question("Does it include 25 participants?", []).valid,
        not validate_question("What treatment was used?", []).valid,
        not validate_question('"Is aspirin discussed?"', []).valid,
        not validate_question("Is aspirin discussed?\nExtra", []).valid,
        not validate_question("Is aspirin discussed?", ["Is aspirin discussed?"]).valid,
        not validate_question("Is the study about aspirin?", []).valid,
        not validate_question("Does THE PASSAGE mention aspirin?", []).valid,
        validate_label("Yes") == (True, "Yes", "OK"),
        validate_label("Unknown") == (True, "Unknown", "OK"),
        validate_label("10 percent")[0] is False,
    ]
    if not all(checks):
        raise SystemExit("unit test failure")
    print(f"PASS {len(checks)}/{len(checks)}")


if __name__ == "__main__":
    main()
