#!/usr/bin/env python3
"""Run all protocol contract tests and persist a machine-readable audit."""

from __future__ import annotations

import io
import json
import sys
import unittest
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class RecordingResult(unittest.TextTestResult):
    def startTest(self, test):  # noqa: N802 - unittest API
        super().startTest(test)
        self.seen_ids.append(test.id())


def main() -> int:
    stream = io.StringIO()
    suite = unittest.defaultTestLoader.discover(
        str(ROOT / "protocols"), pattern="test_protocol.py", top_level_dir=str(ROOT / "protocols")
    )

    class RecordingRunner(unittest.TextTestRunner):
        resultclass = RecordingResult

        def _makeResult(self):  # noqa: N802 - unittest API
            result = super()._makeResult()
            result.seen_ids = []
            return result

    result = RecordingRunner(stream=stream, verbosity=2).run(suite)
    by_attack = Counter(test_id.split(".", 1)[0] for test_id in result.seen_ids)
    payload = {
        "campaign": "CORE6_PROTOCOL_RECOVERY_V1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "tests_run": result.testsRun,
        "successful": result.wasSuccessful(),
        "failures": [test.id() for test, _ in result.failures],
        "errors": [test.id() for test, _ in result.errors],
        "skipped": [test.id() for test, _ in result.skipped],
        "tests_by_attack": dict(sorted(by_attack.items())),
        "command_equivalent": "python -m unittest discover -s protocols -p test_protocol.py -v",
    }
    audits = ROOT / "audits"
    audits.mkdir(exist_ok=True)
    (audits / "UNIT_TEST_RESULTS.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (audits / "UNIT_TEST_RESULTS.txt").write_text(stream.getvalue(), encoding="utf-8")
    print(stream.getvalue(), end="")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    raise SystemExit(main())
