"""Fail CI when the deterministic evaluation is not a submission-ready run."""
import json
import sys
from pathlib import Path


def main() -> int:
    path = Path("evaluation_results/evaluation_report.json")
    if not path.exists():
        print("Missing evaluation report")
        return 1
    summary = json.loads(path.read_text(encoding="utf-8"))["summary"]
    failures = []
    if summary.get("total_cases") != 17:
        failures.append("expected all 17 cases to run")
    if summary.get("skipped_cases"):
        failures.append(f"skipped cases: {summary['skipped_cases']}")
    if summary.get("accuracy") != 1.0:
        failures.append(f"accuracy is {summary.get('accuracy')}, expected 1.0")
    if len(summary.get("abstained_correctly", [])) < 2:
        failures.append("fewer than two expected abstentions were handled correctly")
    if summary.get("citation_coverage") != 1.0:
        failures.append("citation coverage is not 100%")
    if failures:
        print("Release gate failed: " + "; ".join(failures))
        return 1
    print("Release gate passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
