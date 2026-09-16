"""Fail CI when the evaluation report is not a submission-ready run.

The deterministic mock run (FORCE_MOCK_LLM=true) is expected to be exact, so
the gate is strict (accuracy 1.0, all abstentions + recall + coverage).
A real Groq run (llm_mode == "groq") is stochastic and subject to provider
behaviour, so it uses calibrated floors instead of perfection.
"""
import json
import sys
from pathlib import Path


def _check(summary: dict, *, strict: bool) -> list:
    failures = []
    if summary.get("total_cases") != 17:
        failures.append("expected all 17 cases to run")
    if summary.get("skipped_cases"):
        failures.append(f"skipped cases: {summary['skipped_cases']}")
    if strict:
        if summary.get("accuracy") != 1.0:
            failures.append(f"accuracy is {summary.get('accuracy')}, expected 1.0")
        if len(summary.get("abstained_correctly", [])) < 2:
            failures.append("fewer than two expected abstentions were handled correctly")
        if summary.get("citation_coverage") != 1.0:
            failures.append("citation coverage is not 100%")
        recall8 = summary.get("retrieval_recall_at_8")
        if recall8 is None:
            failures.append("retrieval_recall_at_8 not present in report")
        elif recall8 < 0.8:
            failures.append(f"retrieval_recall_at_8 is {recall8}, expected >= 0.8")
    else:
        accuracy = summary.get("accuracy", 0.0)
        if accuracy < 0.85:
            failures.append(f"accuracy is {accuracy}, expected >= 0.85 for a real Groq run")
        if len(summary.get("abstained_correctly", [])) < 2:
            failures.append("fewer than two expected abstentions were handled correctly")
        if summary.get("citation_coverage", 0.0) < 0.9:
            failures.append("citation coverage is below 0.9")
        recall8 = summary.get("retrieval_recall_at_8")
        if recall8 is None:
            failures.append("retrieval_recall_at_8 not present in report")
        elif recall8 < 0.8:
            failures.append(f"retrieval_recall_at_8 is {recall8}, expected >= 0.8")
    return failures


def main() -> int:
    path = Path("evaluation_results/evaluation_report.json")
    if not path.exists():
        print("Missing evaluation report")
        return 1
    summary = json.loads(path.read_text(encoding="utf-8"))["summary"]
    strict = summary.get("llm_mode", "mock") == "mock"
    failures = _check(summary, strict=strict)
    if failures:
        print("Release gate failed: " + "; ".join(failures))
        return 1
    print(f"Release gate passed ({'strict mock' if strict else 'real Groq'} run)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())