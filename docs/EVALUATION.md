# Evaluation and Reliability Notes

## Reproducible command

Run the evaluation without consuming an LLM-provider quota:

```powershell
$env:FORCE_MOCK_LLM='true'
python -m app.evaluation.evaluate
```

The command evaluates all 12 supplied public cases and all 5 candidate-created
cases, then writes `evaluation_results/evaluation_report.json`. A rate-limited
provider result is marked as skipped rather than counted as an incorrect
decision.

## Measures

The report records decision accuracy, average confidence, abstention rate,
citation coverage, per-set results, citations per case, validation status, and
agent trace data. Expected outcomes are versioned in
`app/evaluation/expected_outcomes.json` and the supplied public cases are never
modified.

## Reliability scenarios and improvements

| Scenario | Root cause found | Improvement implemented |
|---|---|---|
| Category-limit decisions missed doctor and medicine charges in offline mode | The mock parser stored prompt labels as `doctor` and `medicines`, but the validated request schema names the fields `doctor_fees` and `medicines_diagnostics`. | Mapped prompt labels to the canonical schema fields and added regression tests for both limits. |
| Provider daily quota interrupted the custom-case evaluation | The original run used Groq for every case and treated rate-limit errors as ordinary failed decisions. | Added `FORCE_MOCK_LLM` for deterministic local and CI runs; evaluator now records 429 failures as skipped. |
| Final decisions could contain no citations while still reporting validation PASS | The deterministic decision response provided a label but no material findings, so validation had nothing to inspect. | Promote cited specialist findings when the provider omits findings; validation rejects uncited material findings; a persistent validation failure becomes `NEEDS_REVIEW`. |
| Public API errors revealed provider exception details | The API returned raw exception strings in HTTP 500 responses. | Log the exception server-side and return a generic 503 response to callers. |

## Release gate

1. `python -m unittest discover -s tests -v` passes.
2. The complete 17-case forced-mock evaluation completes with no skipped cases.
3. Every non-abstaining final response has one or more inspectable policy
   citations per material finding.
4. Any remaining decision mismatches are reviewed against the source policy and
   expected-outcome rationale before changing either implementation or expected
   data.
