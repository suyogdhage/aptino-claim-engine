# Project TODO

This list is based on the current source tree and the existing evaluation report
(`evaluation_results/evaluation_report.json`). Items are ordered by impact.

## P0 — Make decisions reliable

- [x] Fix the offline mock's expense-field mismatch in `app/llm.py`. It reads
  `doctor` and `medicines`, while the validated request model exposes
  `doctor_fees` and `medicines_diagnostics`; consequently, those limits are
  silently evaluated as zero in mock mode.
- [ ] Add deterministic unit tests for each policy rule and boundary: 30-day
  initial waiting period, 48-month PED rule, first-year conditions, day care,
  domiciliary conditions, pre/post-hospitalization windows, hospital criteria,
  portability, and every category sub-limit.
- [x] Add the full 17-case mock evaluation and its 100% accuracy/citation gate
  to CI. Re-run it from a clean Python environment before each submission.
- [x] Record a short rationale and policy-page reference for all 17 expected
  outcomes in `docs/CASE_RATIONALES.md`.

## P1 — Harden the LLM workflow

- [x] Treat Groq rate-limit failures as an explicit, retryable service outcome:
  stop a batch when the daily budget is exhausted, retain partial results, and
  report skipped cases separately instead of counting them as incorrect.
- [x] Add a switch to force mock mode for local development and CI, independent
  of whether a real API key happens to be present.
- [x] Make validation actually gate output: after the bounded retry limit, return
  `NEEDS_REVIEW` (or a clearly flagged validation failure) rather than publishing
  a substantive decision whose validation remains `FAIL`.
- [ ] Tighten prompts and parsing so every finding has a directly supporting
  citation; avoid unsupported blanket findings such as “no other exclusions
  apply” and “all required documents provided.”
- [x] Add request timeouts, structured error logs, and a safe public API error
  response that does not expose provider exception details.

## P2 — Improve service correctness and deployment

- [x] Replace permissive CORS (`allow_origins=["*"]` with credentials enabled)
  with an environment-configured allowlist.
- [x] Validate both Chroma and BM25 against a policy-hash manifest at startup;
  build a staged replacement index when either is missing or stale.
- [x] Pin Python 3.11 and provide `requirements.lock` as the reproducible
  install entry point.
- [x] Add readiness details to `/health`; `/analyze` stays unavailable until the
  policy index is ready.
- [x] Investigate the Pydantic warning in `uvicorn_err.log` and append an
  `AgentTrace` model instance rather than a raw dictionary in `app/main.py`.

## P3 — Engineering hygiene and usability

- [ ] Create an initial commit and push this repository to a public GitHub remote.
- [x] Add CI for unit tests and the mock-mode evaluation gate.
- [x] Move generated runtime logs (`uvicorn_*.log`, `eval_run.*`) out of the
  project root or add them to `.gitignore`; retain reports only when they are
  intentionally published artifacts.
- [x] Document API validation/service failures and first-start deployment needs.
- [x] Surface API failures and validation details in the Streamlit UI with an
  actionable retry path.

## Suggested completion order

1. Correct the mock expense keys and write policy-rule tests.
2. Run and calibrate the deterministic evaluation until all expected cases pass.
3. Add rate-limit handling and validation-safe finalization for the Groq path.
4. Add startup, security, CI, and deployment hardening.
