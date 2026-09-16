# Aptino Claim Decision Engine — Design Document

## 1. Overview

The **Aptino Policy-Aware Multi-Agent RAG Claim Decision Engine** analyzes health-insurance
claim cases against a policy document (PDF) and produces a structured, evidence-grounded
decision.

Inputs:

- A claim case as JSON (diagnosis, treatment, expenses, documents, hospital, coverage history).
- A policy PDF (here: USGIC-CSC Individual Health Insurance, 2017–2018 wording).

Output: a `DecisionResponse` containing:

- Decision status: `ADMISSIBLE`, `ADMISSIBLE_WITH_LIMITS`, `PARTIALLY_ADMISSIBLE`, `NOT_ADMISSIBLE`, or `NEEDS_REVIEW`.
- Confidence, key findings, applicable limits/deductions, missing evidence, citations.
- Validation result (are the findings actually supported by the cited policy chunks?).
- A per-agent execution trace.

Design intent:

- **Policy-aware**: every statement must trace back to specific policy chunks (page/section/clause).
- **Abstain-when-uncertain**: if evidence is insufficient (unconfirmed hospital registration,
  missing medical-necessity confirmation, missing documents), the system returns
  `NEEDS_REVIEW` rather than guessing.
- **Verifiable**: a validation agent checks each material claim against the cited evidence, and
  a failed validation triggers a bounded retry of the decision step.

## 2. System Architecture

```
Claim (JSON) ──► Case Analysis ──► Policy Evidence ──► Coverage & Exclusion
                     │                    │                    │
                     ▼                    ▼                    ▼
                  Invest. Plan      Retriever per         Coverage findings,
                     + gaps          dimension (RAG)       limits, blockers
                                                     │
                                                     ▼
                                                 Decision Agent
                                                     │
                                                     ▼
                                              Validation Agent
                                                     │
                                                PASS or retry ≥ 2
                                                     │
                                                     ▼
                                               Final Structured Response
                                    (decision + findings + citations + limits + trace)
```

Graph implementation: `app/graph/workflow.py` (LangGraph `StateGraph`).

- Nodes: `case_analysis → policy_evidence → coverage_exclusion → decision → validation`.
- Conditional edge from `validation`: on `PASS` (or after 2 decision retries) → `finalize`, else loop back to `decision`.
- `retry_count` is incremented on each decision invocation; the loop is bounded so the workflow always terminates.
- The compiled graph is wrapped by `ClaimEngine`, which also builds the hybrid retriever from the persisted Chroma + BM25 indices.

## 3. Components & Responsibilities

### 3.1 Ingestion — `app/ingestion/policy_ingest.py`

- Extracts page text with `pdfplumber`.
- Filters non-policy noise: page footers, repeated headers, ombudsman/contact appendix, city-pair headings.
- Identifies semantic sections across pages (running state is carried forward):
  `DEFINITIONS`, `SCOPE OF COVER`, `WHAT WE COVER`, `WHAT WE EXCLUDE`,
  `CLAIMS PROCEDURE`, `EXTENSIONS`, etc.
- Tracks sub-structure:
  - definition entries (`<term> means ...`) → `clause_id` = the term;
  - numbered items (`1. Room`, `3. Hospitalization ...`) → `clause_id` = the numbered heading;
  - `Exclusion` keyword → `subsection`.
- Chunking: 500-token chunks with 50-token overlap (per config).
- Chunk ID: `md5("{policy_name}|p{page}|{counter}|{section}")[:12]` — unique and traceable.
- Result: `PolicyChunk` objects (chunk_id, text, page, section, subsection, clause_id, token_count).

### 3.2 Retrieval — `app/retrieval/`

Hybrid dense + sparse retrieval with reciprocal-rank fusion (RRF) and a cross-encoder reranker.

| Stage | Store / Model | Param |
|---|---|---|
| Dense | Chroma (BAAI/bge-small-en-v1.5 embeddings) | `dense_top_k = 20` |
| Sparse | rank-bm25 (`BM25Okapi`) | `sparse_top_k = 20` |
| Fusion | RRF | `fusion_k = 60` |
| Rerank | `cross-encoder/ms-marco-MiniLM-L-6-v2` | `rerank_top_k = 12` → `final_top_k = 8` |

- Files: `chroma_store.py`, `bm25_index.py`, `hybrid_retriever.py`.
- Retrieval runs **per investigation dimension** (e.g. waiting periods, exclusions, category
  limits), producing a ranked list of `Evidence` (chunk_id, text, page, section, clause_id,
  dense/sparse/rerank scores) for each dimension.
- `build_indices` (`hybrid_retriever.py`) is the one-shot path for constructing both indexes from the PDF.

### 3.3 Agents — `app/agents/`

Four chat completions (Groq, JSON mode) plus one deterministic retrieval node. Each
completion is issued through `groq_json_completion` in `app/llm.py`.

| Agent | File | Input → Output |
|---|---|---|
| Case Analysis | `case_analysis.py` | full case → `InvestigationPlan` (dimensions, missing_fields, checklist, priority_order) |
| Policy Evidence | `policy_evidence.py` | plan + case → per-dimension `Evidence[]` via the hybrid retriever |
| Coverage & Exclusion | `coverage_exclusion.py` | case + per-dimension evidence → `CoverageFindings` (findings, limits, blockers, overall_admissible) |
| Decision | `decision.py` | case + coverage findings + available evidence → `DecisionDraft` (status, confidence, key findings, limits) |
| Validation | `validation.py` | draft + cited evidence → `ValidationResult` (PASS/FAIL + unsupported claims) |

Agent defaults pull the model from `settings.GROQ_MODEL` (`openai/gpt-oss-120b`); any agent
can be constructed with an explicit override.

### 3.4 Graph / State — `app/graph/workflow.py`, `app/models/state.py`

`ClaimState` (TypedDict) carries: `case`, `investigation_plan`, `retrieved_evidence`,
`coverage_findings`, `decision_draft`, `validation_result`, `final_decision`, `trace`,
`retry_count`, `error`.

Termination logic: validation PASS or `retry_count >= 2` → `finalize`, which assembles the
flat `DecisionResponse` (fanning findings/limits into the citations list and stamping each
citation with its owning finding's claim text).

### 3.5 API & UI — `app/main.py`, `app/frontend.py`

- FastAPI:
  - `GET /health` → health/version.
  - `POST /analyze` → body is one claim JSON → runs `ClaimEngine.analyze` → `DecisionResponse`.
  - On startup, indexes the policy into the persisted store if empty.
- Streamlit UI:
  - Load a public test case, upload JSON, or paste JSON.
  - Renders decision badge, confidence, key findings, applicable limits, missing evidence,
    citations, validation status, and execution trace; download of the full result JSON.

### 3.6 Evaluation — `app/evaluation/`

- `evaluate.py` runs all public (12) and custom (5) cases, compares against
  `expected_outcomes.json`, and writes `evaluation_results/evaluation_report.json`.
- Metrics: decision accuracy, average confidence, abstention rate (correct/incorrect
  abstentions), citation coverage, and a per-case breakdown (predicted vs expected, citation
  counts, validation status, unsupported claims, trace).

## 4. Data Model — `app/models/claim.py`

Key entities:

- `ClaimCase`: case_id, policy_id, dates, sum_insured_inr, `continuous_coverage_months`,
  `prior_insurer_continuous_years`, patient, hospital, treatment, expenses_inr, documents,
  task, plus optional `evidence_context` (hospital_registered, medical_necessity_confirmed,
  hospital_minimum_criteria_documented), `expense_timing` (pre/post hospital days, same
  condition), and `prior_policy` (portability context).
- `Treatment`: type (inpatient/day_care/domiciliary), admission_hours, diagnosis, procedure,
  pre_existing, experimental, and domiciliary conditions
  (`hospital_room_unavailable`, `patient_cannot_be_moved`).
- `ExpensesINR`: room, doctor_fees, medicines_diagnostics, pre/post hospitalization, ambulance.
- `DecisionResponse`: decision, confidence, key_findings, applicable_limits, missing_evidence,
  citations, validation, trace.

Decision statuses:

| Status | Meaning |
|---|---|
| `ADMISSIBLE` | Fully covered, no material deduction. |
| `ADMISSIBLE_WITH_LIMITS` | Covered but capped/deducted per policy limits. |
| `PARTIALLY_ADMISSIBLE` | Only part of the claim is payable. |
| `NOT_ADMISSIBLE` | Policy supports rejection/exclusion. |
| `NEEDS_REVIEW` | Insufficient evidence — the system abstains. |

## 5. Policy Interpretation & Decision Logic

Key rules derived from the USGIC-CSC 2017–2018 wording and encoded in the expected outcomes:

- **Initial waiting period**: 30 days from policy start.
- **Pre-existing diseases (PED)**: not covered until **48 months** of continuous coverage.
- **First-year disease list** (waiting ≤12 months): cataract, BPH, myomectomy/hysterectomy,
  hernia/hydrocele, fistula/piles, arthritis/gout/rheumatism, sinusitis, stones, D&C,
  tumors/cysts/polyps, dialysis, tonsils, gastric/duodenal ulcers.
- **Category sub-limits** (WHAT WE COVER):
  - Normal room rent: 1% of Basic Sum Insured per day;
  - ICU: 2% of Sum Insured per day;
  - medical practitioner fees: 25% of Sum Assured;
  - medicines/diagnostics/etc.: 40% of Sum Insured;
  - ambulance: 1% of Sum Insured or ₹1,000, whichever is lower.
- **Pre/post-hospitalization**: ≤30 days before admission and ≤60 days after discharge
  (same-condition relatedness required).
- **Domiciliary treatment**: requires **both** (a) hospital room unavailable AND
  (b) patient cannot be moved; subject to a domiciliary sub-limit.
- **Day care**: procedures in the covered day-care list qualify with <24h stays.
- **Exclusions**: cosmetic, dental, experimental/unproven treatments, outpatient treatment.
- **Hospital definition**: registered under the Clinical Establishments (Registration and
  Regulation) Act 2010, or minimum bed/qualified-staffing criteria for unregistered facilities.
- **Portability**: with ≥1 year continuous coverage under another Indian insurer and the
  required claim-history database, waiting periods reduce (first-year list waived; PED reduced).

## 6. Token Budgeting & Rate Limits

Groq free tier is per-request/org limited (TPM/TDP). Prompts are sized to stay comfortably
under the per-request cap:

- Coverage & Exclusion prompt: ~3.8k tokens (per-dimension evidence budget, ≤3 chunks per dimension).
- Decision prompt: ~2k tokens (compact evidence listing, ≤3 chunks/dimension, 220-char snippets).
- Validation prompt: ~0.2k tokens (only cited chunks).
- Case Analysis prompt: small (full case JSON only).

`groq_json_completion` (`app/llm.py`) applies retry/backoff on transient failures
(`429` rate limits, connection errors, `json_validate_failed`).

## 7. Mock LLM Mode

When no valid `GROQ_API_KEY` is configured (empty, `test-key`, or placeholder), the system
falls back to a deterministic **MockLLM** (`app/llm.py`) so the full pipeline, tests, and
evaluation run offline and reproducibly.

- Mimics each agent's JSON contract: dimension planning, coverage findings, and a rule-based
  decision (waiting periods, PED 48-month rule, first-year disease list, exclusions,
  domiciliary conditions, portability, category-limit binding checks, evidence-gap abstention).
- Allows the 17-case suite to be re-run without network or API cost while the real-LLM path
  remains the production branch.

## 8. Security & Failure Handling

- Secrets are never committed: the API key lives in `.env` (gitignored); `.env.example` ships
  with placeholders.
- Abstention policy: unconfirmed critical facts → `NEEDS_REVIEW` (no guessing).
- Validation retry loop is bounded (max 2 decision retries) so the workflow always terminates.
- Defensive JSON parsing: malformed/partial LLM findings and limits (non-dict entries,
  unknown dimensions) are skipped, with fallback findings when nothing parses.
- Rate-limit/connection retries with backoff in all agent completions.

## 9. Deployment (Render)

Two services defined in `render.yaml`:

- `aptino-claim-api` — FastAPI via uvicorn; env `GROQ_API_KEY`, `GROQ_MODEL`,
  `CHROMA_PERSIST_DIR=/data`, `POLICY_PDF_PATH`, embedding/reranker model names; 1GB disk at
  `/data` persists the Chroma + BM25 indexes (auto-built on first startup).
- `aptino-claim-ui` — Streamlit frontend pointed at `https://aptino-claim-api.onrender.com`.

## 10. Directory Map

```
app/
  ingestion/policy_ingest.py   PDF extraction + section detection + chunking
  retrieval/                   Chroma store, BM25, hybrid retriever (RRF + rerank)
  agents/                      case_analysis, policy_evidence, coverage_exclusion,
                               decision, validation
  graph/workflow.py            LangGraph StateGraph + ClaimEngine
  models/                      claim.py (Pydantic schemas), state.py (ClaimState),
                               response.py
  main.py                      FastAPI app
  frontend.py                  Streamlit UI
  llm.py                       Groq client/JSON helper + MockLLM
  evaluation/                  evaluate.py, expected_outcomes.json
  ingest.py                    CLI entrypoint to build indexes
data/                          policy.pdf, public_test_cases.json, custom_test_cases.json
evaluation_results/            generated evaluation reports
render.yaml, requirements.txt, README.md, .env.example
docs/DESIGN.md                 this file
```