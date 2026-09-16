# Aptino Claim Decision Engine Interview Guide

## 30-second introduction

I built a policy-aware, multi-agent RAG system for health-insurance claim
decisions. A claim is not answered by a single retrieval call. Instead, the
system plans the investigation, retrieves policy evidence for each relevant
dimension, evaluates coverage and exclusions, produces a structured decision,
and validates that material findings have policy citations. When the claim facts
or policy evidence are insufficient, it returns `NEEDS_REVIEW` rather than
guessing.

## Problem and success criteria

The supplied policy PDF is the authority. The product must decide one claim at
a time and return a machine-readable result with a decision status, confidence,
findings, limits, missing evidence, citations, and a concise execution trace.

The important reliability requirements were:

- Use meaningful policy chunks with page and section metadata.
- Combine dense semantic retrieval and sparse lexical retrieval, then rerank.
- Separate agent responsibilities and exchange structured state.
- Make every material conclusion inspectable through policy citations.
- Abstain safely when a required fact or clause is unavailable.

## End-to-end flow

```text
Claim JSON
  -> Case Analysis Agent
  -> Policy Evidence Agent
  -> Coverage and Exclusion Agent
  -> Decision Agent
  -> Validation Agent
  -> Structured API response and reviewer UI
```

### 1. Input validation

FastAPI receives `POST /analyze`. Pydantic validates the claim shape before the
workflow begins: dates, treatment type, hospital, expenses, documents, and
optional evidence context. Invalid payloads return HTTP 422.

### 2. Case Analysis Agent

This agent does not decide coverage. It extracts decision dimensions and missing
facts, producing an `InvestigationPlan`. Typical dimensions include initial
waiting period, pre-existing disease, hospital definition, exclusions,
domiciliary conditions, portability, pre/post-hospitalization, and category
limits.

This separation prevents the retriever from being driven by one broad,
underspecified query.

### 3. Policy ingestion and indexing

`app/ingestion/policy_ingest.py` extracts the PDF page by page with
`pdfplumber`. It recognizes policy headings, definitions, numbered clauses, and
exclusions; filters repeated headers/footers and contact appendix noise; then
creates approximately 500-token chunks with a 50-token overlap.

Every chunk stores:

- `chunk_id`
- policy page
- section, subsection, and clause identifier when available
- source text and token count

The dense index is persisted in Chroma. The same chunk set is persisted in a
BM25 index. An index manifest stores the policy SHA-256 and expected chunk
count. Startup checks that both stores and the manifest match the current PDF;
if not, it builds a staged replacement. This avoids serving a stale or
half-built retrieval index.

### 4. Hybrid retrieval and reranking

For each investigation dimension, the system runs two searches:

- Dense semantic retrieval using `BAAI/bge-small-en-v1.5` in Chroma.
- Sparse lexical retrieval using BM25.

The result lists are fused with reciprocal-rank fusion (RRF). The top fused
chunks are reranked by `cross-encoder/ms-marco-MiniLM-L-6-v2`; the best eight
chunks become structured `Evidence` records. This approach handles both
semantic questions and exact policy language such as waiting periods or named
exclusions.

### 5. Coverage and Exclusion Agent

This agent consumes the claim, investigation plan, and retrieved evidence. It
produces structured coverage findings, applicable limits, and blocking issues.
Its purpose is policy interpretation across multiple decision dimensions, not
formatting the final response.

Examples of policy rules considered by the workflow include the 30-day initial
waiting period, the 48-month pre-existing-disease waiting period, first-year
specified diseases, domiciliary conditions, category limits, exclusions,
hospital qualification, and portability.

### 6. Decision Agent

The Decision Agent combines specialist findings into one of five statuses:

| Status | Meaning |
| --- | --- |
| `ADMISSIBLE` | Covered without a material limit identified. |
| `ADMISSIBLE_WITH_LIMITS` | Covered, but a policy cap or deduction affects payment. |
| `PARTIALLY_ADMISSIBLE` | Only part of the request is supported. |
| `NOT_ADMISSIBLE` | A policy exclusion or unmet condition supports rejection. |
| `NEEDS_REVIEW` | Facts or policy support are insufficient for a safe decision. |

It returns a compact rationale, key findings, limits, missing evidence, and
chunk references. It does not expose chain-of-thought.

### 7. Validation Agent

The Validation Agent checks whether each material finding and limit has a cited
retrieved chunk and whether that evidence supports the statement. A failed
validation makes the workflow retry the decision once. If validation remains
unsuccessful, finalization converts the response to `NEEDS_REVIEW` and clearly
flags the validation issue.

That is the key safety property: an unsupported substantive conclusion should
not reach the reviewer as an approved claim decision.

## Data contracts and observability

The `DecisionResponse` contains the case ID, decision, confidence, structured
findings, limits, missing evidence, flattened citations, validation result, and
per-agent trace. The trace exposes agent name, major action, elapsed time,
evidence count, and safe metadata. It is designed to be auditable without
revealing private chain-of-thought.

The Streamlit frontend lets a reviewer select a public case, upload JSON, or
paste JSON. It displays the decision, citations, validation outcome, missing
evidence, limits, and trace. API service errors provide a retry-oriented
message instead of leaking provider internals.

## Evaluation strategy

The repository evaluates all 12 supplied public cases plus five candidate-made
cases. It reports decision accuracy, confidence, abstention behavior, citation
coverage, per-case trace, retrieval counts, and validation status. The expected
outcomes and source-page rationales are documented in
`app/evaluation/expected_outcomes.json` and `docs/CASE_RATIONALES.md`.

For repeatability, `FORCE_MOCK_LLM=true` uses a deterministic local mock rather
than consuming a hosted LLM quota. CI runs the unit suite, the 17-case mock
evaluation, and a release checker that expects all cases, no skips, 100%
decision accuracy, two or more correct abstentions, and 100% citation coverage.

## Design decisions and trade-offs

### Why multiple agents instead of one prompt?

Different tasks need different inputs and failure checks. Planning the
investigation, retrieving evidence, interpreting coverage, choosing a decision,
and validating citations have distinct contracts. Structured state makes those
boundaries inspectable and testable.

### Why hybrid retrieval?

Dense search is useful for paraphrased claim descriptions. BM25 is useful for
precise policy terms and numbers. RRF gives either method a chance to surface a
clause, and a cross-encoder makes the final evidence set more precise.

### Why abstain?

Insurance decisions are high-impact. A plausible answer without evidence is
worse than a clear request for missing documentation. The system therefore
uses `NEEDS_REVIEW` for unknown hospital qualification, missing medical
necessity, missing documentation, or persistent citation-validation failure.

### Why a deterministic mock mode?

It makes local testing and CI reproducible, inexpensive, and independent of
provider rate limits. The trade-off is that mock performance is not proof of
real-model performance, so production use still needs live-model evaluation and
monitoring.

## Known limitations and candid answers

- The first index build downloads embedding/reranker models and can be slow.
  Persistent storage avoids rebuilding it after every deployment.
- The evaluator currently uses expected decision labels and citation presence;
  a stronger next step is human-labelled evidence recall and claim-level
  citation entailment scoring.
- The mock rules are intentionally deterministic but do not replace formal
  policy/legal review or live-model regression testing.
- Local release verification must be rerun after repairing/recreating the
  project virtual environment, which is currently inaccessible in this
  workspace session.
- The application is an evidence-supported decision aid, not an autonomous
  insurer adjudication system.

## Questions to expect

### How do you prevent hallucinated policy claims?

The policy is the retrieval corpus, the response carries page/section/chunk
citations, and a separate validation node checks the final material statements.
A persistent validation failure becomes `NEEDS_REVIEW`.

### How would you improve retrieval quality?

I would add a small human-labelled query-to-clause benchmark, measure recall at
each retrieval stage, tune top-k/RRF settings, and add policy-aware query
rewriting only where the benchmark shows a gap.

### How does the system handle policy updates?

The index manifest contains the policy SHA-256 and chunk count. Startup detects
a changed PDF or inconsistent store and builds a staged replacement index.

### What happens if the provider or index fails?

The API returns a safe 503 rather than internal error details. `/health`
distinguishes a live process from a retrieval-ready service, and the UI tells a
reviewer to retry once readiness is restored.

### What would you do before production use?

Add authenticated access, policy/version lifecycle controls, full audit logs,
human review queues, monitoring for retrieval/citation drift, rate-limit and
latency dashboards, red-team cases, and domain/legal sign-off.

## Demo sequence

1. Open `/health` and show `ready: true`.
2. Select a straightforward public case in Streamlit.
3. Show the decision, supporting citations, and execution trace.
4. Demonstrate a `NEEDS_REVIEW` case to show abstention and missing evidence.
5. Open the evaluation report and explain the deterministic release gate.
6. Finish by explaining the policy-hash manifest and validation retry as the
   main reliability controls.
