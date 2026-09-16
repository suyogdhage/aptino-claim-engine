# Aptino Policy-Aware Multi-Agent RAG Claim Decision Engine

Analyzes health-insurance claim cases against a supplied policy document using hybrid retrieval and a LangGraph multi-agent workflow.

Deploy as a **single Streamlit app** (engine runs in-process — no separate API service, tunnel, or always-on server needed).

## Architecture

```
Claim (JSON)
   │
   ▼
┌─────────────────────┐
│  Case Analysis      │  Extract facts, identify decision dimensions,
│  Agent              │  detect missing fields, build investigation plan
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Policy Evidence    │  Hybrid retrieval (dense+sBM25) per dimension,
│  Agent              │  RRF fusion, cross-encoder rerank -> ranked evidence
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Coverage &         │  Assess waiting periods, exclusions, definitions,
│  Exclusion Agent    │  category limits vs retrieved policy evidence
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Decision Agent     │  Combine findings -> status + confidence + limits
└─────────┬───────────┘
          ▼
┌─────────────────────┐
│  Validation Agent   │  Verify material claims cite supportive evidence
└─────────┬───────────┘
          │  FAIL (retry decision up to 2x)
          ▼  PASS
┌─────────────────────┐
│  Final Structured   │  Decision + findings + citations + limits + trace
│  Response           │
└─────────────────────┘
```

**Stack:** Streamlit (UI + in-process engine) · FastAPI (optional API) · LangGraph (orchestration) · Chroma (dense vectors) · BM25 (sparse) · fastembed + ONNX cross-encoder reranker (no PyTorch) · Groq (LLM).

Without a `GROQ_API_KEY`, the system automatically uses a rule-based mock LLM so the full pipeline and evaluation run offline. With a key, it uses Groq `openai/gpt-oss-20b` (set via `GROQ_MODEL`; the smaller `-20b` model stays under the free-tier token-per-minute cap, which the `-120b` model exceeds on typical prompts).

## Setup

```bash
cd aptino-claim-engine
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.lock
cp .env.example .env   # add your GROQ_API_KEY
python -m app.ingest data/policy.pdf chroma_db   # build the index
```

For a deterministic local run that never calls Groq, set
`FORCE_MOCK_LLM=true` in `.env` (or as an environment variable). This is the
recommended setting for tests and evaluation.

## Run

### All-in-one app (recommended)

```bash
streamlit run streamlit_app.py
```

Runs the full multi-agent engine inside Streamlit — pick a public test case,
upload JSON, or paste JSON, then analyze. The first run builds the retrieval
index if it is missing (`chroma_db/` is not kept in git).

### Two-process mode

Backend:
```bash
uvicorn app.main:app --reload --port 8000
```

Frontend (calls the backend):
```bash
streamlit run app/frontend.py --server.port 8501
```

Health check: `GET http://localhost:8000/health`. A response with `ready: true`
means both the Chroma and BM25 indexes match the current policy PDF. A
`degraded` response is safe to retry, but `/analyze` will return 503 until the
index is ready.

## API

### `POST /analyze`
Body: one claim case JSON (see `data/public_test_cases.json` for schema).

Example response:
```json
{
  "case_id": "PUB-001",
  "decision": "ADMISSIBLE_WITH_LIMITS",
  "confidence": 0.87,
  "key_findings": [...],
  "applicable_limits": [...],
  "missing_evidence": [],
  "citations": [
    {"claim": "...", "source": "policy.pdf", "page": 7, "section": "Scope of Cover", "chunk_id": "..."}
  ],
  "validation": {"status": "PASS", "unsupported_claims": []},
  "trace": [{"agent": "CaseAnalysis", "action": "...", "duration_ms": 120, "evidence_count": 0}]
}
```

### `GET /health`
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "ready": true,
  "detail": "Policy index and both retrieval stores are ready"
}
```

## Evaluation

```bash
python -m app.evaluation.evaluate
```

Runs all public + custom cases, compares against `app/evaluation/expected_outcomes.json`, and writes `evaluation_results/evaluation_report.json` with decision accuracy, abstention, retrieval recall@k (against `app/evaluation/gold_evidence.json` page references), and citation precision.

For the same gate used by CI:

```powershell
$env:FORCE_MOCK_LLM='true'
python -m app.evaluation.evaluate
python -m app.evaluation.check_release
```

### Evaluation results (deterministic mock mode, 18 cases)

| Metric | Value |
| --- | --- |
| Accuracy | 1.0 (12/12 public, 6/6 custom) |
| Abstention rate | 0.167 (3 needed review, all correct) |
| Citation coverage | 1.0 |
| Citation precision | 0.296 |
| Retrieval recall@1 | 0.25 |
| Retrieval recall@2 | 0.528 |
| Retrieval recall@4 | 0.611 |
| Retrieval recall@8 | 0.889 |

The release gate (`app/evaluation/check_release.py`) is mode-aware: the
deterministic mock run must score 1.0 accuracy with full citation coverage and
`retrieval_recall_at_8 >= 0.8`, while a real Groq run must reach an 0.85 accuracy
floor with recall@8 >= 0.8. Refreshing the real-Groq row of this table requires
a run with `FORCE_MOCK_LLM` unset, which is blocked while the Groq token
rate-limit quota is exhausted.

## Error responses

Malformed request bodies receive HTTP 422 with field-level validation details.
If the retrieval index or a model provider is unavailable, `/analyze` returns
HTTP 503 with a safe retry message; provider internals are logged server-side.

## Decision Statuses

- `ADMISSIBLE` — fully covered, no material deduction.
- `ADMISSIBLE_WITH_LIMITS` — covered but capped/deducted per policy limits.
- `PARTIALLY_ADMISSIBLE` — only part of the claim is payable.
- `NOT_ADMISSIBLE` — policy supports rejection/exclusion.
- `NEEDS_REVIEW` — insufficient evidence; system abstains.

## Deployment (Streamlit Community Cloud)

1. Push this repo to GitHub.
2. Go to **https://share.streamlit.io** (or streamlit.io) → **Create app**.
3. Select the repo, branch `master`, and main file **`streamlit_app.py`** → **Deploy**.
4. Add secrets under **Settings → Secrets**, then **Rerun**:
   ```
   GROQ_API_KEY = <your groq key>
   GROQ_MODEL = openai/gpt-oss-20b
   ```

The app builds the retrieval index on first load (downloads the ~67MB embedding
model into its ephemeral storage, then ingests `data/policy.pdf` — allow a few
minutes for the first request). A cold restart may rebuild the index.

The free Streamlit Cloud plan is ~1GB RAM, which fits the fastembed + onnxruntime
stack (PyTorch does **not**); the previous Render **free** (512MB) deployment
out-of-memory'd during cold start, which is why the engine no longer ships PyTorch
or runs as a separate web service.

## Known Limitations / Trade-offs

- LLM adds latency (~10-30s per case in-process, ~60-120s end-to-end); a smaller local model could cut cost at the price of reasoning quality.
- Streamlit Cloud free-tier storage is ephemeral; the retrieval index may be rebuilt on a cold restart.
- Expected outcomes cover 18 cases (12 public + 6 custom) including one `PARTIALLY_ADMISSIBLE` scenario; they encode the intended decision each case is designed to test.
- Reranker model download at first startup requires internet.
- Abstention is preferred over guessing: cases with unconfirmed hospital registration or missing documents return `NEEDS_REVIEW`.
- Real-Groq evaluation is subject to free-tier rate limits (8k tokens/min per request, 200k tokens/day); `openai/gpt-oss-120b` can exceed the per-request cap, so `openai/gpt-oss-20b` is the default. The deterministic mock mode covers the full pipeline offline.

The concise submission architecture note is [docs/ARCHITECTURE_NOTE.md](docs/ARCHITECTURE_NOTE.md).
The policy-page rationale for every expected evaluation outcome is in
[docs/CASE_RATIONALES.md](docs/CASE_RATIONALES.md).

## Evaluation and failure analysis

Run `python -m app.evaluation.evaluate` to generate the case-level report.
See [docs/EVALUATION.md](docs/EVALUATION.md) for deterministic-mode instructions,
the release gate, and documented reliability failures and fixes.
