# SmartIngest — Architecture & Technical Decisions

This document explains *why* SmartIngest is built the way it is. The headline
flow (Client → FastAPI → job store → LangGraph → routing sink) is in the
[README](README.md); here we go a level deeper.

---

## 1. The pipeline as a typed state graph

The core is a [LangGraph](https://langchain-ai.github.io/langgraph/)
`StateGraph` with four nodes operating on a single typed `AgentState`
(`src/smartingest/state.py`). Each node reads some keys and writes others:

| Node        | Reads                                  | Writes                                            |
|-------------|----------------------------------------|---------------------------------------------------|
| Classifier  | `file_path`, `mime_type`               | `document_type`, `classification_confidence`      |
| Extractor   | `file_path`, `document_type`           | `fields`, `extraction_confidence`, `retries`      |
| Validator   | `fields`, `document_type`              | `validation_issues`                               |
| Router      | `document_type`, `fields`, `validation_issues` | `route`, `route_reason`                   |

Using a `TypedDict` for the state (rather than a free-form dict) keeps every
node's contract explicit and gives type-checker support across the graph.

### Edges

```
classifier → extractor → validator ─┬─(low confidence)→ extractor   (retry loop)
                                     └─(ok)────────────→ router → END
```

The `validator → {extractor | router}` edge is a **conditional edge**
(`needs_retry` in `agents/validator.py`). This is the single most important
structural decision in the graph and is covered in §3.

---

## 2. Agents: which use an LLM, and which don't

A deliberate split:

- **Classifier & Extractor are LLM (Gemini, multimodal) calls.** Understanding
  an arbitrary PDF/image is exactly what a vision model is good at, and exactly
  what deterministic code is bad at.
- **Validator & Router are pure, deterministic Python.** Business rules
  ("is the total correct?", "is this vendor approved?", "what action do we
  take?") must be **reproducible, auditable and explainable**. Encoding them in
  a prompt would make routing non-deterministic and impossible to unit-test.
  Instead they read typed rules from `config/rules.yaml`.

This separation is also what makes the whole project testable offline (see §6).

---

## 3. The Validator → Extractor retry loop

Low-confidence extractions are common with messy scans. Rather than passing bad
data downstream, the Validator's conditional edge sends control *back* to the
Extractor:

```python
def needs_retry(state, settings) -> str:
    if (state["extraction_confidence"] < settings.min_confidence
            and state["retries"] < settings.max_retries):
        return "extractor"   # loop
    return "router"          # proceed
```

Two guards keep this safe and observable:

1. A **confidence floor** (`SMARTINGEST_MIN_CONFIDENCE`) decides *whether* to retry.
2. A **retry budget** (`SMARTINGEST_MAX_RETRIES`) bounds the loop — the
   Extractor increments `retries` on every pass, so the loop is guaranteed to
   terminate even if confidence never improves.

---

## 4. Async decoupling: upload now, process later

Document inference takes seconds; HTTP clients shouldn't block on it. So:

1. `POST /upload` writes the file to disk, inserts a `QUEUED` job row, submits
   the job to a background worker, and returns a `job_id` **immediately**.
2. A `PipelineWorker` (thread pool, `worker.py`) runs the graph and persists the
   result, moving the job `QUEUED → RUNNING → COMPLETED/FAILED`.
3. The client polls `GET /status/{job_id}` and then reads `GET /results/{job_id}`.

### Why SQLite for the job store?

The architecture calls for "Redis / SQLite". SQLite is the default because it
makes the project **clone-and-run with zero infrastructure**. The `JobStore`
(`store.py`) exposes a small, swappable interface (`create`, `set_status`,
`save_result`, `get_status`, `get_result`) — backing it with Redis is a
drop-in replacement that requires no changes to the API or worker.

The thread pool plays the same role: it's the simplest thing that decouples
processing from the request, with the same submit-and-poll semantics a
Celery/RQ deployment would have.

---

## 5. Configuration & business rules

All tunables are typed `pydantic-settings` (`config.py`), sourced from the
environment / `.env`. Business rules live separately in `config/rules.yaml` so a
**non-engineer can change routing behaviour** (thresholds, vendor whitelist,
required fields per document type) without a code change or redeploy. The
`Rules` accessor (`rules.py`) wraps the YAML and is injected into the Validator
and Router, which keeps those agents pure and easy to test.

---

## 6. Mock-LLM mode (offline-first)

`use_mock_llm` is on whenever `SMARTINGEST_MOCK_LLM=true` *or* no
`GEMINI_API_KEY` is present. In that mode `get_llm_client` returns a
`MockLLMClient` that classifies and extracts from text documents using
keyword/regex heuristics.

Why this matters:

- The pipeline **never hard-fails for lack of credentials.**
- The full test-suite (28 tests, including the end-to-end graph and API flows)
  runs in CI with **no API key and no network.**
- Reviewers can clone, `make install && make test && make run` and see real
  output in seconds.

Swapping to genuine Gemini extraction is a single env-var flip; the
`GeminiClient` and `MockLLMClient` share the same `LLMClient` protocol, so no
calling code changes.

---

## 7. Observability with LangSmith

When `LANGSMITH_TRACING=true`, `tracing.configure_tracing` sets the standard
`LANGSMITH_*` environment variables before the graph is compiled. LangGraph then
emits a trace per node run, so each Classifier / Extractor / Validator / Router
step — including retry-loop iterations — shows up as a discrete, inspectable
span in LangSmith. This is the hook for production monitoring and for building
evaluation sets over real traffic.

---

## 8. Data contracts (Pydantic v2)

Every boundary is a Pydantic model (`models.py`):

- `ExtractedFields` is a typed superset across document types — the Extractor
  fills only the relevant subset.
- `PipelineResult` is the single object persisted to the store and returned by
  `/results`, so the API response shape is guaranteed to match what the graph
  produced.
- Enums (`DocumentType`, `JobStatus`, `RouteDecision`) keep the vocabulary
  closed and serialisation stable.

---

## 9. Trade-offs & next steps

| Area            | Current                          | Production upgrade path                          |
|-----------------|----------------------------------|--------------------------------------------------|
| Job store       | SQLite + thread pool             | Redis + Celery/RQ; horizontal worker scaling     |
| File storage    | Local disk                       | S3/GCS with signed URLs                          |
| PDF handling    | Passed straight to Gemini        | Add OCR/page-splitting pre-step for huge docs    |
| Auth            | None                             | API keys / OAuth on the FastAPI layer            |
| Rules           | Static YAML                      | DB-backed rules with an admin UI                 |
| Frontend        | Streamlit (fast to ship)         | The React UI noted in the architecture diagram   |
