# Demo

Visual proof that the pipeline runs. Every screenshot below was produced by the
real app (FastAPI backend + Streamlit UI) over the bundled sample documents.

| Auto-approve | Flag for review | Reject (injection) |
|:---:|:---:|:---:|
| ![auto-approve](01-auto-approve.png) | ![flag](02-flag-review.png) | ![reject](03-reject.png) |

![landing](00-landing.png)

## One-click samples

The UI ships with **"try a bundled sample"** buttons — no upload needed. Each
maps to a document that exercises a different routing outcome:

| Button | Document | Outcome |
|--------|----------|---------|
| 🧾 Clean invoice | `data/samples/invoice_acme.txt` | ✅ **Auto-approved** — all checks pass |
| 🏷️ Unknown vendor | `data/samples/invoice_unknown_vendor.txt` | ⚠️ **Flagged** — vendor not in whitelist |
| 🛡️ Injection attempt | `data/eval/docs/invoice_injection.txt` | ❌ **Rejected** — prompt injection detected |
| 📇 Résumé | `data/samples/resume_jane.txt` | ✅ **Auto-approved** |

### Real outputs (mock-LLM mode)

```
Clean invoice        → auto_approve    "All checks passed."
Unknown vendor       → flag_for_review "Vendor 'Sketchy Holdings LLC' is not in the whitelist."
Injection attempt    → reject          "Security: Possible prompt injection (instruction override);
                                        (decision manipulation)."   findings: [injection, injection]
```

## Regenerate the screenshots

The images above are produced automatically — no manual clicking. The script
boots the app in offline mock mode, drives it with a headless browser, and
writes the PNGs:

```bash
uv sync --group capture
uv run python -m playwright install chromium
uv run python scripts/capture_demo.py
```

> 📹 Loom walkthrough (60–90s): _add link here_
> 🔍 LangSmith trace: add `04-langsmith-trace.png` with `LANGSMITH_TRACING=true`.

## Reproduce headless (no UI)

```bash
make run                                                          # → auto_approve
uv run python -m smartingest.cli data/samples/invoice_unknown_vendor.txt  # → flag_for_review
uv run python -m smartingest.cli data/eval/docs/invoice_injection.txt     # → reject
```
