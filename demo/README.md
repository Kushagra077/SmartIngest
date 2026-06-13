# Demo

This folder holds visual proof that the pipeline runs.

## Suggested captures

1. **Streamlit run** — upload `data/samples/invoice_acme.txt`, show the
   progress bar stepping through Classifier → Extractor → Validator → Router,
   then the green **Auto-approved** banner with extracted fields.
2. **Flag-for-review** — upload `data/samples/invoice_unknown_vendor.txt`; the
   non-whitelisted vendor + large amount produces a yellow **Flagged for
   review** result.
3. **LangSmith trace** — a screenshot of the per-node trace for one job.
4. **Loom walkthrough** — a 60–90s screen recording. _Add the link below._

> 📹 Loom: _add link here_

## Reproduce locally

```bash
make api    # terminal 1
make ui     # terminal 2 → http://localhost:8501
```

Or headless, via the CLI:

```bash
make run    # data/samples/invoice_acme.txt → auto_approve
uv run python -m smartingest.cli data/samples/invoice_unknown_vendor.txt   # → flag_for_review
```

Save screenshots into this folder as `01-auto-approve.png`, `02-flag-review.png`,
`03-langsmith-trace.png` and they'll render in the project write-up.
