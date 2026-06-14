# Deploying SmartIngest

SmartIngest runs as **two processes** — the FastAPI backend and the Streamlit
UI that talks to it. This guide covers running both with Docker locally and
hosting them on a PaaS. A `Dockerfile` and `docker-compose.yml` are included; no
host is wired up for you, so you stay in control of accounts and secrets.

---

## 1. Local — Docker Compose (recommended first run)

```bash
cp .env.example .env        # then edit: add GEMINI_API_KEY, or keep mock mode
docker compose up --build
```

- UI  → http://localhost:8501
- API → http://localhost:8000  (OpenAPI docs at `/docs`)

The `ui` service waits for the `api` healthcheck and reaches it by service name
(`SMARTINGEST_API_URL=http://api:8000`). Uploaded files and the SQLite job DB
persist to `./data` via a volume.

To run the public-facing demo against real Gemini, set in `.env`:

```ini
SMARTINGEST_MOCK_LLM=false
GEMINI_API_KEY=your-key
GEMINI_MODEL=gemini-2.0-flash
GEMINI_MODEL_FALLBACKS=gemini-1.5-flash,gemini-1.5-flash-8b
```

---

## 2. Required configuration / secrets

| Variable | Required? | Notes |
|----------|-----------|-------|
| `GEMINI_API_KEY` | Real-LLM mode only | Set as a **secret**, never commit it. |
| `SMARTINGEST_MOCK_LLM` | — | `true` runs offline (no key, no cost). |
| `GEMINI_MODEL` / `GEMINI_MODEL_FALLBACKS` | — | Primary + ordered backups; auto-failover on quota. |
| `SMARTINGEST_RATE_LIMIT_PER_MINUTE` | — | Per-IP cap (default 10). |
| `SMARTINGEST_RATE_LIMIT_PER_DAY` | — | **Global** daily cap that protects the free tier (default 200). |
| `SMARTINGEST_API_URL` | UI only | Public URL of the API service (see below). |

> **Spend guard:** on a public demo, keep `SMARTINGEST_RATE_LIMIT_PER_DAY`
> conservative. It caps total uploads/day across *all* visitors, so the Gemini
> quota can't be drained by the link being shared.

---

## 3. Render / Railway (two services)

Both platforms can build the included `Dockerfile`. Create **two services** from
the same repo:

**Backend (`smartingest-api`)**
- Build: Dockerfile
- Start command: `uvicorn smartingest.api:app --app-dir src --host 0.0.0.0 --port $PORT`
- Secrets/env: `GEMINI_API_KEY`, `SMARTINGEST_MOCK_LLM=false`, rate-limit vars
- Note its public URL, e.g. `https://smartingest-api.onrender.com`

**Frontend (`smartingest-ui`)**
- Build: Dockerfile
- Start command: `streamlit run src/frontend/streamlit_app.py --server.port $PORT --server.address 0.0.0.0`
- Env: `SMARTINGEST_API_URL=https://smartingest-api.onrender.com` (the backend URL above)

The frontend is just a client; only the **backend** needs the Gemini key.

---

## 4. Streamlit Community Cloud (frontend only)

Streamlit Cloud runs a single Python app, so it can host **only the UI**. Deploy
the API elsewhere (section 3), then on Streamlit Cloud:

- App file: `src/frontend/streamlit_app.py`
- Secret: `SMARTINGEST_API_URL = "https://<your-api-host>"`

`requirements.txt` installs the `smartingest` package itself (its last line is
`.`), so the UI's `from smartingest...` imports resolve in Streamlit Cloud's
pip-based environment.

(If you later want a single free deploy with no separate backend, the UI can be
refactored to call the pipeline in-process — ask and I'll wire it up.)

---

## 5. Pre-deploy checklist

```bash
make test                                        # 72 tests, all offline
uv run python -m smartingest.eval.runner --mock  # eval gate, no API spend
docker compose up --build                        # smoke-test both services
```

Then confirm: `GEMINI_API_KEY` is set as a secret (not in the image), the daily
rate cap is sane, and the UI's `SMARTINGEST_API_URL` points at the deployed API.
