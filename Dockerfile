# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------
# SmartIngest image — builds the package with uv into a self-contained venv.
# The same image runs either service; the command picks which (see compose):
#   API : uvicorn smartingest.api:app ...
#   UI  : streamlit run src/frontend/streamlit_app.py ...
# ---------------------------------------------------------------------------
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app

# Reproducible, no-bytecode-surprises, copy (not symlink) the managed Python.
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PYTHONUNBUFFERED=1

# 1) Install dependencies first as a cached layer (changes rarely).
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# 2) Copy the source and install the project itself.
COPY . .
RUN uv sync --frozen --no-dev

# Put the venv on PATH so `uvicorn` / `streamlit` resolve directly.
ENV PATH="/app/.venv/bin:$PATH"

# Run as a non-root user. The data dir is pre-created and owned by that user so
# a fresh named volume mounted there inherits writable ownership.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/data \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000 8501

# Default to the API; docker-compose / the platform overrides for the UI.
CMD ["uvicorn", "smartingest.api:app", "--app-dir", "src", "--host", "0.0.0.0", "--port", "8000"]
