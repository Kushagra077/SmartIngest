.PHONY: install test api ui run eval lock export clean up down

install:  ## Create the uv-managed venv and install all dependencies (incl. dev)
	uv sync

lock:  ## Re-resolve and update uv.lock
	uv lock

export:  ## Regenerate requirements.txt from the lockfile (pip fallback)
	uv export --format requirements-txt --no-emit-project --no-hashes -o requirements.txt

test:  ## Run the test suite
	uv run pytest -q

api:  ## Start the FastAPI backend (http://localhost:8000)
	uv run uvicorn smartingest.api:app --app-dir src --reload --port 8000

ui:  ## Start the Streamlit frontend (http://localhost:8501)
	uv run streamlit run src/frontend/streamlit_app.py

run:  ## Run one sample document through the pipeline (CLI)
	uv run python -m smartingest.cli data/samples/invoice_acme.txt

eval:  ## Run the evaluation harness over the golden dataset
	uv run python -m smartingest.eval.runner

up:  ## Build and run both services (API + UI) via Docker Compose
	docker compose up --build

down:  ## Stop and remove the Docker Compose stack
	docker compose down

clean:  ## Remove caches and local data
	rm -rf .pytest_cache **/__pycache__ data/jobs.db data/uploads
