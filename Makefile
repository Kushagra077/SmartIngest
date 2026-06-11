.PHONY: install test api ui run eval lint clean

VENV ?= .venv
PY := $(VENV)/bin/python
PIP := $(VENV)/bin/pip

install:  ## Create venv and install dependencies
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt

test:  ## Run the test suite
	$(PY) -m pytest -q

api:  ## Start the FastAPI backend (http://localhost:8000)
	$(VENV)/bin/uvicorn smartingest.api:app --app-dir src --reload --port 8000

ui:  ## Start the Streamlit frontend (http://localhost:8501)
	PYTHONPATH=src $(VENV)/bin/streamlit run src/frontend/streamlit_app.py

run:  ## Run one sample document through the pipeline (CLI)
	PYTHONPATH=src $(PY) -m smartingest.cli data/samples/invoice_acme.txt

eval:  ## Run the evaluation harness over the golden dataset
	PYTHONPATH=src $(PY) -m smartingest.eval.runner

clean:  ## Remove caches and local data
	rm -rf .pytest_cache **/__pycache__ data/jobs.db data/uploads
