.PHONY: install ingest serve eval lint test clean

# ── Environment ───────────────────────────────────────────────────────────────
install:
	pip install -e ".[dev]"

# ── Core targets ──────────────────────────────────────────────────────────────
ingest:
	@echo "▶  Running ingestion pipeline..."
	python -m ingestion.pipeline

serve:
	@echo "▶  Starting API server on http://0.0.0.0:8000"
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

eval:
	@echo "▶  Running RAG evaluation suite..."
	python -m eval.run

# ── Quality ───────────────────────────────────────────────────────────────────
lint:
	ruff check . --fix
	ruff format .

test:
	pytest -v

# ── Housekeeping ──────────────────────────────────────────────────────────────
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null; true
	find . -name "*.pyc" -delete 2>/dev/null; true
	rm -rf .pytest_cache dist build *.egg-info
