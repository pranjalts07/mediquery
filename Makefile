# MediQuery Makefile

.PHONY: run dev test eval ingest lint fmt help

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000

dev:
	uvicorn app.main:app --reload --port 8000

test:
	pytest tests/ -v

test-safety:
	pytest tests/test_safety.py -v

test-scoring:
	pytest tests/test_scoring.py -v

eval:
	python scripts/evaluate_mediquery.py

eval-brief:
	python scripts/evaluate_mediquery.py --mode short

eval-save:
	python scripts/evaluate_mediquery.py --out eval_results.json

ingest:
	python scripts/ingest.py

ingest-pdf:
	@if [ -z "$(PDF)" ]; then \
		echo "Usage: make ingest-pdf PDF=path/to/file.pdf [OUT=output.jsonl]"; \
		exit 1; \
	fi
	python scripts/ingest_pdf.py "$(PDF)" $(if $(OUT),"$(OUT)",data/chunks.jsonl)

lint:
	@command -v ruff >/dev/null 2>&1 && ruff check app/ scripts/ tests/ || \
		echo "ruff not installed — run: pip install ruff"

fmt:
	@command -v ruff >/dev/null 2>&1 && ruff format app/ scripts/ tests/ || \
		echo "ruff not installed — run: pip install ruff"

help:
	@echo ""
	@echo "MediQuery — available targets:"
	@echo ""
	@echo "  Server"
	@echo "    make run          Start production server (port 8000)"
	@echo "    make dev          Start dev server with --reload"
	@echo ""
	@echo "  Tests"
	@echo "    make test         Run full test suite"
	@echo "    make test-safety  Run safety module tests only"
	@echo "    make test-scoring Run scoring module tests only"
	@echo ""
	@echo "  Evaluation"
	@echo "    make eval         Run eval suite against local server"
	@echo "    make eval-brief   Run eval in brief/short mode"
	@echo "    make eval-save    Run eval and save results to eval_results.json"
	@echo ""
	@echo "  Ingestion"
	@echo "    make ingest       Run full ingestion pipeline (scripts/ingest.py)"
	@echo "    make ingest-pdf PDF=file.pdf [OUT=out.jsonl]"
	@echo "                      Parse a single PDF into chunks"
	@echo ""
	@echo "  Code quality"
	@echo "    make lint         Lint with ruff"
	@echo "    make fmt          Format with ruff"
	@echo ""