.PHONY: install test lint bench run run-mock docker-build docker-up clean

PYTHON := .venv/bin/python
PYTEST := .venv/bin/pytest
RUFF := .venv/bin/ruff
MYPY := .venv/bin/mypy

install:
	@echo "Creating virtualenv and installing dependencies..."
	python3.11 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install torch --index-url https://download.pytorch.org/whl/cpu
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

test:
	@echo "Running tests with coverage requirement (>85% on app/core)..."
	$(PYTEST) tests/ -v --cov=app/core --cov-report=term-missing --cov-fail-under=85

lint:
	@echo "Running ruff check..."
	$(RUFF) check app/ tests/ benchmarks/
	@echo "Running mypy..."
	$(MYPY) app/

bench:
	@echo "Running semantic threshold sweep..."
	MPLCONFIGDIR=/tmp/matplotlib $(PYTHON) -m benchmarks.threshold_sweep
	@echo "Running latency and concurrency benchmark..."
	MPLCONFIGDIR=/tmp/matplotlib $(PYTHON) -m benchmarks.latency_bench

run:
	@echo "Starting CacheSense Gateway on http://localhost:8000..."
	$(PYTHON) -m uvicorn app.main:app --host 0.0.0.0 --port 8000

run-mock:
	@echo "Starting Mock Upstream Server on http://localhost:8080..."
	$(PYTHON) -m benchmarks.mock_upstream_server

docker-build:
	@echo "Building production Docker image..."
	docker build -t cachesense:latest .

docker-up:
	@echo "Launching CacheSense Gateway and Mock Upstream with zero configuration..."
	docker compose up --build

clean:
	rm -rf .pytest_cache .coverage data/test_* data/*.db-wal data/*.db-shm
