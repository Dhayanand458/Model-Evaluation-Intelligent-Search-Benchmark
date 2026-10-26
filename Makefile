REPO ?= microsoft/vscode
MAX  ?= 3000

.PHONY: setup ingest stats embed eval serve smoke test lint

setup:
	uv sync

ingest:
	uv run python -m triage_mcp.ingest --repo $(REPO) --max $(MAX)

stats:
	uv run python -m triage_mcp.ingest stats

embed:
	uv run python -m triage_mcp.store

eval:
	uv run python -m triage_mcp.evals

serve:
	uv run python -m triage_mcp.server

smoke:
	uv run python scripts/smoke_mcp.py

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .
