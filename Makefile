PYTHON ?= uv run python

.PHONY: sync test lint typecheck serve

sync:
	uv sync --dev

test:
	uv run pytest -v

lint:
	uv run ruff check .

typecheck:
	uv run mypy src

serve:
	uv run market-lab serve
