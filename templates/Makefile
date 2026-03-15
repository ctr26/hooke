.PHONY: all install test lint format typecheck prepush clean

all: install test

install:
	uv sync --all-extras

test:
	uv run pytest

test-fast:
	uv run pytest -m "not slow"

lint:
	uv run ruff check .

format:
	uv run ruff format .

typecheck:
	uv run ty check src/

prepush: lint typecheck test-fast
	@echo "Ready to push"

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .ruff_cache -exec rm -rf {} +
	rm -rf .coverage htmlcov/
