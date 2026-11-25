.PHONY: install test lint format format-check check

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

lint:
	python -m ruff check src tests experiments

format:
	python -m ruff format src tests experiments

format-check:
	python -m ruff format --check src tests experiments

check: lint format-check test