.PHONY: install test lint format check prepare validate pre-commit

install:
	uv sync --all-groups

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

check: lint test

prepare:
	uv run prepare-dataset

validate:
	uv run prepare-dataset --validate-only

pre-commit:
	uv run pre-commit run --all-files
