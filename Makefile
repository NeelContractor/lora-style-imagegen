.PHONY: install test lint format check prepare validate train train-dry-run pre-commit

install:
	uv sync --all-groups --extra train

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

train:
	uv run train-lora --config configs/train_lora.yaml

train-dry-run:
	uv run train-lora --config configs/train_lora.yaml --dry-run

pre-commit:
	uv run pre-commit run --all-files
