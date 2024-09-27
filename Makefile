.PHONY: all init format lint build run_backend dev help tests coverage clean_python_cache clean_all

# Configurations
POETRY := poetry
RUFF := $(POETRY) run ruff
MYPY := $(POETRY) run mypy
PRE_COMMIT := $(POETRY) run pre-commit

# Targets

all: help

init: clean_python_cache
	$(POETRY) install

format:
	$(RUFF) format .

lint:
	$(RUFF) check .
	$(MYPY) .

lint-fix:
	$(RUFF) check --fix .

build:
	$(POETRY) build

install_backend:
	$(POETRY) install

run_backend:
	$(POETRY) run streamlit run src/quoptuna/backend/app.py

dev:
	# Add dev environment setup commands here

help:
	@echo "Available targets:"
	@echo "  init              - Initialize the project"
	@echo "  format            - Format the code"
	@echo "  lint              - Run linters"
	@echo "  lint-fix          - Run linters and fix issues"
	@echo "  build             - Build the project"
	@echo "  install_backend   - Install backend dependencies"
	@echo "  run_backend       - Run the backend"
	@echo "  dev               - Set up development environment"
	@echo "  tests             - Run tests"
	@echo "  coverage          - Run tests with coverage"
	@echo "  clean_python_cache - Clean Python cache"
	@echo "  clean_all         - Clean all caches"

tests:
	$(POETRY) run pytest

coverage:
	$(POETRY) run coverage run -m pytest
	$(POETRY) run coverage report

clean_python_cache:
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type f -name '*.pyc' -exec rm -f {} +

clean_all: clean_python_cache

# Pre-commit hooks
pre-commit:
	$(PRE_COMMIT) run --all-files