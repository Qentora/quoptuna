.PHONY: all init format lint build run_backend dev help tests coverage clean_python_cache clean_all

# Configurations
UV := uv
RUFF := $(UV) run ruff
MYPY := $(UV) run mypy
PRE_COMMIT := $(UV) run pre-commit

# Targets

all: help

init: clean_python_cache
	$(UV) install

format:
	$(RUFF) format .

lint:
	$(RUFF) check .
	$(MYPY) .

lint-fix:
	$(RUFF) check --fix .

build:
	$(UV) build

install_backend:
	$(UV) install

run_backend:
	$(UV) run streamlit run src/quoptuna/frontend/app.py

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
	$(UV) run pytest

coverage:
	$(UV) run coverage run -m pytest
	$(UV) run coverage report

clean_python_cache:
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type f -name '*.pyc' -exec rm -f {} +

clean_all: clean_python_cache

# Pre-commit hooks
pre-commit:
	$(PRE_COMMIT) run --all-files
