.PHONY: all init format lint build run_backend run_frontend run_cli dev help tests coverage clean_python_cache clean_all

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
	@echo "Installing main quoptuna package..."
	$(UV) pip install -e .
	@echo "Installing backend dependencies..."
	cd backend && $(UV) pip install -e .
	@echo "Backend dependencies installed!"

install_frontend:
	@echo "Installing frontend dependencies..."
	cd frontend && npm install
	@echo "Frontend dependencies installed!"

run_backend:
	@echo "Starting FastAPI backend on http://localhost:8000..."
	@echo "API docs: http://localhost:8000/docs"
	cd backend && $(UV) run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

run_frontend:
	@echo "Starting Vite frontend on http://localhost:5173..."
	cd frontend && npm run dev

run_cli:
	@echo "Starting QuOptuna full-stack application..."
	@echo ""
	@echo "Backend: http://localhost:8000"
	@echo "Frontend: http://localhost:5173"
	@echo "API Docs: http://localhost:8000/docs"
	@echo ""
	@echo "Press Ctrl+C to stop both services"
	@echo ""
	@trap 'kill 0' EXIT; \
	$(MAKE) run_backend & \
	$(MAKE) run_frontend & \
	wait

run_streamlit:
	@echo "Starting legacy Streamlit interface..."
	$(UV) run streamlit run src/quoptuna/frontend/app.py

dev:
	# Add dev environment setup commands here

help:
	@echo "QuOptuna Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  init              - Initialize the project"
	@echo "  install_backend   - Install backend dependencies"
	@echo "  install_frontend  - Install frontend dependencies"
	@echo ""
	@echo "Running:"
	@echo "  run_backend       - Run FastAPI backend (port 8000)"
	@echo "  run_frontend      - Run Vite frontend (port 5173)"
	@echo "  run_cli           - Run both backend and frontend"
	@echo "  run_streamlit     - Run legacy Streamlit interface"
	@echo ""
	@echo "Code Quality:"
	@echo "  format            - Format the code"
	@echo "  lint              - Run linters"
	@echo "  lint-fix          - Run linters and fix issues"
	@echo "  tests             - Run tests"
	@echo "  coverage          - Run tests with coverage"
	@echo ""
	@echo "Build & Clean:"
	@echo "  build             - Build the project"
	@echo "  clean_python_cache - Clean Python cache"
	@echo "  clean_all         - Clean all caches"
	@echo ""
	@echo "Docker:"
	@echo "  docker compose up --build - Start with Docker"

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
