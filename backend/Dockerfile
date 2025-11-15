FROM python:3.12-slim

WORKDIR /app

# Install uv for fast package management
RUN pip install uv

# First, install the main quoptuna package
COPY pyproject.toml setup.py* /quoptuna/
COPY src /quoptuna/src/
RUN uv pip install --system -e /quoptuna

# Then install the backend package dependencies
COPY backend/pyproject.toml ./
RUN uv pip install --system -e .

# Copy backend app code
COPY backend/ .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
