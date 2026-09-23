# Portfolio Optimizer API

Python REST API for the Finominal backend assignment.

## Current status

Basic project initialization only: FastAPI application, health endpoint, interactive
API documentation, locked dependencies, and a health endpoint test.
Optimization strategies and workbook processing are not implemented yet.
Reference-tool comparisons have not been performed.

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/), then run
these commands from the repository root:

```bash
uv python install 3.12
uv sync --locked
```

Python 3.12 is required. uv creates the local `.venv`; no global pip installation
is needed. Both application and development dependencies are included.

## Run

```bash
uv run --locked uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

- Interactive documentation: http://127.0.0.1:8000/docs
- OpenAPI schema: http://127.0.0.1:8000/openapi.json
- Liveness check: http://127.0.0.1:8000/health

```bash
curl http://127.0.0.1:8000/health
```

Expected response: `{"status":"ok"}`. This checks the running API only.

## Checks

```bash
uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked ruff format --check .
```

## Data

The supplied assignment document and `Data.xlsx` are kept locally and are not
included in the repository. Neither file is needed to run the health endpoint.
The planned `POST /optimize` endpoint is not available in this scaffold yet.
