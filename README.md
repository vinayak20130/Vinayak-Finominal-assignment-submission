# Portfolio Optimizer API

A FastAPI service that optimizes portfolio weights using market data stored in
PostgreSQL. Requests name securities by ticker; the service reads their daily
returns, names, dividend yields and factor returns from the database.

## Quick start (Docker required)

```bash
docker compose up --build
```

This starts three services:

- `db`: PostgreSQL 17;
- `migrate`: creates the schema, then loads `Data.xlsx`, and exits;
- `api`: starts once `migrate` has succeeded.

Once it's running:

- Interactive docs, with ready-made examples: http://127.0.0.1:8000/docs
- Health check: `curl http://127.0.0.1:8000/health`

```bash
curl http://127.0.0.1:8000/optimize -H 'Content-Type: application/json' \
  -d '{"securities":[{"ticker":"IEFA","current_weight":25},{"ticker":"SPY","current_weight":75}],"optimization_strategy":"equal_weights"}'
```

Stop with `docker compose down`; add `-v` to delete the database volume. The API is
published on port 8000 and PostgreSQL on 5433; `API_PORT` and `DB_PORT` override
them if those ports are taken.

## Development

```bash
uv sync --locked
docker compose up -d db
uv run --locked alembic upgrade head
uv run --locked python -m app.data.load
uv run --locked uvicorn app.main:app --reload
```

`DATABASE_URL` defaults to
`postgresql+psycopg://portfolio:portfolio@localhost:5433/portfolio` (see
`.env.example`). The API refuses to start if the database is unreachable, missing
tables, or empty, and prints the command that fixes it.

Checks (no Docker needed; tests use an in-memory fake of the data source):

```bash
uv run --locked pytest -q
uv run --locked ruff check .
uv run --locked ruff format --check .
```

## Data

`python -m app.data.load [path]` validates the whole workbook first. It then
replaces the data in one transaction, so readers never see a partial load. The
workbook is the complete universe: securities missing from it are removed.
Concurrent loads queue on an advisory lock, and every load is recorded in
`data_loads` with the file's SHA-256 hash.

| Object | Contents |
|---|---|
| `securities` | ticker, name, dividend yield (NULL when blank in the source, e.g. GLD) |
| `fund_daily_returns` | daily total returns per ticker (the source of truth) |
| `factor_daily_returns` | daily Momentum, Value and Size factor returns |
| `data_loads` | audit trail of loads |
| view `fund_annual_returns` | calendar-year returns compounded from the daily rows; partial years flagged |
| view `security_summary` | history range, since-inception and year-to-date returns |

The views compute on read (a few milliseconds at this size), so they can never be
stale. The optimizer uses **daily** returns, not yearly ones: yearly data would hide
intra-year losses and change the results. For example, SPY returned +18.4% in 2020
but fell 33.7% within the year. Computing case 3 from yearly returns moves its weights
by 5.1 percentage points.

## API

| Endpoint | Purpose |
|---|---|
| `POST /optimize` | Optimize a portfolio (below) |
| `GET /securities` | Available tickers with name, yield, history range, since-inception and YTD returns |
| `GET /securities/{ticker}/annual-returns` | Calendar-year returns for one fund |
| `GET /health` | Liveness |

Each security in the request has `ticker`, `current_weight`, and optional
`min_weight`/`max_weight`. Weights are percentages (20 means 20%); constraints are
decimals (0.025 means 2.5%).

- **Strategies:** `equal_weights`, `risk_parity`, `minimize_drawdown`,
  `minimize_volatility`, `maximize_sharpe_ratio`, `optimize_factor_exposure`. The last
  one needs `factor_objective`, e.g.
  `{"direction": "maximize", "coefficients": {"momentum": 1}}`.
- **`constraints`:** `min_cagr`, `min_volatility`, `max_volatility`, `max_drawdown`,
  `min_dividend_yield`.
- **`settings`:** `annualization_factor` (default 252), `risk_free_rate` (annual,
  default 0), `start_date`/`end_date`, and `missing_dividend_yield`. Its default is
  `zero`, which treats a blank yield as 0% and says so in `warnings`; `error` rejects
  yield constraints on such funds instead.

The response contains `allocation_changes` (ticker, security name, current weight,
optimized weight, change), the analysis window, methodology, solver status, current
and optimized metrics, constraint residuals, warnings, and factor betas (Momentum,
Value, Size) for both portfolios.

Errors use `{"error": {"code": "...", "message": "...", "details": []}}`:

| Status | Codes |
|---|---|
| 404 | `ticker_not_found` (the message lists the available tickers), `not_found` |
| 405 | `method_not_allowed` |
| 422 | `invalid_input`, `insufficient_data`, `infeasible_constraints`, `equal_weight_conflict` |
| 500 | `optimization_failed` (not proven infeasible), `internal_error` |
| 503 | `data_unavailable` (the database can't be reached) |

## Assignment scenarios

`examples/` holds the six assignment cases plus a Minimize Drawdown demo. The API's
responses are saved in `validation/responses/`; regenerate them with the server
running:

```bash
uv run --locked python -m scripts.run_scenarios
```

## Methodology

- Dates are intersected for only the selected securities; missing returns are never
  filled.
- Fixed weights are rebalanced each observation; annualization is 252 days; volatility
  and covariance use sample estimates (`ddof=1`).
- CAGR uses the observation count divided by the annualization factor as elapsed years.
  Drawdown includes the initial wealth. Sharpe uses arithmetic mean excess returns, with
  the annual risk-free rate converted geometrically to a daily rate.
- Risk parity equalizes fractional variance contributions. Min drawdown, min
  volatility and max Sharpe use deterministic multistart SLSQP with analytic gradients
  where available. Factor exposure is solved exactly as a linear program (HiGHS)
  because portfolio betas are weighted fund betas.
- Every result is re-verified independently: weights sum to 100%, no negatives,
  bounds and portfolio constraints. Failing to find a solution (500) is kept distinct
  from proven infeasibility (422).

## Deferred

- Annual rebalancing, which is the live tool's default. The current results assume
  daily rebalancing.
- A database-level test suite (the PostgreSQL layer is verified live).
- Caching, authentication, rate limiting.

These results have not yet been compared with the live Finominal tool.
