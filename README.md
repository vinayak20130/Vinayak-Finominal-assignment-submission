# Portfolio Optimizer API

Python REST API for the Finominal backend assignment.

## Current status

Implemented: workbook validation, date alignment, core portfolio metrics, and
`POST /optimize` with all five required strategies (`equal_weights`,
`risk_parity`, `minimize_drawdown`, `minimize_volatility`, and
`maximize_sharpe_ratio`) plus the optional `optimize_factor_exposure` strategy
with Momentum, Value, and Size factor betas.
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
The tests use synthetic data and do not need the workbook.

## Optimize

With the server running:

```bash
curl --fail-with-body http://127.0.0.1:8000/optimize \
  -H 'Content-Type: application/json' \
  --data-binary @examples/equal-weights.json
```

This example returns 50% for each asset. Tickers are identifiers for supplied
data; no external ticker lookup or database is used. Switch
`optimization_strategy` to `risk_parity` to target equal risk contributions.

Assignment scenarios are committed as complete requests in `examples/`
(`case_1_equal_weights.json` to `case_6_maximize_momentum.json`, plus
`minimize_drawdown_demo.json`). They contain the full supplied return history,
and case 6 also contains the supplied factor returns, so they run without the
workbook. `tests/test_scenarios.py` runs every example and checks the
assignment's acceptance rules. For example, case 2:

```bash
curl --fail-with-body http://127.0.0.1:8000/optimize \
  -H 'Content-Type: application/json' \
  --data-binary @examples/case_2_risk_parity.json
```

Only regenerating these files needs the local `Data.xlsx`:

```bash
uv run --locked python -m scripts.build_requests
```

## Units and methodology

- Request/response weights and weight bounds are percentages: 20 means 20%.
- Returns, yields, and portfolio constraints are decimals: 0.025 means 2.5%.
- Returns are already total returns. No price conversion or dividend addition is applied.
- Dates are sorted and intersected for only the selected securities; missing returns
  are never filled with zero. At least two common observations are required.
- Fixed weights rebalance each observation. Daily annualization defaults to 252.
  Volatility/covariance use sample estimates (`ddof=1`).
- CAGR uses observation count divided by annualization factor for elapsed years.
  Drawdown includes initial wealth. Sharpe uses arithmetic excess returns and
  defaults to a zero risk-free rate; undefined reporting metrics are null.
- Risk parity minimizes squared deviations of fractional variance contributions
  from equal shares using deterministic multistart SLSQP. Constrained solutions
  may be approximate; the response reports this explicitly. Zero-risk assets
  are rejected for this strategy because equal risk shares are undefined.
- Minimize Drawdown minimizes the largest historical loss from a prior portfolio
  wealth peak, using compounded portfolio returns. Deterministic multistart SLSQP
  improves the search, but its nonsmooth objective means convergence does not
  certify a global minimum. Failed solves never silently return a starting allocation.
- Minimize Volatility minimizes annualized sample variance using its analytic
  gradient. Positive objective scaling improves numerical conditioning without
  changing the minimizer. With only linear constraints the problem is convex;
  additional nonlinear constraints can introduce local optima.
- Maximize Sharpe uses arithmetic mean excess return divided by sample volatility,
  with an analytic gradient and deterministic multistart SLSQP. An annual effective
  risk-free rate is converted to a period rate before calculation. Zero or near-zero
  volatility is treated as undefined; no epsilon is added to fabricate a ratio.
  Final Sharpe is independently recomputed from the portfolio return series.
- Factor exposure regresses portfolio returns on Momentum, Value, and Size with an
  intercept (OLS), using only dates shared by the selected funds and the supplied
  factor returns; that window is reported separately as `factor_window`. With
  fixed weights, portfolio betas are weighted fund betas, so the objective is
  linear and solved exactly with HiGHS unless nonlinear portfolio constraints
  apply. No diversification caps are added: 100% in the highest-beta fund is a
  valid answer. Supplying `factor_returns` with any other strategy reports
  `factor_betas` without changing its optimization.
- SLSQP uses `ftol=1e-12`, `maxiter=2000`, and `eps=1e-8`.
  Nonlinear slacks are scaled during solving and checked in original units
  afterward, with an absolute feasibility tolerance of `1e-8`.

Optional `min_weight`/`max_weight` belong to each security. Portfolio
`constraints` accepts `min_cagr`, `min_volatility`, `max_volatility`,
`max_drawdown`, and `min_dividend_yield`. Equal weights reports a conflict if
its fixed allocation violates a requested constraint.

`optimize_factor_exposure` requires `factor_returns` (a list of
`{"date", "momentum", "value", "size"}` decimal returns, at least five dates) and
`factor_objective`, for example
`{"direction": "maximize", "coefficients": {"momentum": 1}}`. Coefficients weight
several factors at once; omitted factors count as zero.

Missing dividend yield remains unknown. A yield constraint requires known yields
or explicit `settings.missing_dividend_yield: "zero"`, which produces a warning.
The response includes actual analysis dates, methodology, solver status, metrics,
constraint residuals, and warnings.

Invalid input, equal-weight conflicts, and proven infeasibility return HTTP 422.
Numerical failure returns HTTP 500 with `optimization_failed`; failing to find a
nonlinear feasible allocation is not treated as proof that none exists.
Errors use `{"error": {"code": "...", "message": "...", "details": []}}`.

These financial conventions are local assumptions and have not been verified
against the live reference tool.
