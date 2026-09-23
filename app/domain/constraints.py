"""Independent constraint checks, recomputed from weights after any strategy runs."""

import numpy as np
from numpy.typing import ArrayLike

from app.domain.errors import UndefinedMetricError
from app.domain.metrics import (
    annualized_volatility,
    cagr,
    maximum_drawdown,
    portfolio_dividend_yield,
    portfolio_returns,
)
from app.domain.problem import OptimizationProblem

# Absolute feasibility tolerance for fractional weights and decimal metrics.
FEASIBILITY_TOLERANCE = 1e-8


def constraint_residuals(
    problem: OptimizationProblem, weights: ArrayLike
) -> dict[str, float]:
    """Slack for every applicable constraint; negative slack is a violation.

    Weight bounds and the full-investment rule are always checked. Portfolio
    constraints are checked only when requested.
    """
    w = np.asarray(weights, dtype=np.float64)
    if (
        w.shape != (problem.size,)
        or not np.isfinite(w).all()
        or (w < 0).any()
        or (w > 1).any()
    ):
        return {"valid_weights": -np.inf}
    residuals = {
        # 0.0 - x rather than -x, so an exact sum reports 0.0, not -0.0.
        "weights_sum_to_one": 0.0 - abs(float(w.sum()) - 1.0),
        "min_weight": float(np.min(w - problem.lower_bounds)),
        "max_weight": float(np.min(problem.upper_bounds - w)),
    }
    if violated(residuals) or (w < 0).any():
        # Metrics are undefined for an invalid allocation; report weights only.
        return residuals
    limits = problem.constraints
    returns = portfolio_returns(problem.returns, w)
    factor = problem.annualization_factor
    if limits.min_cagr is not None:
        residuals["min_cagr"] = _metric(cagr, returns, factor) - limits.min_cagr
    if limits.min_volatility is not None or limits.max_volatility is not None:
        volatility = annualized_volatility(returns, factor)
        if limits.min_volatility is not None:
            residuals["min_volatility"] = volatility - limits.min_volatility
        if limits.max_volatility is not None:
            residuals["max_volatility"] = limits.max_volatility - volatility
    if limits.max_drawdown is not None:
        residuals["max_drawdown"] = limits.max_drawdown - maximum_drawdown(returns)
    if limits.min_dividend_yield is not None:
        dividend_yield = portfolio_dividend_yield(problem.yields, w)
        residuals["min_dividend_yield"] = dividend_yield - limits.min_dividend_yield
    return residuals


def violated(residuals: dict[str, float]) -> list[str]:
    return [
        name
        for name, slack in residuals.items()
        if not np.isfinite(slack) or slack < -FEASIBILITY_TOLERANCE
    ]


def _metric(function, returns, factor) -> float:
    # An undefined CAGR (numeric overflow) cannot be shown to satisfy a floor.
    try:
        return function(returns, factor)
    except UndefinedMetricError:
        return -np.inf
