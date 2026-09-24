"""Reference-tool risk allocation, which ignores correlations between assets."""

import numpy as np
from scipy.optimize import Bounds, minimize

from app.domain.constraints import constraint_residuals, violated
from app.domain.errors import OptimizationFailedError, UndefinedMetricError
from app.domain.metrics import MIN_PERIOD_VOLATILITY
from app.domain.problem import OptimizationProblem, StrategyResult
from app.strategies.feasibility import (
    SLSQP_OPTIONS,
    clean_weights,
    feasible_starts,
    linear_constraints,
    metric_slacks,
)


def inverse_volatility(problem: OptimizationProblem) -> StrategyResult:
    volatility = np.sqrt(np.diag(problem.covariance))
    if np.any(volatility <= MIN_PERIOD_VOLATILITY):
        raise UndefinedMetricError(
            "Inverse-volatility allocation requires positive variance for every asset."
        )
    inverse = 1 / volatility
    target = inverse / inverse.sum()
    warning = (
        "Reference risk parity uses inverse-volatility weights, ignoring correlations. "
        "It does not generally equalize covariance-based portfolio risk contributions."
    )
    if not violated(constraint_residuals(problem, target)):
        return StrategyResult(target, "closed_form", warnings=[warning])

    # Where exact inverse-volatility weights are excluded, minimize deviations
    # between standalone volatility contributions subject to all requested limits.
    def objective(weights):
        contributions = weights * volatility
        return float(
            np.sum((contributions / contributions.sum() - 1 / problem.size) ** 2)
        )

    starts = feasible_starts(problem)
    constraints = list(linear_constraints(problem))
    if metric_slacks(problem, starts[0]).size:
        constraints.append({"type": "ineq", "fun": lambda w: metric_slacks(problem, w)})
    best = None
    score = np.inf
    for start in starts:
        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=Bounds(problem.lower_bounds, problem.upper_bounds),
            constraints=constraints,
            options=SLSQP_OPTIONS,
        )
        weights = clean_weights(problem, result.x)
        if result.success and not violated(constraint_residuals(problem, weights)):
            candidate = objective(weights)
            if candidate < score:
                best, score = weights, candidate
    if best is None:
        raise OptimizationFailedError(
            "No verified inverse-volatility allocation found."
        )
    return StrategyResult(
        best,
        "SLSQP",
        status="approximate",
        warnings=[warning, "Constraints prevent exact inverse-volatility weights."],
    )
