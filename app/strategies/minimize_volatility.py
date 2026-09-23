import numpy as np
from scipy.optimize import Bounds, minimize

from app.domain.constraints import constraint_residuals, violated
from app.domain.errors import OptimizationFailedError, UndefinedMetricError
from app.domain.problem import OptimizationProblem, StrategyResult
from app.strategies.feasibility import (
    SLSQP_OPTIONS,
    clean_weights,
    feasible_starts,
    linear_constraints,
    metric_slacks,
)


def minimize_volatility(problem: OptimizationProblem) -> StrategyResult:
    """Minimize variance, which has the same minimizer as volatility."""
    with np.errstate(over="ignore", invalid="ignore"):
        covariance = problem.covariance * problem.annualization_factor
    if not np.isfinite(covariance).all():
        raise UndefinedMetricError(
            "Annualized covariance is outside the numeric range."
        )
    # Positive scaling preserves the optimum and avoids early stopping for tiny returns.
    scale = float(np.max(np.abs(covariance)))
    scaled = covariance / scale if scale > 0 else covariance

    def objective(weights):
        return float(weights @ scaled @ weights)

    def gradient(weights):
        return 2 * scaled @ weights

    starts = feasible_starts(problem)
    constraints = list(linear_constraints(problem))
    nonlinear = bool(metric_slacks(problem, starts[0]).size)
    if nonlinear:
        constraints.append({"type": "ineq", "fun": lambda w: metric_slacks(problem, w)})
    baseline = min(objective(start) for start in starts)
    best_weights = None
    best_score = np.inf

    for start in starts:
        result = minimize(
            objective,
            start,
            jac=gradient,
            method="SLSQP",
            bounds=Bounds(problem.lower_bounds, problem.upper_bounds),
            constraints=constraints,
            options=SLSQP_OPTIONS,
        )
        if not result.success:
            continue
        weights = clean_weights(problem, result.x)
        if violated(constraint_residuals(problem, weights)):
            continue
        score = objective(weights)
        if np.isfinite(score) and score >= -1e-12 and score < best_score - 1e-10:
            best_score, best_weights = score, weights

    if best_weights is None or best_score > baseline + 1e-8:
        raise OptimizationFailedError(
            "Volatility optimization did not produce a converged, verified solution."
        )
    return StrategyResult(
        weights=best_weights,
        method="SLSQP",
        status="converged",
        warnings=(
            [
                "Nonlinear portfolio constraints may introduce local optima; "
                "multistart optimization does not certify a global minimum."
            ]
            if nonlinear
            else []
        ),
    )
