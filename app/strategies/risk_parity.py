import numpy as np
from scipy.optimize import Bounds, minimize

from app.domain.constraints import constraint_residuals, violated
from app.domain.errors import OptimizationFailedError, UndefinedMetricError
from app.domain.metrics import MIN_PERIOD_VOLATILITY, risk_shares
from app.domain.problem import OptimizationProblem, StrategyResult
from app.strategies.feasibility import (
    SLSQP_OPTIONS,
    clean_weights,
    feasible_starts,
    linear_constraints,
    metric_slacks,
)


def risk_parity(problem: OptimizationProblem) -> StrategyResult:
    """Minimize squared deviations from equal contributions to portfolio variance."""
    covariance = problem.covariance
    if np.any(np.diag(covariance) <= MIN_PERIOD_VOLATILITY**2):
        raise UndefinedMetricError(
            "Risk parity requires positive variance for every selected security."
        )
    target = 1 / problem.size

    def objective(weights):
        variance = float(weights @ covariance @ weights)
        if not np.isfinite(variance) or variance <= MIN_PERIOD_VOLATILITY**2:
            return 1e6
        shares = weights * (covariance @ weights) / variance
        return float(np.sum((shares - target) ** 2))

    starts = feasible_starts(problem)
    constraints = list(linear_constraints(problem))
    if metric_slacks(problem, starts[0]).size:
        constraints.append({"type": "ineq", "fun": lambda w: metric_slacks(problem, w)})
    best_weights = None
    best_score = np.inf
    baseline = min(objective(start) for start in starts)

    for start in starts:
        result = minimize(
            objective,
            start,
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
        try:
            shares = risk_shares(covariance, weights)
        except UndefinedMetricError:
            continue
        score = float(np.sum((shares - target) ** 2))
        if np.isfinite(score) and score < best_score - 1e-10:
            best_score, best_weights = score, weights

    if best_weights is None or best_score > baseline + 1e-8:
        raise OptimizationFailedError(
            "Risk parity did not produce a converged, verified solution."
        )
    deviation = float(np.max(np.abs(risk_shares(covariance, best_weights) - target)))
    approximate = deviation > 1e-5
    warnings = (
        [
            "Risk contributions are approximate; "
            f"maximum share deviation is {deviation:.8g}."
        ]
        if approximate
        else []
    )
    return StrategyResult(
        weights=best_weights,
        method="SLSQP",
        status="approximate" if approximate else "converged",
        warnings=warnings,
    )
