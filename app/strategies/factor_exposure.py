import numpy as np
from scipy.optimize import Bounds, minimize

from app.domain.constraints import constraint_residuals, violated
from app.domain.errors import DataValidationError, OptimizationFailedError
from app.domain.problem import OptimizationProblem, StrategyResult
from app.strategies.feasibility import (
    SLSQP_OPTIONS,
    clean_weights,
    feasible_starts,
    linear_constraints,
    linear_solution,
    metric_slacks,
)


def factor_exposure(problem: OptimizationProblem) -> StrategyResult:
    costs = problem.factor_costs
    if costs is None or costs.shape != (problem.size,) or not np.isfinite(costs).all():
        raise DataValidationError("Factor optimization requires finite exposure costs.")
    scale = float(np.max(np.abs(costs)))
    scaled = costs / scale if scale > 0 else costs
    limits = problem.constraints
    nonlinear = any(
        value is not None
        for value in (
            limits.min_cagr,
            limits.min_volatility,
            limits.max_volatility,
            limits.max_drawdown,
        )
    )
    if not nonlinear:
        weights = clean_weights(problem, linear_solution(problem, scaled))
        if violated(constraint_residuals(problem, weights)):
            raise OptimizationFailedError("Factor solution failed constraint checks.")
        return StrategyResult(weights=weights, method="HiGHS")

    starts = feasible_starts(problem)
    constraints = list(linear_constraints(problem))
    constraints.append({"type": "ineq", "fun": lambda w: metric_slacks(problem, w)})
    baseline = min(float(scaled @ start) for start in starts)
    best_weights = None
    best_score = np.inf
    for start in starts:
        result = minimize(
            lambda w: float(scaled @ w),
            start,
            jac=lambda w: scaled,
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
        score = float(scaled @ weights)
        if np.isfinite(score) and score < best_score - 1e-10:
            best_score, best_weights = score, weights
    if best_weights is None or best_score > baseline + 1e-8:
        raise OptimizationFailedError(
            "Factor optimization did not produce a converged, verified solution."
        )
    return StrategyResult(
        weights=best_weights,
        method="SLSQP",
        status="converged",
        warnings=[
            "Nonlinear constraints may introduce local optima; "
            "multistart optimization does not certify a global optimum."
        ],
    )
