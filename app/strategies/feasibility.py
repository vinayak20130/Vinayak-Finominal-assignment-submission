import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, minimize

from app.domain.constraints import constraint_residuals, violated
from app.domain.errors import (
    DataValidationError,
    InfeasibleConstraintsError,
    OptimizationFailedError,
    UndefinedMetricError,
)
from app.domain.metrics import annualized_volatility, cagr, maximum_drawdown
from app.domain.problem import OptimizationProblem

SLSQP_OPTIONS = {"ftol": 1e-12, "maxiter": 2000, "eps": 1e-8}
LP_OPTIONS = {
    "primal_feasibility_tolerance": 1e-9,
    "dual_feasibility_tolerance": 1e-9,
}


def linear_solution(problem: OptimizationProblem, objective: np.ndarray) -> np.ndarray:
    limits = problem.constraints
    yield_floor = limits.min_dividend_yield
    result = linprog(
        objective,
        A_eq=np.ones((1, problem.size)),
        b_eq=[1.0],
        A_ub=-problem.yields[None, :] if yield_floor is not None else None,
        b_ub=[-yield_floor] if yield_floor is not None else None,
        bounds=list(zip(problem.lower_bounds, problem.upper_bounds, strict=True)),
        method="highs",
        options=LP_OPTIONS,
    )
    if result.status == 2:
        raise InfeasibleConstraintsError(
            "Weight bounds and dividend yield constraints admit no allocation."
        )
    if not result.success:
        raise OptimizationFailedError("Linear feasibility check did not converge.")
    return result.x


def linear_constraints(problem: OptimizationProblem) -> list[LinearConstraint]:
    result = [LinearConstraint(np.ones((1, problem.size)), 1, 1)]
    if problem.constraints.min_dividend_yield is not None:
        result.append(
            LinearConstraint(
                problem.yields[None, :], problem.constraints.min_dividend_yield, np.inf
            )
        )
    return result


def metric_slacks(problem: OptimizationProblem, weights: np.ndarray) -> np.ndarray:
    """Evaluate nonlinear slacks even at temporary, not fully invested iterates."""
    limits = problem.constraints
    returns = problem.returns @ weights
    factor = problem.annualization_factor
    values = []
    for bound, function, upper in [
        (limits.min_cagr, lambda: cagr(returns, factor), False),
        (limits.min_volatility, lambda: annualized_volatility(returns, factor), False),
        (limits.max_volatility, lambda: annualized_volatility(returns, factor), True),
        (limits.max_drawdown, lambda: maximum_drawdown(returns), True),
    ]:
        if bound is None:
            continue
        try:
            metric = function()
            slack = bound - metric if upper else metric - bound
            # Scaling keeps small decimal constraints visible to SLSQP.
            scale = max(abs(bound), 1e-4)
            values.append(slack / scale if np.isfinite(slack) else -1e6)
        except (DataValidationError, UndefinedMetricError):
            values.append(-1e6)
    return np.array(values, dtype=float)


def clean_weights(problem: OptimizationProblem, weights: np.ndarray) -> np.ndarray:
    """Remove only tiny floating-point excursions; verification remains mandatory."""
    result = np.asarray(weights, dtype=float).copy()
    if not np.isfinite(result).all() or result.shape != (problem.size,):
        return result
    clipped = np.clip(result, problem.lower_bounds, problem.upper_bounds)
    if np.max(np.abs(result - clipped)) <= 1e-8:
        result = clipped
    total = result.sum()
    if total > 0 and abs(total - 1) <= 1e-8:
        result /= total
    return result


def feasible_starts(problem: OptimizationProblem) -> list[np.ndarray]:
    base = linear_solution(problem, np.zeros(problem.size))
    raw = [
        np.full(problem.size, 1 / problem.size),
        problem.current_weights,
        base,
    ]
    raw.extend(linear_solution(problem, -axis) for axis in np.eye(problem.size))
    raw.extend(np.random.default_rng(42).dirichlet(np.ones(problem.size), size=16))
    linear = linear_constraints(problem)
    bounds = Bounds(problem.lower_bounds, problem.upper_bounds)
    starts = []
    for candidate in raw:
        projected = minimize(
            lambda w: float(np.sum((w - candidate) ** 2)),
            base,
            jac=lambda w: 2 * (w - candidate),
            bounds=bounds,
            constraints=linear,
            method="SLSQP",
            options=SLSQP_OPTIONS,
        )
        if not projected.success:
            continue
        weights = clean_weights(problem, projected.x)
        if violated(constraint_residuals(problem, weights)):
            feasible = minimize(
                lambda w: float(
                    np.minimum(metric_slacks(problem, w), 0)
                    @ np.minimum(metric_slacks(problem, w), 0)
                ),
                weights,
                method="SLSQP",
                bounds=bounds,
                constraints=linear,
                options=SLSQP_OPTIONS,
            )
            if not feasible.success:
                continue
            weights = clean_weights(problem, feasible.x)
        if not violated(constraint_residuals(problem, weights)) and not any(
            np.allclose(weights, old, atol=1e-8, rtol=0) for old in starts
        ):
            starts.append(weights)
    if not starts:
        raise OptimizationFailedError(
            "No feasible start was found for the nonlinear constraints; "
            "infeasibility has not been proven."
        )
    return starts
