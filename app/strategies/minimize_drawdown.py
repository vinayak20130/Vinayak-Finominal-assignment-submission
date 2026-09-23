import numpy as np
from scipy.optimize import Bounds, minimize

from app.domain.constraints import constraint_residuals, violated
from app.domain.errors import (
    DataValidationError,
    OptimizationFailedError,
    UndefinedMetricError,
)
from app.domain.metrics import maximum_drawdown
from app.domain.problem import OptimizationProblem, StrategyResult
from app.strategies.feasibility import (
    SLSQP_OPTIONS,
    clean_weights,
    feasible_starts,
    linear_constraints,
    metric_slacks,
)


def minimize_drawdown(problem: OptimizationProblem) -> StrategyResult:
    """Minimize compounded portfolio drawdown from deterministic feasible starts."""

    def objective(weights):
        try:
            return maximum_drawdown(problem.returns @ weights)
        except (DataValidationError, UndefinedMetricError):
            # Temporary solver iterates may not yet satisfy full investment.
            return 1e6

    starts = feasible_starts(problem)
    constraints = list(linear_constraints(problem))
    if metric_slacks(problem, starts[0]).size:
        constraints.append({"type": "ineq", "fun": lambda w: metric_slacks(problem, w)})
    baseline = min(objective(start) for start in starts)
    best_weights = None
    best_score = np.inf

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
        score = objective(weights)
        if np.isfinite(score) and 0 <= score <= 1 and score < best_score - 1e-10:
            best_score, best_weights = score, weights

    if best_weights is None or best_score > baseline + 1e-8:
        raise OptimizationFailedError(
            "Drawdown optimization did not produce a converged, verified solution."
        )
    return StrategyResult(
        weights=best_weights,
        method="SLSQP",
        status="converged",
        warnings=[
            "Historical maximum drawdown is nonsmooth. Multistart optimization "
            "does not certify a global minimum."
        ],
    )
