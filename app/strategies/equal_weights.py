import numpy as np

from app.domain.constraints import constraint_residuals, violated
from app.domain.errors import EqualWeightConflictError
from app.domain.problem import OptimizationProblem, StrategyResult


def equal_weights(problem: OptimizationProblem) -> StrategyResult:
    """Assign 1/N to every security; no solver is involved.

    Equal weights is a fixed rule, so it cannot bend to satisfy constraints.
    If 1/N breaks any supplied constraint, report the conflict instead.
    """
    weights = np.full(problem.size, 1.0 / problem.size)
    failed = violated(constraint_residuals(problem, weights))
    if failed:
        raise EqualWeightConflictError(
            "Equal weights violate the requested constraints: " + ", ".join(failed)
        )
    return StrategyResult(weights=weights, method="closed_form")
