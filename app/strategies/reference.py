"""Reference-compatible SLSQP conventions; all results still require verification."""

import numpy as np
from scipy.optimize import minimize

from app.domain.constraints import constraint_residuals, violated
from app.domain.errors import OptimizationFailedError
from app.domain.metrics import MIN_PERIOD_VOLATILITY, maximum_drawdown
from app.domain.problem import OptimizationProblem, StrategyResult
from app.strategies.feasibility import clean_weights, metric_slacks


def reference_solution(problem: OptimizationProblem, strategy: str) -> StrategyResult:
    covariance = problem.covariance * problem.annualization_factor
    means = problem.returns.mean(axis=0) * problem.annualization_factor
    risk_free = (
        np.expm1(np.log1p(problem.risk_free_rate) / problem.annualization_factor)
        * problem.annualization_factor
    )

    def objective(weights):
        if strategy == "minimize_drawdown":
            return maximum_drawdown(problem.returns @ weights)
        volatility = np.sqrt(max(float(weights @ covariance @ weights), 0))
        if strategy == "minimize_volatility":
            return volatility
        if volatility <= MIN_PERIOD_VOLATILITY:
            return 1e12
        return -float(means @ weights - risk_free) / volatility

    constraints = [{"type": "eq", "fun": lambda w: w.sum() - 1}]
    if problem.constraints.min_dividend_yield is not None:
        constraints.append(
            {
                "type": "ineq",
                "fun": lambda w: (
                    w @ problem.yields - problem.constraints.min_dividend_yield
                ),
            }
        )
    if metric_slacks(problem, problem.current_weights).size:
        constraints.append({"type": "ineq", "fun": lambda w: metric_slacks(problem, w)})
    result = minimize(
        objective,
        problem.current_weights,
        method="SLSQP",
        bounds=list(zip(problem.lower_bounds, problem.upper_bounds, strict=True)),
        constraints=constraints,
        options={"ftol": 1e-6, "maxiter": 1000},
    )
    weights = clean_weights(problem, result.x)
    if (
        not result.success
        or not np.isfinite(result.fun)
        or result.fun >= 1e12
        or violated(constraint_residuals(problem, weights))
    ):
        raise OptimizationFailedError(
            "Reference-profile optimization did not converge to a verified solution."
        )
    return StrategyResult(
        weights=weights,
        method="SLSQP",
        status="converged",
        warnings=[
            "Reference profile uses the current allocation as a single SLSQP start "
            "and ftol=1e-6; this does not certify a global optimum."
        ],
    )
