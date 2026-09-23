import numpy as np
from scipy.optimize import Bounds, minimize

from app.domain.constraints import constraint_residuals, violated
from app.domain.errors import OptimizationFailedError, UndefinedMetricError
from app.domain.metrics import MIN_PERIOD_VOLATILITY, sharpe_ratio
from app.domain.problem import OptimizationProblem, StrategyResult
from app.strategies.feasibility import (
    SLSQP_OPTIONS,
    clean_weights,
    feasible_starts,
    linear_constraints,
    metric_slacks,
)


def maximize_sharpe(problem: OptimizationProblem) -> StrategyResult:
    """Maximize arithmetic-mean excess return divided by sample volatility."""
    covariance = problem.covariance
    annualization = problem.annualization_factor
    risk_free = np.expm1(np.log1p(problem.risk_free_rate) / annualization)
    excess_means = problem.returns.mean(axis=0) - risk_free
    annual_scale = np.sqrt(annualization)

    def objective(weights):
        variance = float(weights @ covariance @ weights)
        if not np.isfinite(variance) or variance <= MIN_PERIOD_VOLATILITY**2:
            return 1e12
        ratio = annual_scale * (excess_means @ weights) / np.sqrt(variance)
        return -float(ratio) if np.isfinite(ratio) else 1e12

    def gradient(weights):
        marginal = covariance @ weights
        variance = float(weights @ marginal)
        if not np.isfinite(variance) or variance <= MIN_PERIOD_VOLATILITY**2:
            return np.zeros(problem.size)
        numerator = float(excess_means @ weights)
        return -annual_scale * (
            excess_means / np.sqrt(variance) - numerator * marginal / variance**1.5
        )

    def verified_sharpe(weights):
        return sharpe_ratio(
            problem.returns @ weights, annualization, problem.risk_free_rate
        )

    starts = []
    baseline = -np.inf
    for candidate in feasible_starts(problem):
        try:
            score = verified_sharpe(candidate)
        except UndefinedMetricError:
            continue
        starts.append(candidate)
        baseline = max(baseline, score)
    if not starts:
        raise UndefinedMetricError(
            "Sharpe ratio is undefined at all feasible starting allocations."
        )

    constraints = list(linear_constraints(problem))
    if metric_slacks(problem, starts[0]).size:
        constraints.append({"type": "ineq", "fun": lambda w: metric_slacks(problem, w)})
    best_weights = None
    best_score = -np.inf
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
        try:
            score = verified_sharpe(weights)
        except UndefinedMetricError:
            continue
        if score > best_score + 1e-10:
            best_score, best_weights = score, weights

    if best_weights is None or best_score < baseline - 1e-8:
        raise OptimizationFailedError(
            "Sharpe optimization did not produce a converged, verified solution."
        )
    return StrategyResult(
        weights=best_weights,
        method="SLSQP",
        status="converged",
        warnings=["Multistart Sharpe optimization does not certify a global maximum."],
    )
