from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from app.domain.constraints import constraint_residuals, violated
from app.domain.errors import (
    InfeasibleConstraintsError,
    OptimizationFailedError,
    UndefinedMetricError,
)
from app.domain.metrics import risk_shares
from app.domain.problem import OptimizationProblem, PortfolioConstraints
from app.strategies.risk_parity import risk_parity


@pytest.fixture
def problem():
    returns = np.array(
        [
            [0.01, 0, 0],
            [-0.01, 0, 0],
            [0, 0.02, 0],
            [0, -0.02, 0],
            [0, 0, 0.04],
            [0, 0, -0.04],
        ]
    )
    return OptimizationProblem(
        tickers=("AAA", "BBB", "CCC"),
        returns=returns,
        current_weights=np.ones(3) / 3,
        lower_bounds=np.zeros(3),
        upper_bounds=np.ones(3),
        yields=np.array([0.01, 0.02, 0.04]),
        constraints=PortfolioConstraints(),
        annualization_factor=1,
    )


def test_three_asset_risk_contributions_are_equal(problem):
    result = risk_parity(problem)
    np.testing.assert_allclose(result.weights, [4 / 7, 2 / 7, 1 / 7], atol=1e-5)
    np.testing.assert_allclose(
        risk_shares(problem.covariance, result.weights), [1 / 3] * 3, atol=1e-5
    )
    assert result.status == "converged"
    assert not violated(constraint_residuals(problem, result.weights))


def test_two_asset_analytic_solution_and_repeatability(problem):
    problem = replace(
        problem,
        tickers=("AAA", "BBB"),
        returns=problem.returns[:, :2],
        current_weights=np.array([0.25, 0.75]),
        lower_bounds=np.zeros(2),
        upper_bounds=np.ones(2),
        yields=problem.yields[:2],
    )
    first = risk_parity(problem)
    second = risk_parity(problem)
    np.testing.assert_allclose(first.weights, [2 / 3, 1 / 3], atol=1e-5)
    np.testing.assert_array_equal(first.weights, second.weights)


def test_bounded_approximation_is_reported(problem):
    problem = replace(problem, upper_bounds=np.array([0.2, 1, 1]))
    result = risk_parity(problem)
    assert result.weights[0] <= 0.2 + 1e-8
    assert result.status == "approximate"
    assert result.warnings
    assert not violated(constraint_residuals(problem, result.weights))


def test_yield_constraint_is_respected(problem):
    problem = replace(
        problem, constraints=PortfolioConstraints(min_dividend_yield=0.03)
    )
    result = risk_parity(problem)
    assert problem.yields @ result.weights >= 0.03 - 1e-8
    assert result.status == "approximate"


def test_nonlinear_volatility_constraint_is_respected(problem):
    problem = replace(problem, constraints=PortfolioConstraints(max_volatility=0.0056))
    result = risk_parity(problem)
    assert not violated(constraint_residuals(problem, result.weights))
    assert result.status == "approximate"


@pytest.mark.parametrize(
    "limits",
    [
        PortfolioConstraints(max_drawdown=0.0065),
        PortfolioConstraints(min_cagr=-0.000018),
        PortfolioConstraints(min_volatility=0.008),
    ],
)
def test_other_nonlinear_constraints_are_respected(problem, limits):
    problem = replace(problem, constraints=limits)
    result = risk_parity(problem)
    assert not violated(constraint_residuals(problem, result.weights))


def test_impossible_weight_bounds_are_proven_infeasible(problem):
    problem = replace(problem, lower_bounds=np.full(3, 0.4))
    with pytest.raises(InfeasibleConstraintsError):
        risk_parity(problem)


def test_impossible_yield_is_proven_infeasible(problem):
    problem = replace(
        problem, constraints=PortfolioConstraints(min_dividend_yield=0.05)
    )
    with pytest.raises(InfeasibleConstraintsError):
        risk_parity(problem)


def test_failure_to_find_nonlinear_feasibility_is_not_proof(problem):
    problem = replace(problem, constraints=PortfolioConstraints(min_volatility=1))
    with pytest.raises(
        OptimizationFailedError, match="infeasibility has not been proven"
    ):
        risk_parity(problem)


def test_zero_risk_asset_is_explicitly_rejected(problem):
    values = problem.returns.copy()
    values[:, 0] = 0
    with pytest.raises(UndefinedMetricError, match="positive variance"):
        risk_parity(replace(problem, returns=values))


def test_solver_failure_never_returns_starting_weights(problem, monkeypatch):
    monkeypatch.setattr(
        "app.strategies.risk_parity.minimize",
        lambda *args, **kwargs: SimpleNamespace(success=False),
    )
    with pytest.raises(OptimizationFailedError, match="verified solution"):
        risk_parity(problem)
