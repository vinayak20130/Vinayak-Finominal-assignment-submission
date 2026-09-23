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
from app.domain.problem import OptimizationProblem, PortfolioConstraints
from app.strategies.maximize_sharpe import maximize_sharpe


@pytest.fixture
def problem():
    values = np.array(
        [
            [0.01, 0, 0],
            [-0.01, 0, 0],
            [0, 0.02, 0],
            [0, -0.02, 0],
            [0, 0, 0.04],
            [0, 0, -0.04],
        ]
    ) + [0.001, 0.002, 0.003]
    return OptimizationProblem(
        tickers=("AAA", "BBB", "CCC"),
        returns=values,
        current_weights=np.ones(3) / 3,
        lower_bounds=np.zeros(3),
        upper_bounds=np.ones(3),
        yields=np.array([0.01, 0.02, 0.04]),
        constraints=PortfolioConstraints(),
        annualization_factor=1,
    )


def independent_sharpe(values, annualization=1, risk_free=0):
    period_rate = (1 + risk_free) ** (1 / annualization) - 1
    return (values.mean() - period_rate) / values.std(ddof=1) * np.sqrt(annualization)


@pytest.mark.parametrize("annualization,risk_free", [(1, 0), (252, 0.05)])
def test_matches_analytic_tangency_solution(problem, annualization, risk_free):
    problem = replace(
        problem, annualization_factor=annualization, risk_free_rate=risk_free
    )
    period_rate = (1 + risk_free) ** (1 / annualization) - 1
    expected = np.linalg.solve(
        np.cov(problem.returns, rowvar=False, ddof=1),
        problem.returns.mean(axis=0) - period_rate,
    )
    expected /= expected.sum()
    result = maximize_sharpe(problem)
    np.testing.assert_allclose(result.weights, expected, atol=1e-6)
    assert not violated(constraint_residuals(problem, result.weights))


@pytest.mark.parametrize("mean_shift", [0, -0.02])
def test_two_asset_grid_including_negative_sharpe(problem, mean_shift):
    values = problem.returns[:, :2] + mean_shift
    problem = replace(
        problem,
        tickers=("AAA", "BBB"),
        returns=values,
        current_weights=np.array([0.5, 0.5]),
        lower_bounds=np.zeros(2),
        upper_bounds=np.ones(2),
        yields=problem.yields[:2],
    )
    result = maximize_sharpe(problem)
    grid = np.linspace(0, 1, 10_001)
    grid_returns = values @ np.vstack((grid, 1 - grid))
    best = (grid_returns.mean(axis=0) / grid_returns.std(axis=0, ddof=1)).max()
    actual = independent_sharpe(values @ result.weights)
    assert actual >= best - 1e-4


def test_bounds_and_yield_floor(problem):
    problem = replace(
        problem,
        lower_bounds=np.full(3, 0.05),
        upper_bounds=np.full(3, 0.6),
        constraints=PortfolioConstraints(min_dividend_yield=0.025),
    )
    result = maximize_sharpe(problem)
    assert not violated(constraint_residuals(problem, result.weights))
    assert problem.yields @ result.weights >= 0.025 - 1e-8


@pytest.mark.parametrize(
    "limits",
    [
        PortfolioConstraints(max_volatility=0.0056),
        PortfolioConstraints(min_volatility=0.009),
        PortfolioConstraints(max_drawdown=0.005),
        PortfolioConstraints(min_cagr=0.002),
    ],
)
def test_nonlinear_constraints(problem, limits):
    problem = replace(problem, constraints=limits)
    result = maximize_sharpe(problem)
    assert not violated(constraint_residuals(problem, result.weights))


@pytest.mark.parametrize(
    "values",
    [
        np.zeros((6, 3)),
        np.full((6, 3), 0.01),
        np.tile([0.001, 0.002, 0.003], (6, 1)),
    ],
)
def test_undefined_sharpe_is_not_fabricated(problem, values):
    with pytest.raises(UndefinedMetricError):
        maximize_sharpe(replace(problem, returns=values))


def test_zero_excess_means_are_valid_when_volatility_is_positive(problem):
    problem = replace(problem, returns=problem.returns - [0.001, 0.002, 0.003])
    result = maximize_sharpe(problem)
    assert independent_sharpe(problem.returns @ result.weights) == pytest.approx(0)


def test_repeatable(problem):
    np.testing.assert_array_equal(
        maximize_sharpe(problem).weights, maximize_sharpe(problem).weights
    )


def test_impossible_yield_is_rejected(problem):
    problem = replace(
        problem, constraints=PortfolioConstraints(min_dividend_yield=0.05)
    )
    with pytest.raises(InfeasibleConstraintsError):
        maximize_sharpe(problem)


@pytest.mark.parametrize(
    "success,weights",
    [
        (False, [1, 0, 0]),
        (True, [np.nan, 0.5, 0.5]),
        (True, [0, 0, 1]),
    ],
)
def test_solver_failures_never_return_invalid_or_inferior_weights(
    problem, monkeypatch, success, weights
):
    monkeypatch.setattr(
        "app.strategies.maximize_sharpe.minimize",
        lambda *args, **kwargs: SimpleNamespace(success=success, x=np.array(weights)),
    )
    with pytest.raises(OptimizationFailedError):
        maximize_sharpe(problem)
