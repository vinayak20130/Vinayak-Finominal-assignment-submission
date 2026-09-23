from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from app.domain.constraints import constraint_residuals, violated
from app.domain.errors import InfeasibleConstraintsError, OptimizationFailedError
from app.domain.problem import OptimizationProblem, PortfolioConstraints
from app.strategies.minimize_drawdown import minimize_drawdown


def drawdown(returns):
    # Independent reference: compound wealth directly, including initial capital.
    wealth = np.concatenate(([1.0], np.cumprod(1 + returns)))
    return float(np.max(1 - wealth / np.maximum.accumulate(wealth)))


@pytest.fixture
def problem():
    return OptimizationProblem(
        tickers=("AAA", "BBB"),
        returns=np.array([[-0.2, 0.1], [0.1, -0.2]]),
        current_weights=np.array([0.8, 0.2]),
        lower_bounds=np.zeros(2),
        upper_bounds=np.ones(2),
        yields=np.array([0.01, 0.05]),
        constraints=PortfolioConstraints(),
        annualization_factor=1,
    )


@pytest.mark.parametrize(
    "values",
    [
        [[-0.2, 0.1], [0.1, -0.2]],
        [[-0.1, 0.03], [0.04, -0.08], [0.06, -0.01], [-0.03, 0.02]],
        [[0.12, -0.04], [-0.15, 0.01], [0.05, 0.03], [-0.02, -0.01]],
    ],
)
def test_matches_independent_dense_grid(problem, values):
    problem = replace(problem, returns=np.array(values))
    result = minimize_drawdown(problem)
    grid = np.linspace(0, 1, 10_001)
    grid_returns = problem.returns @ np.vstack((grid, 1 - grid))
    wealth = np.vstack((np.ones(len(grid)), np.cumprod(1 + grid_returns, axis=0)))
    grid_best = np.max(1 - wealth / np.maximum.accumulate(wealth, axis=0), axis=0).min()
    actual = drawdown(problem.returns @ result.weights)
    assert actual <= grid_best + 1e-4
    assert actual <= drawdown(problem.returns @ problem.current_weights) + 1e-8
    assert not violated(constraint_residuals(problem, result.weights))


def test_compounded_path_includes_first_period_loss(problem):
    result = minimize_drawdown(problem)
    np.testing.assert_allclose(result.weights, [0.5, 0.5], atol=1e-5)
    assert drawdown(problem.returns @ result.weights) == pytest.approx(0.0975, abs=1e-6)
    assert result.status == "converged"
    assert "global minimum" in result.warnings[0]


def test_repeatable(problem):
    np.testing.assert_array_equal(
        minimize_drawdown(problem).weights, minimize_drawdown(problem).weights
    )


def test_bounds_are_respected(problem):
    problem = replace(problem, upper_bounds=np.array([0.3, 1]))
    result = minimize_drawdown(problem)
    assert result.weights[0] <= 0.3 + 1e-8
    assert not violated(constraint_residuals(problem, result.weights))


def test_yield_floor_is_respected(problem):
    problem = replace(
        problem, constraints=PortfolioConstraints(min_dividend_yield=0.04)
    )
    result = minimize_drawdown(problem)
    assert problem.yields @ result.weights >= 0.04 - 1e-8
    assert result.weights[1] == pytest.approx(0.75, abs=1e-5)


@pytest.mark.parametrize(
    "limits",
    [
        PortfolioConstraints(max_drawdown=0.11),
        PortfolioConstraints(min_cagr=-0.051),
        PortfolioConstraints(max_volatility=0.01),
        PortfolioConstraints(min_volatility=0.03),
    ],
)
def test_nonlinear_constraints_are_respected(problem, limits):
    problem = replace(problem, constraints=limits)
    result = minimize_drawdown(problem)
    assert not violated(constraint_residuals(problem, result.weights))


@pytest.mark.parametrize(
    "values, expected",
    [
        ([[0.01, 0.02], [0.03, 0.01]], 0),
        ([[0, 0], [0, 0]], 0),
        ([[-1, -1], [0.5, 0.5]], 1),
    ],
)
def test_degenerate_paths(problem, values, expected):
    problem = replace(problem, returns=np.array(values))
    result = minimize_drawdown(problem)
    assert drawdown(problem.returns @ result.weights) == pytest.approx(expected)


def test_proven_infeasible_constraints(problem):
    problem = replace(
        problem, constraints=PortfolioConstraints(min_dividend_yield=0.06)
    )
    with pytest.raises(InfeasibleConstraintsError):
        minimize_drawdown(problem)


def test_solver_failure_does_not_return_a_baseline(problem, monkeypatch):
    monkeypatch.setattr(
        "app.strategies.minimize_drawdown.minimize",
        lambda *args, **kwargs: SimpleNamespace(success=False),
    )
    with pytest.raises(OptimizationFailedError):
        minimize_drawdown(problem)


def test_solver_success_is_independently_verified(problem, monkeypatch):
    monkeypatch.setattr(
        "app.strategies.minimize_drawdown.minimize",
        lambda *args, **kwargs: SimpleNamespace(
            success=True, x=np.array([np.nan, 0.5])
        ),
    )
    with pytest.raises(OptimizationFailedError):
        minimize_drawdown(problem)


def test_worse_than_feasible_baseline_is_rejected(problem, monkeypatch):
    monkeypatch.setattr(
        "app.strategies.minimize_drawdown.minimize",
        lambda *args, **kwargs: SimpleNamespace(success=True, x=np.array([1.0, 0.0])),
    )
    with pytest.raises(OptimizationFailedError):
        minimize_drawdown(problem)
