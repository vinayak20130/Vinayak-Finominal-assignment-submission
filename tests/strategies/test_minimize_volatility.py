from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from app.domain.constraints import constraint_residuals, violated
from app.domain.errors import InfeasibleConstraintsError, OptimizationFailedError
from app.domain.problem import OptimizationProblem, PortfolioConstraints
from app.strategies.minimize_volatility import minimize_volatility


@pytest.fixture
def problem():
    return OptimizationProblem(
        tickers=("AAA", "BBB", "CCC"),
        returns=np.array(
            [
                [0.01, 0, 0],
                [-0.01, 0, 0],
                [0, 0.02, 0],
                [0, -0.02, 0],
                [0, 0, 0.04],
                [0, 0, -0.04],
            ]
        ),
        current_weights=np.ones(3) / 3,
        lower_bounds=np.zeros(3),
        upper_bounds=np.ones(3),
        yields=np.array([0.01, 0.02, 0.04]),
        constraints=PortfolioConstraints(),
        annualization_factor=1,
    )


def test_diagonal_covariance_matches_inverse_variance_weights(problem):
    result = minimize_volatility(problem)
    np.testing.assert_allclose(result.weights, [16 / 21, 4 / 21, 1 / 21], atol=1e-6)
    assert not violated(constraint_residuals(problem, result.weights))
    assert result.status == "converged"
    assert not result.warnings


@pytest.mark.parametrize("scale", [1e-6, 0.1, 5])
def test_return_scale_does_not_change_optimal_weights(problem, scale):
    result = minimize_volatility(replace(problem, returns=problem.returns * scale))
    np.testing.assert_allclose(result.weights, [16 / 21, 4 / 21, 1 / 21], atol=1e-6)


def test_annualization_does_not_change_unconstrained_weights(problem):
    original = minimize_volatility(problem)
    annualized = minimize_volatility(replace(problem, annualization_factor=252))
    np.testing.assert_allclose(original.weights, annualized.weights, atol=1e-6)


def test_correlated_assets_match_analytic_solution_and_grid(problem):
    values = np.array([[0.02, -0.01], [-0.01, 0.03], [0.03, 0.01], [-0.02, -0.01]])
    problem = replace(
        problem,
        tickers=("AAA", "BBB"),
        returns=values,
        current_weights=np.array([0.6, 0.4]),
        lower_bounds=np.zeros(2),
        upper_bounds=np.ones(2),
        yields=problem.yields[:2],
    )
    covariance = np.cov(values, rowvar=False, ddof=1)
    first = (covariance[1, 1] - covariance[0, 1]) / (
        covariance[0, 0] + covariance[1, 1] - 2 * covariance[0, 1]
    )
    result = minimize_volatility(problem)
    np.testing.assert_allclose(result.weights, [first, 1 - first], atol=1e-6)
    grid = np.linspace(0, 1, 10_001)
    grid_best = np.std(values @ np.vstack((grid, 1 - grid)), axis=0, ddof=1).min()
    assert np.std(values @ result.weights, ddof=1) <= grid_best + 1e-8


def test_bounds_and_yield_floor(problem):
    problem = replace(
        problem,
        upper_bounds=np.full(3, 0.6),
        constraints=PortfolioConstraints(min_dividend_yield=0.025),
    )
    result = minimize_volatility(problem)
    assert not violated(constraint_residuals(problem, result.weights))
    assert result.weights.max() <= 0.6 + 1e-8
    assert problem.yields @ result.weights >= 0.025 - 1e-8


@pytest.mark.parametrize(
    "limits",
    [
        PortfolioConstraints(max_volatility=0.0056),
        PortfolioConstraints(min_volatility=0.008),
        PortfolioConstraints(max_drawdown=0.0065),
        PortfolioConstraints(min_cagr=-0.000018),
    ],
)
def test_nonlinear_constraints(problem, limits):
    problem = replace(problem, constraints=limits)
    result = minimize_volatility(problem)
    assert not violated(constraint_residuals(problem, result.weights))
    assert result.warnings


def test_zero_risk_asset_can_take_full_weight(problem):
    values = problem.returns.copy()
    values[:, 0] = 0
    problem = replace(problem, returns=values)
    result = minimize_volatility(problem)
    np.testing.assert_allclose(result.weights, [1, 0, 0], atol=1e-6)
    assert np.std(values @ result.weights, ddof=1) < 1e-8


def test_all_constant_assets_have_zero_volatility(problem):
    result = minimize_volatility(replace(problem, returns=np.zeros((6, 3))))
    assert not violated(constraint_residuals(problem, result.weights))


def test_singular_covariance_needs_no_regularization(problem):
    values = problem.returns.copy()
    values[:, 1] = values[:, 0]
    problem = replace(problem, returns=values)
    result = minimize_volatility(problem)
    assert np.std(values @ result.weights, ddof=1) <= (
        np.std(values @ problem.current_weights, ddof=1)
    )
    assert not violated(constraint_residuals(problem, result.weights))


def test_impossible_yield_is_rejected(problem):
    problem = replace(
        problem, constraints=PortfolioConstraints(min_dividend_yield=0.05)
    )
    with pytest.raises(InfeasibleConstraintsError):
        minimize_volatility(problem)


def test_repeatable(problem):
    np.testing.assert_array_equal(
        minimize_volatility(problem).weights, minimize_volatility(problem).weights
    )


@pytest.mark.parametrize(
    "success, weights",
    [
        (False, [0.5, 0.25, 0.25]),
        (True, [float("nan"), 0.5, 0.5]),
        (True, [0, 0, 1]),
    ],
)
def test_failure_invalid_output_and_regression_are_rejected(
    problem, monkeypatch, success, weights
):
    monkeypatch.setattr(
        "app.strategies.minimize_volatility.minimize",
        lambda *args, **kwargs: SimpleNamespace(success=success, x=np.array(weights)),
    )
    with pytest.raises(OptimizationFailedError):
        minimize_volatility(problem)
