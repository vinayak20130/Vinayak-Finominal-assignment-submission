import numpy as np
import pytest

from app.domain.errors import DataValidationError, InsufficientDataError
from app.domain.factors import regress_factors


@pytest.fixture
def factors():
    return np.array(
        [
            [0, 0, 0],
            [0.01, 0, 0],
            [0, 0.01, 0],
            [0, 0, 0.01],
            [-0.01, 0, 0],
            [0.01, -0.01, 0.01],
        ]
    )


def test_recovers_intercept_and_known_coefficients(factors):
    coefficients = np.array([0.003, 2, -0.4, 0.7])
    returns = coefficients[0] + factors @ coefficients[1:]
    np.testing.assert_allclose(
        regress_factors(returns, factors), coefficients, atol=1e-12
    )


def test_direct_portfolio_regression_agrees_with_weighted_fund_betas(factors):
    fund_coefficients = np.array([[0.001, -0.003], [2, 0.5], [-0.2, 0.3], [0.8, -0.1]])
    returns = np.column_stack((np.ones(len(factors)), factors)) @ fund_coefficients
    weights = np.array([0.3, 0.7])
    fitted = regress_factors(returns, factors)
    np.testing.assert_allclose(fitted, fund_coefficients, atol=1e-12)
    np.testing.assert_allclose(
        regress_factors(returns @ weights, factors), fitted @ weights, atol=1e-12
    )


def test_rank_deficiency_is_rejected(factors):
    factors[:, 2] = factors[:, 0] * 2
    with pytest.raises(DataValidationError, match="rank"):
        regress_factors(np.zeros(len(factors)), factors)


def test_too_few_observations(factors):
    with pytest.raises(InsufficientDataError):
        regress_factors(np.zeros(4), factors[:4])


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_nonfinite_factor_data(factors, bad):
    factors[0, 0] = bad
    with pytest.raises(DataValidationError, match="finite"):
        regress_factors(np.zeros(len(factors)), factors)


def test_misaligned_shapes(factors):
    with pytest.raises(DataValidationError, match="aligned"):
        regress_factors(np.zeros(5), factors)
