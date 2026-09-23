import numpy as np
import pytest

from app.domain.errors import DataValidationError, UndefinedMetricError
from app.domain.metrics import (
    annualized_volatility,
    cagr,
    maximum_drawdown,
    portfolio_dividend_yield,
    portfolio_returns,
    portfolio_variance,
    risk_shares,
    sample_covariance,
    sharpe_ratio,
    wealth_index,
)


def test_weighted_returns_and_compounding():
    values = [[0.1, -0.1], [-0.2, 0.2]]
    result = portfolio_returns(values, [0.75, 0.25])
    np.testing.assert_allclose(result, [0.05, -0.1])
    np.testing.assert_allclose(wealth_index(result), [1, 1.05, 0.945])


@pytest.mark.parametrize(
    "values, expected",
    [
        ([-0.2, 0.1], 0.2),
        ([0.1, -0.2, 0.05], 0.2),
        ([0.1, 0.2], 0),
        ([0.1, -1, 0.5], 1),
        ([-0.1], 0.1),
    ],
)
def test_drawdown_includes_initial_wealth(values, expected):
    assert maximum_drawdown(values) == pytest.approx(expected)


def test_cagr_compounds_instead_of_averaging():
    assert cagr([0.1, -0.1], annualization_factor=2) == pytest.approx(-0.01)
    assert cagr([0.1, -0.1], annualization_factor=4) == pytest.approx(0.99**2 - 1)


def test_total_loss_cannot_recover():
    values = [-1, 0.5]
    np.testing.assert_equal(wealth_index(values), [1, 0, 0])
    assert cagr(values) == -1
    assert maximum_drawdown(values) == 1


def test_drawdown_is_stable_after_wealth_underflow():
    values = np.concatenate((np.full(1100, -0.5), [0.1]))
    assert maximum_drawdown(values) == 1
    assert cagr(values) == pytest.approx(np.expm1(np.log1p(values).mean() * 252))


def test_drawdown_is_stable_when_wealth_would_overflow():
    values = np.concatenate((np.full(1100, 1.0), [-0.25]))
    assert maximum_drawdown(values) == pytest.approx(0.25)
    with pytest.raises(UndefinedMetricError):
        wealth_index(values)


def test_sample_volatility_and_covariance_agree():
    values = np.array([[0.01, -0.02], [0.03, 0.04], [-0.02, 0.01]])
    weights = np.array([0.4, 0.6])
    covariance = sample_covariance(values)
    expected = np.sqrt(weights @ covariance @ weights * 252)
    actual = annualized_volatility(portfolio_returns(values, weights))
    assert actual == pytest.approx(expected)
    assert annualized_volatility([0.1, -0.1], 2) == pytest.approx(0.2)


def test_singular_covariance_is_not_modified():
    covariance = sample_covariance([[0.1, 0.1], [-0.1, -0.1]])
    np.testing.assert_allclose(covariance, [[0.02, 0.02], [0.02, 0.02]])


def test_one_asset_covariance_is_a_matrix():
    result = sample_covariance([[0.1], [-0.1]])
    assert result.shape == (1, 1)
    assert result[0, 0] == pytest.approx(0.02)


def test_sharpe_with_effective_annual_risk_free_rate():
    # 21% annual effective rate is 10% per period when there are two periods.
    assert sharpe_ratio([0.1, 0.3], 2, 0.21) == pytest.approx(1)
    assert sharpe_ratio([0.1, 0.3], 2) == pytest.approx(2)


@pytest.mark.parametrize("values", [[0, 0], [0.1, 0.1], [0, 1e-14]])
def test_sharpe_rejects_undefined_volatility(values):
    with pytest.raises(UndefinedMetricError, match="zero volatility"):
        sharpe_ratio(values)


@pytest.mark.parametrize(
    "weights", [[50, 50], [-0.1, 1.1], [0.2, 0.2], [1], [float("nan"), 1]]
)
def test_invalid_weights(weights):
    with pytest.raises(DataValidationError):
        portfolio_returns([[0.01, 0.02], [0.02, 0.01]], weights)


@pytest.mark.parametrize("values", [[], [float("nan")], [float("inf")], [-1.1]])
@pytest.mark.parametrize(
    "metric",
    [wealth_index, maximum_drawdown, cagr, annualized_volatility, sharpe_ratio],
)
def test_invalid_returns(metric, values):
    with pytest.raises(DataValidationError):
        metric(values)


@pytest.mark.parametrize("annualization", [0, -1, 2.5, True])
def test_invalid_annualization(annualization):
    with pytest.raises(DataValidationError):
        cagr([0.1, 0.2], annualization)


@pytest.mark.parametrize("metric", [annualized_volatility, sharpe_ratio])
def test_insufficient_observations(metric):
    with pytest.raises(DataValidationError):
        metric([0.01])


@pytest.mark.parametrize("rate", [-1, -2, float("inf"), float("nan")])
def test_invalid_risk_free_rate(rate):
    with pytest.raises(DataValidationError):
        sharpe_ratio([0.01, 0.02], risk_free_rate=rate)


RETURNS = [[0.01, -0.002], [-0.02, 0.004], [0.015, 0.001], [0.003, -0.003]]


def test_covariance_variance_matches_fixed_weight_returns():
    weights = [0.3, 0.7]
    expected = np.var(portfolio_returns(RETURNS, weights), ddof=1)
    variance = portfolio_variance(sample_covariance(RETURNS), weights)
    assert variance == pytest.approx(expected)


def test_risk_shares_sum_to_one():
    shares = risk_shares(sample_covariance(RETURNS), [0.3, 0.7])
    assert shares.sum() == pytest.approx(1)


def test_two_asset_inverse_volatility_equalizes_risk():
    # For two assets, equal risk contribution is exactly inverse volatility,
    # whatever the correlation.
    covariance = sample_covariance(RETURNS)
    inverse = 1 / np.sqrt(np.diag(covariance))
    shares = risk_shares(covariance, inverse / inverse.sum())
    np.testing.assert_allclose(shares, [0.5, 0.5])


def test_risk_shares_undefined_at_zero_variance():
    with pytest.raises(UndefinedMetricError):
        risk_shares([[0.0, 0.0], [0.0, 0.0]], [0.5, 0.5])


@pytest.mark.parametrize(
    "covariance",
    [[[1.0, 0.2], [0.1, 1.0]], [[1.0, 0.0]], [[float("nan"), 0], [0, 1]], [["a"]]],
)
def test_invalid_covariance(covariance):
    with pytest.raises(DataValidationError):
        portfolio_variance(covariance, [0.5, 0.5])


def test_dividend_yield_is_weighted_sum():
    assert portfolio_dividend_yield([0.04, 0.02], [0.5, 0.5]) == pytest.approx(0.03)


@pytest.mark.parametrize("yields", [[0.02, float("nan")], [0.02, -0.01], []])
def test_invalid_dividend_yields(yields):
    with pytest.raises(DataValidationError):
        portfolio_dividend_yield(yields, [0.5, 0.5])
