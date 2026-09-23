import numpy as np
import pandas as pd
import pytest

from app.data.alignment import align_returns
from app.data.validation import validate_return_table
from app.domain.errors import DataValidationError, InsufficientDataError


@pytest.fixture
def returns():
    return pd.DataFrame(
        [
            ("2025-01-03", "AAA", 0.02),
            ("2025-01-01", "AAA", -0.01),
            ("2025-01-02", "AAA", 0.03),
            ("2025-01-01", "BBB", 0.01),
            ("2025-01-02", "BBB", -0.02),
            ("2025-01-03", "BBB", 0.04),
            ("2025-01-02", "CCC", 0.05),
            ("2025-01-03", "CCC", 0.06),
        ],
        columns=["date", "ticker", "total_return"],
    )


def test_alignment_uses_only_selected_funds_and_preserves_order(returns):
    original = returns.copy(deep=True)
    pair = align_returns(returns, [" bbb ", "aaa"])
    assert list(pair.columns) == ["BBB", "AAA"]
    assert pair.index.strftime("%Y-%m-%d").tolist() == [
        "2025-01-01",
        "2025-01-02",
        "2025-01-03",
    ]
    np.testing.assert_allclose(pair["AAA"], [-0.01, 0.03, 0.02])
    shorter = align_returns(returns, ["AAA", "BBB", "CCC"])
    assert len(shorter) == 2
    pd.testing.assert_frame_equal(returns, original)


def test_date_bounds_are_inclusive(returns):
    result = align_returns(
        returns, ["AAA", "BBB"], start_date="2025-01-02", end_date="2025-01-03"
    )
    assert len(result) == 2
    assert result.index.min() == pd.Timestamp("2025-01-02")


@pytest.mark.parametrize("tickers", [[], ["AAA", "aaa"], ["UNKNOWN"]])
def test_invalid_selection(returns, tickers):
    with pytest.raises(DataValidationError):
        align_returns(returns, tickers)


def test_missing_date_is_not_filled_with_zero(returns):
    returns = returns.drop(index=4)
    result = align_returns(returns, ["AAA", "BBB"])
    assert result.index.strftime("%Y-%m-%d").tolist() == ["2025-01-01", "2025-01-03"]


def test_disjoint_dates_fail(returns):
    returns.loc[returns["ticker"] == "BBB", "date"] = [
        "2024-01-01",
        "2024-01-02",
        "2024-01-03",
    ]
    with pytest.raises(InsufficientDataError):
        align_returns(returns, ["AAA", "BBB"])


def test_too_short_window(returns):
    with pytest.raises(InsufficientDataError):
        align_returns(returns, ["AAA", "BBB"], start_date="2025-01-03")


def test_reversed_window(returns):
    with pytest.raises(DataValidationError):
        align_returns(returns, ["AAA"], start_date="2025-01-03", end_date="2025-01-01")


def test_duplicates_after_ticker_normalization_fail(returns):
    duplicate = returns.iloc[[0]].copy()
    duplicate["ticker"] = " aaa "
    with pytest.raises(DataValidationError, match="Duplicate"):
        validate_return_table(pd.concat([returns, duplicate]))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, -1.01, True, "bad"])
def test_bad_return_values(returns, value):
    returns["total_return"] = returns["total_return"].astype(object)
    returns.loc[0, "total_return"] = value
    with pytest.raises(DataValidationError):
        validate_return_table(returns)


@pytest.mark.parametrize(
    "value",
    [
        None,
        "2025-02-30",
        "20250101",
        "2025-01-01T12:00:00",
        pd.Timestamp("2025-01-01", tz="UTC"),
        45658,
    ],
)
def test_bad_dates(returns, value):
    returns["date"] = returns["date"].astype(object)
    returns.loc[0, "date"] = value
    with pytest.raises(DataValidationError):
        validate_return_table(returns)


def test_missing_column(returns):
    with pytest.raises(DataValidationError, match="Missing columns"):
        validate_return_table(returns.drop(columns="date"))


def test_factor_returns_are_not_given_fund_loss_bounds():
    factors = pd.DataFrame(
        [("2025-01-01", "Value Factor", -1.1)],
        columns=["date", "index_ticker", "total_return"],
    )
    result = validate_return_table(factors, "index_ticker", fund_returns=False)
    assert result.loc[0, "total_return"] == -1.1
