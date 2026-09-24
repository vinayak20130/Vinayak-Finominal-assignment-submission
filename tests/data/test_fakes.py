import numpy as np
import pandas as pd
import pytest

from app.domain.errors import TickerNotFoundError
from tests.fakes import FakeMarketData


@pytest.fixture
def market() -> FakeMarketData:
    dates = pd.to_datetime(["2024-12-31", "2025-01-02", "2025-12-30"])
    returns = pd.DataFrame(
        {"AAA": [0.10, -0.05, 0.02], "BBB": [np.nan, 0.01, 0.01]}, index=dates
    )
    funds = pd.DataFrame(
        {"name": ["Asset A", "Asset B"], "dividend_yield": [0.02, np.nan]},
        index=["AAA", "BBB"],
    )
    return FakeMarketData(funds, returns)


def test_summary_compounds_and_keeps_blank_yield(market):
    aaa, bbb = market.securities()

    assert aaa.since_inception_return == pytest.approx(1.10 * 0.95 * 1.02 - 1)
    assert aaa.ytd_return == pytest.approx(0.95 * 1.02 - 1)
    assert (aaa.observations, bbb.observations) == (3, 2)
    assert bbb.dividend_yield is None


def test_annual_returns_flag_partial_years(market):
    rows = market.annual_returns("AAA")

    assert [(row.year, row.is_partial) for row in rows] == [(2024, True), (2025, False)]
    assert rows[1].total_return == pytest.approx(0.95 * 1.02 - 1)


def test_unknown_ticker_names_the_available_ones(market):
    with pytest.raises(TickerNotFoundError, match="XYZ.*Available: AAA, BBB"):
        market.lookup(["AAA", "XYZ"])


def test_daily_returns_follow_the_requested_order(market):
    assert list(market.daily_returns(["BBB", "AAA"]).columns) == ["BBB", "AAA"]
