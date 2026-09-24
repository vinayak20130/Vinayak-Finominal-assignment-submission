"""The market data interface the service depends on.

The only runtime implementation is PostgresMarketData (app/data/postgres.py);
tests use an in-memory fake (tests/fakes.py).
"""

import datetime as dt
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol

import pandas as pd

from app.domain.errors import TickerNotFoundError


@dataclass(frozen=True)
class Security:
    ticker: str
    name: str
    dividend_yield: float | None  # None when the source leaves it blank
    first_date: dt.date
    last_date: dt.date
    observations: int
    since_inception_return: float
    ytd_return: float  # calendar year of last_date


@dataclass(frozen=True)
class AnnualReturn:
    year: int
    total_return: float
    trading_days: int
    first_date: dt.date
    last_date: dt.date
    is_partial: bool


class MarketData(Protocol):
    def securities(self) -> list[Security]: ...

    def lookup(self, tickers: Sequence[str]) -> dict[str, Security]: ...

    def daily_returns(self, tickers: Sequence[str]) -> pd.DataFrame:
        """Date x ticker returns in the requested order; NaN where a fund has none."""
        ...

    def factor_returns(self) -> pd.DataFrame:
        """Date x factor returns with columns momentum, value, size."""
        ...

    def annual_returns(self, ticker: str) -> list[AnnualReturn]: ...


def select_known(
    available: Mapping[str, Security], tickers: Sequence[str]
) -> dict[str, Security]:
    """Return the requested securities, or raise naming every unknown ticker."""
    unknown = [ticker for ticker in tickers if ticker not in available]
    if unknown:
        plural = "s" if len(unknown) > 1 else ""
        raise TickerNotFoundError(
            f"Unknown ticker{plural}: {', '.join(unknown)}. "
            f"Available: {', '.join(sorted(available))}."
        )
    return {ticker: available[ticker] for ticker in tickers}
