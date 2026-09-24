"""In-memory MarketData for tests; mirrors the SQL views' rules in pandas."""

from collections.abc import Sequence

import numpy as np
import pandas as pd

from app.data.market_data import AnnualReturn, Security, select_known
from app.data.workbook import WorkbookData
from app.domain.factors import FACTOR_NAMES


def compounded(returns: pd.Series) -> float:
    return float(np.prod(1 + returns.to_numpy()) - 1)


class FakeMarketData:
    def __init__(
        self,
        funds: pd.DataFrame,
        returns: pd.DataFrame,
        factors: pd.DataFrame | None = None,
    ) -> None:
        """funds: index ticker, columns name and dividend_yield (NaN = blank).
        returns: date x ticker. factors: date x momentum/value/size."""
        self.funds = funds
        self.returns = returns.sort_index()
        empty = pd.DataFrame(columns=list(FACTOR_NAMES), dtype=float)
        self.factors = (empty if factors is None else factors).sort_index()

    @classmethod
    def from_workbook(cls, data: WorkbookData) -> "FakeMarketData":
        funds = data.funds.rename(columns={"fund_name": "name"}).set_index("ticker")
        returns = data.fund_returns.pivot(
            index="date", columns="ticker", values="total_return"
        )
        factors = data.factor_returns.pivot(
            index="date", columns="index_ticker", values="total_return"
        )
        return cls(funds, returns, factors.reindex(columns=list(FACTOR_NAMES)))

    def securities(self) -> list[Security]:
        result = []
        for ticker, fund in self.funds.sort_index().iterrows():
            series = self.returns[ticker].dropna()
            last = series.index.max()
            yield_value = fund["dividend_yield"]
            result.append(
                Security(
                    ticker=ticker,
                    name=fund["name"],
                    dividend_yield=None if pd.isna(yield_value) else float(yield_value),
                    first_date=series.index.min().date(),
                    last_date=last.date(),
                    observations=len(series),
                    since_inception_return=compounded(series),
                    ytd_return=compounded(series[series.index.year == last.year]),
                )
            )
        return result

    def lookup(self, tickers: Sequence[str]) -> dict[str, Security]:
        return select_known({s.ticker: s for s in self.securities()}, tickers)

    def daily_returns(self, tickers: Sequence[str]) -> pd.DataFrame:
        return self.returns.reindex(columns=list(tickers))

    def factor_returns(self) -> pd.DataFrame:
        return self.factors.reindex(columns=list(FACTOR_NAMES))

    def annual_returns(self, ticker: str) -> list[AnnualReturn]:
        self.lookup([ticker])
        series = self.returns[ticker].dropna()
        rows = []
        for year, values in series.groupby(series.index.year):
            first, last = values.index.min(), values.index.max()
            rows.append(
                AnnualReturn(
                    year=int(year),
                    total_return=compounded(values),
                    trading_days=len(values),
                    first_date=first.date(),
                    last_date=last.date(),
                    is_partial=not (
                        first <= pd.Timestamp(year, 1, 8)
                        and last >= pd.Timestamp(year, 12, 24)
                    ),
                )
            )
        return rows
