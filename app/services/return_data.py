"""Resolve caller-supplied returns without changing the stored market data."""

import pandas as pd

from app.data.market_data import MarketData
from app.schemas.request import OptimizationRequest


def request_returns(request: OptimizationRequest, market: MarketData) -> pd.DataFrame:
    tickers = [security.ticker for security in request.securities]
    if request.securities[0].returns is None:
        return market.daily_returns(tickers)

    # Request validation requires every holding to supply unique dated returns.
    return pd.DataFrame(
        {
            security.ticker: pd.Series(
                {pd.Timestamp(row.date): row.value for row in security.returns or []},
                dtype=float,
            )
            for security in request.securities
        }
    )
