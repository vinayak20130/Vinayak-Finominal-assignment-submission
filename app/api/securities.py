from typing import Annotated

from fastapi import APIRouter, Depends

from app.api.dependencies import get_market_data
from app.data.market_data import MarketData
from app.schemas.securities import AnnualReturnRow, SecuritySummary

router = APIRouter()
Market = Annotated[MarketData, Depends(get_market_data)]


@router.get("/securities", response_model=list[SecuritySummary])
def list_securities(market: Market) -> list[SecuritySummary]:
    """Available tickers with their history range and compounded returns."""
    return [
        SecuritySummary(
            ticker=s.ticker,
            security_name=s.name,
            dividend_yield=s.dividend_yield,
            first_date=s.first_date,
            last_date=s.last_date,
            observations=s.observations,
            since_inception_return=s.since_inception_return,
            ytd_return=s.ytd_return,
        )
        for s in market.securities()
    ]


@router.get("/securities/{ticker}/annual-returns", response_model=list[AnnualReturnRow])
def annual_returns(ticker: str, market: Market) -> list[AnnualReturnRow]:
    """Calendar-year returns compounded from daily data; partial years are flagged."""
    rows = market.annual_returns(ticker.strip().upper())
    return [AnnualReturnRow(**vars(row)) for row in rows]
