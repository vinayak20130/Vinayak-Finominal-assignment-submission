"""Response models for the /securities endpoints. Returns are decimals."""

import datetime as dt

from pydantic import BaseModel


class SecuritySummary(BaseModel):
    ticker: str
    security_name: str
    dividend_yield: float | None
    first_date: dt.date
    last_date: dt.date
    observations: int
    since_inception_return: float
    ytd_return: float


class AnnualReturnRow(BaseModel):
    year: int
    total_return: float
    trading_days: int
    first_date: dt.date
    last_date: dt.date
    is_partial: bool
