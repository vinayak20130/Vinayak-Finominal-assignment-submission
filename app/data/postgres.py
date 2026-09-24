"""PostgreSQL implementation of MarketData (SQLAlchemy Core)."""

from collections.abc import Sequence

import pandas as pd
from sqlalchemy import func, select
from sqlalchemy.engine import Engine
from sqlalchemy.exc import InterfaceError, OperationalError, ProgrammingError
from sqlalchemy.exc import TimeoutError as PoolTimeoutError

from app.data.market_data import AnnualReturn, Security, select_known
from app.data.tables import (
    factor_daily_returns,
    fund_annual_returns,
    fund_daily_returns,
    securities,
    security_summary,
)
from app.domain.errors import DataUnavailableError
from app.domain.factors import FACTOR_NAMES


class PostgresMarketData:
    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def securities(self) -> list[Security]:
        query = select(security_summary).order_by(security_summary.c.ticker)
        return [Security(**row._mapping) for row in self._fetch(query)]

    def lookup(self, tickers: Sequence[str]) -> dict[str, Security]:
        return select_known({s.ticker: s for s in self.securities()}, tickers)

    def daily_returns(self, tickers: Sequence[str]) -> pd.DataFrame:
        table = fund_daily_returns
        query = select(table.c.trade_date, table.c.ticker, table.c.total_return).where(
            table.c.ticker.in_(list(tickers))
        )
        return _wide(self._fetch(query), list(tickers))

    def factor_returns(self) -> pd.DataFrame:
        table = factor_daily_returns
        query = select(table.c.trade_date, table.c.factor, table.c.total_return)
        return _wide(self._fetch(query), list(FACTOR_NAMES))

    def annual_returns(self, ticker: str) -> list[AnnualReturn]:
        self.lookup([ticker])
        view = fund_annual_returns
        query = (
            select(
                view.c.year,
                view.c.total_return,
                view.c.trading_days,
                view.c.first_date,
                view.c.last_date,
                view.c.is_partial,
            )
            .where(view.c.ticker == ticker)
            .order_by(view.c.year)
        )
        return [AnnualReturn(**row._mapping) for row in self._fetch(query)]

    def count_securities(self) -> int:
        return self._fetch(select(func.count()).select_from(securities))[0][0]

    def _fetch(self, query):
        try:
            with self._engine.connect() as connection:
                return connection.execute(query).all()
        except (OperationalError, InterfaceError, PoolTimeoutError) as exc:
            raise DataUnavailableError(
                "Market data is temporarily unavailable."
            ) from exc


def _wide(rows, columns: list[str]) -> pd.DataFrame:
    """Pivot (date, key, value) rows into a date x key table with these columns."""
    frame = pd.DataFrame(rows, columns=["date", "key", "value"])
    # Pin the resolution: pandas infers seconds from Python dates but microseconds
    # from Excel, and date indexes should compare identically either way.
    frame["date"] = pd.to_datetime(frame["date"]).astype("datetime64[us]")
    wide = frame.pivot(index="date", columns="key", values="value").sort_index()
    return wide.reindex(columns=columns)


def connect_market_data(engine: Engine) -> PostgresMarketData:
    """Fail fast at startup, naming the fix, if the database isn't ready."""
    market = PostgresMarketData(engine)
    where = engine.url.render_as_string(hide_password=True)
    try:
        count = market.count_securities()
    except DataUnavailableError as exc:
        raise RuntimeError(
            f"Cannot reach PostgreSQL at {where}. Start it: docker compose up -d db"
        ) from exc
    except ProgrammingError as exc:
        raise RuntimeError(
            f"Tables are missing in {where}. Run: uv run alembic upgrade head"
        ) from exc
    if count == 0:
        raise RuntimeError(
            f"No market data in {where}. Run: uv run python -m app.data.load"
        )
    return market
