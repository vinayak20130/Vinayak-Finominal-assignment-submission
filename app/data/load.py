"""Load the supplied workbook into PostgreSQL.

Usage: uv run python -m app.data.load [path/to/Data.xlsx]

The workbook is validated completely before the database is touched, then written
in one transaction: readers see the old data or the new, never a mix. The workbook
is the complete universe, so securities missing from it are removed.
"""

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sqlalchemy import delete, func, insert, select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.engine import Engine

from app.data.database import create_database_engine
from app.data.tables import (
    data_loads,
    factor_daily_returns,
    fund_daily_returns,
    securities,
)
from app.data.workbook import WorkbookData, load_workbook

DEFAULT_WORKBOOK = Path(__file__).resolve().parents[2] / "Data.xlsx"
# Arbitrary constant identifying market-data loads; concurrent loads queue on it.
LOAD_LOCK_ID = 7_451_001


@dataclass(frozen=True)
class LoadRows:
    securities: list[dict]
    fund_returns: list[dict]
    factor_returns: list[dict]


def build_rows(data: WorkbookData) -> LoadRows:
    """Convert validated workbook tables into plain database rows."""
    return LoadRows(
        securities=[
            {
                "ticker": row.ticker,
                "name": row.fund_name,
                # A blank yield stays NULL; the API decides how to treat it.
                "dividend_yield": (
                    None if pd.isna(row.dividend_yield) else float(row.dividend_yield)
                ),
            }
            for row in data.funds.itertuples(index=False)
        ],
        fund_returns=[
            {
                "ticker": row.ticker,
                "trade_date": row.date.date(),
                "total_return": float(row.total_return),
            }
            for row in data.fund_returns.itertuples(index=False)
        ],
        factor_returns=[
            {
                "factor": row.index_ticker,
                "trade_date": row.date.date(),
                "total_return": float(row.total_return),
            }
            for row in data.factor_returns.itertuples(index=False)
        ],
    )


def load(engine: Engine, path: Path) -> LoadRows:
    rows = build_rows(load_workbook(path))
    tickers = [row["ticker"] for row in rows.securities]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    with engine.begin() as connection:
        connection.execute(select(func.pg_advisory_xact_lock(LOAD_LOCK_ID)))
        upsert = pg_insert(securities).values(rows.securities)
        connection.execute(
            upsert.on_conflict_do_update(
                index_elements=[securities.c.ticker],
                set_={
                    "name": upsert.excluded.name,
                    "dividend_yield": upsert.excluded.dividend_yield,
                },
            )
        )
        connection.execute(
            delete(securities).where(securities.c.ticker.not_in(tickers))
        )
        connection.execute(
            delete(fund_daily_returns).where(fund_daily_returns.c.ticker.in_(tickers))
        )
        connection.execute(insert(fund_daily_returns), rows.fund_returns)
        connection.execute(delete(factor_daily_returns))
        connection.execute(insert(factor_daily_returns), rows.factor_returns)
        connection.execute(
            insert(data_loads).values(
                source_file=path.name,
                sha256=digest,
                fund_rows=len(rows.fund_returns),
                factor_rows=len(rows.factor_returns),
            )
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_WORKBOOK)
    args = parser.parse_args()
    engine = create_database_engine()
    try:
        rows = load(engine, args.path)
    finally:
        engine.dispose()
    print(
        f"Loaded {len(rows.securities)} securities, {len(rows.fund_returns)} fund "
        f"returns and {len(rows.factor_returns)} factor returns from {args.path.name}."
    )


if __name__ == "__main__":
    main()
