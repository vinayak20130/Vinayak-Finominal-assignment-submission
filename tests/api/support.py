"""Helpers that let API tests describe each security's data next to its request.

Test bodies keep an inline style (returns, security_name, dividend_yield and
factor_returns beside each ticker). `post` moves that data into a FakeMarketData
and sends the ticker-only request the API actually accepts.
"""

import copy

import pandas as pd
from fastapi.testclient import TestClient

from app.main import create_app
from tests.fakes import FakeMarketData

DATA_FIELDS = ("returns", "security_name", "dividend_yield")
FACTORS = ("momentum", "value", "size")


def split(body: dict) -> tuple[FakeMarketData, dict]:
    """Return (market data from the body's inline data, ticker-only request)."""
    request = copy.deepcopy(body)
    funds: dict[str, dict] = {}
    series: dict[str, pd.Series] = {}
    for security in request.get("securities", []):
        data = {field: security.pop(field, None) for field in DATA_FIELDS}
        ticker = str(security.get("ticker", "")).strip().upper()
        if ticker in funds:
            continue  # the request's duplicate-ticker check reports this
        funds[ticker] = {
            "name": data["security_name"] or ticker,
            "dividend_yield": data["dividend_yield"],
        }
        series[ticker] = pd.Series(
            {pd.Timestamp(row["date"]): row["return"] for row in data["returns"] or []},
            dtype=float,
        )
    factor_rows = request.pop("factor_returns", None) or []
    factors = pd.DataFrame(
        [
            {"date": pd.Timestamp(row["date"]), **{name: row[name] for name in FACTORS}}
            for row in factor_rows
        ],
        columns=["date", *FACTORS],
    ).set_index("date")
    fund_table = pd.DataFrame.from_dict(
        funds, orient="index", columns=["name", "dividend_yield"]
    ).astype({"dividend_yield": float})
    return FakeMarketData(fund_table, pd.DataFrame(series), factors), request


def client(market: FakeMarketData | None = None) -> TestClient:
    """A TestClient for an app wired to `market` (an empty fake by default)."""
    if market is None:
        market = split({"securities": []})[0]
    return TestClient(create_app(market))


def post(body: dict):
    market, request = split(body)
    with client(market) as api:
        return api.post("/optimize", json=request)
