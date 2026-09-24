import datetime as dt

import pandas as pd

from app.data.load import build_rows
from app.data.workbook import WorkbookData


def workbook() -> WorkbookData:
    return WorkbookData(
        funds=pd.DataFrame(
            {
                "ticker": ["AAA", "GLD"],
                "fund_name": ["Asset A", "Gold"],
                "dividend_yield": [0.02, float("nan")],
            }
        ),
        fund_returns=pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-02", "2025-01-02"]),
                "ticker": ["AAA", "GLD"],
                "total_return": [0.01, -0.02],
            }
        ),
        factor_returns=pd.DataFrame(
            {
                "date": pd.to_datetime(["2025-01-02"]),
                "index_ticker": ["momentum"],
                "total_return": [0.001],
            }
        ),
    )


def test_rows_keep_blank_yield_as_null_and_use_plain_python_values():
    rows = build_rows(workbook())

    assert rows.securities == [
        {"ticker": "AAA", "name": "Asset A", "dividend_yield": 0.02},
        {"ticker": "GLD", "name": "Gold", "dividend_yield": None},
    ]
    assert rows.fund_returns[0] == {
        "ticker": "AAA",
        "trade_date": dt.date(2025, 1, 2),
        "total_return": 0.01,
    }
    assert type(rows.fund_returns[0]["total_return"]) is float
    assert rows.factor_returns == [
        {"factor": "momentum", "trade_date": dt.date(2025, 1, 2), "total_return": 0.001}
    ]
