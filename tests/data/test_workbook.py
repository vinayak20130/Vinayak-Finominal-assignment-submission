import pandas as pd
import pytest

from app.data.workbook import load_workbook
from app.domain.errors import DataValidationError


@pytest.fixture
def sheets():
    return {
        "Fund Info": pd.DataFrame(
            [("AAA", "Asset A", 0.025), ("GLD", "Gold", None)],
            columns=["ticker", "fund_name", "dividend_yield"],
        ),
        "Fund Returns": pd.DataFrame(
            [("2025-01-02", "AAA", 0.01), ("2025-01-02", "GLD", -0.02)],
            columns=["date", "ticker", "total_return"],
        ),
        "Factor Returns": pd.DataFrame(
            [
                ("2025-01-02", factor, 0.001)
                for factor in ["Momentum Factor", "Value Factor", "Size Factor"]
            ],
            columns=["date", "index_ticker", "total_return"],
        ),
    }


@pytest.fixture
def workbook_reader(monkeypatch, sheets):
    class ExcelFile:
        def __init__(self, *args, **kwargs):
            self.sheet_names = list(sheets)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def parse(self, name):
            return sheets[name].copy(deep=True)

    monkeypatch.setattr(pd, "ExcelFile", ExcelFile)


def test_load_preserves_missing_yield_and_decimal_returns(workbook_reader):
    result = load_workbook("fixture.xlsx")
    assert pd.isna(result.funds.loc[1, "dividend_yield"])
    assert result.funds.loc[0, "dividend_yield"] == 0.025
    assert result.fund_returns.loc[0, "total_return"] == 0.01
    assert set(result.factor_returns["index_ticker"]) == {"momentum", "value", "size"}


def test_each_load_owns_its_data(workbook_reader):
    first = load_workbook("fixture.xlsx")
    second = load_workbook("fixture.xlsx")
    first.funds.loc[0, "dividend_yield"] = 9
    assert second.funds.loc[0, "dividend_yield"] == 0.025


@pytest.mark.parametrize("sheet", ["Fund Info", "Fund Returns", "Factor Returns"])
def test_missing_sheet(workbook_reader, sheets, sheet):
    del sheets[sheet]
    with pytest.raises(DataValidationError, match="Missing workbook sheets"):
        load_workbook("fixture.xlsx")


@pytest.mark.parametrize("value", [-0.01, float("inf"), "bad", True])
def test_invalid_yield(workbook_reader, sheets, value):
    sheets["Fund Info"]["dividend_yield"] = sheets["Fund Info"][
        "dividend_yield"
    ].astype(object)
    sheets["Fund Info"].loc[0, "dividend_yield"] = value
    with pytest.raises(DataValidationError, match="Dividend yields"):
        load_workbook("fixture.xlsx")


def test_unknown_fund(workbook_reader, sheets):
    sheets["Fund Returns"].loc[0, "ticker"] = "UNKNOWN"
    with pytest.raises(DataValidationError, match="tickers must match"):
        load_workbook("fixture.xlsx")


def test_unknown_factor(workbook_reader, sheets):
    sheets["Factor Returns"].loc[0, "index_ticker"] = "Other Factor"
    with pytest.raises(DataValidationError, match="factor series"):
        load_workbook("fixture.xlsx")


def test_missing_file(tmp_path):
    with pytest.raises(DataValidationError, match="Cannot read workbook"):
        load_workbook(tmp_path / "missing.xlsx")


def test_invalid_excel_file(tmp_path):
    path = tmp_path / "invalid.xlsx"
    path.write_bytes(b"not an Excel file")
    with pytest.raises(DataValidationError, match="Cannot read workbook"):
        load_workbook(path)
