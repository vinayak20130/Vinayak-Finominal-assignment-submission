from dataclasses import dataclass
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
import pandas as pd

from app.data.validation import (
    normalize_identifier,
    require_columns,
    validate_return_table,
)
from app.domain.errors import DataValidationError

FACTOR_NAMES = {
    "MOMENTUM FACTOR": "momentum",
    "VALUE FACTOR": "value",
    "SIZE FACTOR": "size",
}


@dataclass
class WorkbookData:
    """Tables owned by the caller; each load returns independent data."""

    funds: pd.DataFrame
    fund_returns: pd.DataFrame
    factor_returns: pd.DataFrame


def _validate_funds(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(frame, {"ticker", "fund_name", "dividend_yield"})
    result = frame[["ticker", "fund_name", "dividend_yield"]].copy()
    if result.empty:
        raise DataValidationError("Fund information must not be empty.")
    result["ticker"] = result["ticker"].map(normalize_identifier)
    if result["ticker"].duplicated().any():
        raise DataValidationError("Duplicate tickers in fund information.")
    if (
        not result["fund_name"]
        .map(lambda value: isinstance(value, str) and bool(value.strip()))
        .all()
    ):
        raise DataValidationError("Each fund must have a security name.")
    result["fund_name"] = result["fund_name"].str.strip()
    supplied = result["dividend_yield"].dropna()
    if supplied.map(lambda value: isinstance(value, (bool, np.bool_))).any():
        raise DataValidationError("Dividend yields must be nonnegative numbers.")
    try:
        yields = pd.to_numeric(result["dividend_yield"], errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(
            "Dividend yields must be numeric or missing."
        ) from exc
    known = yields.dropna()
    if not np.isfinite(known).all() or (known < 0).any():
        raise DataValidationError("Dividend yields must be finite and nonnegative.")
    result["dividend_yield"] = yields
    return result


def load_workbook(path: str | Path) -> WorkbookData:
    """Load explicitly, never at import time or through the health endpoint."""
    try:
        with pd.ExcelFile(path, engine="openpyxl") as workbook:
            expected = {"Fund Info", "Fund Returns", "Factor Returns"}
            missing = expected - set(workbook.sheet_names)
            if missing:
                raise DataValidationError(
                    f"Missing workbook sheets: {', '.join(sorted(missing))}"
                )
            funds = _validate_funds(workbook.parse("Fund Info"))
            returns = validate_return_table(workbook.parse("Fund Returns"))
            factors = validate_return_table(
                workbook.parse("Factor Returns"), "index_ticker", fund_returns=False
            )
    except (OSError, BadZipFile) as exc:
        raise DataValidationError(f"Cannot read workbook: {Path(path).name}") from exc

    if set(funds["ticker"]) != set(returns["ticker"]):
        raise DataValidationError("Fund information and return tickers must match.")
    if set(factors["index_ticker"]) != set(FACTOR_NAMES):
        raise DataValidationError("Expected Momentum, Value, and Size factor series.")
    factors["index_ticker"] = factors["index_ticker"].map(FACTOR_NAMES)
    return WorkbookData(funds, returns, factors)
