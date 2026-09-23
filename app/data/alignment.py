from collections.abc import Sequence
from datetime import date

import pandas as pd

from app.data.validation import normalize_identifier, parse_date, validate_return_table
from app.domain.errors import DataValidationError, InsufficientDataError


def align_returns(
    returns: pd.DataFrame,
    tickers: Sequence[str],
    *,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
) -> pd.DataFrame:
    """Return sorted common dates, with columns in the requested ticker order."""
    selected = [normalize_identifier(ticker) for ticker in tickers]
    if not selected or len(selected) != len(set(selected)):
        raise DataValidationError("Select unique, nonempty tickers.")
    start = parse_date(start_date) if start_date is not None else None
    end = parse_date(end_date) if end_date is not None else None
    if start is not None and end is not None and start > end:
        raise DataValidationError("Start date must not follow end date.")

    table = validate_return_table(returns)
    unknown = set(selected) - set(table["ticker"])
    if unknown:
        raise DataValidationError(f"Unknown tickers: {', '.join(sorted(unknown))}")
    table = table[table["ticker"].isin(selected)]
    if start is not None:
        table = table[table["date"] >= start]
    if end is not None:
        table = table[table["date"] <= end]
    aligned = (
        table.pivot(index="date", columns="ticker", values="total_return")
        .reindex(columns=selected)
        .dropna()
        .sort_index()
    )
    if len(aligned) < 2:
        raise InsufficientDataError("At least two common return dates are required.")
    return aligned.copy()
