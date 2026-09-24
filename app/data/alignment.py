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
    """Validate a long-form return table, then align it with `align_table`."""
    table = validate_return_table(returns)
    wide = table.pivot(index="date", columns="ticker", values="total_return")
    return align_table(wide, tickers, start_date=start_date, end_date=end_date)


def align_table(
    table: pd.DataFrame,
    tickers: Sequence[str],
    *,
    start_date: date | str | None = None,
    end_date: date | str | None = None,
) -> pd.DataFrame:
    """Common dates of the selected tickers from a wide date x ticker table.

    A date counts only when every selected fund has a return on it; missing
    returns are never filled. Columns follow the requested ticker order.
    """
    selected = [normalize_identifier(ticker) for ticker in tickers]
    if not selected or len(selected) != len(set(selected)):
        raise DataValidationError("Select unique, nonempty tickers.")
    start = parse_date(start_date) if start_date is not None else None
    end = parse_date(end_date) if end_date is not None else None
    if start is not None and end is not None and start > end:
        raise DataValidationError("Start date must not follow end date.")
    unknown = set(selected) - set(table.columns)
    if unknown:
        raise DataValidationError(f"Unknown tickers: {', '.join(sorted(unknown))}")

    aligned = table[selected].sort_index()
    if start is not None:
        aligned = aligned[aligned.index >= start]
    if end is not None:
        aligned = aligned[aligned.index <= end]
    aligned = aligned.dropna()
    if len(aligned) < 2:
        raise InsufficientDataError("At least two common return dates are required.")
    return aligned.copy()
