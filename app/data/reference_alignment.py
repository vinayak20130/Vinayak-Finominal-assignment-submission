"""Explicit price-path convention used for reference-tool comparisons."""

import pandas as pd

from app.domain.errors import InsufficientDataError


def align_reference_table(table, tickers, *, start_date=None, end_date=None):
    selected = table[tickers].sort_index()
    first = [selected[ticker].first_valid_index() for ticker in tickers]
    last = [selected[ticker].last_valid_index() for ticker in tickers]
    if any(date is None for date in first + last):
        raise InsufficientDataError("Every security requires return history.")
    start, end = max(first), min(last)
    if start_date is not None:
        start = max(start, pd.Timestamp(start_date))
    if end_date is not None:
        end = min(end, pd.Timestamp(end_date))
    selected = selected.loc[start:end].dropna(how="all")
    if len(selected) < 3:
        raise InsufficientDataError(
            "Reference profile requires an initial price date and two return dates."
        )
    initial_date = selected.index[0].date()
    # Forward-filling an unobserved price implies a zero daily return. Never
    # extrapolate before inception or after the last available observation.
    result = selected.iloc[1:].fillna(0).copy()
    result.attrs["start_date"] = initial_date
    return result
