from datetime import date, datetime

import numpy as np
import pandas as pd

from app.domain.errors import DataValidationError


def require_columns(frame: pd.DataFrame, columns: set[str]) -> None:
    missing = columns - set(frame.columns)
    if missing:
        raise DataValidationError(f"Missing columns: {', '.join(sorted(missing))}")


def normalize_identifier(value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataValidationError("Identifiers must be nonempty strings.")
    return value.strip().upper()


def parse_date(value) -> pd.Timestamp:
    """Accept calendar dates without silently discarding time or timezone."""
    if isinstance(value, str):
        try:
            parsed = date.fromisoformat(value)
        except ValueError as exc:
            raise DataValidationError("Dates must use YYYY-MM-DD.") from exc
        if parsed.isoformat() != value:
            raise DataValidationError("Dates must use YYYY-MM-DD.")
        value = parsed
    if not isinstance(value, (date, datetime, pd.Timestamp)) or pd.isna(value):
        raise DataValidationError("Dates must be nonmissing calendar dates.")
    result = pd.Timestamp(value)
    if result.tzinfo is not None or result != result.normalize():
        raise DataValidationError("Dates must not contain times or timezones.")
    return result


def validate_return_table(
    frame: pd.DataFrame, identifier: str = "ticker", *, fund_returns: bool = True
) -> pd.DataFrame:
    """Validate and copy a long-form return table; preserve decimal returns."""
    require_columns(frame, {"date", identifier, "total_return"})
    result = frame[["date", identifier, "total_return"]].copy()
    if result.empty:
        raise DataValidationError("Return data must not be empty.")
    result[identifier] = result[identifier].map(normalize_identifier)
    result["date"] = pd.to_datetime(result["date"].map(parse_date))
    if result.duplicated([identifier, "date"]).any():
        raise DataValidationError("Duplicate dates within a return series.")
    raw = result["total_return"]
    if raw.map(lambda value: isinstance(value, (bool, np.bool_))).any():
        raise DataValidationError("Returns must be finite numbers, not booleans.")
    try:
        values = pd.to_numeric(raw, errors="raise").to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DataValidationError("Returns must be finite numbers.") from exc
    if not np.isfinite(values).all():
        raise DataValidationError("Returns must be finite numbers.")
    if fund_returns and (values < -1).any():
        raise DataValidationError("Fund returns must not be below -100%.")
    result["total_return"] = values
    return result.sort_values([identifier, "date"]).reset_index(drop=True)
