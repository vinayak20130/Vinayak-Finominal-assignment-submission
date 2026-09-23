import numpy as np
from numpy.typing import ArrayLike, NDArray

from app.domain.errors import DataValidationError, InsufficientDataError

FACTOR_NAMES = ("momentum", "value", "size")


def regress_factors(returns: ArrayLike, factors: ArrayLike) -> NDArray[np.float64]:
    """OLS coefficients in intercept, Momentum, Value, Size order."""
    try:
        observations = np.asarray(returns, dtype=np.float64)
        factor_values = np.asarray(factors, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DataValidationError("Factor regression inputs must be numeric.") from exc
    if (
        observations.ndim not in (1, 2)
        or factor_values.ndim != 2
        or factor_values.shape[1] != 3
        or observations.shape[0] != factor_values.shape[0]
        or observations.size == 0
    ):
        raise DataValidationError("Provide aligned returns and three factor columns.")
    if len(observations) < 5:
        raise InsufficientDataError("Factor regression requires five common dates.")
    if not np.isfinite(observations).all() or not np.isfinite(factor_values).all():
        raise DataValidationError("Factor regression inputs must be finite.")
    design = np.column_stack((np.ones(len(observations)), factor_values))
    try:
        coefficients, _, rank, _ = np.linalg.lstsq(design, observations, rcond=None)
    except np.linalg.LinAlgError as exc:
        raise DataValidationError("Factor regression did not converge.") from exc
    if rank != 4:
        raise DataValidationError("Factor model is not identifiable: deficient rank.")
    if not np.isfinite(coefficients).all():
        raise DataValidationError("Factor regression produced nonfinite coefficients.")
    return coefficients
