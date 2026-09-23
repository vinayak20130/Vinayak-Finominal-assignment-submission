import numpy as np
from numpy.typing import ArrayLike, NDArray

from app.domain.errors import DataValidationError, UndefinedMetricError

# Below this period standard deviation, Sharpe is numerically undefined.
MIN_PERIOD_VOLATILITY = 1e-12


def _returns(values: ArrayLike, *, matrix: bool = False) -> NDArray[np.float64]:
    try:
        result = np.asarray(values, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise DataValidationError("Returns must be numeric.") from exc
    dimensions = 2 if matrix else 1
    if result.ndim != dimensions or result.size == 0:
        raise DataValidationError(f"Returns must be a nonempty {dimensions}D array.")
    if not np.isfinite(result).all() or (result < -1).any():
        raise DataValidationError("Returns must be finite and at least -100%.")
    return result


def _annualization(value: int) -> int:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or value <= 0
    ):
        raise DataValidationError("Annualization factor must be a positive integer.")
    return int(value)


def _finite(value: float, metric: str) -> float:
    if not np.isfinite(value):
        raise UndefinedMetricError(f"{metric} is outside the finite numeric range.")
    return float(value)


def _weights(weights: ArrayLike, count: int) -> NDArray[np.float64]:
    try:
        allocation = np.asarray(weights, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise DataValidationError("Weights must be numeric.") from exc
    if allocation.ndim != 1 or len(allocation) != count:
        raise DataValidationError("Provide one weight per return column.")
    if (
        not np.isfinite(allocation).all()
        or (allocation < 0).any()
        or (allocation > 1).any()
        or not np.isclose(allocation.sum(), 1, atol=1e-8, rtol=0)
    ):
        raise DataValidationError("Weights must be nonnegative fractions summing to 1.")
    return allocation


def _covariance(covariance: ArrayLike) -> NDArray[np.float64]:
    try:
        matrix = np.asarray(covariance, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise DataValidationError("Covariance must be numeric.") from exc
    if matrix.ndim != 2 or matrix.size == 0 or matrix.shape[0] != matrix.shape[1]:
        raise DataValidationError("Covariance must be a nonempty square matrix.")
    if not np.isfinite(matrix).all() or not np.allclose(matrix, matrix.T):
        raise DataValidationError("Covariance must be finite and symmetric.")
    tolerance = 10 * np.finfo(float).eps * len(matrix) * np.linalg.norm(matrix, 2)
    if np.linalg.eigvalsh(matrix).min() < -tolerance:
        raise DataValidationError("Covariance must be positive semidefinite.")
    return matrix


def portfolio_returns(returns: ArrayLike, weights: ArrayLike) -> NDArray[np.float64]:
    """Fixed target weights rebalanced at every observation; weights are fractions."""
    values = _returns(returns, matrix=True)
    allocation = _weights(weights, values.shape[1])
    with np.errstate(over="ignore", invalid="ignore"):
        result = values @ allocation
    if not np.isfinite(result).all():
        raise UndefinedMetricError("Portfolio returns exceed the finite numeric range.")
    return result


def sample_covariance(returns: ArrayLike) -> NDArray[np.float64]:
    values = _returns(returns, matrix=True)
    if len(values) < 2:
        raise DataValidationError("Covariance requires at least two observations.")
    with np.errstate(over="ignore", invalid="ignore"):
        covariance = np.atleast_2d(np.cov(values, rowvar=False, ddof=1))
    if not np.isfinite(covariance).all():
        raise UndefinedMetricError("Covariance exceeds the finite numeric range.")
    return covariance


def annualized_volatility(returns: ArrayLike, annualization_factor: int = 252) -> float:
    values = _returns(returns)
    annualization = _annualization(annualization_factor)
    if len(values) < 2:
        raise DataValidationError("Volatility requires at least two observations.")
    with np.errstate(over="ignore", invalid="ignore"):
        volatility = np.std(values, ddof=1) * np.sqrt(annualization)
    return _finite(volatility, "Volatility")


def _log_wealth(returns: ArrayLike) -> NDArray[np.float64]:
    values = _returns(returns)
    # log(0) = -inf represents a total loss that cannot subsequently recover.
    with np.errstate(divide="ignore"):
        return np.concatenate(([0.0], np.cumsum(np.log1p(values))))


def wealth_index(returns: ArrayLike) -> NDArray[np.float64]:
    """Include initial wealth of one before the first return."""
    with np.errstate(over="ignore"):
        wealth = np.exp(_log_wealth(returns))
    if not np.isfinite(wealth).all():
        raise UndefinedMetricError("Wealth exceeds the finite numeric range.")
    return wealth


def maximum_drawdown(returns: ArrayLike) -> float:
    log_wealth = _log_wealth(returns)
    log_peaks = np.maximum.accumulate(log_wealth)
    # Log differences avoid overflow/underflow in wealth-to-peak ratios.
    drawdowns = -np.expm1(log_wealth - log_peaks)
    return _finite(np.max(drawdowns), "Maximum drawdown")


def cagr(returns: ArrayLike, annualization_factor: int = 252) -> float:
    """Annualize geometric growth using observation count / annualization factor."""
    annualization = _annualization(annualization_factor)
    log_wealth = _log_wealth(returns)
    with np.errstate(over="ignore"):
        growth = np.expm1(log_wealth[-1] * annualization / (len(log_wealth) - 1))
    return _finite(growth, "CAGR")


def sharpe_ratio(
    returns: ArrayLike,
    annualization_factor: int = 252,
    risk_free_rate: float = 0,
) -> float:
    values = _returns(returns)
    annualization = _annualization(annualization_factor)
    if not np.isfinite(risk_free_rate) or risk_free_rate <= -1:
        raise DataValidationError("Annual risk-free rate must be finite and above -1.")
    period_volatility = annualized_volatility(values, annualization) / np.sqrt(
        annualization
    )
    if period_volatility <= MIN_PERIOD_VOLATILITY:
        raise UndefinedMetricError("Sharpe ratio is undefined at zero volatility.")
    period_risk_free = np.expm1(np.log1p(risk_free_rate) / annualization)
    ratio = (values.mean() - period_risk_free) / period_volatility
    return _finite(ratio * np.sqrt(annualization), "Sharpe ratio")


def portfolio_variance(covariance: ArrayLike, weights: ArrayLike) -> float:
    """Fixed-weight portfolio variance, in the same units as the covariance."""
    matrix = _covariance(covariance)
    allocation = _weights(weights, len(matrix))
    # A valid covariance gives a nonnegative result; clip float rounding below zero.
    return _finite(max(allocation @ matrix @ allocation, 0.0), "Portfolio variance")


def risk_shares(covariance: ArrayLike, weights: ArrayLike) -> NDArray[np.float64]:
    """Fraction of portfolio variance contributed by each asset; shares sum to 1."""
    matrix = _covariance(covariance)
    allocation = _weights(weights, len(matrix))
    variance = portfolio_variance(matrix, allocation)
    if variance <= MIN_PERIOD_VOLATILITY**2:
        raise UndefinedMetricError("Risk shares are undefined at zero variance.")
    return allocation * (matrix @ allocation) / variance


def portfolio_dividend_yield(yields: ArrayLike, weights: ArrayLike) -> float:
    """Weighted yield of target weights; every yield must be a known decimal."""
    try:
        values = np.asarray(yields, dtype=np.float64)
    except (ValueError, TypeError) as exc:
        raise DataValidationError("Dividend yields must be numeric.") from exc
    if values.ndim != 1 or values.size == 0:
        raise DataValidationError("Provide a nonempty list of dividend yields.")
    # NaN marks a missing yield; the caller resolves its policy before this point.
    if not np.isfinite(values).all() or (values < 0).any():
        raise DataValidationError("Dividend yields must be known and nonnegative.")
    return float(values @ _weights(weights, len(values)))
