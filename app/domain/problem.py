"""Framework-free description of one optimization request and its result."""

from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from app.domain.metrics import sample_covariance


@dataclass(frozen=True)
class PortfolioConstraints:
    """Portfolio-level limits as decimals; None means not requested."""

    min_cagr: float | None = None
    min_volatility: float | None = None
    max_volatility: float | None = None
    max_drawdown: float | None = None
    min_dividend_yield: float | None = None


@dataclass(frozen=True)
class OptimizationProblem:
    """Aligned inputs for a strategy. All weights and rates are decimal fractions.

    `returns` has one row per common date and one column per ticker, in request
    order. `yields` holds NaN for a missing yield the caller chose not to resolve.
    """

    tickers: tuple[str, ...]
    returns: NDArray[np.float64]
    current_weights: NDArray[np.float64]
    lower_bounds: NDArray[np.float64]
    upper_bounds: NDArray[np.float64]
    yields: NDArray[np.float64]
    constraints: PortfolioConstraints
    annualization_factor: int = 252
    risk_free_rate: float = 0.0
    factor_costs: NDArray[np.float64] | None = None

    @property
    def size(self) -> int:
        return len(self.tickers)

    @property
    def yields_known(self) -> bool:
        return bool(np.isfinite(self.yields).all())

    @cached_property
    def covariance(self) -> NDArray[np.float64]:
        """Sample covariance of period returns (not annualized)."""
        return sample_covariance(self.returns)


@dataclass(frozen=True)
class StrategyResult:
    weights: NDArray[np.float64]
    method: str
    status: str = "optimal"
    warnings: list[str] = field(default_factory=list)
