"""Response models for POST /optimize.

Weights and changes are percentages at full precision; metrics are decimals.
"""

import datetime as dt

from pydantic import BaseModel

from app.schemas.request import Strategy


class AllocationChange(BaseModel):
    ticker: str
    security_name: str
    current_weight: float
    optimized_weight: float
    change: float


class DataWindow(BaseModel):
    start_date: dt.date
    end_date: dt.date
    observations: int


class Methodology(BaseModel):
    frequency: str
    annualization_factor: int
    rebalancing: str
    risk_free_rate: float
    missing_dividend_yield: str


class SolverInfo(BaseModel):
    method: str
    status: str


class PortfolioMetrics(BaseModel):
    """None marks a metric that is undefined for this portfolio or input."""

    cagr: float | None
    volatility: float | None
    sharpe_ratio: float | None
    max_drawdown: float | None
    dividend_yield: float | None


class MetricsComparison(BaseModel):
    current_portfolio: PortfolioMetrics
    optimized_portfolio: PortfolioMetrics


class BetaValues(BaseModel):
    momentum: float
    value: float
    size: float


class FactorBetas(BaseModel):
    current_portfolio: BetaValues
    optimized_portfolio: BetaValues


class OptimizationResponse(BaseModel):
    optimization_strategy: Strategy
    allocation_changes: list[AllocationChange]
    window: DataWindow
    methodology: Methodology
    solver: SolverInfo
    metrics: MetricsComparison
    # Slack per checked constraint; nonnegative (within tolerance) means satisfied.
    constraint_residuals: dict[str, float]
    warnings: list[str]
    factor_betas: FactorBetas | None = None
    factor_window: DataWindow | None = None
