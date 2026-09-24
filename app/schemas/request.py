"""Request models for POST /optimize.

Securities are referenced by ticker; callers can supply dated daily returns.
Otherwise returns, names and dividend yields come from the database.
Units: weights and weight bounds are percentages (20 means 20%);
the risk-free rate and portfolio-level constraints are decimals (0.025 means 2.5%).
"""

import datetime as dt
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# Tolerance for current weights summing to 100, in percentage points.
WEIGHT_SUM_TOLERANCE = 1e-6

Number = Annotated[float, Field(strict=True)]
Percent = Annotated[float, Field(strict=True, ge=0, le=100)]
NonNegative = Annotated[float, Field(strict=True, ge=0)]


class Strategy(StrEnum):
    EQUAL_WEIGHTS = "equal_weights"
    RISK_PARITY = "risk_parity"
    MINIMIZE_DRAWDOWN = "minimize_drawdown"
    MINIMIZE_VOLATILITY = "minimize_volatility"
    MAXIMIZE_SHARPE_RATIO = "maximize_sharpe_ratio"
    OPTIMIZE_FACTOR_EXPOSURE = "optimize_factor_exposure"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


def _iso_date(value: object) -> dt.date:
    """Accept only canonical YYYY-MM-DD strings, not timestamps or datetimes."""
    if not isinstance(value, str):
        raise ValueError("Dates must be YYYY-MM-DD strings.")
    try:
        parsed = dt.date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError("Dates must use YYYY-MM-DD.") from exc
    if parsed.isoformat() != value:
        raise ValueError("Dates must use YYYY-MM-DD.")
    return parsed


IsoDate = Annotated[dt.date, BeforeValidator(_iso_date)]


class ReturnObservation(StrictModel):
    date: IsoDate
    value: Number = Field(alias="return", ge=-1)


class SecurityInput(StrictModel):
    """A known holding with optional request-specific daily returns."""

    ticker: str
    current_weight: Percent
    min_weight: Percent = 0
    max_weight: Percent = 100
    returns: list[ReturnObservation] | None = Field(default=None, min_length=1)
    dividend_yield: NonNegative | None = None

    @field_validator("ticker")
    @classmethod
    def _normalize_ticker(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Ticker must not be empty.")
        return value.strip().upper()

    @model_validator(mode="after")
    def _check_bounds(self) -> "SecurityInput":
        if self.min_weight > self.max_weight:
            raise ValueError("min_weight must not exceed max_weight.")
        if self.returns is not None:
            dates = [row.date for row in self.returns]
            if len(dates) != len(set(dates)):
                raise ValueError("Return dates must be unique for each security.")
        return self


class PortfolioConstraints(StrictModel):
    min_cagr: Number | None = Field(default=None, ge=-1)
    min_volatility: NonNegative | None = None
    max_volatility: NonNegative | None = None
    max_drawdown: NonNegative | None = Field(default=None, le=1)
    min_dividend_yield: NonNegative | None = None

    @model_validator(mode="after")
    def _check_volatility_range(self) -> "PortfolioConstraints":
        low, high = self.min_volatility, self.max_volatility
        if low is not None and high is not None and low > high:
            raise ValueError("min_volatility must not exceed max_volatility.")
        return self


class CalculationSettings(StrictModel):
    calculation_profile: Literal["standard", "reference"] = "standard"
    frequency: Literal["daily"] = "daily"
    annualization_factor: int = Field(default=252, strict=True, gt=0)
    rebalancing: Literal["each_observation"] = "each_observation"
    risk_free_rate: Number = Field(default=0, gt=-1)
    start_date: IsoDate | None = None
    end_date: IsoDate | None = None
    missing_dividend_yield: Literal["error", "zero"] = "zero"

    @model_validator(mode="after")
    def _check_window(self) -> "CalculationSettings":
        if (
            self.calculation_profile == "reference"
            and "risk_free_rate" not in self.model_fields_set
        ):
            self.risk_free_rate = 0.0175
        if self.start_date and self.end_date and self.start_date > self.end_date:
            raise ValueError("start_date must not follow end_date.")
        return self


FactorName = Literal["momentum", "value", "size"]


class FactorObjective(StrictModel):
    direction: Literal["maximize", "minimize"]
    coefficients: dict[FactorName, Number] = Field(min_length=1)

    @model_validator(mode="after")
    def _nonzero_coefficients(self) -> "FactorObjective":
        if not any(value != 0 for value in self.coefficients.values()):
            raise ValueError("At least one factor coefficient must be nonzero.")
        return self


class OptimizationRequest(StrictModel):
    securities: list[SecurityInput] = Field(min_length=2)
    optimization_strategy: Strategy
    constraints: PortfolioConstraints = Field(default_factory=PortfolioConstraints)
    settings: CalculationSettings = Field(default_factory=CalculationSettings)
    factor_objective: FactorObjective | None = None

    @model_validator(mode="after")
    def _check_portfolio(self) -> "OptimizationRequest":
        tickers = [security.ticker for security in self.securities]
        if len(tickers) != len(set(tickers)):
            raise ValueError("Tickers must be unique.")
        if self.optimization_strategy == Strategy.OPTIMIZE_FACTOR_EXPOSURE:
            if self.factor_objective is None:
                raise ValueError("optimize_factor_exposure requires factor_objective.")
        elif self.factor_objective is not None:
            raise ValueError("factor_objective requires optimize_factor_exposure.")
        supplied = [security.returns is not None for security in self.securities]
        if any(supplied) and not all(supplied):
            raise ValueError("Supply returns for every security or omit them for all.")
        total = sum(security.current_weight for security in self.securities)
        if abs(total - 100) > WEIGHT_SUM_TOLERANCE:
            raise ValueError(f"Current weights must sum to 100 (got {total:g}).")
        return self
