from dataclasses import dataclass

import numpy as np
import pandas as pd

from app.domain.errors import DataValidationError, InsufficientDataError
from app.domain.factors import FACTOR_NAMES, regress_factors
from app.schemas.request import OptimizationRequest
from app.schemas.response import BetaValues, DataWindow, FactorBetas


@dataclass
class FactorContext:
    returns: np.ndarray
    factors: np.ndarray
    fund_betas: np.ndarray
    window: DataWindow

    def compare(self, current: np.ndarray, optimized: np.ndarray) -> FactorBetas:
        def beta_values(weights):
            coefficients = regress_factors(self.returns @ weights, self.factors)
            return BetaValues(**dict(zip(FACTOR_NAMES, coefficients[1:], strict=True)))

        return FactorBetas(
            current_portfolio=beta_values(current),
            optimized_portfolio=beta_values(optimized),
        )


def prepare_factors(
    request: OptimizationRequest, aligned_funds: pd.DataFrame
) -> FactorContext | None:
    if request.factor_returns is None:
        return None
    factors = pd.DataFrame(
        [
            {
                "date": observation.date,
                **{factor: getattr(observation, factor) for factor in FACTOR_NAMES},
            }
            for observation in request.factor_returns
        ]
    )
    factors["date"] = pd.to_datetime(factors["date"])
    factors = factors.set_index("date").sort_index()
    common = aligned_funds.index.intersection(factors.index).sort_values()
    if len(common) < 5:
        raise InsufficientDataError("Factor regression requires five common dates.")
    fund_values = aligned_funds.loc[common].to_numpy()
    factor_values = factors.loc[common, list(FACTOR_NAMES)].to_numpy()
    betas = regress_factors(fund_values, factor_values)[1:]
    return FactorContext(
        returns=fund_values,
        factors=factor_values,
        fund_betas=betas,
        window=DataWindow(
            start_date=common[0].date(),
            end_date=common[-1].date(),
            observations=len(common),
        ),
    )


def exposure_costs(request: OptimizationRequest, context: FactorContext) -> np.ndarray:
    objective = request.factor_objective
    coefficients = np.array(
        [objective.coefficients.get(name, 0) for name in FACTOR_NAMES]
    )
    with np.errstate(over="ignore", invalid="ignore"):
        costs = context.fund_betas.T @ coefficients
    if not np.isfinite(costs).all():
        raise DataValidationError("Factor objective exceeds the finite numeric range.")
    return -costs if objective.direction == "maximize" else costs
