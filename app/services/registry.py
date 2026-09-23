from collections.abc import Callable

from app.domain.errors import DataValidationError
from app.domain.problem import OptimizationProblem, StrategyResult
from app.schemas.request import Strategy
from app.strategies.equal_weights import equal_weights
from app.strategies.risk_parity import risk_parity

StrategyFunction = Callable[[OptimizationProblem], StrategyResult]

STRATEGIES: dict[Strategy, StrategyFunction] = {
    Strategy.EQUAL_WEIGHTS: equal_weights,
    Strategy.RISK_PARITY: risk_parity,
}


def get_strategy(strategy: Strategy) -> StrategyFunction:
    try:
        return STRATEGIES[strategy]
    except KeyError:
        raise DataValidationError(
            f"Strategy '{strategy.value}' is not available yet."
        ) from None
