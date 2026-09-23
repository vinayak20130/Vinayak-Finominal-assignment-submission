from collections.abc import Callable

from app.domain.errors import DataValidationError
from app.domain.problem import OptimizationProblem, StrategyResult
from app.schemas.request import Strategy
from app.strategies.equal_weights import equal_weights
from app.strategies.factor_exposure import factor_exposure
from app.strategies.maximize_sharpe import maximize_sharpe
from app.strategies.minimize_drawdown import minimize_drawdown
from app.strategies.minimize_volatility import minimize_volatility
from app.strategies.risk_parity import risk_parity

StrategyFunction = Callable[[OptimizationProblem], StrategyResult]

STRATEGIES: dict[Strategy, StrategyFunction] = {
    Strategy.EQUAL_WEIGHTS: equal_weights,
    Strategy.RISK_PARITY: risk_parity,
    Strategy.MINIMIZE_DRAWDOWN: minimize_drawdown,
    Strategy.MINIMIZE_VOLATILITY: minimize_volatility,
    Strategy.MAXIMIZE_SHARPE_RATIO: maximize_sharpe,
    Strategy.OPTIMIZE_FACTOR_EXPOSURE: factor_exposure,
}


def get_strategy(strategy: Strategy) -> StrategyFunction:
    try:
        return STRATEGIES[strategy]
    except KeyError:
        raise DataValidationError(
            f"Strategy '{strategy.value}' is not available yet."
        ) from None
