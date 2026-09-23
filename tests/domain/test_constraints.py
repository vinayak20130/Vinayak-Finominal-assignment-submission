import numpy as np
import pytest

from app.domain.constraints import constraint_residuals, violated
from app.domain.problem import OptimizationProblem, PortfolioConstraints


@pytest.fixture
def problem():
    return OptimizationProblem(
        tickers=("AAA", "BBB"),
        returns=np.array([[0.01, -0.02], [-0.01, 0.02]]),
        current_weights=np.array([0.5, 0.5]),
        lower_bounds=np.zeros(2),
        upper_bounds=np.ones(2),
        yields=np.array([0.01, 0.02]),
        constraints=PortfolioConstraints(),
    )


@pytest.mark.parametrize(
    "weights",
    [
        [float("nan"), 0.5],
        [float("inf"), 0.5],
        [-1e-12, 1 + 1e-12],
        [[0.5, 0.5]],
        [1],
        [0.4, 0.4],
    ],
)
def test_invalid_solver_weights_fail_independent_validation(problem, weights):
    assert violated(constraint_residuals(problem, weights))


@pytest.mark.parametrize("slack", [float("nan"), float("inf"), -float("inf"), -0.01])
def test_nonfinite_residuals_fail(slack):
    assert violated({"constraint": slack}) == ["constraint"]
