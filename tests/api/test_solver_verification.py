import numpy as np
import pytest

from app.domain.problem import StrategyResult
from tests.api.support import post


@pytest.mark.parametrize(
    "weights",
    [
        [float("nan"), 0.5],
        [-1e-12, 1 + 1e-12],
        [float("inf"), 0],
        [0.4, 0.4],
        [[0.5, 0.5]],
    ],
)
def test_invalid_strategy_output_is_never_success(monkeypatch, weights):
    monkeypatch.setattr(
        "app.services.optimizer.get_strategy",
        lambda name: lambda problem: StrategyResult(np.array(weights), "test"),
    )
    body = {
        "optimization_strategy": "equal_weights",
        "securities": [
            {
                "ticker": ticker,
                "security_name": ticker,
                "current_weight": 50,
                "returns": [
                    {"date": "2025-01-02", "return": 0.01},
                    {"date": "2025-01-03", "return": -0.01},
                ],
            }
            for ticker in ["AAA", "BBB"]
        ],
    }
    response = post(body)
    assert response.status_code == 500
    assert response.json()["error"]["code"] == "optimization_failed"
