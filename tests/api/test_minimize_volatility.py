import pytest

from tests.api.support import post


def test_minimum_volatility_endpoint():
    body = {
        "optimization_strategy": "minimize_volatility",
        "securities": [
            {
                "ticker": ticker,
                "security_name": ticker,
                "current_weight": 50,
                "returns": [
                    {"date": date, "return": value}
                    for date, value in zip(
                        ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"],
                        returns,
                        strict=True,
                    )
                ],
            }
            for ticker, returns in [
                ("AAA", [0.01, -0.01, 0, 0]),
                ("BBB", [0, 0, 0.02, -0.02]),
            ]
        ],
    }
    response = post(body)
    assert response.status_code == 200, response.text
    result = response.json()
    weights = [row["optimized_weight"] for row in result["allocation_changes"]]
    assert weights == pytest.approx([80, 20], abs=1e-4)
    assert sum(weights) == pytest.approx(100)
    metrics = result["metrics"]
    assert (
        metrics["optimized_portfolio"]["volatility"]
        < (metrics["current_portfolio"]["volatility"])
    )
    assert result["solver"] == {"method": "SLSQP", "status": "converged"}
