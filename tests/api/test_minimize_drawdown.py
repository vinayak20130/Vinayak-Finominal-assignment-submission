import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def body():
    return {
        "optimization_strategy": "minimize_drawdown",
        "securities": [
            {
                "ticker": ticker,
                "security_name": ticker,
                "current_weight": weight,
                "returns": [
                    {"date": "2025-01-02", "return": first},
                    {"date": "2025-01-03", "return": second},
                ],
            }
            for ticker, weight, first, second in [
                ("AAA", 80, -0.2, 0.1),
                ("BBB", 20, 0.1, -0.2),
            ]
        ],
    }


def test_drawdown_endpoint(body):
    with TestClient(app) as client:
        response = client.post("/optimize", json=body)
    assert response.status_code == 200, response.text
    result = response.json()
    weights = [row["optimized_weight"] for row in result["allocation_changes"]]
    assert weights == pytest.approx([50, 50], abs=1e-3)
    assert sum(weights) == pytest.approx(100)
    metrics = result["metrics"]
    assert metrics["optimized_portfolio"]["max_drawdown"] == pytest.approx(
        0.0975, abs=1e-6
    )
    assert (
        metrics["optimized_portfolio"]["max_drawdown"]
        < (metrics["current_portfolio"]["max_drawdown"])
    )
    assert result["solver"] == {"method": "SLSQP", "status": "converged"}


def test_impossible_bounds(body):
    for security in body["securities"]:
        security["max_weight"] = 40
    with TestClient(app) as client:
        response = client.post("/optimize", json=body)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "infeasible_constraints"
