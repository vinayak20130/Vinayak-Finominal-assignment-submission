import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def body():
    dates = ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]
    return {
        "optimization_strategy": "risk_parity",
        "securities": [
            {
                "ticker": ticker,
                "security_name": ticker,
                "current_weight": 50,
                "returns": [
                    {"date": day, "return": value}
                    for day, value in zip(dates, values, strict=True)
                ],
            }
            for ticker, values in [
                ("AAA", [0.01, -0.01, 0.01, -0.01]),
                ("BBB", [0.02, -0.02, 0.02, -0.02]),
            ]
        ],
    }


def test_risk_parity_endpoint(body):
    with TestClient(app) as client:
        response = client.post("/optimize", json=body)
    assert response.status_code == 200, response.text
    result = response.json()
    weights = [row["optimized_weight"] for row in result["allocation_changes"]]
    assert weights == pytest.approx([200 / 3, 100 / 3], abs=1e-3)
    assert sum(weights) == pytest.approx(100)
    assert result["solver"] == {"method": "SLSQP", "status": "converged"}


def test_infeasible_constraint_error(body):
    for security in body["securities"]:
        security["min_weight"] = 60
    with TestClient(app) as client:
        response = client.post("/optimize", json=body)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "infeasible_constraints"
