import pytest

from tests.api.support import post


@pytest.fixture
def body():
    dates = ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]
    return {
        "optimization_strategy": "maximize_sharpe_ratio",
        "securities": [
            {
                "ticker": ticker,
                "security_name": ticker,
                "current_weight": 50,
                "dividend_yield": yield_value,
                "returns": [
                    {"date": day, "return": value}
                    for day, value in zip(dates, values, strict=True)
                ],
            }
            for ticker, yield_value, values in [
                ("AAA", 0.01, [0.011, -0.009, 0.001, 0.001]),
                ("BBB", 0.04, [0.002, 0.002, 0.022, -0.018]),
            ]
        ],
    }


def test_sharpe_endpoint(body):
    response = post(body)
    assert response.status_code == 200, response.text
    result = response.json()
    weights = [item["optimized_weight"] for item in result["allocation_changes"]]
    assert weights == pytest.approx([200 / 3, 100 / 3], abs=1e-4)
    assert (
        result["metrics"]["optimized_portfolio"]["sharpe_ratio"]
        >= (result["metrics"]["current_portfolio"]["sharpe_ratio"])
    )


def test_constrained_sharpe_endpoint(body):
    body["constraints"] = {"min_dividend_yield": 0.025}
    response = post(body)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["metrics"]["optimized_portfolio"]["dividend_yield"] >= 0.025 - 1e-8


def test_undefined_ratio_is_clear_validation_error(body):
    for security in body["securities"]:
        for observation in security["returns"]:
            observation["return"] = 0
    response = post(body)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_input"
