import copy
import math

import pytest

from tests.api.support import post


@pytest.fixture
def body():
    days = [
        "2025-01-02",
        "2025-01-03",
        "2025-01-06",
        "2025-01-07",
        "2025-01-08",
        "2025-01-09",
    ]
    factors = [[0, 0, 0], [0.01, 0, 0], [0, 0.01, 0], [0, 0, 0.01], [-0.01, 0, 0]]
    return {
        "optimization_strategy": "optimize_factor_exposure",
        "factor_objective": {"direction": "maximize", "coefficients": {"momentum": 1}},
        "factor_returns": [
            {"date": day, "momentum": values[0], "value": values[1], "size": values[2]}
            for day, values in zip(days, factors)
        ],
        "securities": [
            {
                "ticker": ticker,
                "security_name": ticker,
                "current_weight": 50,
                "dividend_yield": yield_value,
                "returns": [
                    {"date": day, "return": 0.001 + beta * factor[0]}
                    for day, factor in zip(days, factors)
                ]
                + [{"date": days[-1], "return": last_return}],
            }
            for ticker, beta, yield_value, last_return in [
                ("AAA", 2, 0.01, -0.5),
                ("BBB", 0.5, 0.04, 0),
            ]
        ],
    }


def test_maximum_momentum_has_correct_betas_and_separate_windows(body):
    response = post(body)
    assert response.status_code == 200, response.text
    result = response.json()
    assert [row["optimized_weight"] for row in result["allocation_changes"]] == [100, 0]
    assert result["factor_betas"]["current_portfolio"]["momentum"] == pytest.approx(
        1.25
    )
    assert result["factor_betas"]["optimized_portfolio"]["momentum"] == pytest.approx(2)
    assert result["factor_betas"]["optimized_portfolio"]["value"] == pytest.approx(0)
    assert result["window"]["observations"] == 6
    assert result["window"]["end_date"] == "2025-01-09"
    assert result["factor_window"]["observations"] == 5
    assert result["factor_window"]["end_date"] == "2025-01-08"
    assert result["solver"] == {"method": "HiGHS", "status": "optimal"}


def test_minimum_momentum(body):
    body["factor_objective"]["direction"] = "minimize"
    result = post(body).json()
    assert [row["optimized_weight"] for row in result["allocation_changes"]] == [0, 100]
    assert result["factor_betas"]["optimized_portfolio"]["momentum"] == pytest.approx(
        0.5
    )


def test_signed_multiple_factor_objective(body):
    body["factor_objective"]["coefficients"] = {"momentum": -1, "value": 0.25}
    result = post(body).json()
    assert result["allocation_changes"][1]["optimized_weight"] == 100


def test_factor_beta_reporting_on_other_strategy_is_optional(body):
    body["optimization_strategy"] = "equal_weights"
    del body["factor_objective"]
    result = post(body).json()
    assert (
        result["factor_betas"]["current_portfolio"]
        == (result["factor_betas"]["optimized_portfolio"])
    )
    del body["factor_returns"]
    result = post(body).json()
    assert result["factor_betas"] is None
    assert result["factor_window"] is None


def test_bounds_and_yield_floor(body):
    for security in body["securities"]:
        security["min_weight"] = 10
        security["max_weight"] = 80
    body["constraints"] = {"min_dividend_yield": 0.025}
    response = post(body)
    assert response.status_code == 200, response.text
    result = response.json()
    weights = [row["optimized_weight"] for row in result["allocation_changes"]]
    assert weights == pytest.approx([50, 50])
    assert result["metrics"]["optimized_portfolio"]["dividend_yield"] >= 0.025 - 1e-8


def test_nonlinear_constraint_uses_full_fund_window(body):
    body["constraints"] = {"max_drawdown": 0.1}
    response = post(body)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["solver"]["method"] == "SLSQP"
    # The binding drawdown spans the last two fund days: day 5 returns
    # -0.004 - 0.015a and the extra fund-only day returns -0.5a, so the largest
    # AAA weight solves (0.996 - 0.015a)(1 - 0.5a) = 0.9. It is not simply 20%,
    # which ignores day 5 and would breach the cap (drawdown 10.63%).
    a, b, c = 0.0075, -0.513, 0.096
    expected = (-b - math.sqrt(b * b - 4 * a * c)) / (2 * a) * 100
    assert result["allocation_changes"][0]["optimized_weight"] == pytest.approx(
        expected, abs=1e-4
    )
    assert result["metrics"]["optimized_portfolio"]["max_drawdown"] <= 0.1 + 1e-8


def test_factor_dates_are_sorted(body):
    expected = post(body).json()["factor_betas"]
    body["factor_returns"].reverse()
    assert post(body).json()["factor_betas"] == expected


def test_requested_window_applies_to_factors(body):
    body["settings"] = {"start_date": "2025-01-03"}
    response = post(body)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "insufficient_data"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda b: b.pop("factor_objective"),
        lambda b: b["factor_objective"].update(coefficients={}),
        lambda b: b["factor_objective"].update(coefficients={"momentum": 0}),
        lambda b: b["factor_objective"].update(coefficients={"unknown": 1}),
        lambda b: b["factor_objective"].update(direction="invalid"),
        lambda b: b.update(optimization_strategy="equal_weights"),
    ],
)
def test_bad_factor_inputs(body, mutate):
    mutate(body)
    response = post(body)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_input"


def test_factor_strategy_needs_factor_data(body):
    del body["factor_returns"]

    response = post(body)

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "insufficient_data"


def test_rank_failure(body):
    for row in body["factor_returns"]:
        row["size"] = row["momentum"]
    response = post(body)
    assert response.status_code == 422
    assert "rank" in response.json()["error"]["message"]


def test_no_factor_overlap(body):
    for row in body["factor_returns"]:
        row["date"] = row["date"].replace("2025", "2024")
    response = post(body)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "insufficient_data"


def test_requested_factor_reporting_does_not_change_nonfactor_optimization(body):
    body["optimization_strategy"] = "minimize_volatility"
    del body["factor_objective"]
    plain = copy.deepcopy(body)
    del plain["factor_returns"]
    with_factors = post(body)
    without_factors = post(plain)
    assert with_factors.status_code == without_factors.status_code == 200
    assert (
        with_factors.json()["allocation_changes"]
        == (without_factors.json()["allocation_changes"])
    )
