import copy

import pytest
from fastapi.testclient import TestClient

from app.main import app

DATES = ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]


def security(ticker, weight, returns, **extra):
    return {
        "ticker": ticker,
        "security_name": f"Synthetic Asset {ticker}",
        "current_weight": weight,
        "returns": [
            {"date": day, "return": value}
            for day, value in zip(DATES, returns, strict=False)
        ],
        **extra,
    }


@pytest.fixture
def request_body():
    return {
        "securities": [
            security("AAA", 25, [0.01, -0.005, 0.002, 0.004]),
            security("BBB", 75, [-0.002, 0.004, -0.001, 0.003]),
        ],
        "optimization_strategy": "equal_weights",
    }


def post(body):
    with TestClient(app) as client:
        return client.post("/optimize", json=body)


def assert_error(response, status, code):
    assert response.status_code == status, response.json()
    error = response.json()["error"]
    assert error["code"] == code
    assert error["message"]
    return error


def test_equal_weights_contract_example(request_body):
    response = post(request_body)

    assert response.status_code == 200
    body = response.json()
    assert body["optimization_strategy"] == "equal_weights"
    assert body["allocation_changes"] == [
        {
            "ticker": "AAA",
            "security_name": "Synthetic Asset AAA",
            "current_weight": 25,
            "optimized_weight": 50,
            "change": 25,
        },
        {
            "ticker": "BBB",
            "security_name": "Synthetic Asset BBB",
            "current_weight": 75,
            "optimized_weight": 50,
            "change": -25,
        },
    ]
    assert body["solver"] == {"method": "closed_form", "status": "optimal"}
    assert body["window"] == {
        "start_date": "2025-01-02",
        "end_date": "2025-01-07",
        "observations": 4,
    }


def test_three_securities_split_evenly(request_body):
    request_body["securities"][0]["current_weight"] = 20
    request_body["securities"][1]["current_weight"] = 50
    request_body["securities"].append(security("CCC", 30, [0.001, 0.002, 0.0, 0.001]))

    changes = post(request_body).json()["allocation_changes"]

    weights = [change["optimized_weight"] for change in changes]
    assert weights == pytest.approx([100 / 3] * 3)
    assert sum(weights) == pytest.approx(100)


def test_window_uses_only_common_dates(request_body):
    # BBB lacks the first date, so the aligned window starts one day later.
    second = request_body["securities"][1]
    second["returns"] = second["returns"][1:]

    window = post(request_body).json()["window"]

    assert window["start_date"] == "2025-01-03"
    assert window["observations"] == 3


def test_requested_window_is_applied_and_reported(request_body):
    request_body["settings"] = {"start_date": "2025-01-03", "end_date": "2025-01-06"}

    window = post(request_body).json()["window"]

    assert window == {
        "start_date": "2025-01-03",
        "end_date": "2025-01-06",
        "observations": 2,
    }


def test_tickers_are_normalized(request_body):
    request_body["securities"][0]["ticker"] = " aaa "

    assert post(request_body).json()["allocation_changes"][0]["ticker"] == "AAA"


def test_metrics_reported_for_both_portfolios(request_body):
    metrics = post(request_body).json()["metrics"]

    for portfolio in ("current_portfolio", "optimized_portfolio"):
        assert metrics[portfolio]["volatility"] > 0
        assert metrics[portfolio]["max_drawdown"] >= 0
        # No yields were supplied, so the portfolio yield is unknown.
        assert metrics[portfolio]["dividend_yield"] is None


@pytest.mark.parametrize(
    "mutate",
    [
        lambda body: body.update(unexpected=1),
        lambda body: body["securities"][0].update(current_weight=30),
        lambda body: body["securities"][1].update(ticker="aaa"),
        lambda body: body.update(securities=body["securities"][:1]),
        lambda body: body.update(optimization_strategy="maximize_returns"),
        lambda body: body["securities"][0]["returns"][0].update({"return": True}),
        lambda body: body["securities"][0]["returns"][0].update({"return": "0.01"}),
        lambda body: body["securities"][0]["returns"][0].update({"return": -1.5}),
        lambda body: body["securities"][0]["returns"][0].update(date="2025-1-02"),
        lambda body: body["securities"][0]["returns"][0].update(date=20250102),
        lambda body: body["securities"][0]["returns"][1].update(date="2025-01-02"),
        lambda body: body["securities"][0].update(min_weight=60, max_weight=40),
        lambda body: body["securities"][0].update(ticker="  "),
        lambda body: body.update(settings={"rebalancing": "annual"}),
        lambda body: body.update(settings={"annualization_factor": 252.0}),
        lambda body: body.update(
            settings={"start_date": "2025-02-01", "end_date": "2025-01-01"}
        ),
        lambda body: body.update(constraints={"max_drawdown": 1.5}),
        lambda body: body.update(
            constraints={"min_volatility": 0.2, "max_volatility": 0.1}
        ),
    ],
)
def test_invalid_requests_are_rejected(request_body, mutate):
    mutate(request_body)

    error = assert_error(post(request_body), 422, "invalid_input")

    assert error["details"]
    assert all({"field", "message"} <= set(detail) for detail in error["details"])


def test_error_details_do_not_echo_input(request_body):
    request_body["securities"][0]["returns"][0]["return"] = "bad"

    error = assert_error(post(request_body), 422, "invalid_input")

    assert error["details"][0]["field"] == "securities.0.returns.0.return"
    assert "input" not in error["details"][0]


def test_disjoint_dates_are_insufficient_data(request_body):
    request_body["securities"][1]["returns"] = [{"date": "2024-06-03", "return": 0.01}]

    assert_error(post(request_body), 422, "insufficient_data")


def test_equal_weights_conflicting_with_bound(request_body):
    request_body["securities"][0]["max_weight"] = 40

    error = assert_error(post(request_body), 422, "equal_weight_conflict")

    assert "max_weight" in error["message"]


def test_equal_weights_conflicting_with_portfolio_constraint(request_body):
    for item, value in zip(request_body["securities"], [0.01, 0.02], strict=True):
        item["dividend_yield"] = value
    request_body["constraints"] = {"min_dividend_yield": 0.05}

    error = assert_error(post(request_body), 422, "equal_weight_conflict")

    assert "min_dividend_yield" in error["message"]


def test_missing_yield_needed_by_constraint_is_an_error(request_body):
    request_body["securities"][0]["dividend_yield"] = 0.03
    request_body["constraints"] = {"min_dividend_yield": 0.01}

    error = assert_error(post(request_body), 422, "invalid_input")

    assert "BBB" in error["message"]


def test_missing_yield_zero_policy_is_explicit_and_warned(request_body):
    request_body["securities"][0]["dividend_yield"] = 0.03
    request_body["constraints"] = {"min_dividend_yield": 0.01}
    request_body["settings"] = {"missing_dividend_yield": "zero"}

    body = post(request_body).json()

    optimized = body["metrics"]["optimized_portfolio"]
    assert optimized["dividend_yield"] == pytest.approx(0.015)
    assert body["constraint_residuals"]["min_dividend_yield"] == pytest.approx(0.005)
    assert any("BBB" in warning for warning in body["warnings"])


def test_unimplemented_strategy_is_rejected_clearly(request_body):
    body = copy.deepcopy(request_body)
    body["optimization_strategy"] = "optimize_factor_exposure"

    error = assert_error(post(body), 422, "invalid_input")

    assert "optimize_factor_exposure" in error["message"]
