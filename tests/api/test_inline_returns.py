"""Exercise the actual HTTP input; no helper strips the supplied return data."""

import copy

import pandas as pd
import pytest

from tests.api.support import client, split
from tests.api.test_market_data_api import data_body, ticker_request


def inline_request():
    body = ticker_request(optimization_strategy="risk_parity")
    dates = ["2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]
    for security, scale in zip(body["securities"], [4, 1], strict=True):
        security["returns"] = [
            {"date": date, "return": value * scale}
            for date, value in zip(dates, [0.01, -0.01, 0.02, -0.02], strict=True)
        ]
    return body


def test_inline_returns_drive_solver_without_reading_stored_returns(monkeypatch):
    market, _ = split(data_body())

    def unexpected_read(tickers):
        pytest.fail("Supplied returns must replace the stored return series.")

    monkeypatch.setattr(market, "daily_returns", unexpected_read)
    with client(market) as api:
        response = api.post("/optimize", json=inline_request())

    assert response.status_code == 200, response.text
    result = response.json()
    # Perfectly correlated series, with A four times as volatile as B.
    assert [
        r["optimized_weight"] for r in result["allocation_changes"]
    ] == pytest.approx([20, 80], abs=1e-5)
    assert result["window"]["observations"] == 4
    assert result["allocation_changes"][0]["security_name"] == "Asset A"


@pytest.mark.parametrize(
    "mutation",
    [
        lambda b: b["securities"][1].pop("returns"),
        lambda b: b["securities"][0]["returns"].append(
            copy.deepcopy(b["securities"][0]["returns"][0])
        ),
        lambda b: b["securities"][0]["returns"][0].update({"return": -1.01}),
        lambda b: b["securities"][0]["returns"][0].update({"return": "0.01"}),
        lambda b: b["securities"][0]["returns"][0].update({"return": True}),
        lambda b: b["securities"][0]["returns"][0].update(date="2025-01-02T00:00:00"),
    ],
)
def test_invalid_inline_data_is_rejected(mutation):
    market, _ = split(data_body())
    body = inline_request()
    mutation(body)
    with client(market) as api:
        response = api.post("/optimize", json=body)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_input"


def test_inline_dates_are_intersected_then_filtered():
    market, _ = split(data_body())
    body = inline_request()
    body["securities"][0]["returns"].reverse()
    body["securities"][1]["returns"] = body["securities"][1]["returns"][1:]
    body["settings"] = {"end_date": "2025-01-06"}
    with client(market) as api:
        result = api.post("/optimize", json=body)
    assert result.status_code == 200, result.text
    assert result.json()["window"] == {
        "start_date": "2025-01-03",
        "end_date": "2025-01-06",
        "observations": 2,
    }


def test_inline_returns_produce_both_factor_betas_and_improve_momentum():
    market, _ = split(data_body())
    dates = pd.date_range("2025-01-01", periods=8)
    factors = pd.DataFrame(
        {
            "momentum": [0.01, -0.01, 0, 0, 0, 0, 0.01, -0.01],
            "value": [0, 0, 0.01, -0.01, 0, 0, 0.01, -0.01],
            "size": [0, 0, 0, 0, 0.01, -0.01, 0.01, -0.01],
        },
        index=dates,
    )
    market.factors = factors
    body = ticker_request(
        optimization_strategy="optimize_factor_exposure",
        factor_objective={"direction": "maximize", "coefficients": {"momentum": 1}},
    )
    series = [
        3 * factors["momentum"] + factors["value"],
        -factors["momentum"] + factors["size"],
    ]
    for security, returns in zip(body["securities"], series, strict=True):
        security["returns"] = [
            {"date": day.date().isoformat(), "return": value}
            for day, value in returns.items()
        ]
    with client(market) as api:
        response = api.post("/optimize", json=body)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["factor_betas"]["current_portfolio"]["momentum"] == pytest.approx(1)
    assert result["factor_betas"]["optimized_portfolio"]["momentum"] == pytest.approx(3)
    assert result["allocation_changes"][0]["optimized_weight"] == pytest.approx(100)
