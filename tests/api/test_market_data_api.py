import pytest

from app.domain.errors import DataUnavailableError
from tests.api.support import client, split
from tests.fakes import FakeMarketData

DATES = ["2024-12-31", "2025-01-02", "2025-01-03", "2025-01-06", "2025-01-07"]


def data_body() -> dict:
    return {
        "securities": [
            {
                "ticker": "AAA",
                "security_name": "Asset A",
                "current_weight": 50,
                "dividend_yield": 0.03,
                "returns": [
                    {"date": d, "return": r}
                    for d, r in zip(
                        DATES, [0.01, -0.02, 0.015, 0.004, -0.003], strict=True
                    )
                ],
            },
            {
                "ticker": "BBB",
                "security_name": "Asset B",
                "current_weight": 50,
                "returns": [
                    {"date": d, "return": r}
                    for d, r in zip(
                        DATES, [0.002, 0.001, -0.001, 0.003, 0.0], strict=True
                    )
                ],
            },
        ],
        "optimization_strategy": "equal_weights",
    }


@pytest.fixture
def market() -> FakeMarketData:
    return split(data_body())[0]


def ticker_request(**extra) -> dict:
    return {
        "securities": [
            {"ticker": "AAA", "current_weight": 50},
            {"ticker": "BBB", "current_weight": 50},
        ],
        "optimization_strategy": "equal_weights",
        **extra,
    }


def test_unknown_ticker_is_404(market):
    body = ticker_request()
    body["securities"][1]["ticker"] = "XYZ"
    with client(market) as api:
        response = api.post("/optimize", json=body)

    assert response.status_code == 404
    error = response.json()["error"]
    assert error["code"] == "ticker_not_found"
    assert "XYZ" in error["message"] and "Available: AAA, BBB" in error["message"]


@pytest.mark.parametrize("field", ["returns", "security_name", "dividend_yield"])
def test_data_fields_are_rejected(market, field):
    body = ticker_request()
    body["securities"][0][field] = [] if field == "returns" else "x"
    with client(market) as api:
        response = api.post("/optimize", json=body)

    assert response.status_code == 422
    detail = response.json()["error"]["details"][0]
    assert detail["field"] == f"securities.0.{field}"


def test_factor_returns_field_is_rejected(market):
    with client(market) as api:
        response = api.post("/optimize", json=ticker_request(factor_returns=[]))

    assert response.status_code == 422


def test_blank_yield_defaults_to_zero_with_warning(market):
    body = ticker_request(constraints={"min_dividend_yield": 0.01})
    with client(market) as api:
        response = api.post("/optimize", json=body)

    assert response.status_code == 200, response.text
    result = response.json()
    assert result["metrics"]["optimized_portfolio"]["dividend_yield"] == pytest.approx(
        0.015
    )
    assert any("BBB" in warning for warning in result["warnings"])


def test_securities_endpoint(market):
    with client(market) as api:
        rows = api.get("/securities").json()

    assert [row["ticker"] for row in rows] == ["AAA", "BBB"]
    aaa, bbb = rows
    assert aaa["security_name"] == "Asset A"
    assert aaa["observations"] == 5
    assert aaa["since_inception_return"] == pytest.approx(
        1.01 * 0.98 * 1.015 * 1.004 * 0.997 - 1
    )
    assert aaa["ytd_return"] == pytest.approx(0.98 * 1.015 * 1.004 * 0.997 - 1)
    assert bbb["dividend_yield"] is None


def test_annual_returns_endpoint(market):
    with client(market) as api:
        rows = api.get("/securities/ aaa /annual-returns").json()
        missing = api.get("/securities/XYZ/annual-returns")

    assert [(row["year"], row["is_partial"]) for row in rows] == [
        (2024, True),
        (2025, True),
    ]
    assert rows[0]["total_return"] == pytest.approx(0.01)
    assert missing.status_code == 404


def test_data_unavailable_is_503(market):
    class DownMarket(FakeMarketData):
        def lookup(self, tickers):
            raise DataUnavailableError("Market data is temporarily unavailable.")

    down = DownMarket(market.funds, market.returns)
    with client(down) as api:
        response = api.post("/optimize", json=ticker_request())

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "data_unavailable"
