"""Run every example against the real supplied data and check the assignment's
acceptance rules plus the regression baseline in validation/responses/."""

import json
from pathlib import Path

import pytest

from app.data.workbook import load_workbook
from tests.api.support import client
from tests.fakes import FakeMarketData

ROOT = Path(__file__).parents[1]
EXAMPLES = sorted((ROOT / "examples").glob("*.json"))
BASELINE = ROOT / "validation" / "responses"
REQUIRED_FIELDS = {
    "ticker",
    "security_name",
    "current_weight",
    "optimized_weight",
    "change",
}
EXACT_METRICS = ("cagr", "volatility", "sharpe_ratio", "max_drawdown")


@pytest.fixture(scope="module")
def results() -> dict[str, dict]:
    market = FakeMarketData.from_workbook(load_workbook(ROOT / "Data.xlsx"))
    output = {}
    with client(market) as api:
        for path in EXAMPLES:
            response = api.post("/optimize", json=json.loads(path.read_text()))
            assert response.status_code == 200, response.text
            output[path.stem] = response.json()
    return output


def weights(result: dict) -> dict[str, float]:
    return {
        row["ticker"]: row["optimized_weight"] for row in result["allocation_changes"]
    }


def test_all_assignment_cases_exist():
    names = {path.stem[:6] for path in EXAMPLES}
    assert {f"case_{number}" for number in range(1, 7)} <= names


@pytest.mark.parametrize("name", [path.stem for path in EXAMPLES])
def test_valid_allocations(results, name):
    rows = results[name]["allocation_changes"]
    assert all(REQUIRED_FIELDS <= set(row) for row in rows)
    assert sum(row["optimized_weight"] for row in rows) == pytest.approx(100, abs=1e-6)
    assert all(row["optimized_weight"] >= 0 for row in rows)
    for row in rows:
        expected = row["optimized_weight"] - row["current_weight"]
        assert row["change"] == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize("name", [path.stem for path in EXAMPLES])
def test_matches_regression_baseline(results, name):
    baseline = json.loads((BASELINE / f"{name}.json").read_text())
    assert weights(results[name]) == pytest.approx(weights(baseline), abs=1e-9)
    for portfolio in ("current_portfolio", "optimized_portfolio"):
        got = results[name]["metrics"][portfolio]
        want = baseline["metrics"][portfolio]
        for metric in EXACT_METRICS:
            assert got[metric] == pytest.approx(want[metric], abs=1e-12), metric
        if want["dividend_yield"] is not None:
            assert got["dividend_yield"] == pytest.approx(want["dividend_yield"])


def test_case_1_is_exactly_equal(results):
    assert weights(results["case_1_equal_weights"]) == {"IEFA": 50, "SPY": 50}


def test_case_5_respects_constraints(results):
    result = results["case_5_maximize_sharpe_constrained"]
    assert all(5 - 1e-6 <= w <= 40 + 1e-6 for w in weights(result).values())
    assert result["metrics"]["optimized_portfolio"]["dividend_yield"] >= 0.025 - 1e-8


def test_case_6_increases_momentum(results):
    betas = results["case_6_maximize_momentum"]["factor_betas"]
    current = betas["current_portfolio"]["momentum"]
    assert betas["optimized_portfolio"]["momentum"] > current
