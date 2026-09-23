"""Run every committed example request and check the assignment's acceptance rules.

These checks are invariants only; they never hardcode expected optimized weights.
"""

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.main import app

EXAMPLES = sorted((Path(__file__).parents[1] / "examples").glob("*.json"))
REQUIRED_FIELDS = {
    "ticker",
    "security_name",
    "current_weight",
    "optimized_weight",
    "change",
}


def run(path: Path) -> dict:
    with TestClient(app) as client:
        response = client.post("/optimize", json=json.loads(path.read_text()))
    assert response.status_code == 200, response.text
    return response.json()


def weights(result: dict) -> dict[str, float]:
    return {
        row["ticker"]: row["optimized_weight"] for row in result["allocation_changes"]
    }


@pytest.fixture(scope="module")
def results() -> dict[str, dict]:
    return {path.stem: run(path) for path in EXAMPLES}


def test_examples_exist():
    names = {path.stem for path in EXAMPLES}
    assert {f"case_{number}" for number in range(1, 7)} <= {name[:6] for name in names}
    assert "minimize_drawdown_demo" in names


@pytest.mark.parametrize("name", [path.stem for path in EXAMPLES])
def test_weights_are_valid_allocations(results, name):
    rows = results[name]["allocation_changes"]

    assert all(REQUIRED_FIELDS <= set(row) for row in rows)
    assert sum(row["optimized_weight"] for row in rows) == pytest.approx(100, abs=1e-6)
    assert all(row["optimized_weight"] >= 0 for row in rows)
    for row in rows:
        expected = row["optimized_weight"] - row["current_weight"]
        assert row["change"] == pytest.approx(expected, abs=1e-9)


def test_case_1_is_exactly_equal(results):
    assert weights(results["case_1_equal_weights"]) == {"IEFA": 50, "SPY": 50}


def test_case_5_respects_security_and_portfolio_constraints(results):
    result = results["case_5_maximize_sharpe_constrained"]

    assert all(5 - 1e-6 <= weight <= 40 + 1e-6 for weight in weights(result).values())
    optimized = result["metrics"]["optimized_portfolio"]
    assert optimized["dividend_yield"] >= 0.025 - 1e-8


def test_case_6_increases_momentum_exposure(results):
    betas = results["case_6_maximize_momentum"]["factor_betas"]

    assert betas is not None
    current = betas["current_portfolio"]["momentum"]
    assert betas["optimized_portfolio"]["momentum"] > current
