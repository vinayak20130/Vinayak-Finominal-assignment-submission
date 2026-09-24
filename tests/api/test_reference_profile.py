"""Offline regression checks against captured live API responses."""

import json
from pathlib import Path

import pandas as pd
import pytest

from app.data.reference_alignment import align_reference_table
from app.data.workbook import load_workbook
from app.domain.errors import InsufficientDataError
from app.schemas.request import CalculationSettings
from tests.api.support import client
from tests.fakes import FakeMarketData

ROOT = Path(__file__).parents[2]
CASES = [
    "case_1_equal_weights",
    "case_2_risk_parity",
    "case_3_minimize_volatility",
    "case_4_maximize_sharpe",
    "case_5_maximize_sharpe_constrained",
    "case_6_maximize_momentum",
    "minimize_drawdown_demo",
]


@pytest.fixture(scope="module")
def market():
    return FakeMarketData.from_workbook(load_workbook(ROOT / "Data.xlsx"))


@pytest.mark.parametrize("name", CASES)
def test_reference_allocations_and_constraints(market, name):
    fixture = json.loads(
        (ROOT / "tests/fixtures/allocations/required" / f"{name}.json").read_text()
    )
    with client(market) as api:
        response = api.post("/optimize", json=fixture["request"])
    assert response.status_code == 200, response.text
    result = response.json()
    weights = {r["ticker"]: r["optimized_weight"] for r in result["allocation_changes"]}
    assert sum(weights.values()) == pytest.approx(100, abs=1e-6)
    assert min(weights.values()) >= 0
    assert min(result["constraint_residuals"].values()) >= -1e-8
    if name == "case_6_maximize_momentum":
        betas = result["factor_betas"]
        assert (
            betas["optimized_portfolio"]["momentum"]
            > betas["current_portfolio"]["momentum"]
        )
    else:
        reference = fixture["reference_response"]["data"]["optimizedPortfolio"]
        for row in reference:
            assert abs(weights[row["ticker"]] - row["optimizedWeight"] * 100) < 0.1
    if name == "case_5_maximize_sharpe_constrained":
        assert all(5 - 1e-8 <= w <= 40 + 1e-8 for w in weights.values())
        supplied_yield = sum(
            weights[s["ticker"]] / 100 * s["dividend_yield"]
            for s in fixture["request"]["securities"]
        )
        assert supplied_yield >= 0.025 - 1e-8
        assert any("Caller-supplied dividend yields" in w for w in result["warnings"])


def test_reference_alignment_excludes_initial_date_without_extrapolation():
    dates = pd.date_range("2025-01-01", periods=6)
    frame = pd.DataFrame(
        {"A": [0.1, 0.2, 0.3, None, 0.5, 0.6], "B": [None, 0.1, 0.2, 0.3, 0.4, None]},
        index=dates,
    )
    aligned = align_reference_table(frame, ["A", "B"])
    assert aligned.attrs["start_date"] == dates[1].date()
    assert list(aligned.index) == list(dates[2:5])
    assert aligned["A"].tolist() == [0.3, 0.0, 0.5]
    assert aligned["B"].tolist() == [0.2, 0.3, 0.4]


def test_reference_alignment_requires_two_returns_after_initial_price():
    frame = pd.DataFrame(
        {"A": [0.1, 0.2], "B": [0.2, 0.3]},
        index=pd.date_range("2025-01-01", periods=2),
    )
    with pytest.raises(InsufficientDataError):
        align_reference_table(frame, ["A", "B"])


def test_risk_free_defaults_are_explicit_and_overridable():
    assert CalculationSettings().risk_free_rate == 0
    assert CalculationSettings(calculation_profile="reference").risk_free_rate == 0.0175
    assert (
        CalculationSettings(
            calculation_profile="reference", risk_free_rate=0
        ).risk_free_rate
        == 0
    )


def test_infeasible_reference_yield_is_rejected(market):
    fixture = json.loads(
        (
            ROOT
            / "tests/fixtures/allocations/required"
            / "case_5_maximize_sharpe_constrained.json"
        ).read_text()
    )
    request = fixture["request"]
    request["constraints"]["min_dividend_yield"] = 0.9
    with client(market) as api:
        response = api.post("/optimize", json=request)
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "infeasible_constraints"
