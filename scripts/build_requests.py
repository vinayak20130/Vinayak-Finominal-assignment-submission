"""Build self-contained POST /optimize request files from the supplied workbook.

Usage (from the repository root, with Data.xlsx present locally):

    uv run python -m scripts.build_requests [--workbook Data.xlsx] [--out examples]

Each request carries every security's full return history; the API itself finds
the common dates. The generated files let anyone run the assignment scenarios
without the workbook.
"""

import argparse
import json
import math
from pathlib import Path

from app.data.workbook import WorkbookData, load_workbook

ALL_FIVE = {"IEFA": 20, "GLD": 20, "AGG": 20, "VEA": 20, "SPY": 20}

# name -> (current weights in percent, strategy, extra request fields)
SCENARIOS: dict[str, tuple[dict[str, float], str, dict]] = {
    "case_1_equal_weights": ({"IEFA": 25, "SPY": 75}, "equal_weights", {}),
    "case_2_risk_parity": ({"VEA": 25, "AGG": 75}, "risk_parity", {}),
    "case_3_minimize_volatility": (
        {"SPY": 60, "AGG": 30, "GLD": 10},
        "minimize_volatility",
        {},
    ),
    "case_4_maximize_sharpe": (ALL_FIVE, "maximize_sharpe_ratio", {}),
    "case_5_maximize_sharpe_constrained": (
        ALL_FIVE,
        "maximize_sharpe_ratio",
        {
            "bounds": (5, 40),
            "constraints": {"min_dividend_yield": 0.025},
            # GLD has no yield in the workbook; treating it as zero is a stated
            # assumption, not a silent default.
            "settings": {"missing_dividend_yield": "zero"},
        },
    ),
    "minimize_drawdown_demo": (ALL_FIVE, "minimize_drawdown", {}),
}


def build_request(
    data: WorkbookData, weights: dict[str, float], strategy: str, extra: dict
) -> dict:
    funds = data.funds.set_index("ticker")
    bounds = extra.get("bounds")
    securities = []
    for ticker, weight in weights.items():
        if ticker not in funds.index:
            raise ValueError(f"{ticker} is not in the workbook.")
        history = data.fund_returns[data.fund_returns["ticker"] == ticker]
        security = {
            "ticker": ticker,
            "security_name": funds.at[ticker, "fund_name"],
            "current_weight": weight,
            "dividend_yield": _optional(funds.at[ticker, "dividend_yield"]),
            "returns": [
                {"date": day.date().isoformat(), "return": float(value)}
                for day, value in zip(
                    history["date"], history["total_return"], strict=True
                )
            ],
        }
        if bounds:
            security["min_weight"], security["max_weight"] = bounds
        securities.append(security)

    request = {"securities": securities, "optimization_strategy": strategy}
    for key in ("constraints", "settings"):
        if key in extra:
            request[key] = extra[key]
    return request


def _optional(value: float) -> float | None:
    return None if math.isnan(value) else float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--workbook", default="Data.xlsx", type=Path)
    parser.add_argument("--out", default="examples", type=Path)
    args = parser.parse_args()

    data = load_workbook(args.workbook)
    args.out.mkdir(parents=True, exist_ok=True)
    for name, (weights, strategy, extra) in SCENARIOS.items():
        path = args.out / f"{name}.json"
        request = build_request(data, weights, strategy, extra)
        # Compact separators keep the full-history files reasonably small.
        path.write_text(json.dumps(request, separators=(",", ":")) + "\n")
        print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
