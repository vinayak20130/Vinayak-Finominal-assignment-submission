"""Ready-to-run requests shown in the interactive docs (/docs)."""

ALL_FIVE = [
    {"ticker": ticker, "current_weight": 20}
    for ticker in ("IEFA", "GLD", "AGG", "VEA", "SPY")
]

OPTIMIZE_EXAMPLES = {
    "equal_weights": {
        "summary": "Case 1: equal weights (IEFA 25%, SPY 75%)",
        "value": {
            "securities": [
                {"ticker": "IEFA", "current_weight": 25},
                {"ticker": "SPY", "current_weight": 75},
            ],
            "optimization_strategy": "equal_weights",
        },
    },
    "constrained_sharpe": {
        "summary": "Case 5: max Sharpe, 5-40% per fund, yield >= 2.5%",
        "value": {
            "securities": [
                {**security, "min_weight": 5, "max_weight": 40} for security in ALL_FIVE
            ],
            "optimization_strategy": "maximize_sharpe_ratio",
            "constraints": {"min_dividend_yield": 0.025},
        },
    },
    "momentum": {
        "summary": "Case 6: maximize momentum exposure",
        "value": {
            "securities": ALL_FIVE,
            "optimization_strategy": "optimize_factor_exposure",
            "factor_objective": {
                "direction": "maximize",
                "coefficients": {"momentum": 1},
            },
        },
    },
}
