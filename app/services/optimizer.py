"""Turn a validated request into a verified optimization response.

Flow: look up tickers -> read daily returns -> build problem -> run strategy ->
independently verify every constraint -> report weights, metrics, and methodology.
"""

from dataclasses import replace

import numpy as np

from app.data.alignment import align_table
from app.data.market_data import MarketData, Security
from app.data.reference_alignment import align_reference_table
from app.domain.constraints import constraint_residuals, violated
from app.domain.errors import (
    DataValidationError,
    OptimizationFailedError,
    UndefinedMetricError,
)
from app.domain.metrics import (
    annualized_volatility,
    cagr,
    maximum_drawdown,
    portfolio_dividend_yield,
    portfolio_returns,
    sharpe_ratio,
)
from app.domain.problem import OptimizationProblem, PortfolioConstraints
from app.schemas.request import OptimizationRequest
from app.schemas.response import (
    AllocationChange,
    DataWindow,
    Methodology,
    MetricsComparison,
    OptimizationResponse,
    PortfolioMetrics,
    SolverInfo,
)
from app.services.factors import exposure_costs, prepare_factors
from app.services.registry import get_strategy
from app.services.return_data import request_returns
from app.strategies.feasibility import linear_solution
from app.strategies.inverse_volatility import inverse_volatility
from app.strategies.reference import reference_solution


def optimize(request: OptimizationRequest, market: MarketData) -> OptimizationResponse:
    tickers = [security.ticker for security in request.securities]
    known = market.lookup(tickers)
    settings = request.settings
    reference = settings.calculation_profile == "reference"
    align = align_reference_table if reference else align_table
    aligned = align(
        request_returns(request, market),
        tickers,
        start_date=settings.start_date,
        end_date=settings.end_date,
    )
    problem, warnings = _build_problem(request, known, aligned.to_numpy())
    factors, factor_warnings = prepare_factors(
        market.factor_returns(), aligned, required=request.factor_objective is not None
    )
    if request.factor_objective is not None:
        problem = replace(problem, factor_costs=exposure_costs(request, factors))

    # Impossible weight bounds or yield floors are infeasible for every strategy,
    # so prove that before any strategy-specific error can mask it.
    linear_solution(problem, np.zeros(problem.size))

    result = (
        inverse_volatility(problem)
        if reference and request.optimization_strategy == "risk_parity"
        else reference_solution(problem, request.optimization_strategy)
        if reference
        and request.optimization_strategy
        in {"minimize_volatility", "minimize_drawdown", "maximize_sharpe_ratio"}
        else get_strategy(request.optimization_strategy)(problem)
    )
    if reference:
        warnings.append(
            "Reference profile excludes the start-date return and treats missing "
            "observations inside the shared history as unchanged prices."
        )

    # Never trust a strategy's own claim of success: recompute every constraint.
    residuals = constraint_residuals(problem, result.weights)
    failed = violated(residuals)
    if failed:
        raise OptimizationFailedError(
            "The result failed independent verification: " + ", ".join(failed)
        )

    return OptimizationResponse(
        optimization_strategy=request.optimization_strategy,
        allocation_changes=_allocation_changes(request, known, result.weights),
        window=DataWindow(
            start_date=aligned.attrs.get("start_date", aligned.index[0].date()),
            end_date=aligned.index[-1].date(),
            observations=len(aligned),
        ),
        methodology=Methodology(
            calculation_profile=settings.calculation_profile,
            frequency=settings.frequency,
            annualization_factor=settings.annualization_factor,
            rebalancing=settings.rebalancing,
            risk_free_rate=settings.risk_free_rate,
            missing_dividend_yield=settings.missing_dividend_yield,
        ),
        solver=SolverInfo(method=result.method, status=result.status),
        metrics=MetricsComparison(
            current_portfolio=_metrics(problem, problem.current_weights),
            optimized_portfolio=_metrics(problem, result.weights),
        ),
        constraint_residuals=residuals,
        warnings=warnings + factor_warnings + result.warnings,
        factor_betas=(
            factors.compare(problem.current_weights, result.weights)
            if factors
            else None
        ),
        factor_window=factors.window if factors else None,
    )


def _build_problem(
    request: OptimizationRequest, known: dict[str, Security], returns: np.ndarray
) -> tuple[OptimizationProblem, list[str]]:
    securities = request.securities
    limits = request.constraints
    yields, warnings = _resolve_yields(request, known)
    current = np.array([s.current_weight for s in securities])

    problem = OptimizationProblem(
        tickers=tuple(security.ticker for security in securities),
        returns=returns,
        # The API speaks percentages; everything internal is a fraction. Rescaling
        # absorbs only the validated sub-1e-6 percentage-point rounding.
        current_weights=current / current.sum(),
        lower_bounds=np.array([s.min_weight for s in securities]) / 100,
        upper_bounds=np.array([s.max_weight for s in securities]) / 100,
        yields=yields,
        constraints=PortfolioConstraints(
            min_cagr=limits.min_cagr,
            min_volatility=limits.min_volatility,
            max_volatility=limits.max_volatility,
            max_drawdown=limits.max_drawdown,
            min_dividend_yield=limits.min_dividend_yield,
        ),
        annualization_factor=request.settings.annualization_factor,
        risk_free_rate=request.settings.risk_free_rate,
    )
    return problem, warnings


def _resolve_yields(
    request: OptimizationRequest, known: dict[str, Security]
) -> tuple[np.ndarray, list[str]]:
    """Apply the missing-yield policy; the default treats blank as 0% and says so."""
    supplied = [
        s.dividend_yield
        if s.dividend_yield is not None
        else known[s.ticker].dividend_yield
        for s in request.securities
    ]
    overrides = [s.ticker for s in request.securities if s.dividend_yield is not None]
    source_warnings = (
        ["Caller-supplied dividend yields used for: " + ", ".join(overrides) + "."]
        if overrides
        else []
    )
    missing = [
        s.ticker
        for s, value in zip(request.securities, supplied, strict=True)
        if value is None
    ]
    yields = np.array([np.nan if value is None else value for value in supplied])
    if not missing:
        return yields, source_warnings

    names = ", ".join(missing)
    if request.settings.missing_dividend_yield == "zero":
        return np.nan_to_num(yields, nan=0.0), source_warnings + [
            f"Missing dividend yield treated as 0 for: {names} "
            "(settings.missing_dividend_yield='zero')."
        ]
    if request.constraints.min_dividend_yield is not None:
        raise DataValidationError(
            f"Dividend yield is missing for {names}, but min_dividend_yield needs "
            "it. Set settings.missing_dividend_yield to 'zero' to treat it as 0%."
        )
    return yields, source_warnings + [
        f"Portfolio dividend yield is unavailable; missing for: {names}."
    ]


def _allocation_changes(
    request: OptimizationRequest, known: dict[str, Security], weights: np.ndarray
) -> list[AllocationChange]:
    changes = []
    for security, weight in zip(request.securities, weights, strict=True):
        optimized = float(weight) * 100
        changes.append(
            AllocationChange(
                ticker=security.ticker,
                security_name=known[security.ticker].name,
                current_weight=security.current_weight,
                optimized_weight=optimized,
                change=optimized - security.current_weight,
            )
        )
    return changes


def _metrics(problem: OptimizationProblem, weights: np.ndarray) -> PortfolioMetrics:
    returns = portfolio_returns(problem.returns, weights)
    factor = problem.annualization_factor
    return PortfolioMetrics(
        cagr=_defined(cagr, returns, factor),
        volatility=_defined(annualized_volatility, returns, factor),
        sharpe_ratio=_defined(sharpe_ratio, returns, factor, problem.risk_free_rate),
        max_drawdown=_defined(maximum_drawdown, returns),
        dividend_yield=(
            portfolio_dividend_yield(problem.yields, weights)
            if problem.yields_known
            else None
        ),
    )


def _defined(function, *args) -> float | None:
    """Report an undefined metric (e.g. Sharpe at zero volatility) as None."""
    try:
        return function(*args)
    except UndefinedMetricError:
        return None
