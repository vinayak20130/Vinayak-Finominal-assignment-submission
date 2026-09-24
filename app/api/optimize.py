from typing import Annotated

from fastapi import APIRouter, Body, Depends

from app.api.dependencies import get_market_data
from app.api.examples import OPTIMIZE_EXAMPLES
from app.data.market_data import MarketData
from app.schemas.request import OptimizationRequest
from app.schemas.response import OptimizationResponse
from app.services.optimizer import optimize

router = APIRouter()


# A plain `def` runs in FastAPI's thread pool, so CPU-bound optimization
# does not block the event loop.
@router.post("/optimize", response_model=OptimizationResponse)
def optimize_portfolio(
    request: Annotated[OptimizationRequest, Body(openapi_examples=OPTIMIZE_EXAMPLES)],
    market: Annotated[MarketData, Depends(get_market_data)],
) -> OptimizationResponse:
    """Optimize portfolio weights for the requested strategy.

    Securities are referenced by ticker; their returns, names and dividend yields
    come from the database. Weights are percentages (20 means 20%); portfolio
    constraints are decimals (0.025 means 2.5%).
    """
    return optimize(request, market)
