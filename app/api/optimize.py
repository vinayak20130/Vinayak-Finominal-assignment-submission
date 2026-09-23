from fastapi import APIRouter

from app.schemas.request import OptimizationRequest
from app.schemas.response import OptimizationResponse
from app.services.optimizer import optimize

router = APIRouter()


# A plain `def` runs in FastAPI's thread pool, so CPU-bound optimization
# does not block the event loop.
@router.post("/optimize", response_model=OptimizationResponse)
def optimize_portfolio(request: OptimizationRequest) -> OptimizationResponse:
    """Optimize portfolio weights for the requested strategy.

    Weights are percentages (20 means 20%); returns, yields, and portfolio
    constraints are decimals (0.025 means 2.5%).
    """
    return optimize(request)
