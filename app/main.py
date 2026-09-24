from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.errors import register_error_handlers
from app.api.optimize import router as optimize_router
from app.api.securities import router as securities_router
from app.data.database import create_database_engine
from app.data.market_data import MarketData
from app.data.postgres import connect_market_data


def create_app(market_data: MarketData | None = None) -> FastAPI:
    """Build the API.

    Without `market_data`, startup connects to PostgreSQL (DATABASE_URL) and fails
    fast, naming the fix, if the database is unreachable or empty.
    """

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if market_data is not None:
            app.state.market_data = market_data
            yield
            return
        engine = create_database_engine()
        try:
            app.state.market_data = connect_market_data(engine)
            yield
        finally:
            engine.dispose()

    app = FastAPI(
        title="Portfolio Optimizer API",
        description="Optimizes portfolio weights using market data from PostgreSQL.",
        lifespan=lifespan,
    )
    register_error_handlers(app)
    app.include_router(optimize_router)
    app.include_router(securities_router)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    return app


app = create_app()
