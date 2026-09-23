from fastapi import FastAPI

from app.api.errors import register_error_handlers
from app.api.optimize import router as optimize_router

app = FastAPI(
    title="Portfolio Optimizer API",
    description="Optimizes portfolio weights from historical returns.",
)
register_error_handlers(app)
app.include_router(optimize_router)


@app.get("/health")
def health():
    return {"status": "ok"}
