"""Map domain and validation errors to one JSON error envelope.

Every error body is {"error": {"code", "message", "details"}}. Details name the
offending field; they never echo input arrays, and no traceback is exposed.
"""

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.domain.errors import (
    DataUnavailableError,
    DataValidationError,
    EqualWeightConflictError,
    InfeasibleConstraintsError,
    InsufficientDataError,
    OptimizationFailedError,
    TickerNotFoundError,
    UndefinedMetricError,
)

logger = logging.getLogger(__name__)

# Most specific first; handlers are matched on the exception's own class.
DOMAIN_ERRORS: dict[type[Exception], tuple[int, str]] = {
    InsufficientDataError: (422, "insufficient_data"),
    DataValidationError: (422, "invalid_input"),
    UndefinedMetricError: (422, "invalid_input"),
    InfeasibleConstraintsError: (422, "infeasible_constraints"),
    EqualWeightConflictError: (422, "equal_weight_conflict"),
    TickerNotFoundError: (404, "ticker_not_found"),
    DataUnavailableError: (503, "data_unavailable"),
    OptimizationFailedError: (500, "optimization_failed"),
}

HTTP_ERROR_CODES = {404: "not_found", 405: "method_not_allowed"}


def error_response(
    status: int, code: str, message: str, details: list[dict] | None = None
) -> JSONResponse:
    body = {"error": {"code": code, "message": message, "details": details or []}}
    return JSONResponse(status_code=status, content=body)


async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
    details = [
        {"field": _field(error), "message": error["msg"]} for error in exc.errors()
    ]
    return error_response(422, "invalid_input", "Request validation failed.", details)


def _field(error: dict) -> str:
    # A JSON decode error's location holds a character offset, not a field name.
    if error["type"] == "json_invalid":
        return "body"
    # Drop the leading "body" segment so fields read like request paths.
    return ".".join(str(part) for part in error["loc"][1:]) or "body"


def _domain_handler(status: int, code: str):
    async def handle(_: Request, exc: Exception) -> JSONResponse:
        return error_response(status, code, str(exc))

    return handle


async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
    """Wrap routing errors (unknown path, wrong method) in the same envelope."""
    code = HTTP_ERROR_CODES.get(exc.status_code, "http_error")
    response = error_response(exc.status_code, code, str(exc.detail))
    # Keep headers such as Allow on 405 responses.
    response.headers.update(exc.headers or {})
    return response


async def _unexpected_error(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error", exc_info=exc)
    return error_response(500, "internal_error", "An unexpected error occurred.")


def register_error_handlers(app: FastAPI) -> None:
    app.add_exception_handler(RequestValidationError, _validation_error)
    app.add_exception_handler(StarletteHTTPException, _http_error)
    for error_type, (status, code) in DOMAIN_ERRORS.items():
        app.add_exception_handler(error_type, _domain_handler(status, code))
    app.add_exception_handler(Exception, _unexpected_error)
