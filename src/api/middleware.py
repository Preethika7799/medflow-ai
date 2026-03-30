from __future__ import annotations

import uuid
from typing import Callable

import structlog
from fastapi import Request, Response
from slowapi import Limiter
from slowapi.util import get_remote_address
from starlette.middleware.base import BaseHTTPMiddleware

logger = structlog.get_logger(__name__)


def configure_structlog() -> None:
    """Configure structlog for JSON-friendly process logs."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(20),
        context_class=dict,
        cache_logger_on_first_use=True,
    )


class CorrelationIdMiddleware(BaseHTTPMiddleware):
    """Attach correlation ID header and context var."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        cid = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(correlation_id=cid)
        response: Response = await call_next(request)
        response.headers["X-Correlation-ID"] = cid
        return response


def build_limiter(settings_rate_per_minute: int) -> Limiter:
    """Rate limiter keyed by client IP."""
    return Limiter(key_func=get_remote_address, default_limits=[f"{settings_rate_per_minute}/minute"])
