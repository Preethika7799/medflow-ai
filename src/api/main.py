from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from medflow.config import get_settings
from medflow.exceptions import MedFlowError
from medflow.generation.pipeline import GenerationPipeline
from medflow.ingestion.pipeline import IngestionPipeline
from medflow.qdrant_utils import build_async_qdrant_client
from api.middleware import CorrelationIdMiddleware, build_limiter, configure_structlog
from api.routes import documents, evaluate, health, query

logger = structlog.get_logger(__name__)
configure_structlog()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    qdrant = build_async_qdrant_client(settings)
    ingestion = IngestionPipeline(settings, qdrant=qdrant)
    generation = GenerationPipeline(settings)
    app.state.store = {
        "settings": settings,
        "qdrant": qdrant,
        "ingestion": ingestion,
        "generation": generation,
    }
    logger.info("app_startup", env=settings.env)
    yield
    await qdrant.close()
    logger.info("app_shutdown")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title="MedFlow AI API",
        version="0.1.0",
        lifespan=lifespan,
    )
    limiter = build_limiter(settings.api.rate_limit_per_minute)
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    app.add_middleware(CorrelationIdMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api.cors_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(MedFlowError)
    async def medflow_err(_: Request, exc: MedFlowError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"detail": exc.message, "error": exc.__class__.__name__})

    @app.exception_handler(RequestValidationError)
    async def validation_err(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": exc.errors()})

    app.include_router(health.router)
    app.include_router(documents.router, prefix="/api/v1")
    app.include_router(query.router, prefix="/api/v1")
    app.include_router(evaluate.router, prefix="/api/v1")
    return app


app = create_app()
