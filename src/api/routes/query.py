from __future__ import annotations

from fastapi import APIRouter, Request

from api.schemas import QueryRequest, QueryResponse
from medflow.config import RetrievalStrategy
from medflow.observability.metrics_collector import get_metrics_collector

router = APIRouter(prefix="/query", tags=["query"])


def _parse_strategy(s: str | None) -> RetrievalStrategy | None:
    if not s:
        return None
    try:
        return RetrievalStrategy(s.lower())
    except ValueError:
        return RetrievalStrategy.AUTO


@router.post("", response_model=QueryResponse)
async def run_query(request: Request, body: QueryRequest) -> QueryResponse:
    """Run retrieval + generation for a user question."""
    generation = request.app.state.store["generation"]
    filters = body.filters.model_dump(exclude_none=True) if body.filters else None
    strat = _parse_strategy(body.strategy)
    res = await generation.run(body.query, filters=filters, strategy=strat)
    total_ms = float(res.metrics.get("retrieval_ms", 0)) + float(res.metrics.get("generation_ms", 0))
    get_metrics_collector().record_query(total_ms=total_ms)
    return QueryResponse(
        answer=res.answer,
        citations=res.citations,
        strategy_used=res.strategy_used,
        metrics=res.metrics,
    )
