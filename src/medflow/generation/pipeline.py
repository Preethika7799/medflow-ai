from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from medflow.config import MedFlowSettings, RetrievalStrategy
from medflow.generation.generator import GenerationResult, ResponseGenerator
from medflow.retrieval.pipeline import RetrievalPipeline, RetrievalPipelineResult


@dataclass
class GenerationPipelineResult:
    """End-to-end RAG output."""

    answer: str
    citations: list[dict[str, Any]]
    strategy_used: str
    metrics: dict[str, float | int | str]


class GenerationPipeline:
    """retrieve → generate."""

    def __init__(self, settings: MedFlowSettings) -> None:
        self._settings = settings
        self._retrieval = RetrievalPipeline(settings)
        self._gen = ResponseGenerator(settings)

    @property
    def retrieval(self) -> RetrievalPipeline:
        """Expose retrieval for evaluation and observability."""
        return self._retrieval

    async def run(
        self,
        query: str,
        *,
        filters: dict[str, Any] | None = None,
        strategy: RetrievalStrategy | None = None,
    ) -> GenerationPipelineResult:
        """Execute full RAG."""
        retr: RetrievalPipelineResult = await self._retrieval.retrieve(query, strategy=strategy, filters=filters)
        gen: GenerationResult = await self._gen.generate(query, retr.hits)
        metrics: dict[str, float | int | str] = {
            "retrieval_ms": retr.retrieval_latency_ms,
            "reranking_ms": retr.reranking_latency_ms,
            "generation_ms": gen.latency_ms,
            "prompt_tokens": gen.token_usage.get("prompt_tokens", 0),
            "completion_tokens": gen.token_usage.get("completion_tokens", 0),
            "total_tokens": gen.token_usage.get("total_tokens", 0),
        }
        return GenerationPipelineResult(
            answer=gen.answer,
            citations=gen.citations,
            strategy_used=retr.strategy_used,
            metrics=metrics,
        )
