from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog
from qdrant_client import AsyncQdrantClient

from medflow.qdrant_utils import build_async_qdrant_client

from medflow.config import MedFlowSettings, RetrievalStrategy
from medflow.providers.embedding import EmbeddingProvider
from medflow.providers.factory import ProviderFactory
from medflow.retrieval.dense import DenseRetriever, RetrievalHit
from medflow.retrieval.hybrid import reciprocal_rank_fusion
from medflow.retrieval.reranker import CrossEncoderReranker
from medflow.retrieval.router import QueryRouter
from medflow.retrieval.sparse import BM25Retriever

logger = structlog.get_logger(__name__)


@dataclass
class RetrievalPipelineResult:
    """Retrieval stage output."""

    hits: list[RetrievalHit]
    strategy_used: str
    retrieval_latency_ms: float
    reranking_latency_ms: float


class RetrievalPipeline:
    """Configurable retrieval orchestrator."""

    def __init__(
        self,
        settings: MedFlowSettings,
        *,
        qdrant: AsyncQdrantClient | None = None,
        embedder: EmbeddingProvider | None = None,
    ) -> None:
        self._settings = settings
        self._router = QueryRouter(settings)
        self._qdrant = qdrant or build_async_qdrant_client(settings)
        self._embedder = embedder or ProviderFactory.create_embedding(settings)
        self._dense = DenseRetriever(settings, self._qdrant, self._embedder)
        self._sparse = BM25Retriever(settings)
        self._sparse.load_from_disk()
        self._reranker = CrossEncoderReranker(settings)

    async def retrieve(
        self,
        query: str,
        *,
        strategy: RetrievalStrategy | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalPipelineResult:
        """Run retrieval with optional forced strategy."""
        import time

        t0 = time.perf_counter()
        strat = strategy or self._settings.retrieval.strategy
        if strat == RetrievalStrategy.AUTO:
            strat = await self._router.route(query)

        top_k = self._settings.retrieval.top_k
        if strat == RetrievalStrategy.DENSE:
            hits = await self._dense.retrieve(query, top_k, filters)
        elif strat == RetrievalStrategy.SPARSE:
            if not self._sparse.ready:
                await self._sparse.sync_from_qdrant()
            hits = await self._sparse.retrieve(query, top_k)
        else:
            dense_hits = await self._dense.retrieve(query, top_k, filters)
            if not self._sparse.ready:
                await self._sparse.sync_from_qdrant()
            sparse_hits = await self._sparse.retrieve(query, top_k)
            hits = reciprocal_rank_fusion(
                [dense_hits, sparse_hits],
                k=self._settings.retrieval.rrf_k,
                top_k=top_k,
            )

        t1 = time.perf_counter()
        rerank_start = time.perf_counter()
        reranked = self._reranker.rerank(query, hits, self._settings.retrieval.rerank_top_k)
        t2 = time.perf_counter()
        logger.info(
            "retrieval_complete",
            strategy=strat.value,
            hits=len(reranked),
            retrieval_ms=(t1 - t0) * 1000,
            rerank_ms=(t2 - rerank_start) * 1000,
        )
        return RetrievalPipelineResult(
            hits=reranked,
            strategy_used=strat.value,
            retrieval_latency_ms=(t1 - t0) * 1000,
            reranking_latency_ms=(t2 - rerank_start) * 1000,
        )
