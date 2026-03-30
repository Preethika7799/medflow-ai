from __future__ import annotations

import time

import structlog
from sentence_transformers import CrossEncoder

from medflow.config import MedFlowSettings
from medflow.retrieval.dense import RetrievalHit

logger = structlog.get_logger(__name__)


class CrossEncoderReranker:
    """ms-marco cross-encoder scoring."""

    def __init__(self, settings: MedFlowSettings) -> None:
        self._name = settings.reranker.model_name
        self._model: CrossEncoder | None = None

    def _ensure(self) -> CrossEncoder:
        if self._model is None:
            logger.info("loading_reranker", model=self._name)
            self._model = CrossEncoder(self._name)
        return self._model

    def rerank(self, query: str, results: list[RetrievalHit], top_k: int) -> list[RetrievalHit]:
        """Re-score hits and trim to ``top_k``."""
        if not results:
            return []
        start = time.perf_counter()
        model = self._ensure()
        pairs = [(query, h.text) for h in results]
        scores = model.predict(pairs)
        latency_ms = (time.perf_counter() - start) * 1000
        enriched: list[RetrievalHit] = []
        for h, s in zip(results, scores, strict=False):
            meta = dict(h.metadata)
            meta["reranking_score"] = float(s)
            meta["reranking_latency_ms"] = latency_ms
            enriched.append(RetrievalHit(id=h.id, score=float(s), text=h.text, metadata=meta))
        enriched.sort(key=lambda x: x.score, reverse=True)
        return enriched[:top_k]
