from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from medflow.config import MedFlowSettings
from medflow.exceptions import RetrievalError
from medflow.providers.embedding import EmbeddingProvider

logger = structlog.get_logger(__name__)


@dataclass
class RetrievalHit:
    id: str
    score: float
    text: str
    metadata: dict[str, Any]


class DenseRetriever:
    def __init__(self, settings: MedFlowSettings, client: AsyncQdrantClient, embedder: EmbeddingProvider) -> None:
        self._settings = settings
        self._client = client
        self._embedder = embedder

    async def retrieve(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[RetrievalHit]:
        try:
            qv = (await self._embedder.embed([query]))[0]
        except Exception as e:
            raise RetrievalError(str(e)) from e

        q_filter = self._build_filter(filters)
        try:
            res = await self._client.query_points(
                collection_name=self._settings.qdrant.collection_name,
                query=qv,
                limit=top_k,
                query_filter=q_filter,
                with_payload=True,
            )
        except Exception as e:
            logger.exception("dense_retrieve_failed")
            raise RetrievalError(str(e)) from e

        hits: list[RetrievalHit] = []
        for r in res.points:
            payload = r.payload or {}
            text = str(payload.get("text", ""))
            hits.append(
                RetrievalHit(
                    id=str(r.id),
                    score=float(r.score),
                    text=text,
                    metadata=dict(payload),
                ),
            )
        return hits

    def _build_filter(self, filters: dict[str, Any] | None) -> Filter | None:
        if not filters:
            return None
        must: list[FieldCondition] = []
        if dt := filters.get("doc_type"):
            must.append(FieldCondition(key="doc_type", match=MatchValue(value=str(dt))))
        return Filter(must=must) if must else None
