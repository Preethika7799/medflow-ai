from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import structlog
from rank_bm25 import BM25Okapi

from medflow.config import MedFlowSettings
from medflow.qdrant_utils import build_async_qdrant_client
from medflow.retrieval.dense import RetrievalHit

logger = structlog.get_logger(__name__)


class BM25Retriever:
    """In-memory BM25 index built from JSON snapshot or Qdrant scroll."""

    def __init__(self, settings: MedFlowSettings) -> None:
        self._settings = settings
        self._path = Path(settings.retrieval.bm25_index_path)
        self._corpus: list[dict[str, Any]] = []
        self._bm25: BM25Okapi | None = None
        self._tokenized: list[list[str]] = []

    def load_from_disk(self) -> None:
        """Load precomputed corpus + tokenized texts."""
        if not self._path.exists():
            return
        data = json.loads(self._path.read_text(encoding="utf-8"))
        self._corpus = list(data.get("corpus", []))
        self._tokenized = [c["tokens"] for c in self._corpus]
        if self._tokenized:
            self._bm25 = BM25Okapi(self._tokenized)

    def save_to_disk(self) -> None:
        """Persist corpus for offline BM25."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"corpus": self._corpus}
        self._path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def index_documents(self, docs: list[dict[str, Any]]) -> None:
        """Replace index with ``docs`` entries containing ``text`` and metadata."""
        self._corpus = []
        self._tokenized = []
        for d in docs:
            text = str(d.get("text", ""))
            tokens = text.lower().split()
            entry = {"id": d.get("id"), "text": text, "metadata": d.get("metadata", {}), "tokens": tokens}
            self._corpus.append(entry)
            self._tokenized.append(tokens)
        self._bm25 = BM25Okapi(self._tokenized) if self._tokenized else None

    async def sync_from_qdrant(self, scroll_batch: int = 256) -> None:
        """Rebuild BM25 from all points in the configured Qdrant collection."""
        client = build_async_qdrant_client(self._settings)
        docs: list[dict[str, Any]] = []
        offset = None
        try:
            while True:
                batch = await client.scroll(
                    collection_name=self._settings.qdrant.collection_name,
                    limit=scroll_batch,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                )
                if hasattr(batch, "points"):
                    points = batch.points
                    offset = getattr(batch, "next_page_offset", None)
                else:
                    points, offset = batch[0], batch[1]
                for p in points:
                    pl = p.payload or {}
                    docs.append(
                        {
                            "id": str(p.id),
                            "text": str(pl.get("text", "")),
                            "metadata": dict(pl),
                        },
                    )
                if offset is None:
                    break
        finally:
            await client.close()
        self.index_documents(docs)
        self.save_to_disk()
        logger.info("bm25_synced", docs=len(docs))

    @property
    def ready(self) -> bool:
        """Whether a BM25 index is loaded."""
        return self._bm25 is not None

    async def retrieve(self, query: str, top_k: int) -> list[RetrievalHit]:
        """BM25 top-k."""
        if not self._bm25:
            self.load_from_disk()
        if not self._bm25:
            return []
        q_tokens = query.lower().split()
        scores = self._bm25.get_scores(q_tokens)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        hits: list[RetrievalHit] = []
        for idx in ranked:
            row = self._corpus[idx]
            meta = dict(row.get("metadata", {}))
            hits.append(
                RetrievalHit(
                    id=str(row.get("id", idx)),
                    score=float(scores[idx]),
                    text=str(row.get("text", "")),
                    metadata=meta,
                ),
            )
        return hits
