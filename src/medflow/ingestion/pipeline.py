from __future__ import annotations

import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
from qdrant_client import AsyncQdrantClient

from medflow.qdrant_utils import build_async_qdrant_client
from qdrant_client.models import Distance, PointStruct, VectorParams

from medflow.classifier.categories import DocumentCategory
from medflow.classifier.llm_classifier import LLMClassifier
from medflow.config import MedFlowSettings
from medflow.deidentify.pipeline import DeIDPipeline
from medflow.exceptions import IngestionError
from medflow.ingestion.chunkers import get_chunker
from medflow.ingestion.loaders import ImageLoader, PDFLoader, TextLoader
from medflow.ingestion.metadata import extract_metadata
from medflow.ocr.pipeline import OCRPipeline
from medflow.providers.embedding import EmbeddingProvider
from medflow.providers.factory import ProviderFactory

logger = structlog.get_logger(__name__)


@dataclass
class IngestionResult:
    """Summary returned to API callers."""

    doc_id: str
    classification: str
    chunk_count: int
    processing_time_ms: float


class IngestionPipeline:
    """Load → OCR (if needed) → de-ID → classify → chunk → embed → Qdrant upsert."""

    def __init__(
        self,
        settings: MedFlowSettings,
        *,
        qdrant: AsyncQdrantClient | None = None,
        embedder: EmbeddingProvider | None = None,
    ) -> None:
        self._settings = settings
        self._ocr = OCRPipeline(settings)
        self._deid = DeIDPipeline(settings)
        self._classifier = LLMClassifier(settings)
        self._chunker = get_chunker(settings)
        self._embedder = embedder or ProviderFactory.create_embedding(settings)
        self._qdrant = qdrant or build_async_qdrant_client(settings)

    def _loader_for(self, path: Path) -> Any:
        suf = path.suffix.lower()
        if suf == ".pdf":
            return PDFLoader(self._ocr)
        if suf in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
            return ImageLoader(self._ocr)
        if suf in {".txt", ".md"}:
            return TextLoader()
        msg = f"Unsupported extension: {suf}"
        raise IngestionError(msg)

    async def _ensure_collection(self, dim: int) -> None:
        name = self._settings.qdrant.collection_name
        try:
            cols = await self._qdrant.get_collections()
            names = {c.name for c in cols.collections}
            if name in names:
                return
        except Exception as e:
            logger.warning("qdrant_list_collections_failed", error=str(e))
        dist = Distance.COSINE if self._settings.qdrant.distance_metric == "cosine" else Distance.DOT
        await self._qdrant.create_collection(
            collection_name=name,
            vectors_config=VectorParams(size=dim, distance=dist),
        )

    async def ingest(self, file_path: str | Path) -> IngestionResult:
        """Ingest a single file."""
        start = time.perf_counter()
        path = Path(file_path)
        doc_uuid = str(uuid.uuid4())
        loader = self._loader_for(path)
        try:
            raw = await loader.load(path)
        except Exception as e:
            logger.exception("ingestion_load_failed", path=str(path))
            raise IngestionError(str(e)) from e

        deid = self._deid.process(raw)
        safe_text = deid.anonymized_text
        try:
            category = await self._classifier.classify(safe_text)
        except Exception as e:
            logger.warning("ingestion_classify_failed", err=str(e))
            category = DocumentCategory.OTHER

        meta = extract_metadata(path, category)
        base_meta: dict[str, Any] = {
            "source_doc_id": meta.source_doc_id,
            "title": meta.title,
            "doc_type": category.value,
            "qdrant_doc_id": doc_uuid,
        }
        if meta.created_at:
            base_meta["created_at"] = meta.created_at

        chunks = self._chunker.chunk(safe_text, base_meta)
        if not chunks:
            return IngestionResult(
                doc_id=doc_uuid,
                classification=category.value,
                chunk_count=0,
                processing_time_ms=(time.perf_counter() - start) * 1000,
            )

        texts = [c.text for c in chunks]
        try:
            vectors = await self._embedder.embed(texts)
        except Exception as e:
            logger.exception("ingestion_embed_failed")
            raise IngestionError(str(e)) from e

        dim = len(vectors[0])
        await self._ensure_collection(dim)

        points: list[PointStruct] = []
        for ch, vec in zip(chunks, vectors, strict=False):
            payload = {
                "text": ch.text,
                "chunk_id": ch.chunk_id,
                "source_doc_id": ch.source_doc_id,
                "position": ch.position,
                "doc_type": category.value,
                "qdrant_doc_id": doc_uuid,
            }
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload=payload,
                ),
            )

        try:
            await self._qdrant.upsert(
                collection_name=self._settings.qdrant.collection_name,
                points=points,
            )
        except Exception as e:
            logger.exception("ingestion_qdrant_failed")
            raise IngestionError(str(e)) from e

        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            "ingestion_complete",
            doc_id=doc_uuid,
            classification=category.value,
            chunks=len(points),
            ms=elapsed,
        )
        return IngestionResult(
            doc_id=doc_uuid,
            classification=category.value,
            chunk_count=len(points),
            processing_time_ms=elapsed,
        )
