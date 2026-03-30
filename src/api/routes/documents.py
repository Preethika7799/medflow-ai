from __future__ import annotations

import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import structlog
from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from api.schemas import DocumentDetail, DocumentSummary, UploadResponse
from medflow.exceptions import IngestionError
from medflow.observability.audit_log import audit_event

logger = structlog.get_logger(__name__)

router = APIRouter(prefix="/documents", tags=["documents"])


@router.post("/upload", response_model=UploadResponse)
async def upload_document(request: Request, file: UploadFile = File(...)) -> UploadResponse:
    """Accept PDF/image/text, run ingestion pipeline."""
    ingestion = request.app.state.store["ingestion"]
    suffix = Path(file.filename or "upload").suffix or ".bin"
    data = await file.read()
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        result = await ingestion.ingest(tmp_path)
    except IngestionError:
        raise
    except Exception as e:
        logger.exception("upload_failed")
        raise IngestionError(str(e)) from e
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    audit_event(
        "document_ingest",
        subject=result.doc_id,
        outcome="success",
        details={"classification": result.classification, "chunks": result.chunk_count},
    )
    return UploadResponse(
        doc_id=result.doc_id,
        classification=result.classification,
        chunk_count=result.chunk_count,
        processing_time_ms=result.processing_time_ms,
    )


async def _scan_documents(request: Request) -> dict[str, dict[str, Any]]:
    client = request.app.state.store["qdrant"]
    settings = request.app.state.store["settings"]
    grouped: dict[str, dict[str, Any]] = defaultdict(lambda: {"chunks": [], "meta": {}})
    offset = None
    while True:
        batch = await client.scroll(
            collection_name=settings.qdrant.collection_name,
            limit=256,
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
            doc_id = str(pl.get("qdrant_doc_id", p.id))
            grouped[doc_id]["chunks"].append(
                {
                    "chunk_id": pl.get("chunk_id"),
                    "position": pl.get("position"),
                    "text_preview": str(pl.get("text", ""))[:500],
                },
            )
            grouped[doc_id]["meta"] = {
                "title": pl.get("source_doc_id", doc_id),
                "doc_type": pl.get("doc_type", ""),
            }
        if offset is None:
            break
    return grouped


@router.get("", response_model=list[DocumentSummary])
async def list_documents(request: Request) -> list[DocumentSummary]:
    """List ingested documents aggregated from vectors."""
    grouped = await _scan_documents(request)
    out: list[DocumentSummary] = []
    for doc_id, data in grouped.items():
        meta = data["meta"]
        out.append(
            DocumentSummary(
                doc_id=doc_id,
                title=str(meta.get("title", doc_id)),
                doc_type=str(meta.get("doc_type", "")),
                chunk_count=len(data["chunks"]),
            ),
        )
    return sorted(out, key=lambda x: x.title)


@router.get("/{document_id}", response_model=DocumentDetail)
async def get_document(document_id: str, request: Request) -> DocumentDetail:
    """Return chunk previews for a document id."""
    grouped = await _scan_documents(request)
    if document_id not in grouped:
        raise HTTPException(status_code=404, detail="Document not found")
    data = grouped[document_id]
    meta = data["meta"]
    return DocumentDetail(
        doc_id=document_id,
        title=str(meta.get("title", document_id)),
        doc_type=str(meta.get("doc_type", "")),
        chunks=sorted(data["chunks"], key=lambda c: int(c.get("position") or 0)),
    )
