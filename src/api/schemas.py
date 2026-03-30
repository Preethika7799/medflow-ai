from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Result of document ingestion."""

    doc_id: str
    classification: str
    chunk_count: int
    processing_time_ms: float

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "doc_id": "f47ac10b-58cc-4372-a567-0e02b2c3d479",
                    "classification": "PRIOR_AUTH",
                    "chunk_count": 12,
                    "processing_time_ms": 8420.5,
                }
            ]
        }
    }


class QueryFilters(BaseModel):
    """Optional metadata filters for retrieval."""

    doc_type: str | None = None
    date_range: dict[str, str] | None = None


class QueryRequest(BaseModel):
    """RAG query body."""

    query: str = Field(..., min_length=1)
    filters: QueryFilters | None = None
    strategy: str | None = Field(
        default=None,
        description="dense | sparse | hybrid | auto",
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "query": "What CPT code is requested for the lumbar MRI?",
                    "filters": {"doc_type": "PRIOR_AUTH"},
                    "strategy": "hybrid",
                }
            ]
        }
    }


class QueryResponse(BaseModel):
    """RAG answer payload."""

    answer: str
    citations: list[dict[str, Any]]
    strategy_used: str
    metrics: dict[str, Any]

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "answer": "The request lists CPT 72148 for MRI lumbar spine [Doc-doc_pa_003, Chunk-2].",
                    "citations": [{"doc": "doc_pa_003", "chunk": 2, "valid": True}],
                    "strategy_used": "hybrid",
                    "metrics": {
                        "retrieval_ms": 120.4,
                        "reranking_ms": 45.2,
                        "generation_ms": 900.1,
                        "total_tokens": 812,
                    },
                }
            ]
        }
    }


class DocumentSummary(BaseModel):
    """List view row."""

    doc_id: str
    title: str
    doc_type: str
    chunk_count: int


class DocumentDetail(BaseModel):
    """Detail with chunk previews."""

    doc_id: str
    title: str
    doc_type: str
    chunks: list[dict[str, Any]]


class EvaluateResponse(BaseModel):
    """Evaluation summary."""

    aggregate_metrics: dict[str, float]
    timestamp: str
    num_questions: int
