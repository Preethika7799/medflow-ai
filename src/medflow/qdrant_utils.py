from __future__ import annotations

from typing import TYPE_CHECKING

from qdrant_client import AsyncQdrantClient, QdrantClient

if TYPE_CHECKING:
    from medflow.config import MedFlowSettings


def build_async_qdrant_client(settings: MedFlowSettings) -> AsyncQdrantClient:
    qc = settings.qdrant
    if qc.url and str(qc.url).strip():
        return AsyncQdrantClient(
            url=str(qc.url).strip(),
            api_key=qc.api_key,
        )
    return AsyncQdrantClient(
        host=qc.host,
        port=qc.port,
        api_key=qc.api_key,
    )


def build_sync_qdrant_client(settings: MedFlowSettings) -> QdrantClient:
    qc = settings.qdrant
    if qc.url and str(qc.url).strip():
        return QdrantClient(
            url=str(qc.url).strip(),
            api_key=qc.api_key,
        )
    return QdrantClient(
        host=qc.host,
        port=qc.port,
        api_key=qc.api_key,
    )
