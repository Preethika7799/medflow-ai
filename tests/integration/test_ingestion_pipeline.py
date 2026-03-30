from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from medflow.classifier.categories import DocumentCategory
from medflow.config import MedFlowSettings
from medflow.ingestion.pipeline import IngestionPipeline


@pytest.mark.asyncio
async def test_ingest_txt_minimal(project_root: Path) -> None:
    settings = MedFlowSettings.from_yaml(project_root / "configs", profile="development")
    txt_dir = project_root / "data" / "synthetic" / "documents"
    files = list(txt_dir.glob("*.txt"))
    if not files:
        pytest.skip("synthetic corpus not generated")

    fake_embed = MagicMock()
    fake_embed.embed = AsyncMock(return_value=[[0.1] * 384])

    qdrant = MagicMock()
    qdrant.get_collections = AsyncMock(return_value=MagicMock(collections=[MagicMock(name=settings.qdrant.collection_name)]))
    qdrant.upsert = AsyncMock()

    with (
        patch("medflow.ingestion.pipeline.LLMClassifier") as clf,
        patch("medflow.ingestion.pipeline.DeIDPipeline") as deid,
    ):
        clf.return_value.classify = AsyncMock(return_value=DocumentCategory.OTHER)
        deid_inst = MagicMock()
        deid_ret = MagicMock()
        deid_ret.anonymized_text = "synthetic de-identified body text " * 20
        deid_inst.process.return_value = deid_ret
        deid.return_value = deid_inst
        pipeline = IngestionPipeline(settings, qdrant=qdrant, embedder=fake_embed)
        res = await pipeline.ingest(files[0])
        assert res.doc_id
        assert res.chunk_count >= 0
