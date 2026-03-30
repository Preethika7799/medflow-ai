from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from medflow.config import MedFlowSettings, RetrievalStrategy
from medflow.retrieval.dense import RetrievalHit
from medflow.retrieval.pipeline import RetrievalPipeline, RetrievalPipelineResult


@pytest.mark.asyncio
async def test_retrieval_pipeline_dense() -> None:
    root = Path(__file__).resolve().parents[2]
    settings = MedFlowSettings.from_yaml(root / "configs", profile="development")
    hits = [RetrievalHit(id="1", score=0.9, text="ctx", metadata={"source_doc_id": "d", "position": 0})]

    async def fake_dense(*_a, **_k):
        return hits

    with (
        patch("medflow.retrieval.pipeline.QueryRouter") as qr,
        patch("medflow.retrieval.pipeline.CrossEncoderReranker") as cr,
    ):
        qr.return_value.route = AsyncMock(return_value=RetrievalStrategy.DENSE)
        cr.return_value.rerank = lambda q, h, k: h[:k]  # noqa: ARG005
        rp = RetrievalPipeline(settings, qdrant=MagicMock(), embedder=MagicMock())
        rp._dense.retrieve = fake_dense  # noqa: SLF001
        out: RetrievalPipelineResult = await rp.retrieve("hello", strategy=RetrievalStrategy.DENSE)
        assert out.hits
