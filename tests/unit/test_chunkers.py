from __future__ import annotations

from medflow.config import MedFlowSettings
from medflow.ingestion.chunkers import FixedSizeChunker, RecursiveChunker


def test_fixed_chunker() -> None:
    c = FixedSizeChunker(100, 10)
    chunks = c.chunk("a" * 250, {"source_doc_id": "d1"})
    assert len(chunks) >= 2
    assert all(ch.source_doc_id == "d1" for ch in chunks)


def test_recursive_chunker() -> None:
    settings = MedFlowSettings()
    c = RecursiveChunker(
        settings.chunking.chunk_size,
        settings.chunking.chunk_overlap,
        settings.chunking.separators,
    )
    text = "Para one.\n\nPara two.\n\nPara three with more words."
    chunks = c.chunk(text, {"source_doc_id": "d2"})
    assert len(chunks) >= 1
