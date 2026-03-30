from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import structlog
from langchain_text_splitters import RecursiveCharacterTextSplitter

from medflow.config import ChunkingStrategy, MedFlowSettings

logger = structlog.get_logger(__name__)


@dataclass
class Chunk:
    """Segment of a source document."""

    text: str
    metadata: dict[str, Any]
    chunk_id: str
    source_doc_id: str
    position: int


class Chunker(ABC):
    """Split document text into retrievable chunks."""

    @abstractmethod
    def chunk(self, text: str, metadata: dict[str, Any]) -> list[Chunk]:
        """Produce ordered chunks with shared base metadata."""


class FixedSizeChunker(Chunker):
    """Fixed-size windows with overlap (character-based approximation)."""

    def __init__(self, chunk_size: int, overlap: int) -> None:
        self._size = chunk_size
        self._overlap = overlap

    def chunk(self, text: str, metadata: dict[str, Any]) -> list[Chunk]:
        """Chunk by character spans."""
        doc_id = str(metadata.get("source_doc_id", "unknown"))
        out: list[Chunk] = []
        start = 0
        pos = 0
        while start < len(text):
            end = min(len(text), start + self._size)
            piece = text[start:end].strip()
            if piece:
                cid = str(uuid4())
                out.append(
                    Chunk(
                        text=piece,
                        metadata=dict(metadata),
                        chunk_id=cid,
                        source_doc_id=doc_id,
                        position=pos,
                    ),
                )
                pos += 1
            if end >= len(text):
                break
            start = end - self._overlap
        return out


class RecursiveChunker(Chunker):
    """LangChain recursive character splitter."""

    def __init__(self, chunk_size: int, overlap: int, separators: list[str]) -> None:
        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=overlap,
            separators=separators,
        )

    def chunk(self, text: str, metadata: dict[str, Any]) -> list[Chunk]:
        """Recursive split into chunks."""
        doc_id = str(metadata.get("source_doc_id", "unknown"))
        docs = self._splitter.create_documents([text], metadatas=[metadata])
        out: list[Chunk] = []
        for i, d in enumerate(docs):
            cid = str(uuid4())
            out.append(
                Chunk(
                    text=d.page_content,
                    metadata=dict(d.metadata or {}),
                    chunk_id=cid,
                    source_doc_id=doc_id,
                    position=i,
                ),
            )
        return out


class SemanticChunker(Chunker):
    """Sentence-based chunker via LlamaIndex (falls back to recursive on failure)."""

    def __init__(self, settings: MedFlowSettings) -> None:
        self._fallback = RecursiveChunker(
            settings.chunking.chunk_size,
            settings.chunking.chunk_overlap,
            settings.chunking.separators,
        )
        self._settings = settings

    def chunk(self, text: str, metadata: dict[str, Any]) -> list[Chunk]:
        """Use sentence-aware boundaries when LlamaIndex is available."""
        try:
            from llama_index.core import Document
            from llama_index.core.node_parser import SentenceSplitter

            splitter = SentenceSplitter(
                chunk_size=self._settings.chunking.chunk_size,
                chunk_overlap=self._settings.chunking.chunk_overlap,
            )
            doc = Document(text=text, metadata=dict(metadata))
            nodes = splitter.get_nodes_from_documents([doc])
        except Exception:
            logger.warning("semantic_chunker_fallback")
            return self._fallback.chunk(text, metadata)

        doc_id = str(metadata.get("source_doc_id", "unknown"))
        out: list[Chunk] = []
        for i, n in enumerate(nodes):
            cid = str(uuid4())
            piece = n.get_content(metadata_mode="none") if hasattr(n, "get_content") else str(n.text or "")
            out.append(
                Chunk(
                    text=piece,
                    metadata=dict(metadata),
                    chunk_id=cid,
                    source_doc_id=doc_id,
                    position=i,
                ),
            )
        return out if out else self._fallback.chunk(text, metadata)


def get_chunker(settings: MedFlowSettings) -> Chunker:
    """Factory for configured chunker."""
    if settings.chunking.strategy == ChunkingStrategy.FIXED:
        return FixedSizeChunker(settings.chunking.chunk_size, settings.chunking.chunk_overlap)
    if settings.chunking.strategy == ChunkingStrategy.SEMANTIC:
        return SemanticChunker(settings)
    return RecursiveChunker(
        settings.chunking.chunk_size,
        settings.chunking.chunk_overlap,
        settings.chunking.separators,
    )
