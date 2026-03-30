from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Any

import structlog

from medflow.config import MedFlowSettings
from medflow.generation.prompts import SYSTEM_PROMPT, format_context
from medflow.providers.base import Message
from medflow.providers.factory import ProviderFactory
from medflow.retrieval.dense import RetrievalHit

logger = structlog.get_logger(__name__)

_CITE_RE = re.compile(r"\[Doc-([^,\]]+),\s*Chunk-(\d+)\]|\[Doc-([^\]]+)\]")


@dataclass
class GenerationResult:
    """Model response with extracted citations."""

    answer: str
    citations: list[dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    token_usage: dict[str, int] = field(default_factory=dict)
    latency_ms: float = 0.0


class ResponseGenerator:
    """Call LLM with grounded context and validate citation markers."""

    def __init__(self, settings: MedFlowSettings) -> None:
        self._settings = settings
        self._llm = ProviderFactory.create_llm(settings)

    def _extract_citations(self, answer: str) -> list[dict[str, Any]]:
        refs: list[dict[str, Any]] = []
        for m in _CITE_RE.finditer(answer):
            if m.group(1) and m.group(2):
                refs.append({"doc": m.group(1).strip(), "chunk": int(m.group(2))})
            elif m.group(3):
                refs.append({"doc": m.group(3).strip(), "chunk": None})
        return refs

    def _validate_citations(self, refs: list[dict[str, Any]], hits: list[RetrievalHit]) -> list[dict[str, Any]]:
        allowed = {(h.metadata.get("source_doc_id"), int(h.metadata.get("position", -1))) for h in hits}
        valid: list[dict[str, Any]] = []
        for r in refs:
            doc = r.get("doc")
            chunk = r.get("chunk")
            ok = (doc, chunk if chunk is not None else -1) in allowed or any(doc == str(h.metadata.get("source_doc_id")) for h in hits)
            valid.append({**r, "valid": bool(ok)})
        return valid

    async def generate(self, query: str, context_hits: list[RetrievalHit]) -> GenerationResult:
        """Produce an answer with citations."""
        chunk_dicts: list[dict[str, str | int]] = []
        for h in context_hits:
            chunk_dicts.append(
                {
                    "text": h.text,
                    "source_doc_id": str(h.metadata.get("source_doc_id", "")),
                    "position": int(h.metadata.get("position", 0)),
                },
            )
        user = f"Context:\n{format_context(chunk_dicts)}\n\nQuestion: {query}\n"
        start = time.perf_counter()
        resp = await self._llm.chat(
            [Message(role="system", content=SYSTEM_PROMPT), Message(role="user", content=user)],
            temperature=self._settings.llm.temperature,
            max_tokens=min(self._settings.llm.max_tokens, 2048),
        )
        latency = (time.perf_counter() - start) * 1000
        refs = self._extract_citations(resp.content)
        validated = self._validate_citations(refs, context_hits)
        conf = sum(1 for r in validated if r.get("valid")) / len(validated) if validated else 0.7
        logger.info("generation_done", latency_ms=latency, citations=len(validated))
        return GenerationResult(
            answer=resp.content,
            citations=validated,
            confidence=float(conf),
            token_usage={
                "prompt_tokens": resp.usage.prompt_tokens,
                "completion_tokens": resp.usage.completion_tokens,
                "total_tokens": resp.usage.total_tokens,
            },
            latency_ms=latency,
        )
