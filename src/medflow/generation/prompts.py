from __future__ import annotations

SYSTEM_PROMPT = """You are a healthcare document assistant working on DE-IDENTIFIED text only.
Answer based ONLY on the provided context. Cite sources using [Doc-<source_doc_id>, Chunk-<position>] format.
If the answer is not in the context, say you cannot find it in the documents. Never invent PHI."""


def format_context(chunks: list[dict[str, str | int]]) -> str:
    """Render retrieval chunks with explicit identifiers for citations."""
    parts: list[str] = []
    for i, ch in enumerate(chunks, start=1):
        sid = str(ch.get("source_doc_id", "unknown"))
        pos = int(ch.get("position", i))
        txt = str(ch.get("text", "")).strip()
        parts.append(f"[{i}] source_doc_id={sid} chunk_position={pos}\n{txt}")
    return "\n\n".join(parts)
