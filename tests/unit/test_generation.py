from __future__ import annotations

import re

from medflow.generation.prompts import format_context


def test_format_context_includes_ids() -> None:
    ctx = format_context(
        [{"text": "hello", "source_doc_id": "doc1", "position": 0}],
    )
    assert "doc1" in ctx
    assert "hello" in ctx


def test_citation_pattern() -> None:
    text = "See [Doc-doc1, Chunk-2] for details."
    assert re.search(r"\[Doc-([^,\]]+),\s*Chunk-(\d+)\]", text)
