from __future__ import annotations

from typing import Any


def citation_accuracy(answer: str, citations: list[dict[str, Any]], actual_sources: list[str]) -> float:
    """Fraction of citations referencing allowed source document ids."""
    if not citations:
        return 1.0 if not actual_sources else 0.0
    allowed = set(actual_sources)
    ok = 0
    for c in citations:
        doc = str(c.get("doc", ""))
        if doc in allowed or any(doc in s for s in allowed):
            ok += 1
    return ok / max(len(citations), 1)
