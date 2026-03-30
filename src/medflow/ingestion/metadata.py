from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from medflow.classifier.categories import DocumentCategory


@dataclass
class DocumentMetadata:
    """Indexing metadata stored alongside chunks."""

    source_doc_id: str
    title: str
    doc_type: DocumentCategory
    created_at: str | None = None
    extra: dict[str, Any] | None = None


_DATE_RE = re.compile(r"(20\d{2}|19\d{2})[./-](\d{1,2})[./-](\d{1,2})")


def extract_metadata(path: str | Path, category: DocumentCategory) -> DocumentMetadata:
    """Derive metadata from filesystem path and classified category."""
    p = Path(path)
    stem = p.stem
    created = None
    m = _DATE_RE.search(stem)
    if m:
        try:
            created = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3))).date().isoformat()
        except ValueError:
            created = None
    doc_id = stem.replace(" ", "_")
    return DocumentMetadata(
        source_doc_id=doc_id,
        title=stem,
        doc_type=category,
        created_at=created,
        extra={"filename": p.name},
    )
