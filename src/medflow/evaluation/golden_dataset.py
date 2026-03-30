from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import structlog

from medflow.exceptions import EvaluationError

logger = structlog.get_logger(__name__)


@dataclass
class GoldenQA:
    """Single evaluation record."""

    id: str
    question: str
    ground_truth: str
    source_doc_ids: list[str]
    category: str
    difficulty: str


def load_golden_dataset(path: str | Path, documents_dir: Path | None = None) -> list[GoldenQA]:
    """Load and validate ``golden_qa.json``."""
    p = Path(path)
    if not p.exists():
        msg = f"Missing golden dataset: {p}"
        raise EvaluationError(msg)
    rows = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        msg = "Golden dataset must be a JSON array"
        raise EvaluationError(msg)

    doc_ids_exist: set[str] = set()
    if documents_dir and documents_dir.exists():
        for f in documents_dir.glob("*.json"):
            try:
                meta = json.loads(f.read_text(encoding="utf-8"))
                if "doc_id" in meta:
                    doc_ids_exist.add(str(meta["doc_id"]))
            except json.JSONDecodeError:
                continue

    out: list[GoldenQA] = []
    for row in rows:
        gt = row.get("ground_truth") or row.get("ground_truth_answer")
        if not row.get("question") or not gt:
            logger.warning("golden_qa_skip_empty", row_id=row.get("id"))
            continue
        refs = list(row.get("source_doc_ids", []))
        if doc_ids_exist:
            for rid in refs:
                if rid not in doc_ids_exist:
                    logger.warning("golden_qa_missing_doc", ref=rid, qa_id=row.get("id"))
        out.append(
            GoldenQA(
                id=str(row["id"]),
                question=str(row["question"]),
                ground_truth=str(gt),
                source_doc_ids=refs,
                category=str(row.get("category", "")),
                difficulty=str(row.get("difficulty", "medium")),
            ),
        )
    return out
