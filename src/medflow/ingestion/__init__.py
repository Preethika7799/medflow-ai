from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from medflow.ingestion.pipeline import IngestionPipeline, IngestionResult

__all__ = ["IngestionPipeline", "IngestionResult"]


def __getattr__(name: str):
    if name in ("IngestionPipeline", "IngestionResult"):
        from medflow.ingestion import pipeline as pipeline_mod

        return getattr(pipeline_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
