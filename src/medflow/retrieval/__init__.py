from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from medflow.retrieval.pipeline import RetrievalPipeline

__all__ = ["RetrievalPipeline"]


def __getattr__(name: str):  # PEP 562
    if name == "RetrievalPipeline":
        from medflow.retrieval.pipeline import RetrievalPipeline

        return RetrievalPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
