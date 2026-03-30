from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from medflow.ocr.pipeline import DocumentText, OCRPipeline

__all__ = ["DocumentText", "OCRPipeline"]


def __getattr__(name: str):
    if name in ("DocumentText", "OCRPipeline"):
        from medflow.ocr import pipeline as pipeline_mod

        return getattr(pipeline_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
