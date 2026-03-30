from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from medflow.generation.pipeline import GenerationPipeline, GenerationPipelineResult

__all__ = ["GenerationPipeline", "GenerationPipelineResult"]


def __getattr__(name: str):
    if name in ("GenerationPipeline", "GenerationPipelineResult"):
        from medflow.generation import pipeline as pipeline_mod

        return getattr(pipeline_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
