from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from medflow.deidentify.pipeline import DeIDPipeline, DeIDResult

__all__ = ["DeIDPipeline", "DeIDResult"]


def __getattr__(name: str):
    if name in ("DeIDPipeline", "DeIDResult"):
        from medflow.deidentify import pipeline as pipeline_mod

        return getattr(pipeline_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
