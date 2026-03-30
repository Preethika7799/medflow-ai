from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from medflow.evaluation.runner import EvaluationReport, EvaluationRunner

__all__ = ["EvaluationReport", "EvaluationRunner"]


def __getattr__(name: str):
    if name in ("EvaluationReport", "EvaluationRunner"):
        from medflow.evaluation import runner as runner_mod

        return getattr(runner_mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
