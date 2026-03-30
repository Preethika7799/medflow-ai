from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from medflow.classifier.categories import DOCUMENT_CATEGORY_DESCRIPTIONS, DocumentCategory
    from medflow.classifier.llm_classifier import LLMClassifier

__all__ = ["DOCUMENT_CATEGORY_DESCRIPTIONS", "DocumentCategory", "LLMClassifier"]


def __getattr__(name: str):
    if name in ("DOCUMENT_CATEGORY_DESCRIPTIONS", "DocumentCategory"):
        from medflow.classifier import categories as categories_mod

        return getattr(categories_mod, name)
    if name == "LLMClassifier":
        from medflow.classifier.llm_classifier import LLMClassifier

        return LLMClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
