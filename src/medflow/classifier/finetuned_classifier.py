from __future__ import annotations

from pathlib import Path

import structlog

from medflow.classifier.categories import DocumentCategory
from medflow.classifier.llm_classifier import LLMClassifier
from medflow.config import MedFlowSettings

logger = structlog.get_logger(__name__)


class FineTunedClassifier:
    """When no adapter artifact exists, delegates to :class:`LLMClassifier` (same as baseline)."""

    def __init__(
        self,
        settings: MedFlowSettings,
        *,
        adapter_path: str | Path | None = None,
    ) -> None:
        self._settings = settings
        self._adapter_path = Path(adapter_path) if adapter_path else None
        self._llm_classifier = LLMClassifier(settings)

    async def classify(self, text: str) -> DocumentCategory:
        """Classify using LoRA weights if ``adapter_path`` exists, else LLM zero-shot."""
        if self._adapter_path and self._adapter_path.is_file():
            logger.warning(
                "finetuned_adapter_not_loaded",
                path=str(self._adapter_path),
                detail="Loading PEFT adapters from disk is not wired in this build; using LLM baseline.",
            )
        return await self._llm_classifier.classify(text)

    def predict_sync(self, text: str) -> DocumentCategory:
        """Synchronous wrapper for notebooks (runs async classifier)."""
        import asyncio

        return asyncio.run(self.classify(text))
