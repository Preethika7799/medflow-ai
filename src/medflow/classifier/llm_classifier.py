from __future__ import annotations

import json
import re
import structlog

from medflow.classifier.categories import DOCUMENT_CATEGORY_DESCRIPTIONS, DocumentCategory
from medflow.config import MedFlowSettings
from medflow.providers.base import Message
from medflow.providers.factory import ProviderFactory

logger = structlog.get_logger(__name__)


class LLMClassifier:
    def __init__(self, settings: MedFlowSettings) -> None:
        self._settings = settings
        self._llm = ProviderFactory.create_llm(settings)

    async def classify(self, text: str, *, max_chars: int = 6000) -> DocumentCategory:
        """Map ``text`` to a :class:`DocumentCategory` (truncated to ``max_chars``)."""
        excerpt = text[:max_chars]
        desc = "\n".join(f"- {k.value}: {v}" for k, v in DOCUMENT_CATEGORY_DESCRIPTIONS.items())
        system = (
            "You are a healthcare document triage assistant. "
            "Choose exactly one category from the list. Respond ONLY as JSON: "
            '{"category": "<ENUM>", "confidence": 0.0}'
        )
        user = f"Categories:\n{desc}\n\nDocument excerpt:\n{excerpt}\n"
        resp = await self._llm.chat(
            [Message(role="system", content=system), Message(role="user", content=user)],
            temperature=0.0,
            max_tokens=128,
        )
        raw = resp.content.strip()
        try:
            m = re.search(r"\{[^}]+\}", raw, re.DOTALL)
            payload = json.loads(m.group(0) if m else raw)
            name = str(payload.get("category", "")).strip().upper()
            for cat in DocumentCategory:
                if cat.name == name or cat.value == name:
                    return cat
            return DocumentCategory.OTHER
        except (json.JSONDecodeError, KeyError, ValueError):
            logger.warning("llm_classify_fallback", raw=raw[:500])
            return DocumentCategory.OTHER
