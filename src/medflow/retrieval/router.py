from __future__ import annotations

import enum
import json
import re
import structlog

from medflow.config import MedFlowSettings, RetrievalStrategy
from medflow.providers.base import Message
from medflow.providers.factory import ProviderFactory

logger = structlog.get_logger(__name__)


class QueryIntent(str, enum.Enum):
    """High-level query archetypes."""

    FACTUAL = "FACTUAL"
    ANALYTICAL = "ANALYTICAL"
    KEYWORD = "KEYWORD"
    COMPARISON = "COMPARISON"


class QueryRouter:
    """Map natural language queries to retrieval strategies."""

    def __init__(self, settings: MedFlowSettings) -> None:
        self._settings = settings
        self._llm = ProviderFactory.create_llm(settings)

    async def route(self, query: str) -> RetrievalStrategy:
        """Classify ``query`` and return a :class:`RetrievalStrategy`."""
        system = (
            "You route healthcare document RAG queries. Respond ONLY JSON: "
            '{"intent":"FACTUAL|ANALYTICAL|KEYWORD|COMPARISON","confidence":0.0}'
        )
        user = f"Query:\n{query}\n"
        resp = await self._llm.chat(
            [Message(role="system", content=system), Message(role="user", content=user)],
            temperature=0.0,
            max_tokens=64,
        )
        raw = resp.content.strip()
        try:
            m = re.search(r"\{[^}]+\}", raw, re.DOTALL)
            payload = json.loads(m.group(0) if m else raw)
            raw_intent = str(payload.get("intent", "FACTUAL")).upper()
            try:
                intent = QueryIntent(raw_intent)
            except ValueError:
                intent = QueryIntent.FACTUAL
        except (json.JSONDecodeError, KeyError, ValueError):
            logger.warning("query_router_fallback", raw=raw[:300])
            intent = QueryIntent.FACTUAL

        if intent == QueryIntent.KEYWORD:
            strat = RetrievalStrategy.SPARSE
        elif intent == QueryIntent.ANALYTICAL or intent == QueryIntent.COMPARISON:
            strat = RetrievalStrategy.DENSE
        else:
            strat = RetrievalStrategy.HYBRID

        logger.info("query_routed", intent=intent.value, strategy=strat.value)
        return strat
