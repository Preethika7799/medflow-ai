from __future__ import annotations

from typing import Any

import structlog
from langfuse import Langfuse

from medflow.config import MedFlowSettings

logger = structlog.get_logger(__name__)


class LangfuseTracer:
    """Optional Langfuse tracing when enabled in settings."""

    def __init__(self, settings: MedFlowSettings) -> None:
        self._settings = settings
        self._client: Langfuse | None = None
        if settings.langfuse.enabled and settings.langfuse.public_key and settings.langfuse.secret_key:
            self._client = Langfuse(
                public_key=settings.langfuse.public_key,
                secret_key=settings.langfuse.secret_key,
                host=settings.langfuse.host,
            )

    def log_generation(self, name: str, **kwargs: Any) -> None:
        """Emit a generation span if Langfuse is active."""
        if not self._client:
            return
        try:
            self._client.generation(name=name, **kwargs)
        except Exception:
            logger.warning("langfuse_log_failed", name=name)
