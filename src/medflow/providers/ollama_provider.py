from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
import structlog
from ollama import AsyncClient
from tenacity import retry, stop_after_attempt, wait_exponential

from medflow.exceptions import ProviderError
from medflow.providers.base import LLMProvider, LLMResponse, Message, TokenUsage, normalize_messages
from medflow.providers.metrics import get_provider_metrics

logger = structlog.get_logger(__name__)


class OllamaProvider(LLMProvider):
    """Ollama async chat client."""

    def __init__(
        self,
        *,
        model: str,
        host: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self._model = model
        self._host = host
        self._client = AsyncClient(host=host, timeout=timeout_seconds)
        self._timeout = timeout_seconds

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=20),
    )
    async def chat(self, messages: list[Message] | list[Any], **kwargs: Any) -> LLMResponse:
        """Run chat against Ollama."""
        norm = normalize_messages(list(messages))
        start = time.perf_counter()
        ollama_msgs = [{"role": m.role, "content": m.content} for m in norm]
        try:
            resp = await self._client.chat(
                model=kwargs.get("model", self._model),
                messages=ollama_msgs,
                options={"temperature": kwargs.get("temperature", 0.1)},
            )
        except (httpx.HTTPError, ConnectionError, TimeoutError, Exception) as e:
            logger.exception("ollama_chat_failed")
            raise ProviderError(str(e), provider="ollama") from e

        text = resp.get("message", {}).get("content", "") or ""
        pt = int(resp.get("prompt_eval_count", 0))
        ct = int(resp.get("eval_count", 0))
        latency = (time.perf_counter() - start) * 1000
        get_provider_metrics().record(
            provider="ollama",
            prompt_tokens=pt,
            completion_tokens=ct,
            cost_usd=0.0,
            latency_ms=latency,
        )
        model_name = str(resp.get("model", self._model))
        return LLMResponse(
            content=text,
            model=model_name,
            usage=TokenUsage(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct),
            cost_usd=0.0,
            latency_ms=latency,
            raw=dict(resp),
        )

    async def stream(self, messages: list[Message] | list[Any], **kwargs: Any) -> AsyncIterator[str]:
        """Stream tokens from Ollama."""
        norm = normalize_messages(list(messages))
        ollama_msgs = [{"role": m.role, "content": m.content} for m in norm]
        try:
            stream = await self._client.chat(
                model=kwargs.get("model", self._model),
                messages=ollama_msgs,
                stream=True,
            )
            async for chunk in stream:
                msg = chunk.get("message", {})
                c = msg.get("content")
                if c:
                    yield c
        except Exception as e:
            logger.exception("ollama_stream_failed")
            raise ProviderError(str(e), provider="ollama") from e
