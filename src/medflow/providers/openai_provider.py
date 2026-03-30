from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

import structlog
from openai import APIError, AsyncOpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from medflow.exceptions import ProviderError
from medflow.providers.base import LLMProvider, LLMResponse, Message, TokenUsage, normalize_messages
from medflow.providers.metrics import get_provider_metrics

logger = structlog.get_logger(__name__)

# Rough per-1K token defaults for cost estimation (override via env in production pricing tables).
OPENAI_PRICE_PER_1K: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.005, 0.015),
}


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Estimate USD cost from token counts."""
    key = model.lower()
    rates = OPENAI_PRICE_PER_1K.get(key, (0.00015, 0.0006))
    return (prompt_tokens / 1000.0) * rates[0] + (completion_tokens / 1000.0) * rates[1]


class OpenAIProvider(LLMProvider):
    """Async OpenAI client with retries and metrics."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self._model = model
        self._client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=timeout_seconds)

    def _messages_to_openai(self, messages: list[Message]) -> list[dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in messages]

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((RateLimitError, APIError)),
    )
    async def chat(self, messages: list[Message] | list[Any], **kwargs: Any) -> LLMResponse:
        """Call OpenAI chat completions API."""
        norm = normalize_messages(list(messages))
        start = time.perf_counter()
        try:
            resp = await self._client.chat.completions.create(
                model=kwargs.get("model", self._model),
                messages=self._messages_to_openai(norm),
                temperature=kwargs.get("temperature", 0.1),
                max_tokens=kwargs.get("max_tokens", 1024),
            )
        except Exception as e:
            logger.exception("openai_chat_failed")
            raise ProviderError(str(e), provider="openai") from e

        choice = resp.choices[0]
        text = choice.message.content or ""
        usage = resp.usage
        pt = usage.prompt_tokens if usage else 0
        ct = usage.completion_tokens if usage else 0
        tt = usage.total_tokens if usage else pt + ct
        latency = (time.perf_counter() - start) * 1000
        cost = _estimate_cost(resp.model or self._model, pt, ct)
        get_provider_metrics().record(
            provider="openai",
            prompt_tokens=pt,
            completion_tokens=ct,
            cost_usd=cost,
            latency_ms=latency,
        )
        return LLMResponse(
            content=text,
            model=resp.model or self._model,
            usage=TokenUsage(prompt_tokens=pt, completion_tokens=ct, total_tokens=tt),
            cost_usd=cost,
            latency_ms=latency,
            raw={"id": resp.id},
        )

    async def stream(self, messages: list[Message] | list[Any], **kwargs: Any) -> AsyncIterator[str]:
        """Stream chat completion chunks."""
        norm = normalize_messages(list(messages))
        try:
            stream = await self._client.chat.completions.create(
                model=kwargs.get("model", self._model),
                messages=self._messages_to_openai(norm),
                temperature=kwargs.get("temperature", 0.1),
                max_tokens=kwargs.get("max_tokens", 1024),
                stream=True,
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield delta
        except Exception as e:
            logger.exception("openai_stream_failed")
            raise ProviderError(str(e), provider="openai") from e

    async def embed(self, texts: list[str], *, embedding_model: str = "text-embedding-3-small") -> list[list[float]]:
        """Call OpenAI embeddings API."""
        try:
            resp = await self._client.embeddings.create(
                model=embedding_model,
                input=texts,
            )
        except Exception as e:
            logger.exception("openai_embed_failed")
            raise ProviderError(str(e), provider="openai") from e

        return [list(d.embedding) for d in resp.data]
