from __future__ import annotations

import time
from collections.abc import AsyncIterator
from typing import Any

import structlog
from anthropic import APIError, AsyncAnthropic, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from medflow.exceptions import ProviderError
from medflow.providers.base import LLMProvider, LLMResponse, Message, TokenUsage, normalize_messages
from medflow.providers.metrics import get_provider_metrics

logger = structlog.get_logger(__name__)

ANTHROPIC_PRICE_PER_1K: dict[str, tuple[float, float]] = {
    "claude-3-5-sonnet-20241022": (0.003, 0.015),
    "claude-3-haiku-20240307": (0.00025, 0.00125),
}


def _anthropic_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost for Anthropic models."""
    key = model.lower()
    for k, v in ANTHROPIC_PRICE_PER_1K.items():
        if k in key:
            return (input_tokens / 1000.0) * v[0] + (output_tokens / 1000.0) * v[1]
    return (input_tokens / 1000.0) * 0.003 + (output_tokens / 1000.0) * 0.015


class AnthropicProvider(LLMProvider):
    """Async Anthropic client."""

    def __init__(
        self,
        *,
        model: str,
        api_key: str | None = None,
        timeout_seconds: float = 120.0,
    ) -> None:
        self._model = model
        self._client = AsyncAnthropic(api_key=api_key, timeout=timeout_seconds)

    @staticmethod
    def _split_system(messages: list[Message]) -> tuple[str | None, list[Message]]:
        system_parts: list[str] = []
        rest: list[Message] = []
        for m in messages:
            if m.role == "system":
                system_parts.append(m.content)
            else:
                rest.append(m)
        system = "\n".join(system_parts) if system_parts else None
        return system, rest

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((RateLimitError, APIError)),
    )
    async def chat(self, messages: list[Message] | list[Any], **kwargs: Any) -> LLMResponse:
        """Call Anthropic messages API."""
        norm = normalize_messages(list(messages))
        system, rest = self._split_system(norm)
        anth_msgs: list[dict[str, Any]] = []
        for m in rest:
            if m.role not in ("user", "assistant"):
                anth_msgs.append({"role": "user", "content": m.content})
            else:
                anth_msgs.append({"role": m.role, "content": m.content})

        start = time.perf_counter()
        try:
            resp = await self._client.messages.create(
                model=kwargs.get("model", self._model),
                max_tokens=kwargs.get("max_tokens", 1024),
                temperature=kwargs.get("temperature", 0.1),
                system=system or "",
                messages=anth_msgs,
            )
        except Exception as e:
            logger.exception("anthropic_chat_failed")
            raise ProviderError(str(e), provider="anthropic") from e

        text_blocks = [b.text for b in resp.content if b.type == "text"]
        text = "".join(text_blocks)
        pt = resp.usage.input_tokens
        ct = resp.usage.output_tokens
        latency = (time.perf_counter() - start) * 1000
        cost = _anthropic_cost(resp.model, pt, ct)
        get_provider_metrics().record(
            provider="anthropic",
            prompt_tokens=pt,
            completion_tokens=ct,
            cost_usd=cost,
            latency_ms=latency,
        )
        return LLMResponse(
            content=text,
            model=resp.model,
            usage=TokenUsage(prompt_tokens=pt, completion_tokens=ct, total_tokens=pt + ct),
            cost_usd=cost,
            latency_ms=latency,
            raw={"id": resp.id},
        )

    async def stream(self, messages: list[Message] | list[Any], **kwargs: Any) -> AsyncIterator[str]:
        """Stream Anthropic message deltas."""
        norm = normalize_messages(list(messages))
        system, rest = self._split_system(norm)
        anth_msgs: list[dict[str, Any]] = []
        for m in rest:
            role = "user" if m.role != "assistant" else "assistant"
            if m.role == "system":
                continue
            anth_msgs.append({"role": role, "content": m.content})

        try:
            async with self._client.messages.stream(
                model=kwargs.get("model", self._model),
                max_tokens=kwargs.get("max_tokens", 1024),
                temperature=kwargs.get("temperature", 0.1),
                system=system or "",
                messages=anth_msgs,
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        except Exception as e:
            logger.exception("anthropic_stream_failed")
            raise ProviderError(str(e), provider="anthropic") from e
