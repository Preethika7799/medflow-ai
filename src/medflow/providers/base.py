from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from medflow.exceptions import ProviderError


@dataclass(frozen=True)
class Message:
    role: str
    content: str


@dataclass(frozen=True)
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


@dataclass(frozen=True)
class LLMResponse:
    content: str
    model: str
    usage: TokenUsage
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    raw: dict[str, Any] = field(default_factory=dict)


def normalize_messages(messages: Sequence[Message | Mapping[str, str]]) -> list[Message]:
    """Dict ``role``/``content`` → ``Message``."""
    out: list[Message] = []
    for m in messages:
        if isinstance(m, Message):
            out.append(m)
        else:
            role = str(m.get("role", "user"))
            content = str(m.get("content", ""))
            out.append(Message(role=role, content=content))
    return out


class LLMProvider(ABC):
    @abstractmethod
    async def chat(self, messages: list[Message] | Sequence[Message | Mapping[str, str]], **kwargs: Any) -> LLMResponse:
        ...

    async def stream(
        self,
        messages: list[Message] | Sequence[Message | Mapping[str, str]],
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """Default implementation: one chunk with full text."""
        response = await self.chat(normalize_messages(list(messages)), **kwargs)
        yield response.content

    async def embed(self, texts: list[str]) -> list[list[float]]:
        msg = "no embeddings on this provider; use SentenceTransformerEmbeddingProvider"
        raise ProviderError(msg, provider=type(self).__name__)
