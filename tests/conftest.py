from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

os.environ.setdefault("MEDFLOW_ENV", "development")
os.environ.setdefault("OPENAI_API_KEY", "sk-test")


@pytest.fixture()
def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture()
def mock_llm_response():
    from medflow.providers.base import LLMResponse, TokenUsage

    return LLMResponse(
        content='{"category":"OTHER","confidence":0.9}',
        model="gpt-4o-mini",
        usage=TokenUsage(10, 5, 15),
        cost_usd=0.0,
        latency_ms=1.0,
    )


@pytest.fixture()
def mock_openai_provider(mock_llm_response, monkeypatch):
    from medflow.providers import openai_provider

    async def fake_chat(*_a, **_k):
        return mock_llm_response

    monkeypatch.setattr(openai_provider.OpenAIProvider, "chat", fake_chat)
    return mock_llm_response


@pytest.fixture()
def qdrant_mock():
    m = MagicMock()
    m.get_collections = AsyncMock(return_value=MagicMock(collections=[]))
    m.create_collection = AsyncMock()
    m.upsert = AsyncMock()
    m.search = AsyncMock(return_value=[])
    m.scroll = AsyncMock(return_value=([], None))
    m.close = AsyncMock()
    return m
