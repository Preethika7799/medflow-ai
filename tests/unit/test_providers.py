from __future__ import annotations

import pytest

from medflow.config import MedFlowSettings, LLMProviderName
from medflow.exceptions import ConfigurationError
from medflow.providers.base import Message
from medflow.providers.factory import ProviderFactory


def test_message_dataclass() -> None:
    m = Message(role="user", content="hello")
    assert m.role == "user"


def test_factory_openai_requires_key(monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    settings = MedFlowSettings()
    settings.llm.provider = LLMProviderName.OPENAI
    with pytest.raises(ConfigurationError):
        ProviderFactory.create_llm(settings)
