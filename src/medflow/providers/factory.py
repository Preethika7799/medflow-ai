from __future__ import annotations

import os

import structlog

from medflow.config import LLMProviderName, MedFlowSettings
from medflow.exceptions import ConfigurationError
from medflow.providers.anthropic_provider import AnthropicProvider
from medflow.providers.base import LLMProvider
from medflow.providers.embedding import EmbeddingProvider, SentenceTransformerEmbeddingProvider
from medflow.providers.ollama_provider import OllamaProvider
from medflow.providers.openai_provider import OpenAIProvider

logger = structlog.get_logger(__name__)


class ProviderFactory:
    @staticmethod
    def create(settings: MedFlowSettings) -> LLMProvider:
        return ProviderFactory.create_llm(settings)

    @staticmethod
    def create_llm(settings: MedFlowSettings) -> LLMProvider:
        cfg = settings.llm
        provider = cfg.provider
        if provider == LLMProviderName.OPENAI:
            key = os.environ.get("OPENAI_API_KEY")
            if not key:
                msg = "OPENAI_API_KEY is not set"
                raise ConfigurationError(msg)
            return OpenAIProvider(
                model=cfg.model,
                api_key=key,
                timeout_seconds=cfg.timeout_seconds,
            )
        if provider == LLMProviderName.ANTHROPIC:
            key = os.environ.get("ANTHROPIC_API_KEY")
            if not key:
                msg = "ANTHROPIC_API_KEY is not set"
                raise ConfigurationError(msg)
            return AnthropicProvider(
                model=cfg.model,
                api_key=key,
                timeout_seconds=cfg.timeout_seconds,
            )
        if provider == LLMProviderName.OLLAMA:
            host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
            return OllamaProvider(
                model=cfg.model,
                host=host,
                timeout_seconds=cfg.timeout_seconds,
            )
        msg = f"Unknown LLM provider: {provider}"
        raise ConfigurationError(msg)

    @staticmethod
    def create_embedding(settings: MedFlowSettings) -> EmbeddingProvider:
        return SentenceTransformerEmbeddingProvider(
            model_name=settings.embedding.model,
            batch_size=settings.embedding.batch_size,
        )
