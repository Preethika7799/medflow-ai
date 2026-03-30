from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

from medflow.exceptions import ProviderError

logger = structlog.get_logger(__name__)


class EmbeddingProvider(ABC):
    """Encode text batches to dense vectors."""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Return embedding vectors for each input string."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimensionality."""


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Local sentence-transformers model (runs encode in thread pool from async callers)."""

    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        self._model_name = model_name
        self._batch_size = batch_size
        self._model: SentenceTransformer | None = None

    def _ensure_model(self) -> SentenceTransformer:
        if self._model is None:
            logger.info("loading_embedding_model", model=self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    @property
    def dimension(self) -> int:
        """Return model output dimension."""
        m = self._ensure_model()
        dim_fn: Any = getattr(m, "get_sentence_embedding_dimension", None)
        if callable(dim_fn):
            return int(dim_fn())
        msg = "Could not determine embedding dimension"
        raise ProviderError(msg, provider="sentence-transformers")

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Encode texts; CPU-bound work runs in default executor when called from async."""
        import asyncio

        if not texts:
            return []

        def _encode() -> np.ndarray:
            model = self._ensure_model()
            return model.encode(
                texts,
                batch_size=self._batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )

        try:
            loop = asyncio.get_running_loop()
            vec: np.ndarray = await loop.run_in_executor(None, _encode)
        except Exception as e:
            logger.exception("embedding_failed")
            raise ProviderError(str(e), provider="sentence-transformers") from e

        return [row.astype(float).tolist() for row in np.atleast_2d(vec)]
