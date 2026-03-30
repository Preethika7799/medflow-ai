from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import structlog
import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

load_dotenv()

logger = structlog.get_logger(__name__)


class LLMProviderName(str, Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    OLLAMA = "ollama"


class OCREngineName(str, Enum):
    PADDLE = "paddle"
    EASYOCR = "easyocr"


class ChunkingStrategy(str, Enum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


class RetrievalStrategy(str, Enum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"
    AUTO = "auto"


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in overlay.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml_config(config_dir: Path, profile: str) -> dict[str, Any]:
    default_path = config_dir / "default.yaml"
    if not default_path.exists():
        msg = f"Missing required config: {default_path}"
        raise FileNotFoundError(msg)
    with default_path.open(encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    if profile and profile != "default":
        overlay_path = config_dir / f"{profile}.yaml"
        if overlay_path.exists():
            with overlay_path.open(encoding="utf-8") as f:
                overlay = yaml.safe_load(f) or {}
            data = _deep_merge(data, overlay)
            logger.info("config_overlay_loaded", profile=profile, path=str(overlay_path))
    return data


class LLMConfig(BaseModel):
    provider: LLMProviderName = LLMProviderName.OPENAI
    model: str = "gpt-4o-mini"
    temperature: float = 0.1
    max_tokens: int = 4096
    timeout_seconds: float = 120.0
    max_retries: int = 3


class EmbeddingConfig(BaseModel):
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    dimension: int = 384
    batch_size: int = 32


class OCRPreprocessConfig(BaseModel):
    deskew: bool = True
    enhance_contrast: bool = True
    remove_noise: bool = False


class OCRConfig(BaseModel):
    engine: OCREngineName = OCREngineName.PADDLE
    language: str = "en"
    fallback_confidence_threshold: float = 0.75
    preprocessing: OCRPreprocessConfig = Field(default_factory=OCRPreprocessConfig)


class QdrantConfig(BaseModel):
    host: str = "localhost"
    port: int = 6333
    url: str | None = Field(default=None, description="Qdrant Cloud URL, e.g. https://cluster.aws.cloud.qdrant.io:6333")
    api_key: str | None = None
    collection_name: str = "medflow_chunks"
    distance_metric: Literal["cosine", "euclid", "dot"] = "cosine"


class RetrievalConfig(BaseModel):
    strategy: RetrievalStrategy = RetrievalStrategy.AUTO
    top_k: int = 8
    rerank_top_k: int = 4
    rrf_k: int = 60
    bm25_index_path: str = "data/bm25_index.json"


class ChunkingConfig(BaseModel):
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    chunk_size: int = 512
    chunk_overlap: int = 64
    separators: list[str] = Field(
        default_factory=lambda: ["\n\n", "\n", ". ", " "],
    )


class EvaluationThresholds(BaseModel):
    faithfulness_min: float = 0.7
    hallucination_max: float = 0.25


class EvaluationConfig(BaseModel):
    metrics: list[str] = Field(default_factory=list)
    thresholds: EvaluationThresholds = Field(default_factory=EvaluationThresholds)
    results_dir: str = "evaluation_results"


class LangfuseConfig(BaseModel):
    enabled: bool = False
    host: str = "http://localhost:3000"
    public_key: str = ""
    secret_key: str = ""


class APIConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = Field(default_factory=list)
    rate_limit_per_minute: int = 120


class DeidentifyConfig(BaseModel):
    replacement_strategy: Literal["redact", "mask", "fake"] = "mask"
    log_detections: bool = True
    entity_types: list[str] = Field(default_factory=list)


class PathsConfig(BaseModel):
    synthetic_documents: str = "data/synthetic/documents"
    golden_qa: str = "data/synthetic/golden_qa.json"


class RerankerConfig(BaseModel):
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"


class MedFlowSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="MEDFLOW_",
        env_nested_delimiter="__",
        extra="ignore",
        populate_by_name=True,
    )

    env: str = Field(default="default")
    config_dir: Path = Field(default=Path("configs"))

    llm: LLMConfig = Field(default_factory=LLMConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    ocr: OCRConfig = Field(default_factory=OCRConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    chunking: ChunkingConfig = Field(default_factory=ChunkingConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    langfuse: LangfuseConfig = Field(default_factory=LangfuseConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    deidentify: DeidentifyConfig = Field(default_factory=DeidentifyConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    reranker: RerankerConfig = Field(default_factory=RerankerConfig)

    @classmethod
    def from_yaml(cls, config_dir: Path | str | None = None, profile: str | None = None) -> MedFlowSettings:
        cfg_path = Path(config_dir or os.environ.get("MEDFLOW_CONFIG_DIR", "configs"))
        prof = profile or os.environ.get("MEDFLOW_ENV", "default")
        yaml_dict = load_yaml_config(cfg_path, prof)
        return cls(**yaml_dict)


@lru_cache
def get_settings_cached(config_dir_str: str, profile: str) -> MedFlowSettings:
    return MedFlowSettings.from_yaml(Path(config_dir_str), profile)


def get_settings() -> MedFlowSettings:
    return MedFlowSettings.from_yaml()
