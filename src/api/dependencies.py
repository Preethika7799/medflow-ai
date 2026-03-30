from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from medflow.config import MedFlowSettings, get_settings
from medflow.ingestion.pipeline import IngestionPipeline
from medflow.generation.pipeline import GenerationPipeline


def settings_dep() -> MedFlowSettings:
    """Return cached application settings."""
    return get_settings()


def ingestion_dep(request: Request) -> IngestionPipeline:
    return request.app.state.store["ingestion"]


def generation_dep(request: Request) -> GenerationPipeline:
    return request.app.state.store["generation"]


SettingsDep = Annotated[MedFlowSettings, Depends(settings_dep)]
IngestionDep = Annotated[IngestionPipeline, Depends(ingestion_dep)]
GenerationDep = Annotated[GenerationPipeline, Depends(generation_dep)]
