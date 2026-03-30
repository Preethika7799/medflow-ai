from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from medflow.config import MedFlowSettings
from medflow.deidentify.presidio_engine import AnonymizedResult, PresidioDeIdentifier

logger = structlog.get_logger(__name__)


@dataclass
class DeIDResult:
    """Structured de-identification output."""

    anonymized_text: str
    entity_map: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    detections_logged: bool = False


class DeIDPipeline:
    def __init__(self, settings: MedFlowSettings) -> None:
        self._settings = settings
        self._engine = PresidioDeIdentifier()

    def process(self, text: str) -> DeIDResult:
        entities = self._settings.deidentify.entity_types or None
        masked: AnonymizedResult = self._engine.anonymize(text, entities)
        if self._settings.deidentify.log_detections:
            logger.info(
                "deid_audit",
                entity_types=list(masked.entity_map.keys()),
                counts={k: len(v) for k, v in masked.entity_map.items()},
            )
        return DeIDResult(
            anonymized_text=masked.anonymized_text,
            entity_map=masked.entity_map,
            detections_logged=self._settings.deidentify.log_detections,
        )
