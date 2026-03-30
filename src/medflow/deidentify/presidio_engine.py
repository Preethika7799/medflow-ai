from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog
from presidio_analyzer import AnalyzerEngine, RecognizerRegistry
from presidio_anonymizer import AnonymizerEngine

from medflow.deidentify.healthcare_recognizers import (
    build_dob_recognizer,
    build_insurance_recognizer,
    build_mrn_recognizer,
    build_provider_id_recognizer,
)
from medflow.exceptions import DeidentifyError

logger = structlog.get_logger(__name__)


@dataclass
class AnonymizedResult:
    """Output of PHI masking."""

    anonymized_text: str
    entity_map: dict[str, list[dict[str, Any]]] = field(default_factory=dict)


class PresidioDeIdentifier:
    """Analyze and anonymize PHI using Presidio plus custom recognizers."""

    def __init__(self, *, language: str = "en") -> None:
        registry = RecognizerRegistry()
        registry.load_predefined_recognizers()
        for fn in (
            build_mrn_recognizer,
            build_provider_id_recognizer,
            build_insurance_recognizer,
            build_dob_recognizer,
        ):
            registry.add_recognizer(fn())
        self._analyzer = AnalyzerEngine(registry=registry, supported_languages=[language])
        self._anonymizer = AnonymizerEngine()
        self._language = language

    def analyze(self, text: str, entities: list[str] | None = None) -> list[Any]:
        """Return Presidio recognizer results."""
        try:
            return self._analyzer.analyze(
                text=text,
                language=self._language,
                entities=entities,
            )
        except Exception as e:
            logger.exception("presidio_analyze_failed")
            raise DeidentifyError(str(e)) from e

    def anonymize(self, text: str, entities: list[str] | None = None) -> AnonymizedResult:
        """Mask entities in ``text`` and return mapping metadata."""
        results = self.analyze(text, entities)
        try:
            res = self._anonymizer.anonymize(text=text, analyzer_results=results)
        except Exception as e:
            logger.exception("presidio_anonymize_failed")
            raise DeidentifyError(str(e)) from e

        entity_map: dict[str, list[dict[str, Any]]] = {}
        for r in results:
            entity_map.setdefault(r.entity_type, []).append(
                {"start": r.start, "end": r.end, "score": float(r.score)},
            )

        return AnonymizedResult(anonymized_text=res.text, entity_map=entity_map)
