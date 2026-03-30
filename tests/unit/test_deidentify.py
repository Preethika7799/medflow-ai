from __future__ import annotations

from medflow.deidentify.healthcare_recognizers import build_mrn_recognizer


def test_mrn_recognizer_entity() -> None:
    r = build_mrn_recognizer()
    assert "MRN" in r.supported_entities
