from __future__ import annotations

import json

from medflow.classifier.categories import DocumentCategory


def test_enum_values() -> None:
    assert DocumentCategory.PRIOR_AUTH.value == "PRIOR_AUTH"


def test_category_json_roundtrip() -> None:
    payload = {"category": "LAB_RESULTS", "confidence": 0.95}
    assert json.dumps(payload)
