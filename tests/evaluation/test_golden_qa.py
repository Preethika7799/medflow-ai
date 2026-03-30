from __future__ import annotations

from pathlib import Path

import pytest

from medflow.evaluation.golden_dataset import load_golden_dataset


def test_load_golden(project_root: Path) -> None:
    p = project_root / "data" / "synthetic" / "golden_qa.json"
    if not p.exists():
        pytest.skip("golden file missing")
    rows = load_golden_dataset(p, project_root / "data" / "synthetic" / "documents")
    assert len(rows) >= 1
