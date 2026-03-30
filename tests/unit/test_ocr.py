from __future__ import annotations

import numpy as np

from medflow.config import MedFlowSettings
from medflow.ocr.preprocessing import deskew_image, enhance_contrast, preprocess


def test_preprocess_runs() -> None:
    img = np.zeros((50, 50, 3), dtype=np.uint8)
    settings = MedFlowSettings()
    out = preprocess(img, settings.ocr)
    assert out.shape[0] == 50
