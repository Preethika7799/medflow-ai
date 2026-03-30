from __future__ import annotations

import time
from typing import Any

import structlog

from medflow.exceptions import OCRError
from medflow.ocr.base import OCREngine, OCRResult

logger = structlog.get_logger(__name__)


class EasyOCREngine(OCREngine):
    """EasyOCR with lazy reader initialization."""

    def __init__(self, *, lang_list: list[str] | None = None, gpu: bool = False) -> None:
        self._lang_list = lang_list or ["en"]
        self._gpu = gpu
        self._reader: Any = None

    def _ensure(self) -> Any:
        if self._reader is None:
            try:
                import easyocr  # type: ignore[import-untyped]

                self._reader = easyocr.Reader(self._lang_list, gpu=self._gpu, verbose=False)
            except Exception as e:
                logger.exception("easyocr_init_failed")
                raise OCRError(str(e), engine="easyocr") from e
        return self._reader

    def extract_text(self, image_path: str) -> OCRResult:
        """Run EasyOCR on image file."""
        start = time.perf_counter()
        reader = self._ensure()
        try:
            det = reader.readtext(image_path)
        except Exception as e:
            logger.exception("easyocr_failed", path=image_path)
            raise OCRError(str(e), engine="easyocr") from e

        lines: list[str] = []
        boxes: list[list[float]] = []
        confs: list[float] = []
        for item in det:
            box, text, conf = item
            lines.append(text)
            confs.append(float(conf))
            flat = [float(c) for pt in box for c in pt]
            boxes.append(flat)

        body = "\n".join(lines).strip()
        conf = float(sum(confs) / len(confs)) if confs else 0.0
        latency = (time.perf_counter() - start) * 1000
        return OCRResult(
            text=body,
            confidence=conf,
            bounding_boxes=boxes,
            processing_time_ms=latency,
            engine="easyocr",
        )
