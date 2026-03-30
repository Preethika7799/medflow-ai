from __future__ import annotations

import time
from typing import Any

import structlog

from medflow.exceptions import OCRError
from medflow.ocr.base import OCREngine, OCRResult

logger = structlog.get_logger(__name__)


class PaddleOCREngine(OCREngine):
    """Thin wrapper around PaddleOCR with lazy model load."""

    def __init__(self, *, lang: str = "en", use_gpu: bool = False) -> None:
        self._lang = lang
        self._use_gpu = use_gpu
        self._ocr: Any = None

    def _ensure(self) -> Any:
        if self._ocr is None:
            try:
                from paddleocr import PaddleOCR  # type: ignore[import-untyped]

                self._ocr = PaddleOCR(
                    lang=self._lang,
                    use_gpu=self._use_gpu,
                    show_log=False,
                )
            except Exception as e:
                logger.exception("paddle_init_failed")
                raise OCRError(str(e), engine="paddle") from e
        return self._ocr

    def extract_text(self, image_path: str) -> OCRResult:
        """Run PaddleOCR on ``image_path``."""
        start = time.perf_counter()
        ocr = self._ensure()
        try:
            result = ocr.ocr(image_path)
        except Exception as e:
            logger.exception("paddle_ocr_failed", path=image_path)
            raise OCRError(str(e), engine="paddle") from e

        lines: list[str] = []
        boxes: list[list[float]] = []
        confs: list[float] = []
        if not result or result[0] is None:
            latency = (time.perf_counter() - start) * 1000
            return OCRResult(
                text="",
                confidence=0.0,
                bounding_boxes=boxes,
                processing_time_ms=latency,
                engine="paddle",
            )

        for line in result[0]:
            try:
                box, (text, conf) = line
            except (ValueError, TypeError):
                continue
            lines.append(str(text))
            flat_box: list[float] = []
            for pt in box:
                for c in pt:
                    flat_box.append(float(c))
            boxes.append(flat_box)
            confs.append(float(conf))

        text = "\n".join(lines).strip()
        conf = float(sum(confs) / len(confs)) if confs else 0.0
        latency = (time.perf_counter() - start) * 1000
        return OCRResult(
            text=text,
            confidence=conf,
            bounding_boxes=boxes,
            processing_time_ms=latency,
            engine="paddle",
        )
