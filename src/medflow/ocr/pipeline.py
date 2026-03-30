from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import fitz
import structlog

from medflow.config import MedFlowSettings, OCREngineName
from medflow.exceptions import OCRError
from medflow.ocr.base import OCREngine
from medflow.ocr.easyocr_engine import EasyOCREngine
from medflow.ocr.paddle_engine import PaddleOCREngine
from medflow.ocr.preprocessing import load_image_bgr, preprocess, write_temp_image

logger = structlog.get_logger(__name__)


@dataclass
class DocumentText:
    text: str
    pages: list[dict[str, Any]] = field(default_factory=list)
    source_path: str = ""


class OCRPipeline:
    """Preprocess, OCR with Paddle by default, fall back to EasyOCR on low confidence."""

    def __init__(self, settings: MedFlowSettings) -> None:
        self._settings = settings
        self._paddle = PaddleOCREngine(lang=settings.ocr.language)
        self._easy = EasyOCREngine(lang_list=[settings.ocr.language])
        self._threshold = settings.ocr.fallback_confidence_threshold

    def _select_primary(self) -> OCREngine:
        if self._settings.ocr.engine == OCREngineName.EASYOCR:
            return self._easy
        return self._paddle

    def _fallback(self) -> OCREngine:
        if self._settings.ocr.engine == OCREngineName.EASYOCR:
            return self._paddle
        return self._easy

    def process(self, file_path: str | Path) -> DocumentText:
        path = Path(file_path)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._process_pdf(path)
        if suffix in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}:
            return self._process_image(path)
        msg = f"Unsupported file type for OCR: {suffix}"
        raise OCRError(msg, details={"path": str(path)})

    def _run_page(self, page_image_path: str, page_num: int) -> dict[str, Any]:
        primary = self._select_primary()
        try:
            img = load_image_bgr(page_image_path)
            proc = preprocess(img, self._settings.ocr)
            tmp = write_temp_image(proc)
            try:
                res = primary.extract_text(tmp)
            finally:
                Path(tmp).unlink(missing_ok=True)
        except Exception:
            res = primary.extract_text(page_image_path)

        used = primary.__class__.__name__
        if res.confidence < self._threshold:
            fb = self._fallback()
            logger.warning(
                "ocr_fallback_triggered",
                page=page_num,
                confidence=res.confidence,
                fallback=fb.__class__.__name__,
            )
            try:
                img = load_image_bgr(page_image_path)
                proc = preprocess(img, self._settings.ocr)
                tmp = write_temp_image(proc)
                try:
                    res2 = fb.extract_text(tmp)
                finally:
                    Path(tmp).unlink(missing_ok=True)
            except Exception:
                res2 = fb.extract_text(page_image_path)
            if res2.confidence >= res.confidence:
                res = res2
                used = fb.__class__.__name__

        logger.info(
            "ocr_page_done",
            page=page_num,
            confidence=res.confidence,
            engine=res.engine,
            ms=res.processing_time_ms,
        )
        return {
            "page": page_num,
            "text": res.text,
            "confidence": res.confidence,
            "engine": res.engine or used,
            "ms": res.processing_time_ms,
        }

    def _process_pdf(self, path: Path) -> DocumentText:
        pages_out: list[dict[str, Any]] = []
        texts: list[str] = []
        doc = fitz.open(path)
        try:
            for i in range(doc.page_count):
                page = doc.load_page(i)
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                tmp = write_temp_image_bytes(pix.tobytes("png"))
                try:
                    meta = self._run_page(tmp, i + 1)
                    pages_out.append(meta)
                    if meta["text"]:
                        texts.append(meta["text"])
                finally:
                    Path(tmp).unlink(missing_ok=True)
        finally:
            doc.close()

        body = "\n\n".join(texts).strip()
        return DocumentText(text=body, pages=pages_out, source_path=str(path))

    def _process_image(self, path: Path) -> DocumentText:
        meta = self._run_page(str(path), 1)
        return DocumentText(text=meta["text"], pages=[meta], source_path=str(path))


def write_temp_image_bytes(png_bytes: bytes) -> str:
    """Write PNG bytes to temp path."""
    import tempfile

    fd, p = tempfile.mkstemp(suffix=".png")
    import os

    os.close(fd)
    Path(p).write_bytes(png_bytes)
    return p
