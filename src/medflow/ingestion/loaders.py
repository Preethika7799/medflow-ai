from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import pdfplumber
import structlog

from medflow.ocr.pipeline import OCRPipeline

logger = structlog.get_logger(__name__)


class BaseLoader(ABC):
    """Load raw text from a supported artifact."""

    @abstractmethod
    async def load(self, path: str | Path) -> str:
        """Return extracted text."""


class PDFLoader(BaseLoader):
    """Extract digital text via pdfplumber; delegate scanned pages to OCR."""

    def __init__(self, ocr: OCRPipeline | None = None) -> None:
        self._ocr = ocr

    async def load(self, path: str | Path) -> str:
        """Load PDF text, falling back to OCR when pages lack extractable text."""
        p = Path(path)
        parts: list[str] = []
        with pdfplumber.open(p) as pdf:
            for page in pdf.pages:
                t = page.extract_text() or ""
                if len(t.strip()) < 40 and self._ocr:
                    logger.info("pdf_page_ocr_fallback", page=page.page_number)
                    doc = self._ocr.process(p)
                    return doc.text
                parts.append(t)
        return "\n\n".join(parts).strip()


class ImageLoader(BaseLoader):
    """OCR images into text."""

    def __init__(self, ocr: OCRPipeline) -> None:
        self._ocr = ocr

    async def load(self, path: str | Path) -> str:
        """Return OCR text for an image path."""
        return self._ocr.process(Path(path)).text


class TextLoader(BaseLoader):
    """UTF-8 plain text files."""

    async def load(self, path: str | Path) -> str:
        """Read text file contents."""
        return Path(path).read_text(encoding="utf-8", errors="replace")
