from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class OCRResult:
    """Structured output for a single OCR pass."""

    text: str
    confidence: float
    bounding_boxes: list[list[float]] = field(default_factory=list)
    processing_time_ms: float = 0.0
    engine: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


class OCREngine(ABC):
    """Extract text from a raster image path or ``numpy`` array."""

    @abstractmethod
    def extract_text(self, image_path: str) -> OCRResult:
        """Run OCR on an image file path."""
