from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import structlog

from medflow.config import OCRConfig

logger = structlog.get_logger(__name__)


def deskew_image(image: np.ndarray) -> np.ndarray:
    """Detect dominant text skew via min area rectangle and rotate."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if coords.size == 0:
        return image
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    if abs(angle) < 0.1:
        return image
    h, w = image.shape[:2]
    m = cv2.getRotationMatrix2D((w // 2, h // 2), angle, 1.0)
    return cv2.warpAffine(image, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def enhance_contrast(image: np.ndarray) -> np.ndarray:
    """Apply CLAHE to luminance channel."""
    if image.ndim == 2:
        lab = cv2.cvtColor(cv2.cvtColor(image, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2LAB)
    else:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def remove_noise(image: np.ndarray) -> np.ndarray:
    """Light denoise + adaptive threshold for noisy scans."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    return cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2,
    )


def preprocess(image: np.ndarray, config: OCRConfig) -> np.ndarray:
    """Apply preprocessing chain based on configuration."""
    out = image
    opts = config.preprocessing
    if opts.deskew:
        out = deskew_image(out)
    if opts.enhance_contrast:
        out = enhance_contrast(out)
    if opts.remove_noise:
        den = remove_noise(out)
        out = cv2.cvtColor(den, cv2.COLOR_GRAY2BGR) if out.ndim == 3 else den
    return out


def load_image_bgr(path: str) -> np.ndarray:
    """Load image as BGR ``numpy`` array."""
    img = cv2.imread(path)
    if img is None:
        msg = f"Could not read image: {path}"
        raise ValueError(msg)
    return img


def write_temp_image(image: np.ndarray, suffix: str = ".png") -> str:
    """Write array to a temp file and return path."""
    import tempfile

    fd, p = tempfile.mkstemp(suffix=suffix)
    import os

    os.close(fd)
    cv2.imwrite(p, image)
    return p
