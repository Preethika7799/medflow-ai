from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(__name__)


def audit_event(action: str, *, subject: str, outcome: str, details: dict[str, Any] | None = None) -> None:
    """Log a tamper-evident style audit record (structured JSON)."""
    logger.info(
        "hipaa_audit",
        action=action,
        subject=subject,
        outcome=outcome,
        details=details or {},
    )
