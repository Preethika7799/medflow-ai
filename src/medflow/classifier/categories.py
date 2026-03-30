from __future__ import annotations

from enum import Enum


class DocumentCategory(str, Enum):
    """High-level routing labels for intake documents."""

    PRIOR_AUTH = "PRIOR_AUTH"
    REFERRAL = "REFERRAL"
    RECORDS_REQUEST = "RECORDS_REQUEST"
    LAB_RESULTS = "LAB_RESULTS"
    INSURANCE = "INSURANCE"
    OTHER = "OTHER"


DOCUMENT_CATEGORY_DESCRIPTIONS: dict[DocumentCategory, str] = {
    DocumentCategory.PRIOR_AUTH: "Prior authorization or pre-certification requests for procedures or imaging.",
    DocumentCategory.REFERRAL: "Referral letters or specialist handoff documentation.",
    DocumentCategory.RECORDS_REQUEST: "ROI / medical records request or release forms.",
    DocumentCategory.LAB_RESULTS: "Laboratory, pathology, or structured diagnostic results.",
    DocumentCategory.INSURANCE: "Eligibility, benefits, EOB, or payer correspondence.",
    DocumentCategory.OTHER: "Clinical correspondence that does not match the above.",
}
