from __future__ import annotations

from presidio_analyzer import Pattern, PatternRecognizer

# Medical record numbers like MRN-1234567
MRN_PATTERNS = [
    Pattern("MRN dash digits", r"\bMRN[-\s:]?\d{6,10}\b", 0.85),
    Pattern("MRN label", r"\bMedical Record (?:Number|#)?\s*[:\-]?\s*\d{6,10}\b", 0.75),
]

# NPI: 10-digit healthcare provider identifier (simplified Luhn-adjacent check omitted for speed)
NPI_PATTERNS = [
    Pattern("NPI", r"\bNPI\s*[:\-]?\s*\d{10}\b", 0.9),
    Pattern("NPI bare", r"\b(?<!\.)\d{10}\b", 0.35),
]

# DEA number rough pattern: 2 letters + 7 digits
DEA_PATTERNS = [
    Pattern("DEA", r"\bDEA\s*[:\-]?\s*[A-Z]{2}\d{7}\b", 0.85),
]

INSURANCE_PATTERNS = [
    Pattern("Member ID", r"\b(?:Member|Subscriber) ?ID\s*[:\-]?\s*[A-Z0-9]{6,20}\b", 0.8),
    Pattern("Group Number", r"\bGroup\s*(?:No\.?|Number)?\s*[:\-]?\s*[A-Z0-9]{4,15}\b", 0.75),
]

DOB_PATTERNS = [
    Pattern("DOB slash", r"\b(?:DOB|D\.O\.B\.|Date of Birth)\s*[:\-]?\s*\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b", 0.9),
    Pattern("DOB month", r"\b(?:DOB|Date of Birth)\s*[:\-]?\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s+\d{4}\b", 0.85),
]


def build_mrn_recognizer() -> PatternRecognizer:
    """Recognizer for synthetic MRNs."""
    return PatternRecognizer(
        supported_entity="MRN",
        patterns=MRN_PATTERNS,
        context=["medical", "record", "patient"],
    )


def build_provider_id_recognizer() -> PatternRecognizer:
    """Recognizer for NPI/DEA style identifiers."""
    return PatternRecognizer(
        supported_entity="PROVIDER_ID",
        patterns=NPI_PATTERNS + DEA_PATTERNS,
        context=["provider", "prescriber", "npi", "dea"],
    )


def build_insurance_recognizer() -> PatternRecognizer:
    """Recognizer for member/group insurance identifiers."""
    return PatternRecognizer(
        supported_entity="INSURANCE_ID",
        patterns=INSURANCE_PATTERNS,
        context=["insurance", "payer", "plan"],
    )


def build_dob_recognizer() -> PatternRecognizer:
    """Recognizer for date-of-birth phrases."""
    return PatternRecognizer(
        supported_entity="DOB",
        patterns=DOB_PATTERNS,
        context=["birth", "patient"],
    )
