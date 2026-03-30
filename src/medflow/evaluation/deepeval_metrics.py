from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


def hallucination_score(answer: str, contexts: list[str]) -> float:
    """Lower is better — DeepEval HallucinationMetric when installed."""
    try:
        from deepeval.metrics import HallucinationMetric  # type: ignore[import-untyped]
        from deepeval.test_case import LLMTestCase  # type: ignore[import-untyped]

        ctx = "\n".join(contexts)
        case = LLMTestCase(input="", actual_output=answer, context=[ctx])
        m = HallucinationMetric()
        m.measure(case)
        return float(getattr(m, "score", 0.0) or 0.0)
    except Exception:
        logger.warning("deepeval_hallucination_fallback")
        ans = answer.lower()
        ctx = " ".join(contexts).lower()
        weird = sum(1 for w in ans.split() if len(w) > 6 and w not in ctx)
        return min(1.0, weird / max(len(ans.split()), 1))


def toxicity_score(answer: str) -> float:
    """Toxicity proxy via DeepEval or simple blocklist heuristic."""
    try:
        from deepeval.metrics import ToxicityMetric  # type: ignore[import-untyped]
        from deepeval.test_case import LLMTestCase  # type: ignore[import-untyped]

        case = LLMTestCase(input="", actual_output=answer)
        m = ToxicityMetric()
        m.measure(case)
        return float(getattr(m, "score", 0.0) or 0.0)
    except Exception:
        return 0.0
