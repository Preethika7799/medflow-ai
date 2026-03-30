from __future__ import annotations

import structlog

logger = structlog.get_logger(__name__)


def faithfulness(question: str, answer: str, contexts: list[str]) -> float:
    """Proxy faithfulness using RAGAS when available, else lexical overlap."""
    try:
        from ragas import evaluate  # type: ignore[import-untyped]
        from ragas.metrics import faithfulness as ragas_faith  # type: ignore[import-untyped]
        from datasets import Dataset  # type: ignore[import-untyped]

        ds = Dataset.from_dict({"question": [question], "answer": [answer], "contexts": [contexts]})
        result = evaluate(ds, metrics=[ragas_faith()])
        return float(result["faithfulness"][0])
    except Exception:
        logger.warning("ragas_faithfulness_fallback")
        overlap = 0
        toks = answer.lower().split()
        ctx = " ".join(contexts).lower()
        for t in toks:
            if len(t) > 4 and t in ctx:
                overlap += 1
        return min(1.0, overlap / max(len(toks), 1))


def answer_relevancy(question: str, answer: str) -> float:
    """Answer relevancy via RAGAS or Jaccard overlap on tokens."""
    try:
        from ragas import evaluate  # type: ignore[import-untyped]
        from ragas.metrics import answer_relevancy as ragas_ar  # type: ignore[import-untyped]
        from datasets import Dataset  # type: ignore[import-untyped]

        ds = Dataset.from_dict({"question": [question], "answer": [answer]})
        result = evaluate(ds, metrics=[ragas_ar()])
        return float(result["answer_relevancy"][0])
    except Exception:
        q = set(question.lower().split())
        a = set(answer.lower().split())
        inter = len(q & a)
        return inter / max(len(q), 1)


def context_precision(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
) -> float:
    """Context precision — RAGAS if available."""
    try:
        from ragas import evaluate  # type: ignore[import-untyped]
        from ragas.metrics import context_precision as ragas_cp  # type: ignore[import-untyped]
        from datasets import Dataset  # type: ignore[import-untyped]

        ds = Dataset.from_dict(
            {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
                "ground_truth": [ground_truth],
            },
        )
        result = evaluate(ds, metrics=[ragas_cp()])
        return float(result["context_precision"][0])
    except Exception:
        gt = ground_truth.lower().split()
        joined = " ".join(contexts).lower()
        hits = sum(1 for g in gt if g in joined and len(g) > 3)
        return min(1.0, hits / max(len(gt), 1))


def context_recall(question: str, contexts: list[str], ground_truth: str) -> float:
    """Context recall — RAGAS if available."""
    try:
        from ragas import evaluate  # type: ignore[import-untyped]
        from ragas.metrics import context_recall as ragas_cr  # type: ignore[import-untyped]
        from datasets import Dataset  # type: ignore[import-untyped]

        ds = Dataset.from_dict(
            {"question": [question], "contexts": [contexts], "ground_truth": [ground_truth]},
        )
        result = evaluate(ds, metrics=[ragas_cr()])
        return float(result["context_recall"][0])
    except Exception:
        gt = ground_truth.lower()
        joined = " ".join(contexts).lower()
        return 1.0 if gt.strip() and gt in joined else 0.5
