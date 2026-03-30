from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import structlog

from medflow.config import MedFlowSettings
from medflow.evaluation import custom_metrics, deepeval_metrics, ragas_metrics
from medflow.evaluation.golden_dataset import GoldenQA, load_golden_dataset
from medflow.generation.pipeline import GenerationPipeline

logger = structlog.get_logger(__name__)


@dataclass
class EvaluationReport:
    """Structured evaluation output."""

    per_question_results: list[dict[str, Any]] = field(default_factory=list)
    aggregate_metrics: dict[str, float] = field(default_factory=dict)
    timestamp: str = ""


class EvaluationRunner:
    """Run configured metrics over golden Q&A."""

    def __init__(self, settings: MedFlowSettings) -> None:
        self._settings = settings
        self._rag = GenerationPipeline(settings)

    async def run(self, dataset: list[GoldenQA] | None = None) -> EvaluationReport:
        """Execute metrics for all golden entries."""
        if dataset is None:
            dataset = load_golden_dataset(
                self._settings.paths.golden_qa,
                Path(self._settings.paths.synthetic_documents),
            )
        ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        per_q: list[dict[str, Any]] = []
        agg: dict[str, list[float]] = {}

        for row in dataset:
            res = await self._rag.run(row.question)
            retrieval = await self._rag.retrieval.retrieve(row.question)
            contexts = [h.text for h in retrieval.hits]

            metrics_row: dict[str, Any] = {"id": row.id, "question": row.question}

            def _avg(name: str, fn: Callable[..., float], *args: Any) -> None:
                try:
                    v = float(fn(*args))
                except Exception as e:
                    logger.warning("metric_failed", metric=name, err=str(e))
                    v = 0.0
                metrics_row[name] = v
                agg.setdefault(name, []).append(v)

            _avg("faithfulness", ragas_metrics.faithfulness, row.question, res.answer, contexts)
            _avg("answer_relevancy", ragas_metrics.answer_relevancy, row.question, res.answer)
            _avg(
                "context_precision",
                ragas_metrics.context_precision,
                row.question,
                res.answer,
                contexts,
                row.ground_truth,
            )
            _avg("context_recall", ragas_metrics.context_recall, row.question, contexts, row.ground_truth)
            _avg("hallucination", deepeval_metrics.hallucination_score, res.answer, contexts)
            _avg("toxicity", deepeval_metrics.toxicity_score, res.answer)
            _avg(
                "citation_accuracy",
                custom_metrics.citation_accuracy,
                res.answer,
                res.citations,
                row.source_doc_ids,
            )
            per_q.append(metrics_row)

        aggregate_metrics = {k: sum(v) / max(len(v), 1) for k, v in agg.items()}
        report = EvaluationReport(
            per_question_results=per_q,
            aggregate_metrics=aggregate_metrics,
            timestamp=ts,
        )
        out_dir = Path(self._settings.evaluation.results_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"eval_{ts.replace(':', '-')}.json"
        out_path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
        logger.info("evaluation_saved", path=str(out_path))
        return report
