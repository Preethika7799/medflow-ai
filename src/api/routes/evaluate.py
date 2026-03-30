from __future__ import annotations

from fastapi import APIRouter, Request

from api.schemas import EvaluateResponse
from medflow.evaluation.runner import EvaluationRunner

router = APIRouter(tags=["evaluate"])


@router.post("/evaluate", response_model=EvaluateResponse)
async def evaluate(request: Request) -> EvaluateResponse:
    """Run golden Q&A evaluation (may take several minutes)."""
    settings = request.app.state.store["settings"]
    runner = EvaluationRunner(settings)
    report = await runner.run()
    return EvaluateResponse(
        aggregate_metrics=report.aggregate_metrics,
        timestamp=report.timestamp,
        num_questions=len(report.per_question_results),
    )
