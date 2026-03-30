from __future__ import annotations

import argparse
import asyncio
from pathlib import Path

from medflow.config import MedFlowSettings
from medflow.evaluation.runner import EvaluationRunner


async def _run(profile: str) -> None:
    settings = MedFlowSettings.from_yaml(Path("configs"), profile=profile)
    runner = EvaluationRunner(settings)
    report = await runner.run()
    print("\n=== MedFlow evaluation report ===")
    print(f"Timestamp: {report.timestamp}")
    print(f"Questions evaluated: {len(report.per_question_results)}")
    print("\nAggregate metrics:")
    for name in sorted(report.aggregate_metrics.keys()):
        val = report.aggregate_metrics[name]
        print(f"  {name}: {val:.4f}")
    print(f"\nSaved under: {settings.evaluation.results_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml", help="Ignored — use MEDFLOW_ENV profile")
    parser.add_argument("--profile", default=None)
    args = parser.parse_args()
    import os

    profile = args.profile or os.environ.get("MEDFLOW_ENV", "default")
    asyncio.run(_run(profile))


if __name__ == "__main__":
    main()
