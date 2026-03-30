from __future__ import annotations

import argparse
import asyncio
import os
from pathlib import Path

import structlog

from medflow.config import MedFlowSettings
from medflow.ingestion.pipeline import IngestionPipeline

logger = structlog.get_logger(__name__)


async def main_async(settings: MedFlowSettings) -> None:
    root = Path(settings.paths.synthetic_documents)
    paths = sorted(root.glob("*.pdf"))
    if not paths:
        paths = sorted(root.glob("*.txt"))
    pipeline = IngestionPipeline(settings)
    for p in paths:
        if p.name.startswith("."):
            continue
        logger.info("seeding_file", path=str(p))
        try:
            await pipeline.ingest(p)
        except Exception as e:
            logger.error("seed_failed", path=str(p), err=str(e))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml", help="Reserved for future YAML override")
    _ = parser.parse_args()
    profile = os.environ.get("MEDFLOW_ENV", "development")
    settings = MedFlowSettings.from_yaml(Path("configs"), profile=profile)
    asyncio.run(main_async(settings))


if __name__ == "__main__":
    main()
