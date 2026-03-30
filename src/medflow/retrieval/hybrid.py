from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from medflow.retrieval.dense import RetrievalHit


def reciprocal_rank_fusion(
    ranked_lists: Iterable[list[RetrievalHit]],
    *,
    k: int = 60,
    top_k: int = 10,
) -> list[RetrievalHit]:
    """RRF fusion; ``k`` is the usual 60."""
    scores: dict[str, float] = defaultdict(float)
    by_id: dict[str, RetrievalHit] = {}
    for lst in ranked_lists:
        for rank, hit in enumerate(lst, start=1):
            scores[hit.id] += 1.0 / (k + rank)
            by_id.setdefault(hit.id, hit)

    fused_ids = sorted(scores.keys(), key=lambda i: scores[i], reverse=True)[:top_k]
    out: list[RetrievalHit] = []
    for fid in fused_ids:
        base = by_id[fid]
        out.append(
            RetrievalHit(
                id=base.id,
                score=float(scores[fid]),
                text=base.text,
                metadata=dict(base.metadata) | {"rrf_score": float(scores[fid])},
            ),
        )
    return out
