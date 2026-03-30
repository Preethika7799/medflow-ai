from __future__ import annotations

from medflow.retrieval.dense import RetrievalHit
from medflow.retrieval.hybrid import reciprocal_rank_fusion


def test_rrf_orders_union() -> None:
    a = [RetrievalHit(id="1", score=1.0, text="t1", metadata={})]
    b = [RetrievalHit(id="2", score=0.5, text="t2", metadata={})]
    fused = reciprocal_rank_fusion([a, b], k=60, top_k=2)
    assert {h.id for h in fused} == {"1", "2"}
