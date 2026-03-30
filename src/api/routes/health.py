from __future__ import annotations

from fastapi import APIRouter, Request

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe (load balancers / playbook expect ``healthy``)."""
    return {"status": "healthy"}


@router.get("/ready")
async def ready(request: Request) -> dict[str, str]:
    """Readiness: verify Qdrant connectivity."""
    client = request.app.state.store["qdrant"]
    try:
        await client.get_collections()
        return {"status": "ready"}
    except Exception as e:
        return {"status": "not_ready", "detail": str(e)}
