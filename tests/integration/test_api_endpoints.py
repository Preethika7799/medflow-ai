from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from api.main import create_app


@pytest.mark.asyncio
async def test_health_ok() -> None:
    app = create_app()
    app.state.store = {
        "settings": MagicMock(),
        "qdrant": MagicMock(get_collections=AsyncMock(return_value=MagicMock(collections=[]))),
        "ingestion": MagicMock(),
        "generation": MagicMock(),
    }
    transport = ASGITransport(app=app, lifespan="off")
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "healthy"
