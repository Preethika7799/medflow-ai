from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any

from medflow.providers.metrics import get_provider_metrics


@dataclass
class MetricsCollector:
    """Aggregate latency and cost metrics from subsystems."""

    _lock: Lock = field(default_factory=Lock, repr=False)
    query_latencies_ms: list[float] = field(default_factory=list)

    def record_query(self, **kwargs: Any) -> None:
        """Append a query timing snapshot."""
        with self._lock:
            if "total_ms" in kwargs:
                self.query_latencies_ms.append(float(kwargs["total_ms"]))

    def snapshot(self) -> dict[str, Any]:
        """Serialize current metrics."""
        llm = get_provider_metrics().snapshot()
        with self._lock:
            lat = list(self.query_latencies_ms)[-100:]
        avg_latency = sum(lat) / len(lat) if lat else 0.0
        return {"llm": llm, "recent_query_avg_ms": round(avg_latency, 2), "recent_query_n": len(lat)}


_GLOBAL = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    return _GLOBAL
