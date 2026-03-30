from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock


@dataclass
class ProviderMetrics:
    """Aggregated counters for observability."""

    total_calls: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    by_provider: dict[str, dict[str, float]] = field(default_factory=dict)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def record(
        self,
        *,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float,
        latency_ms: float,
    ) -> None:
        """Record a single completion."""
        with self._lock:
            self.total_calls += 1
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens
            self.total_cost_usd += cost_usd
            self.total_latency_ms += latency_ms
            bucket = self.by_provider.setdefault(
                provider,
                {"calls": 0.0, "cost_usd": 0.0, "tokens": 0.0},
            )
            bucket["calls"] += 1
            bucket["cost_usd"] += cost_usd
            bucket["tokens"] += float(prompt_tokens + completion_tokens)

    def snapshot(self) -> dict[str, object]:
        """Return JSON-serializable snapshot."""
        with self._lock:
            return {
                "total_calls": self.total_calls,
                "total_prompt_tokens": self.total_prompt_tokens,
                "total_completion_tokens": self.total_completion_tokens,
                "total_cost_usd": round(self.total_cost_usd, 6),
                "avg_latency_ms": round(
                    self.total_latency_ms / self.total_calls,
                    2,
                )
                if self.total_calls
                else 0.0,
                "by_provider": dict(self.by_provider),
            }


_GLOBAL_METRICS = ProviderMetrics()


def get_provider_metrics() -> ProviderMetrics:
    """Shared metrics singleton for the process."""
    return _GLOBAL_METRICS
