import time
from collections import deque
from threading import Lock


class _Metrics:
    def __init__(self) -> None:
        self._lock = Lock()
        self._started_at = time.monotonic()
        self._requests: int = 0
        self._errors: int = 0
        self._latencies: deque[float] = deque(maxlen=1000)
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    def record_request(self, latency_ms: float, *, error: bool = False) -> None:
        with self._lock:
            self._requests += 1
            if error:
                self._errors += 1
            self._latencies.append(latency_ms)

    def record_cache_hit(self) -> None:
        with self._lock:
            self._cache_hits += 1

    def record_cache_miss(self) -> None:
        with self._lock:
            self._cache_misses += 1

    def summary(self) -> dict:
        with self._lock:
            latencies = sorted(self._latencies)
            n = len(latencies)
            total_cache = self._cache_hits + self._cache_misses
            return {
                "uptime_seconds": round(time.monotonic() - self._started_at),
                "note": "per-worker metrics (each gunicorn worker tracks independently)",
                "requests": {
                    "total": self._requests,
                    "errors": self._errors,
                    "error_rate": round(self._errors / max(self._requests, 1), 4),
                },
                "latency_ms": {
                    "p50": latencies[n // 2] if n else 0,
                    "p95": latencies[int(n * 0.95)] if n else 0,
                    "p99": latencies[int(n * 0.99)] if n else 0,
                    "samples": n,
                },
                "embedding_cache": {
                    "hits": self._cache_hits,
                    "misses": self._cache_misses,
                    "hit_rate": round(self._cache_hits / max(total_cache, 1), 4),
                },
            }


metrics = _Metrics()
