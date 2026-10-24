"""Real-time metrics collection, percentiles calculation, and Prometheus exporter."""

import threading
import time
from collections import deque
from typing import Any

import numpy as np


class MetricsCollector:
    """
    Thread-safe metrics aggregator tracking throughput, hit/miss ratios,
    token & cost savings, latency distributions, and similarity score histograms.
    """

    def __init__(self, window_size: int = 2000):
        self.window_size = window_size
        self._lock = threading.Lock()

        # Counters
        self.total_requests: int = 0
        self.cache_hits: int = 0
        self.cache_misses: int = 0
        self.bypass_count: int = 0
        self.upstream_errors: int = 0
        self.evictions_count: int = 0

        # Token and cost savings
        self.prompt_tokens_saved: int = 0
        self.completion_tokens_saved: int = 0
        self.total_tokens_saved: int = 0
        self.total_cost_saved_usd: float = 0.0

        # Latency sliding windows (milliseconds)
        self.embedding_latencies: deque[float] = deque(maxlen=window_size)
        self.search_latencies: deque[float] = deque(maxlen=window_size)
        self.hit_latencies: deque[float] = deque(maxlen=window_size)
        self.miss_latencies: deque[float] = deque(maxlen=window_size)

        # Similarity score distribution
        self.similarity_scores: deque[float] = deque(maxlen=window_size)

        # Historical time-series points for dashboard charts (retains last 120 points ~ 4 mins at 2s poll)
        self.time_series_history: deque[dict[str, Any]] = deque(maxlen=120)
        self._last_ts_snapshot = time.time()

    def record_hit(
        self,
        similarity: float,
        embedding_ms: float,
        search_ms: float,
        total_ms: float,
        prompt_tokens: int,
        completion_tokens: int,
        cost_savings: float,
    ) -> None:
        with self._lock:
            self.total_requests += 1
            self.cache_hits += 1
            self.prompt_tokens_saved += prompt_tokens
            self.completion_tokens_saved += completion_tokens
            self.total_tokens_saved += (prompt_tokens + completion_tokens)
            self.total_cost_saved_usd += cost_savings

            self.embedding_latencies.append(embedding_ms)
            self.search_latencies.append(search_ms)
            self.hit_latencies.append(total_ms)
            self.similarity_scores.append(similarity)
            self._maybe_record_time_series_sync()

    def record_miss(
        self,
        embedding_ms: float,
        search_ms: float,
        total_ms: float,
    ) -> None:
        with self._lock:
            self.total_requests += 1
            self.cache_misses += 1
            self.embedding_latencies.append(embedding_ms)
            self.search_latencies.append(search_ms)
            self.miss_latencies.append(total_ms)
            self._maybe_record_time_series_sync()

    def record_bypass(self, total_ms: float) -> None:
        with self._lock:
            self.total_requests += 1
            self.bypass_count += 1
            self.miss_latencies.append(total_ms)
            self._maybe_record_time_series_sync()

    def record_upstream_error(self) -> None:
        with self._lock:
            self.upstream_errors += 1

    def record_eviction(self, count: int = 1) -> None:
        with self._lock:
            self.evictions_count += count

    def _maybe_record_time_series_sync(self) -> None:
        now = time.time()
        # Record time series point at most once per second
        if now - self._last_ts_snapshot >= 1.0 or not self.time_series_history:
            self._last_ts_snapshot = now
            hit_ratio = (self.cache_hits / self.total_requests) if self.total_requests > 0 else 0.0
            self.time_series_history.append({
                "timestamp": now,
                "hit_ratio": round(hit_ratio * 100, 2),
                "total_requests": self.total_requests,
                "cost_saved": round(self.total_cost_saved_usd, 4),
            })

    def _calc_percentiles(self, values: deque[float]) -> dict[str, float]:
        if not values:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "avg": 0.0}
        arr = np.array(values, dtype=np.float32)
        return {
            "p50": float(round(np.percentile(arr, 50), 2)),
            "p95": float(round(np.percentile(arr, 95), 2)),
            "p99": float(round(np.percentile(arr, 99), 2)),
            "avg": float(round(np.mean(arr), 2)),
        }

    def _calc_similarity_histogram(self) -> dict[str, int]:
        buckets = {
            "0.70-0.80": 0,
            "0.80-0.85": 0,
            "0.85-0.90": 0,
            "0.90-0.92": 0,
            "0.92-0.95": 0,
            "0.95-0.98": 0,
            "0.98-1.00": 0,
        }
        for score in self.similarity_scores:
            if score < 0.80:
                buckets["0.70-0.80"] += 1
            elif score < 0.85:
                buckets["0.80-0.85"] += 1
            elif score < 0.90:
                buckets["0.85-0.90"] += 1
            elif score < 0.92:
                buckets["0.90-0.92"] += 1
            elif score < 0.95:
                buckets["0.92-0.95"] += 1
            elif score < 0.98:
                buckets["0.95-0.98"] += 1
            else:
                buckets["0.98-1.00"] += 1
        return buckets

    def get_metrics_json(self, cache_size: int = 0) -> dict[str, Any]:
        with self._lock:
            self._maybe_record_time_series_sync()
            hit_ratio = (self.cache_hits / self.total_requests) if self.total_requests > 0 else 0.0

            return {
                "counters": {
                    "total_requests": self.total_requests,
                    "cache_hits": self.cache_hits,
                    "cache_misses": self.cache_misses,
                    "bypass_count": self.bypass_count,
                    "hit_ratio": round(hit_ratio, 4),
                    "hit_ratio_percent": round(hit_ratio * 100, 2),
                    "prompt_tokens_saved": self.prompt_tokens_saved,
                    "completion_tokens_saved": self.completion_tokens_saved,
                    "total_tokens_saved": self.total_tokens_saved,
                    "total_cost_saved_usd": round(self.total_cost_saved_usd, 4),
                    "upstream_errors": self.upstream_errors,
                    "evictions_count": self.evictions_count,
                    "cache_size": cache_size,
                },
                "latencies_ms": {
                    "embedding": self._calc_percentiles(self.embedding_latencies),
                    "search": self._calc_percentiles(self.search_latencies),
                    "hit_e2e": self._calc_percentiles(self.hit_latencies),
                    "miss_e2e": self._calc_percentiles(self.miss_latencies),
                },
                "similarity_histogram": self._calc_similarity_histogram(),
                "time_series": list(self.time_series_history),
            }

    def get_prometheus_metrics(self, cache_size: int = 0) -> str:
        with self._lock:
            hit_ratio = (self.cache_hits / self.total_requests) if self.total_requests > 0 else 0.0
            hit_p = self._calc_percentiles(self.hit_latencies)
            miss_p = self._calc_percentiles(self.miss_latencies)

            lines = [
                "# HELP cachesense_requests_total Total number of proxy requests received.",
                "# TYPE cachesense_requests_total counter",
                f"cachesense_requests_total {self.total_requests}",
                "",
                "# HELP cachesense_cache_hits_total Total number of cache hits.",
                "# TYPE cachesense_cache_hits_total counter",
                f"cachesense_cache_hits_total {self.cache_hits}",
                "",
                "# HELP cachesense_cache_misses_total Total number of cache misses.",
                "# TYPE cachesense_cache_misses_total counter",
                f"cachesense_cache_misses_total {self.cache_misses}",
                "",
                "# HELP cachesense_cache_hit_ratio Current overall hit ratio (0.0 to 1.0).",
                "# TYPE cachesense_cache_hit_ratio gauge",
                f"cachesense_cache_hit_ratio {round(hit_ratio, 4)}",
                "",
                "# HELP cachesense_tokens_saved_total Total prompt and completion tokens saved.",
                "# TYPE cachesense_tokens_saved_total counter",
                f"cachesense_tokens_saved_total {self.total_tokens_saved}",
                "",
                "# HELP cachesense_cost_saved_usd_total Estimated total dollars saved.",
                "# TYPE cachesense_cost_saved_usd_total counter",
                f"cachesense_cost_saved_usd_total {round(self.total_cost_saved_usd, 4)}",
                "",
                "# HELP cachesense_cache_size Current number of items in cache.",
                "# TYPE cachesense_cache_size gauge",
                f"cachesense_cache_size {cache_size}",
                "",
                "# HELP cachesense_evictions_total Total entries evicted from cache.",
                "# TYPE cachesense_evictions_total counter",
                f"cachesense_evictions_total {self.evictions_count}",
                "",
                "# HELP cachesense_upstream_errors_total Total errors encountered from upstream provider.",
                "# TYPE cachesense_upstream_errors_total counter",
                f"cachesense_upstream_errors_total {self.upstream_errors}",
                "",
                "# HELP cachesense_hit_latency_ms Hit path latency percentiles in ms.",
                "# TYPE cachesense_hit_latency_ms gauge",
                f'cachesense_hit_latency_ms{{quantile="0.5"}} {hit_p["p50"]}',
                f'cachesense_hit_latency_ms{{quantile="0.95"}} {hit_p["p95"]}',
                f'cachesense_hit_latency_ms{{quantile="0.99"}} {hit_p["p99"]}',
                "",
                "# HELP cachesense_miss_latency_ms Miss path latency percentiles in ms.",
                "# TYPE cachesense_miss_latency_ms gauge",
                f'cachesense_miss_latency_ms{{quantile="0.5"}} {miss_p["p50"]}',
                f'cachesense_miss_latency_ms{{quantile="0.95"}} {miss_p["p95"]}',
                f'cachesense_miss_latency_ms{{quantile="0.99"}} {miss_p["p99"]}',
                "",
            ]
            return "\n".join(lines)


# Global singleton metrics collector
metrics_collector = MetricsCollector()
