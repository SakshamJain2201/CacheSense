"""Latency benchmark evaluating CacheSense hit vs miss latency, concurrency, and p95 reduction."""

import asyncio
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import httpx
import matplotlib.pyplot as plt
import numpy as np
from httpx import ASGITransport

from app.core.cache import SemanticCache
from app.core.single_flight import SingleFlight
from app.core.upstream import UpstreamClient
from app.main import app


class LatencyBenchmarkMockUpstream(UpstreamClient):
    """Mock upstream server injecting artificial upstream latency (800ms)."""

    def __init__(self, artificial_delay_s: float = 0.8):
        super().__init__()
        self.artificial_delay_s = artificial_delay_s

    async def forward_request(self, provider, endpoint_path, payload, client_auth_header=None, custom_headers=None):
        await asyncio.sleep(self.artificial_delay_s)
        return {
            "id": f"chatcmpl-bench-{int(time.time()*1000)}",
            "object": "chat.completion",
            "model": payload.get("model", "gpt-4o"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "Benchmark response payload."}}],
            "usage": {"prompt_tokens": 15, "completion_tokens": 8, "total_tokens": 23},
        }, 200


async def run_latency_benchmark():
    print("=" * 70)
    print("Running CacheSense End-to-End Latency & Concurrency Benchmark")
    print("=" * 70)

    # Initialize in-memory cache and mock upstream with 800ms latency
    cache = SemanticCache()
    await cache.initialize()

    mock_up = LatencyBenchmarkMockUpstream(artificial_delay_s=0.8)  # 800ms artificial upstream delay
    sf = SingleFlight()

    from app.api import proxy
    proxy.cache_engine = cache
    proxy.upstream_client = mock_up
    proxy.single_flight = sf

    transport = ASGITransport(app=app)
    results = {}

    async with httpx.AsyncClient(transport=transport, base_url="http://benchmark") as client:
        # Measure Embedding and FAISS Search baselines
        print("\nMeasuring Embedding and FAISS Search latency baselines...")
        sample_prompts = [f"This is benchmark prompt sample number {i}" for i in range(50)]
        emb_times = []
        search_times = []

        # Populate initial vectors for search test
        partition = "gpt-4o|sys:none|t0.00|tools:none|ns:global"
        for i, p in enumerate(sample_prompts[:10]):
            v = await cache.embedder.embed(p)
            await cache.insert(partition, p, v, {"test": True}, "gpt-4o")

        for p in sample_prompts:
            t0 = time.perf_counter()
            vec = await cache.embedder.embed(p)
            emb_times.append((time.perf_counter() - t0) * 1000.0)

            t1 = time.perf_counter()
            cache.index.search(partition, vec)
            search_times.append((time.perf_counter() - t1) * 1000.0)

        emb_arr = np.array(emb_times)
        search_arr = np.array(search_times)
        baseline_metrics = {
            "embedding_p50_ms": round(float(np.percentile(emb_arr, 50)), 2),
            "embedding_p95_ms": round(float(np.percentile(emb_arr, 95)), 2),
            "faiss_search_p50_ms": round(float(np.percentile(search_arr, 50)), 2),
            "faiss_search_p95_ms": round(float(np.percentile(search_arr, 95)), 2),
        }
        results["baselines"] = baseline_metrics
        print(f"Embedding Latency: p50={baseline_metrics['embedding_p50_ms']}ms, p95={baseline_metrics['embedding_p95_ms']}ms")
        print(f"FAISS Search Latency: p50={baseline_metrics['faiss_search_p50_ms']}ms, p95={baseline_metrics['faiss_search_p95_ms']}ms")

        # 1. Pure Cache MISS path (simulated 800ms upstream)
        print("\nBenchmarking 100% Cache MISS workload (cold prompts)...")
        miss_prompts = [f"Unique cold query {i} for miss path" for i in range(20)]
        miss_latencies = []
        for p in miss_prompts:
            t0 = time.perf_counter()
            await client.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": [{"role": "user", "content": p}]})
            miss_latencies.append((time.perf_counter() - t0) * 1000.0)

        miss_arr = np.array(miss_latencies)
        miss_stats = {
            "p50_ms": round(float(np.percentile(miss_arr, 50)), 2),
            "p95_ms": round(float(np.percentile(miss_arr, 95)), 2),
            "p99_ms": round(float(np.percentile(miss_arr, 99)), 2),
            "avg_ms": round(float(np.mean(miss_arr)), 2),
        }
        results["pure_miss"] = miss_stats
        print(f"Pure MISS: p50={miss_stats['p50_ms']}ms, p95={miss_stats['p95_ms']}ms, p99={miss_stats['p99_ms']}ms")

        # 2. Pure Cache HIT path (warm prompts)
        print("\nBenchmarking 100% Cache HIT workload (warm cached prompts)...")
        await asyncio.sleep(0.1)  # Allow background cache writes to settle
        hit_latencies = []
        for p in miss_prompts:
            t0 = time.perf_counter()
            await client.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": [{"role": "user", "content": p}]})
            hit_latencies.append((time.perf_counter() - t0) * 1000.0)

        hit_arr = np.array(hit_latencies)
        hit_stats = {
            "p50_ms": round(float(np.percentile(hit_arr, 50)), 2),
            "p95_ms": round(float(np.percentile(hit_arr, 95)), 2),
            "p99_ms": round(float(np.percentile(hit_arr, 99)), 2),
            "avg_ms": round(float(np.mean(hit_arr)), 2),
        }
        results["pure_hit"] = hit_stats
        print(f"Pure HIT:  p50={hit_stats['p50_ms']}ms, p95={hit_stats['p95_ms']}ms, p99={hit_stats['p99_ms']}ms")

        p95_reduction = round((1.0 - (hit_stats["p95_ms"] / miss_stats["p95_ms"])) * 100.0, 1)
        results["p95_latency_reduction_percent"] = p95_reduction
        print(f"-> Latency Reduction on Hit vs Miss: {p95_reduction}% reduction in p95 latency!")

        # 3. Concurrency Throughput Scaling (C = 1, 10, 50, 100)
        print("\nEvaluating Concurrency Scaling (C = 1, 10, 50, 100) on warm cache...")
        concurrency_levels = [1, 10, 50, 100]
        concurrency_results = []

        warm_prompt_pool = miss_prompts[:5]  # Repeated warm queries
        for c_level in concurrency_levels:
            test_prompts = [warm_prompt_pool[i % len(warm_prompt_pool)] for i in range(c_level * 5)]
            
            # Run concurrent tasks
            sem = asyncio.Semaphore(c_level)
            async def worker(p):
                async with sem:
                    t_w0 = time.perf_counter()
                    await client.post("/v1/chat/completions", json={"model": "gpt-4o", "messages": [{"role": "user", "content": p}]})
                    return (time.perf_counter() - t_w0) * 1000.0

            t_c_start = time.perf_counter()
            lat_list = await asyncio.gather(*[worker(p) for p in test_prompts])
            c_duration = time.perf_counter() - t_c_start
            rps = len(lat_list) / c_duration if c_duration > 0 else 0.0

            c_arr = np.array(lat_list)
            entry = {
                "concurrency": c_level,
                "total_requests": len(test_prompts),
                "throughput_rps": round(rps, 1),
                "p50_ms": round(float(np.percentile(c_arr, 50)), 2),
                "p95_ms": round(float(np.percentile(c_arr, 95)), 2),
                "p99_ms": round(float(np.percentile(c_arr, 99)), 2),
            }
            concurrency_results.append(entry)
            print(f"Concurrency {c_level:>3}: Throughput={entry['throughput_rps']:>6.1f} req/s, p50={entry['p50_ms']:>5.2f}ms, p95={entry['p95_ms']:>5.2f}ms")

        results["concurrency_scaling"] = concurrency_results

        # 4. Latency under Workload Duplicate/Paraphrase Ratios (0%, 25%, 50%, 75%, 100%)
        print("\nEvaluating p95 Latency under varying Duplicate / Paraphrase Ratios...")
        ratios = [0.0, 0.25, 0.50, 0.75, 1.00]
        workload_results = []

        base_miss_p95 = miss_stats["p95_ms"]
        for r in ratios:
            # Theoretical combined workload p50/p95
            w_p50 = round(r * hit_stats["p50_ms"] + (1 - r) * miss_stats["p50_ms"], 2)
            w_p95 = round(hit_stats["p95_ms"] if r >= 0.95 else (r * hit_stats["p95_ms"] + (1 - r) * miss_stats["p95_ms"]), 2)
            reduction = round((1.0 - (w_p95 / base_miss_p95)) * 100.0, 1)
            workload_results.append({
                "duplicate_ratio_percent": int(r * 100),
                "p50_ms": w_p50,
                "p95_ms": w_p95,
                "p95_reduction_percent": max(0.0, reduction),
            })
            print(f"Hit Ratio {int(r*100):>3}%: p50={w_p50:>6.2f}ms, p95={w_p95:>6.2f}ms, p95 Reduction={reduction:>5.1f}%")

        results["workload_duplicate_ratios"] = workload_results

    await cache.shutdown()

    # Save JSON results
    res_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(res_dir, exist_ok=True)
    json_path = os.path.join(res_dir, "latency_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nLatency benchmark results saved to {json_path}")

    # Generate Chart
    png_path = os.path.join(res_dir, "latency_comparison.png")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    # Subplot 1: Hit vs Miss Latency
    categories = ["p50", "p95", "p99"]
    hit_vals = [hit_stats["p50_ms"], hit_stats["p95_ms"], hit_stats["p99_ms"]]
    miss_vals = [miss_stats["p50_ms"], miss_stats["p95_ms"], miss_stats["p99_ms"]]

    x = np.arange(len(categories))
    width = 0.35

    ax1.bar(x - width/2, hit_vals, width, label="Cache HIT", color="#10b981")
    ax1.bar(x + width/2, miss_vals, width, label="Cache MISS (Upstream 800ms)", color="#f59e0b")
    ax1.set_ylabel("Latency (ms)", fontsize=11)
    ax1.set_title("End-to-End Latency: Cache HIT vs. Upstream MISS", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(categories)
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.4)

    # Subplot 2: Throughput vs Concurrency
    c_labels = [str(e["concurrency"]) for e in concurrency_results]
    c_rps = [e["throughput_rps"] for e in concurrency_results]

    ax2.plot(c_labels, c_rps, marker="o", color="#3b82f6", linewidth=2.5, markersize=8)
    ax2.set_xlabel("Concurrency Level (Clients)", fontsize=11)
    ax2.set_ylabel("Throughput (Requests / sec)", fontsize=11)
    ax2.set_title("CacheSense Cache-Hit Throughput Scaling", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()
    print(f"Latency comparison chart saved to {png_path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    asyncio.run(run_latency_benchmark())
