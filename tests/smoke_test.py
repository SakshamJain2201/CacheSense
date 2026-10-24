"""End-to-end smoke test verifying deployed or local CacheSense instance."""

import argparse
import sys
import time

import httpx


def run_smoke_test(target_url: str):
    print("=" * 65)
    print(f"Executing CacheSense Smoke Test against: {target_url}")
    print("=" * 65)

    client = httpx.Client(base_url=target_url, timeout=30.0)

    # 1. Health Probe Check
    print("\n[Step 1] Checking /health endpoint and resetting test cache...")
    try:
        r_health = client.get("/health")
        assert r_health.status_code == 200, f"Expected 200, got {r_health.status_code}"
        health_data = r_health.json()
        assert health_data.get("status") == "healthy", "Service is not reported as healthy"
        
        # Purge cache to start test in clean state
        r_purge = client.delete("/cache")
        assert r_purge.status_code == 200
        print(f" PASS: Service healthy! Uptime: {health_data['uptime_seconds']}s | Cache successfully reset.")
    except Exception as exc:
        print(f" FAIL: Health check or cache reset failed: {exc}")
        sys.exit(1)

    # 2. Initial Request -> MISS
    prompt_1 = "Explain Dijkstra's shortest path algorithm concisely."
    print("\n[Step 2] Sending initial cold prompt (Expect X-Cache: MISS)...")
    print(f"  Prompt: '{prompt_1}'")
    try:
        t0 = time.perf_counter()
        r_miss = client.post("/v1/chat/completions", json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": prompt_1}],
            "temperature": 0.0,
        })
        lat_miss = (time.perf_counter() - t0) * 1000.0
        assert r_miss.status_code == 200, f"Expected 200, got {r_miss.status_code}"
        cache_hdr = r_miss.headers.get("X-Cache")
        assert cache_hdr == "MISS", f"Expected X-Cache: MISS, got {cache_hdr}"
        print(f" PASS: X-Cache: {cache_hdr} | Latency: {lat_miss:.2f}ms")
    except Exception as exc:
        print(f" FAIL: Cold request failed: {exc}")
        sys.exit(1)

    # Settle background store insert
    time.sleep(0.3)

    # 3. Identical Request -> HIT
    print("\n[Step 3] Sending identical prompt (Expect X-Cache: HIT)...")
    try:
        t0 = time.perf_counter()
        r_hit = client.post("/v1/chat/completions", json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": prompt_1}],
            "temperature": 0.0,
        })
        lat_hit = (time.perf_counter() - t0) * 1000.0
        assert r_hit.status_code == 200, f"Expected 200, got {r_hit.status_code}"
        cache_hdr = r_hit.headers.get("X-Cache")
        similarity = float(r_hit.headers.get("X-Cache-Similarity", 0.0))
        assert cache_hdr == "HIT", f"Expected X-Cache: HIT, got {cache_hdr}"
        assert similarity >= 0.99, f"Expected similarity >= 0.99, got {similarity}"
        print(f" PASS: X-Cache: {cache_hdr} | Similarity: {similarity:.4f} | Latency: {lat_hit:.2f}ms")
    except Exception as exc:
        print(f" FAIL: Identical prompt cache hit failed: {exc}")
        sys.exit(1)

    # 4. Paraphrased Request -> HIT
    prompt_para = "Concisely explain Dijkstra's shortest path algorithm."
    print("\n[Step 4] Sending paraphrased prompt (Expect X-Cache: HIT with similarity >= 0.92)...")
    print(f"  Prompt: '{prompt_para}'")
    try:
        t0 = time.perf_counter()
        r_para = client.post("/v1/chat/completions", json={
            "model": "gpt-4o",
            "messages": [{"role": "user", "content": prompt_para}],
            "temperature": 0.0,
        })
        lat_para = (time.perf_counter() - t0) * 1000.0
        assert r_para.status_code == 200, f"Expected 200, got {r_para.status_code}"
        cache_hdr = r_para.headers.get("X-Cache")
        similarity = float(r_para.headers.get("X-Cache-Similarity", 0.0))
        assert cache_hdr == "HIT", f"Expected X-Cache: HIT, got {cache_hdr}"
        assert similarity >= 0.92, f"Expected similarity >= 0.92, got {similarity}"
        print(f" PASS: X-Cache: {cache_hdr} | Similarity: {similarity:.4f} | Latency: {lat_para:.2f}ms")
    except Exception as exc:
        print(f" FAIL: Paraphrase cache hit failed: {exc}")
        sys.exit(1)

    # 5. Different Partition Request -> MISS
    print("\n[Step 5] Sending request with DIFFERENT model (Expect X-Cache: MISS)...")
    try:
        r_diff_model = client.post("/v1/chat/completions", json={
            "model": "gpt-4o-mini",
            "messages": [{"role": "user", "content": prompt_1}],
            "temperature": 0.0,
        })
        assert r_diff_model.status_code == 200
        cache_hdr = r_diff_model.headers.get("X-Cache")
        assert cache_hdr == "MISS", f"Expected partition isolation MISS, got {cache_hdr}"
        print(f" PASS: Model partition isolation verified (X-Cache: {cache_hdr})")
    except Exception as exc:
        print(f" FAIL: Partition isolation check failed: {exc}")
        sys.exit(1)

    print("\n" + "=" * 65)
    print(" ALL SMOKE TESTS PASSED SUCCESSFULLY! Service verified.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CacheSense Smoke Test")
    parser.add_argument("--url", default="http://localhost:8000", help="Target base URL")
    args = parser.parse_args()
    run_smoke_test(args.url.rstrip("/"))
