"""Threshold sweep benchmark for precision, recall, F1, and PR curve on QQP, STS-B, and PAWS."""

import asyncio
import csv
import os

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from app.core.embedder import PromptEmbedder
from benchmarks.datasets import load_benchmark_dataset


async def run_threshold_sweep():
    print("=" * 70)
    print("Running CacheSense Semantic Similarity Threshold Sweep Benchmark")
    print("=" * 70)

    dataset = load_benchmark_dataset()
    print(f"Loaded {len(dataset)} sentence pairs from QQP, STS-B, and PAWS.")

    embedder = PromptEmbedder()
    embedder.load_model()

    # Collect texts
    s1_list = [p.sentence1 for p in dataset]
    s2_list = [p.sentence2 for p in dataset]
    ground_truth = np.array([1 if p.is_duplicate else 0 for p in dataset], dtype=np.int32)

    print("Encoding sentence pairs...")
    v1_matrix = await embedder.embed_batch(s1_list)
    v2_matrix = await embedder.embed_batch(s2_list)

    # Dot product of normalized vectors = Cosine Similarity
    similarities = np.sum(v1_matrix * v2_matrix, axis=1)

    # Prepare results directory
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    csv_file = os.path.join(results_dir, "threshold_sweep.csv")
    png_file = os.path.join(results_dir, "threshold_pr_curve.png")

    thresholds = [round(t, 2) for t in np.arange(0.70, 1.00, 0.01)]
    records = []

    best_f1 = -1.0
    best_thresh = 0.92

    for thresh in thresholds:
        predicted = (similarities >= thresh).astype(np.int32)

        tp = int(np.sum((predicted == 1) & (ground_truth == 1)))
        fp = int(np.sum((predicted == 1) & (ground_truth == 0)))
        tn = int(np.sum((predicted == 0) & (ground_truth == 0)))
        fn = int(np.sum((predicted == 0) & (ground_truth == 1)))

        precision = (tp / (tp + fp)) if (tp + fp) > 0 else 1.0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        fpr = (fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

        records.append({
            "threshold": thresh,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "fpr": round(fpr, 4),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        })

    # Write CSV
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["threshold", "precision", "recall", "f1", "fpr", "tp", "fp", "tn", "fn"]
        )
        writer.writeheader()
        writer.writerows(records)

    print(f"Threshold sweep CSV saved to {csv_file}")

    # Plot PR & F1 curves
    thresh_vals = [r["threshold"] for r in records]
    precisions = [r["precision"] for r in records]
    recalls = [r["recall"] for r in records]
    f1_scores = [r["f1"] for r in records]
    fpr_rates = [r["fpr"] for r in records]

    plt.figure(figsize=(10, 6), dpi=150)
    plt.plot(thresh_vals, precisions, label="Precision", color="#3b82f6", linewidth=2.5)
    plt.plot(thresh_vals, recalls, label="Recall", color="#10b981", linewidth=2.5)
    plt.plot(thresh_vals, f1_scores, label="F1 Score", color="#f59e0b", linewidth=2.5)
    plt.plot(thresh_vals, fpr_rates, label="False Positive Rate", color="#ef4444", linestyle="--", linewidth=1.5)

    # Mark chosen threshold 0.92
    chosen_rec = next(r for r in records if r["threshold"] == 0.92)
    plt.axvline(x=0.92, color="#8b5cf6", linestyle=":", linewidth=2, label=f"Configured Default (0.92, F1={chosen_rec['f1']:.2f})")
    plt.scatter([0.92], [chosen_rec["f1"]], color="#8b5cf6", s=80, zorder=5)

    plt.title("CacheSense Precision, Recall, and F1 vs. Semantic Similarity Threshold", fontsize=14, pad=12)
    plt.xlabel("Cosine Similarity Threshold (tau)", fontsize=12)
    plt.ylabel("Metric Score (0.0 - 1.0)", fontsize=12)
    plt.xlim(0.70, 0.99)
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.legend(loc="lower left", fontsize=10)
    plt.tight_layout()
    plt.savefig(png_file)
    plt.close()

    print(f"Threshold PR Curve chart saved to {png_file}")

    # Print summary table
    print("\n" + "=" * 60)
    print(f"{'Threshold':<10} {'Precision':<12} {'Recall':<10} {'F1':<10} {'FPR':<10}")
    print("-" * 60)
    for r in records:
        if r["threshold"] in [0.75, 0.80, 0.85, 0.90, 0.92, 0.95, 0.98]:
            mark = " <-- DEFAULT" if r["threshold"] == 0.92 else ""
            print(f"{r['threshold']:<10.2f} {r['precision']:<12.4f} {r['recall']:<10.4f} {r['f1']:<10.4f} {r['fpr']:<10.4f}{mark}")
    print("=" * 60)
    print(f"Optimal F1 Threshold in empirical sweep: {best_thresh:.2f} (F1 = {best_f1:.4f})")
    print(f"Default 0.92 operates at: Precision={chosen_rec['precision']:.4f}, Recall={chosen_rec['recall']:.4f}, FPR={chosen_rec['fpr']:.4f}")
    print("=" * 60 + "\n")

    embedder.shutdown()


if __name__ == "__main__":
    asyncio.run(run_threshold_sweep())
