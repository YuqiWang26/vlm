"""Visualization helpers for efficiency-accuracy trade-offs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


def _successful_grouped(df: pd.DataFrame) -> pd.DataFrame:
    if "success" in df.columns:
        df = df[df["success"].astype(bool)]
    return (
        df.groupby(["compression_method", "retention_ratio"], dropna=False)
        .agg(
            latency_ms=("latency_ms", "mean"),
            memory_mb=("peak_gpu_memory_mb", "mean"),
            quality_score=("quality_score", "mean"),
            throughput=("throughput_tokens_per_second", "mean"),
        )
        .reset_index()
        .sort_values(["compression_method", "retention_ratio"])
    )


def _line_plot(
    grouped: pd.DataFrame,
    y_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    plt.figure(figsize=(7, 4.5))
    for method, sub in grouped.groupby("compression_method"):
        sub = sub.sort_values("retention_ratio")
        plt.plot(sub["retention_ratio"], sub[y_col], marker="o", label=method)
    plt.xlabel("Retention ratio")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_all(
    results_csv: str | Path = "results/benchmark_results.csv",
    output_dir: str | Path = "results",
) -> Iterable[Path]:
    results_csv = Path(results_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(results_csv)
    grouped = _successful_grouped(df)

    outputs = []
    latency_path = output_dir / "latency_vs_retention_ratio.png"
    memory_path = output_dir / "memory_vs_retention_ratio.png"
    quality_path = output_dir / "quality_vs_retention_ratio.png"
    tradeoff_path = output_dir / "efficiency_accuracy_tradeoff.png"

    _line_plot(grouped, "latency_ms", "Latency (ms)", "Latency vs retention ratio", latency_path)
    _line_plot(grouped, "memory_mb", "Peak GPU memory (MB)", "Memory vs retention ratio", memory_path)
    _line_plot(grouped, "quality_score", "Quality score", "Quality vs retention ratio", quality_path)
    outputs.extend([latency_path, memory_path, quality_path])

    plt.figure(figsize=(7, 4.5))
    for method, sub in grouped.groupby("compression_method"):
        plt.scatter(sub["latency_ms"], sub["quality_score"], s=60, label=method)
        for _, row in sub.iterrows():
            plt.annotate(f"{row['retention_ratio']:.2g}", (row["latency_ms"], row["quality_score"]))
    plt.xlabel("Latency (ms)")
    plt.ylabel("Quality score")
    plt.title("Efficiency-accuracy trade-off")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(tradeoff_path, dpi=180)
    plt.close()
    outputs.append(tradeoff_path)
    return outputs


if __name__ == "__main__":
    for path in plot_all():
        print(path)
