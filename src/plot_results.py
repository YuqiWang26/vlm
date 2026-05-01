"""Visualization helpers for efficiency-accuracy trade-offs."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd


METHOD_COLORS = {
    "none": "#4D4D4D",
    "fixed": "#0072B2",
    "importance": "#D55E00",
    "merging": "#009E73",
}

METHOD_LABELS = {
    "none": "no compression",
    "fixed": "fixed pruning",
    "importance": "importance proxy",
    "merging": "merging proxy",
}


def _empty_grouped() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "compression_method",
            "retention_ratio",
            "latency_ms",
            "memory_mb",
            "quality_score",
            "throughput",
            "num_runs",
        ]
    )


def _successful_grouped(df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "compression_method",
        "retention_ratio",
        "latency_ms",
        "peak_gpu_memory_mb",
        "quality_score",
        "throughput_tokens_per_second",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        print(f"Cannot plot: results CSV is missing columns: {missing}")
        return _empty_grouped()

    if "success" in df.columns:
        df = df[df["success"].astype(bool)]
    if df.empty:
        print("Cannot plot: no successful benchmark rows found.")
        return _empty_grouped()

    grouped = (
        df.groupby(["compression_method", "retention_ratio"], dropna=False)
        .agg(
            latency_ms=("latency_ms", "mean"),
            memory_mb=("peak_gpu_memory_mb", "mean"),
            quality_score=("quality_score", "mean"),
            throughput=("throughput_tokens_per_second", "mean"),
            num_runs=("latency_ms", "count"),
        )
        .reset_index()
    )
    grouped["compression_method"] = grouped["compression_method"].astype(str)
    return grouped.sort_values(["compression_method", "retention_ratio"])


def _baseline_value(grouped: pd.DataFrame, metric: str) -> float | None:
    baseline = grouped[grouped["compression_method"] == "none"].sort_values("retention_ratio")
    if baseline.empty or metric not in baseline.columns:
        return None
    exact = baseline[baseline["retention_ratio"] == 1.0]
    row = exact.iloc[0] if not exact.empty else baseline.iloc[-1]
    return float(row[metric])


def _ratios(grouped: pd.DataFrame) -> list[float]:
    values = sorted(float(x) for x in grouped["retention_ratio"].dropna().unique())
    return values or [1.0]


def _setup_ax(title: str, ylabel: str) -> tuple[plt.Figure, plt.Axes]:
    fig, ax = plt.subplots(figsize=(9.5, 5.6), dpi=180)
    ax.set_title(title, fontsize=16, pad=12)
    ax.set_xlabel("Visual token retention ratio", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, which="major", color="#D0D0D0", alpha=0.55, linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    return fig, ax


def _annotate_last_point(ax: plt.Axes, sub: pd.DataFrame, y_col: str, label: str) -> None:
    if sub.empty:
        return
    row = sub.sort_values("retention_ratio").iloc[-1]
    ax.annotate(
        label,
        xy=(row["retention_ratio"], row[y_col]),
        xytext=(7, 0),
        textcoords="offset points",
        fontsize=9,
        va="center",
    )


def _line_plot(
    grouped: pd.DataFrame,
    y_col: str,
    ylabel: str,
    title: str,
    output_path: Path,
    baseline_label: str | None = None,
    y_min_zero: bool = False,
) -> None:
    ratios = _ratios(grouped)
    fig, ax = _setup_ax(title, ylabel)

    baseline = _baseline_value(grouped, y_col)
    if baseline is not None:
        ax.axhline(
            baseline,
            linestyle="--",
            linewidth=1.8,
            color=METHOD_COLORS["none"],
            alpha=0.85,
            label=baseline_label or f"no compression: {baseline:.1f}",
        )

    plotted_any = False
    for method in ["fixed", "importance", "merging", "none"]:
        sub = grouped[grouped["compression_method"] == method].sort_values("retention_ratio")
        if sub.empty:
            continue
        if method == "none":
            ax.scatter(
                sub["retention_ratio"],
                sub[y_col],
                s=62,
                color=METHOD_COLORS["none"],
                zorder=4,
            )
            continue

        label = METHOD_LABELS.get(method, method)
        ax.plot(
            sub["retention_ratio"],
            sub[y_col],
            marker="o",
            markersize=6,
            linewidth=2.4,
            color=METHOD_COLORS.get(method),
            label=label,
        )
        _annotate_last_point(ax, sub, y_col, label)
        plotted_any = True

    ax.set_xticks(ratios)
    ax.set_xlim(min(ratios) - 0.03, max(ratios) + 0.08)
    if y_min_zero:
        ax.set_ylim(bottom=0)
    if plotted_any or baseline is not None:
        ax.legend(frameon=False, loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _savings_plot(
    grouped: pd.DataFrame,
    metric: str,
    ylabel: str,
    title: str,
    output_path: Path,
) -> None:
    baseline = _baseline_value(grouped, metric)
    if baseline is None or baseline == 0:
        return

    savings = grouped[grouped["compression_method"] != "none"].copy()
    if savings.empty:
        return
    savings["savings_pct"] = (baseline - savings[metric]) / baseline * 100.0

    fig, ax = _setup_ax(title, ylabel)
    ratios = _ratios(grouped)
    ax.axhline(0, color="#808080", linestyle="--", linewidth=1.2, alpha=0.8)

    for method in ["fixed", "importance", "merging"]:
        sub = savings[savings["compression_method"] == method].sort_values("retention_ratio")
        if sub.empty:
            continue
        label = METHOD_LABELS.get(method, method)
        ax.plot(
            sub["retention_ratio"],
            sub["savings_pct"],
            marker="o",
            markersize=6,
            linewidth=2.4,
            color=METHOD_COLORS.get(method),
            label=label,
        )
        _annotate_last_point(ax, sub, "savings_pct", label)

    ax.set_xticks(ratios)
    ax.set_xlim(min(ratios) - 0.03, max(ratios) + 0.08)
    ax.legend(frameon=False, loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def _tradeoff_plot(grouped: pd.DataFrame, output_path: Path) -> None:
    fig, ax = _setup_ax("Efficiency-accuracy trade-off", "Quality score")
    ax.set_xlabel("Latency (ms)", fontsize=12)

    for method in ["fixed", "importance", "merging", "none"]:
        sub = grouped[grouped["compression_method"] == method].sort_values("retention_ratio")
        if sub.empty:
            continue
        color = METHOD_COLORS.get(method)
        label = METHOD_LABELS.get(method, method)
        ax.plot(
            sub["latency_ms"],
            sub["quality_score"],
            marker="o",
            markersize=7,
            linewidth=2.0 if method != "none" else 0,
            linestyle="-" if method != "none" else "None",
            color=color,
            label=label,
        )
        for _, row in sub.iterrows():
            ax.annotate(
                f"{row['retention_ratio']:.2g}",
                (row["latency_ms"], row["quality_score"]),
                xytext=(6, 5),
                textcoords="offset points",
                fontsize=8,
                color=color,
            )

    ax.legend(frameon=False, loc="best", fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def plot_all(
    results_csv: str | Path = "results/benchmark_results.csv",
    output_dir: str | Path = "results",
) -> Iterable[Path]:
    results_csv = Path(results_csv)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(results_csv)
    grouped = _successful_grouped(df)
    if grouped.empty:
        if "error" in df.columns:
            failed = df[df["success"].astype(bool) == False] if "success" in df.columns else df  # noqa: E712
            if not failed.empty:
                print("\nFirst benchmark errors:")
                print(
                    failed[
                        [
                            col
                            for col in [
                                "compression_method",
                                "retention_ratio",
                                "sample_id",
                                "oom",
                                "error",
                            ]
                            if col in failed.columns
                        ]
                    ]
                    .head(5)
                    .to_string(index=False)
                )
        return []

    outputs: list[Path] = []
    latency_path = output_dir / "latency_vs_retention_ratio.png"
    latency_savings_path = output_dir / "latency_savings_vs_retention_ratio.png"
    memory_path = output_dir / "memory_vs_retention_ratio.png"
    memory_savings_path = output_dir / "memory_savings_vs_retention_ratio.png"
    quality_path = output_dir / "quality_vs_retention_ratio.png"
    tradeoff_path = output_dir / "efficiency_accuracy_tradeoff.png"

    _line_plot(
        grouped,
        "latency_ms",
        "Latency (ms)",
        "Latency vs visual token retention",
        latency_path,
        baseline_label="no compression baseline",
        y_min_zero=True,
    )
    outputs.append(latency_path)

    _savings_plot(
        grouped,
        "latency_ms",
        "Latency reduction vs no compression (%)",
        "Latency savings vs visual token retention",
        latency_savings_path,
    )
    if latency_savings_path.exists():
        outputs.append(latency_savings_path)

    _line_plot(
        grouped,
        "memory_mb",
        "Peak GPU memory (MB)",
        "Peak GPU memory vs visual token retention",
        memory_path,
        baseline_label="no compression baseline",
    )
    outputs.append(memory_path)

    _savings_plot(
        grouped,
        "memory_mb",
        "Peak memory reduction vs no compression (%)",
        "Memory savings vs visual token retention",
        memory_savings_path,
    )
    if memory_savings_path.exists():
        outputs.append(memory_savings_path)

    _line_plot(
        grouped,
        "quality_score",
        "Quality score",
        "Quality vs visual token retention",
        quality_path,
        baseline_label="no compression baseline",
    )
    outputs.append(quality_path)

    _tradeoff_plot(grouped, tradeoff_path)
    outputs.append(tradeoff_path)
    return outputs


if __name__ == "__main__":
    for path in plot_all():
        print(path)
