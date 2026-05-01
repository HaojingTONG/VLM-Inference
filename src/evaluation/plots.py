"""Plot helpers for the visual-token compression notebook."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _save(fig, output_dir, filename):
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    fig.savefig(path / filename, dpi=160, bbox_inches="tight")


def plot_quality_vs_compression(df_quality: pd.DataFrame, metric_label: str, output_dir="results"):
    """Plot task quality as token retention changes."""
    fig, ax = plt.subplots(figsize=(8, 5))
    baseline = df_quality[df_quality["method"] == "none"]
    if not baseline.empty:
        baseline_score = baseline["score"].mean()
        ax.axhline(
            baseline_score,
            color="black",
            linestyle="--",
            alpha=0.6,
            label=f"baseline ({baseline_score:.3f})",
        )

    for method, group in df_quality[df_quality["method"] != "none"].groupby("method"):
        group = group.sort_values("retention_ratio")
        ax.plot(group["retention_ratio"], group["score"], marker="o", linewidth=2, label=method)

    ax.set_xlabel("Visual token retention ratio")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} vs compression")
    ax.invert_xaxis()
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, output_dir, "quality_vs_compression.png")
    return fig


def plot_quality_ci_vs_compression(
    df_quality_ci: pd.DataFrame,
    metric_label: str,
    output_dir="results",
):
    """Plot task quality with bootstrap confidence intervals."""
    fig, ax = plt.subplots(figsize=(8, 5))
    baseline = df_quality_ci[df_quality_ci["method"] == "none"]
    if not baseline.empty:
        row = baseline.iloc[0]
        ax.axhline(
            row["mean_score"],
            color="black",
            linestyle="--",
            alpha=0.6,
            label=f"baseline ({row['mean_score']:.3f})",
        )
        ax.axhspan(row["ci_low"], row["ci_high"], color="black", alpha=0.08)

    for method, group in df_quality_ci[df_quality_ci["method"] != "none"].groupby("method"):
        group = group.sort_values("retention_ratio")
        yerr = [
            group["mean_score"] - group["ci_low"],
            group["ci_high"] - group["mean_score"],
        ]
        ax.errorbar(
            group["retention_ratio"],
            group["mean_score"],
            yerr=yerr,
            marker="o",
            capsize=3,
            linewidth=2,
            label=method,
        )

    ax.set_xlabel("Visual token retention ratio")
    ax.set_ylabel(metric_label)
    ax.set_title(f"{metric_label} vs compression with bootstrap CI")
    ax.invert_xaxis()
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, output_dir, "quality_ci_vs_compression.png")
    return fig


def plot_accuracy_delta_vs_baseline(
    df_quality: pd.DataFrame,
    metric_label: str,
    output_dir="results",
):
    """Plot accuracy change relative to the uncompressed baseline."""
    baseline = df_quality[df_quality["method"] == "none"]["score"].mean()
    compressed = df_quality[df_quality["method"] != "none"].copy()
    compressed["delta_vs_baseline"] = compressed["score"] - baseline

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    for method, group in compressed.groupby("method"):
        group = group.sort_values("retention_ratio")
        ax.plot(
            group["retention_ratio"],
            group["delta_vs_baseline"],
            marker="o",
            linewidth=2,
            label=method,
        )

    ax.set_xlabel("Visual token retention ratio")
    ax.set_ylabel(f"Delta {metric_label} vs baseline")
    ax.set_title("Accuracy change relative to uncompressed baseline")
    ax.invert_xaxis()
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, output_dir, "accuracy_delta_vs_baseline.png")
    return fig


def plot_accuracy_heatmap(
    df_quality: pd.DataFrame,
    metric_label: str,
    output_dir="results",
):
    """Show method/retention accuracy as a compact heatmap."""
    compressed = df_quality[df_quality["method"] != "none"].copy()
    pivot = compressed.pivot_table(
        index="method",
        columns="retention_ratio",
        values="score",
        aggfunc="mean",
    )
    pivot = pivot.reindex(sorted(pivot.columns, reverse=True), axis=1)

    fig, ax = plt.subplots(figsize=(8, 3.6))
    im = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0.4, vmax=0.9)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{c:.2f}" for c in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Visual token retention ratio")
    ax.set_title(f"{metric_label} by late-compression method")

    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.3f}", ha="center", va="center", fontsize=8)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(metric_label)
    fig.tight_layout()
    _save(fig, output_dir, "accuracy_heatmap.png")
    return fig


def plot_latency_vs_compression(df_perf: pd.DataFrame, output_dir="results"):
    """Plot latency as token retention changes for each resolution."""
    resolutions = [r for r in ["low", "medium", "high"] if r in set(df_perf["resolution"])]
    if not resolutions:
        resolutions = sorted(df_perf["resolution"].dropna().unique())
    fig, axes = plt.subplots(1, len(resolutions), figsize=(5 * len(resolutions), 4), squeeze=False)

    for ax, resolution in zip(axes[0], resolutions):
        sub = df_perf[df_perf["resolution"] == resolution]
        baseline = sub[sub["method"] == "none"]["latency_ms"].mean()
        if pd.notna(baseline):
            ax.axhline(baseline, color="black", linestyle="--", alpha=0.6, label=f"baseline ({baseline:.0f} ms)")
        for method, group in sub[sub["method"] != "none"].groupby("method"):
            group = group.sort_values("retention_ratio")
            yerr = group["latency_std_ms"] if "latency_std_ms" in group else None
            ax.errorbar(group["retention_ratio"], group["latency_ms"], yerr=yerr, marker="o", capsize=3, label=method)
        ax.set_title(f"{resolution} resolution")
        ax.set_xlabel("Visual token retention ratio")
        ax.set_ylabel("Latency (ms)")
        ax.invert_xaxis()
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Latency vs compression")
    fig.tight_layout()
    _save(fig, output_dir, "latency_vs_compression.png")
    return fig


def plot_memory_vs_compression(df_perf: pd.DataFrame, output_dir="results"):
    """Plot peak GPU memory as token retention changes."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for (resolution, method), group in df_perf.groupby(["resolution", "method"]):
        group = group.sort_values("retention_ratio")
        label = f"{resolution}/{method}"
        ax.plot(group["retention_ratio"], group["peak_memory_mb"], marker="o", linewidth=1.5, label=label)
    ax.set_xlabel("Visual token retention ratio")
    ax.set_ylabel("Peak GPU memory (MB)")
    ax.set_title("Memory vs compression")
    ax.invert_xaxis()
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    _save(fig, output_dir, "memory_vs_compression.png")
    return fig


def plot_tradeoff(
    df_quality: pd.DataFrame,
    df_perf: pd.DataFrame,
    metric_label: str,
    resolution: str = "medium",
    output_dir="results",
):
    """Plot quality against latency for matching method / ratio settings."""
    perf = (
        df_perf[df_perf["resolution"] == resolution]
        .groupby(["method", "retention_ratio"], as_index=False)
        .agg(latency_ms=("latency_ms", "mean"), peak_memory_mb=("peak_memory_mb", "max"))
    )
    merged = df_quality.merge(perf, on=["method", "retention_ratio"], how="inner")

    fig, ax = plt.subplots(figsize=(7, 5))
    for method, group in merged.groupby("method"):
        group = group.sort_values("retention_ratio")
        ax.plot(group["latency_ms"], group["score"], marker="o", linewidth=1.8, label=method)
        for _, row in group.iterrows():
            ax.annotate(
                f"{row['retention_ratio']:.2f}",
                (row["latency_ms"], row["score"]),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )
    ax.set_xlabel(f"Latency (ms, {resolution} resolution)")
    ax.set_ylabel(metric_label)
    ax.set_title("Quality-latency tradeoff")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, output_dir, "quality_latency_tradeoff.png")
    return fig


def plot_quality_vs_sequence_reduction(
    df_quality: pd.DataFrame,
    df_effectiveness: pd.DataFrame,
    metric_label: str,
    output_dir="results",
):
    """Plot accuracy against actual LLM sequence reduction for late compression."""
    effectiveness_cols = [
        "method",
        "retention_ratio",
        "sequence_reduction_pct",
        "visual_token_reduction_pct",
    ]
    merged = df_quality.merge(
        df_effectiveness[[c for c in effectiveness_cols if c in df_effectiveness.columns]],
        on=["method", "retention_ratio"],
        how="inner",
    )
    merged = merged[merged["method"] != "none"].dropna(subset=["score", "sequence_reduction_pct"])

    fig, ax = plt.subplots(figsize=(7, 5))
    baseline = df_quality[df_quality["method"] == "none"]["score"].mean()
    if pd.notna(baseline):
        ax.axhline(
            baseline,
            color="black",
            linestyle="--",
            alpha=0.6,
            label=f"baseline ({baseline:.3f})",
        )

    for method, group in merged.groupby("method"):
        group = group.sort_values("sequence_reduction_pct")
        ax.plot(
            group["sequence_reduction_pct"],
            group["score"],
            marker="o",
            linewidth=2,
            label=method,
        )
        for _, row in group.iterrows():
            ax.annotate(
                f"{row['retention_ratio']:.2f}",
                (row["sequence_reduction_pct"], row["score"]),
                fontsize=7,
                xytext=(3, 3),
                textcoords="offset points",
            )

    ax.set_xlabel("Actual LLM sequence reduction (%)")
    ax.set_ylabel(metric_label)
    ax.set_title("Accuracy vs actual LLM sequence reduction")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save(fig, output_dir, "quality_vs_sequence_reduction.png")
    return fig


def plot_late_compression_effectiveness(df_effectiveness: pd.DataFrame, output_dir="results"):
    """Plot readable late-compression optimization metrics."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sub = df_effectiveness[df_effectiveness["method"] != "none"].copy()
    if sub.empty:
        axes[0].set_title("Late compression effectiveness unavailable")
        fig.tight_layout()
        _save(fig, output_dir, "late_compression_effectiveness.png")
        return fig

    for method, group in sub.groupby("method"):
        group = group.sort_values("retention_ratio")
        axes[0].plot(
            group["retention_ratio"],
            group["visual_token_reduction_pct"],
            marker="o",
            linewidth=2,
            label=f"{method} visual tokens",
        )
        axes[0].plot(
            group["retention_ratio"],
            group["sequence_reduction_pct"],
            marker="s",
            linestyle="--",
            linewidth=1.5,
            label=f"{method} LLM sequence",
        )
        if "measured_llm_prefill_speedup" in group:
            axes[1].plot(
                group["retention_ratio"],
                group["measured_llm_prefill_speedup"],
                marker="s",
                linestyle="--",
                linewidth=1.5,
                label=f"{method} isolated LLM prefill",
            )
        if "measured_end_to_end_speedup" in group:
            axes[1].plot(
                group["retention_ratio"],
                group["measured_end_to_end_speedup"],
                marker="x",
                linestyle=":",
                linewidth=1.5,
                label=f"{method} end-to-end",
            )

    axes[0].set_xlabel("Visual token retention ratio")
    axes[0].set_ylabel("Reduction (%)")
    axes[0].set_title("Late compression reduces LLM-side sequence")
    axes[0].invert_xaxis()
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)

    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.6)
    axes[1].set_xlabel("Visual token retention ratio")
    axes[1].set_ylabel("Measured speedup")
    axes[1].set_title("Prefill gains do not become end-to-end speedup")
    axes[1].invert_xaxis()
    axes[1].grid(alpha=0.3)
    axes[1].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    _save(fig, output_dir, "late_compression_effectiveness.png")
    return fig
