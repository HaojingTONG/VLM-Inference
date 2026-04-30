"""Plot helpers for the visual-token compression notebook."""

from pathlib import Path

import matplotlib.pyplot as plt
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
