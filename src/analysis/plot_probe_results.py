from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


METRIC_LABELS = {
    "accuracy": "Accuracy",
    "balanced_accuracy": "Balanced accuracy",
    "macro_f1": "Macro F1",
    "weighted_f1": "Weighted F1",
}


def plot_layerwise_probe_summary(
    summary: pd.DataFrame,
    output_path: str | Path,
    metric: str = "macro_f1",
    title: str | None = None,
) -> Path:
    mean_column = f"{metric}_mean"
    std_column = f"{metric}_std"
    if mean_column not in summary.columns:
        raise KeyError(f"Probe summary is missing column: {mean_column}")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    probes = summary[summary["model"].eq("linear_probe")].sort_values("layer")
    ax.errorbar(
        probes["layer"],
        probes[mean_column],
        yerr=probes[std_column] if std_column in probes.columns else None,
        marker="o",
        linewidth=2,
        capsize=3,
        label="Linear probe",
        color="#1f77b4",
    )

    baseline_colors = {"majority": "#7f7f7f", "stratified": "#bcbd22"}
    for baseline_name, baseline in summary[summary["model"].ne("linear_probe")].groupby("model"):
        value = float(baseline[mean_column].mean())
        ax.axhline(
            value,
            linestyle="--",
            linewidth=1.5,
            color=baseline_colors.get(baseline_name, "#999999"),
            label=f"{baseline_name} baseline",
        )

    metric_label = METRIC_LABELS.get(metric, metric)
    ax.set_xlabel("Wav2Vec2 layer")
    ax.set_ylabel(metric_label)
    ax.set_title(title or f"Layer-wise probe {metric_label}")
    ax.set_xticks(probes["layer"].astype(int).tolist())
    ax.set_ylim(bottom=0.0, top=min(1.0, max(0.2, float(probes[mean_column].max()) + 0.1)))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_probe_comparison(
    summaries: list[tuple[str, pd.DataFrame]],
    output_path: str | Path,
    metric: str = "macro_f1",
    title: str | None = None,
) -> Path:
    mean_column = f"{metric}_mean"
    std_column = f"{metric}_std"
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    for label, summary in summaries:
        probes = summary[summary["model"].eq("linear_probe")].sort_values("layer")
        if probes.empty:
            continue
        ax.errorbar(
            probes["layer"],
            probes[mean_column],
            yerr=probes[std_column] if std_column in probes.columns else None,
            marker="o",
            linewidth=2,
            capsize=3,
            label=label,
        )

    metric_label = METRIC_LABELS.get(metric, metric)
    ax.set_xlabel("Wav2Vec2 layer")
    ax.set_ylabel(metric_label)
    ax.set_title(title or f"Layer-wise probe comparison ({metric_label})")
    ax.set_ylim(bottom=0.0, top=1.0)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path
