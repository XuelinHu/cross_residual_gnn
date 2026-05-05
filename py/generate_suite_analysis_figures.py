from __future__ import annotations

import glob
import json
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geomatric.experiment_catalog import (
    ALL_ACTIVE_DATASETS,
    EXTERNAL_BASELINES,
    FOCUSED_MODELS,
    MAIN_BIOLOGICAL_DATASETS,
    MODEL_DISPLAY,
)
from geomatric.experiment_paths import DEFAULT_EXPERIMENT_VERSION, ensure_version_manifest, log_dir, normalize_version
from py.plot_style import MODEL_COLORS, MODEL_MARKERS, apply_paper_style, style_axis

FIG_DIRS = [
    ROOT / "figures" / "exp",
    ROOT / "paper" / "figures" / "exp",
]

DATASETS = ALL_ACTIVE_DATASETS
TOPIC_DATASETS = MAIN_BIOLOGICAL_DATASETS
CR_MODELS = list(FOCUSED_MODELS)
EXT_MODELS = [name for name, _ in EXTERNAL_BASELINES]
ALL_MODELS = CR_MODELS + EXT_MODELS
MODELS = CR_MODELS  # CR-GNN family only
OUTPUT_SUFFIX = ""

_BASELINE_COLORS = {
    "GraphSAGEBaseline": "#E69F00",
    "GINBaseline": "#56B4E9",
    "JKNetBaseline": "#F0E442",
    "APPNPBaseline": "#D55E00",
}
_BASELINE_MARKERS = {
    "GraphSAGEBaseline": "*",
    "GINBaseline": "X",
    "JKNetBaseline": "p",
    "APPNPBaseline": "h",
}
_BASELINE_DISPLAY = {
    "GraphSAGEBaseline": "GraphSAGE",
    "GINBaseline": "GIN",
    "JKNetBaseline": "JKNet",
    "APPNPBaseline": "APPNP",
}
ALL_COLORS = {**MODEL_COLORS, **_BASELINE_COLORS}
ALL_MARKERS = {**MODEL_MARKERS, **_BASELINE_MARKERS}
ALL_DISPLAY = {**MODEL_DISPLAY, **_BASELINE_DISPLAY}

_MODEL_OPERATOR = {m: "GCNConv" for m in CR_MODELS}
_MODEL_OPERATOR.update({name: op for name, op in EXTERNAL_BASELINES})


def latest_matching_log(dataset: str, model: str, fold: int, active_log_dir: Path) -> Path:
    op = _MODEL_OPERATOR.get(model, "GCNConv")
    pattern = str(active_log_dir / f"train_{dataset}_{model}_{op}_fold{fold}__*.json")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(pattern)
    return Path(matches[-1])


def load_rows(active_log_dir: Path) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for dataset in DATASETS:
        for model in MODELS:
            for fold in range(5):
                try:
                    payload = json.loads(latest_matching_log(dataset, model, fold, active_log_dir).read_text(encoding="utf-8"))
                except FileNotFoundError:
                    continue
                rows.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "fold": fold,
                        "best_test_acc": float(payload["best_test_acc"]),
                        "test_loss": float(payload["test_loss"]),
                        "best_epoch": int(payload["best_epoch"]) + 1,
                        "params": int(payload["parameter_stats"]["total_params"]),
                    }
                )
    return rows


def summarize(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, Dict[str, float]]]:
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    for dataset in DATASETS:
        summary[dataset] = {}
        for model in MODELS:
            subset = [row for row in rows if row["dataset"] == dataset and row["model"] == model]
            if not subset:
                summary[dataset][model] = {"pending": True}
                continue
            accs = [float(row["best_test_acc"]) for row in subset]
            losses = [float(row["test_loss"]) for row in subset]
            epochs = [int(row["best_epoch"]) for row in subset]
            summary[dataset][model] = {
                "mean_acc": float(st.mean(accs)),
                "std_acc": float(st.pstdev(accs)),
                "mean_loss": float(st.mean(losses)),
                "mean_best_epoch": float(st.mean(epochs)),
                "params": int(subset[0]["params"]),
            }
    return summary


def completed_datasets(summary: Dict[str, Dict[str, Dict[str, float]]]) -> List[str]:
    datasets: List[str] = []
    for dataset in DATASETS:
        if all(not summary[dataset][model].get("pending") for model in MODELS):
            datasets.append(dataset)
    return datasets


def ensure_dirs() -> None:
    for path in FIG_DIRS:
        path.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, filename: str) -> None:
    actual_name = f"{filename}_{OUTPUT_SUFFIX}" if OUTPUT_SUFFIX else filename
    for out_dir in FIG_DIRS:
        fig.savefig(out_dir / f"{actual_name}.pdf", dpi=300, bbox_inches="tight")
        fig.savefig(out_dir / f"{actual_name}.png", dpi=300, bbox_inches="tight")


def add_bar_labels(
    ax: plt.Axes,
    bars,
    fmt: str = "{:.3f}",
    fontsize: int = 10,
    rotation: int = 90,
    inside: bool = False,
) -> None:
    for bar in bars:
        height = bar.get_height()
        if inside:
            y = max(height - 0.018, height * 0.65)
            va = "top"
            color = "white"
        else:
            y = height + 0.006
            va = "bottom"
            color = "black"
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            y,
            fmt.format(height),
            ha="center",
            va=va,
            fontsize=fontsize,
            rotation=rotation,
            color=color,
            fontweight="bold" if inside else None,
            clip_on=False,
            zorder=5,
        )


def plot_full_suite(summary: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    apply_paper_style()
    datasets = completed_datasets(summary)
    if not datasets:
        return
    fig, ax = plt.subplots(figsize=(14, 6.5))
    x = np.arange(len(datasets))
    n_models = len(MODELS)
    width = 0.9 / n_models  # bars fill the group with no gap
    offsets = np.linspace(-(n_models-1)/2, (n_models-1)/2, n_models) * width

    for offset, model in zip(offsets, MODELS):
        means = [summary[dataset][model]["mean_acc"] for dataset in datasets]
        bars = ax.bar(
            x + offset,
            means,
            width=width,
            color=ALL_COLORS[model],
            label=ALL_DISPLAY[model],
            edgecolor="black",
            linewidth=0.5,
            zorder=3,
        )
        add_bar_labels(ax, bars, fmt="{:.4f}", fontsize=8, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets, rotation=15, fontsize=16)
    ax.set_ylabel("Mean best test accuracy", fontsize=18)
    ax.set_title("Benchmark across the active dataset package", fontsize=20, fontweight="bold")
    # Adaptive y-axis based on data
    all_means = [summary[ds][m]["mean_acc"] for ds in datasets for m in MODELS]
    y_min = max(0, min(all_means) - 0.10)
    y_max = max(all_means) + 0.08
    ax.set_ylim(y_min, y_max)
    ax.tick_params(axis='y', labelsize=14)
    style_axis(ax)
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.16), fontsize=14)
    fig.tight_layout()
    save(fig, "fig1_full_suite_results")
    plt.close(fig)


def plot_cross_heatmap(summary: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    apply_paper_style()
    datasets = completed_datasets(summary)
    if not datasets:
        return
    matrix = []
    for dataset in datasets:
        plain = summary[dataset]["PlainGNN"]["mean_acc"]
        node_res = summary[dataset]["NodeResGNN"]["mean_acc"]
        node_cross = summary[dataset]["NodeCrossGNN"]["mean_acc"]
        graph_res = summary[dataset]["GraphResGNN"]["mean_acc"]
        graph_cross = summary[dataset]["GraphCrossGNN"]["mean_acc"]
        best_cross = max(node_cross, graph_cross)
        best_residual = max(node_res, graph_res)
        matrix.append(
            [
                best_cross - plain,
                best_cross - best_residual,
                node_cross - node_res,
                graph_cross - graph_res,
            ]
        )

    values = np.array(matrix)
    labels = ["BestCross-Plain", "BestCross-BestRes", "NodeCross-NodeRes", "GraphCross-GraphRes"]

    fig, ax = plt.subplots(figsize=(8.8, 5.8))
    im = ax.imshow(values, cmap="coolwarm", aspect="auto")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=12)
    ax.set_yticks(np.arange(len(datasets)))
    ax.set_yticklabels(datasets, fontsize=12)
    ax.set_title("Accuracy deltas defining the cross-residual advantage", fontsize=16, fontweight="bold")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:+.3f}", ha="center", va="center", fontsize=10)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Accuracy delta")
    fig.tight_layout()
    save(fig, "fig2_cross_advantage_heatmap")
    plt.close(fig)


def plot_rank_summary(summary: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    apply_paper_style()
    datasets = completed_datasets(summary)
    if not datasets:
        return
    rank_sum = {model: 0 for model in MODELS}
    win_count = {model: 0 for model in MODELS}
    for dataset in datasets:
        rows = sorted(
            (
                {"model": model, **summary[dataset][model]}
                for model in MODELS
            ),
            key=lambda row: (-row["mean_acc"], row["mean_loss"]),
        )
        for rank, row in enumerate(rows, 1):
            rank_sum[row["model"]] += rank
        win_count[rows[0]["model"]] += 1

    avg_rank = np.array([rank_sum[model] / len(datasets) for model in MODELS])
    wins = np.array([win_count[model] for model in MODELS])

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    axes[0].bar(
        np.arange(len(MODELS)),
        avg_rank,
        color=[ALL_COLORS[model] for model in MODELS],
        edgecolor="black",
        linewidth=0.8,
    )
    axes[0].set_xticks(np.arange(len(MODELS)))
    axes[0].set_xticklabels([ALL_DISPLAY[model] for model in MODELS], rotation=20, fontsize=12)
    axes[0].set_ylabel("Average rank", fontsize=14)
    axes[0].set_title("Average rank across datasets", fontsize=15, fontweight="bold")
    # Adaptive y-axis: invert so rank 1 is best
    axes[0].set_ylim(max(avg_rank) + 0.5, min(avg_rank) - 0.5)
    style_axis(axes[0])

    axes[1].bar(
        np.arange(len(MODELS)),
        wins,
        color=[ALL_COLORS[model] for model in MODELS],
        edgecolor="black",
        linewidth=0.8,
    )
    axes[1].set_xticks(np.arange(len(MODELS)))
    axes[1].set_xticklabels([ALL_DISPLAY[model] for model in MODELS], rotation=20, fontsize=12)
    axes[1].set_ylabel("Winner count", fontsize=14)
    axes[1].set_title("Number of dataset wins", fontsize=15, fontweight="bold")
    axes[1].set_ylim(0, max(wins) + 1.2)
    style_axis(axes[1])

    fig.tight_layout()
    save(fig, "fig3_rank_winner_summary")
    plt.close(fig)


def plot_topic_focus(summary: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    apply_paper_style()
    datasets = [dataset for dataset in TOPIC_DATASETS if dataset in completed_datasets(summary)]
    if not datasets:
        return
    fig, axes = plt.subplots(1, len(datasets), figsize=(11.2, 4.5), sharey=False)
    if len(datasets) == 1:
        axes = [axes]
    # Compute global y-range for consistent comparison
    all_means = [summary[ds][m]["mean_acc"] for ds in datasets for m in MODELS]
    global_y_max = max(all_means) + 0.12
    global_y_min = max(0, min(all_means) - 0.06)
    for ax, dataset in zip(axes, datasets):
        means = [summary[dataset][model]["mean_acc"] for model in MODELS]
        bars = ax.bar(
            np.arange(len(MODELS)),
            means,
            color=[ALL_COLORS[model] for model in MODELS],
            edgecolor="black",
            linewidth=0.8,
            zorder=3,
        )
        for bar in bars:
            value = bar.get_height()
            ax.annotate(
                f"{value:.3f}",
                xy=(bar.get_x() + bar.get_width() / 2, value),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=11,
                color="black",
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.2},
                clip_on=False,
                zorder=6,
            )
        ax.set_title(dataset, fontsize=15, fontweight="bold")
        ax.set_xticks(np.arange(len(MODELS)))
        ax.set_xticklabels([ALL_DISPLAY[model] for model in MODELS], rotation=30, ha="right", fontsize=11)
        ax.set_ylim(global_y_min, global_y_max)
        ax.margins(y=0.12)
        style_axis(ax)
    axes[0].set_ylabel("Mean 5-fold best test accuracy", fontsize=14)
    fig.subplots_adjust(left=0.08, right=0.995, top=0.88, bottom=0.35, wspace=0.12)
    save(fig, "fig4_topic_focus_results")
    plt.close(fig)


def plot_protein_package(summary: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    apply_paper_style()
    datasets = [dataset for dataset in TOPIC_DATASETS if dataset in completed_datasets(summary)]
    if not datasets:
        return
    fig, ax = plt.subplots(figsize=(10.8, 4.8))
    x = np.arange(len(datasets))
    all_acc_values = []
    for idx, model in enumerate(MODELS):
        accs = [summary[dataset][model]["mean_acc"] for dataset in datasets]
        all_acc_values.extend(accs)
        ax.plot(
            x,
            accs,
            color=ALL_COLORS[model],
            marker=ALL_MARKERS[model],
            linewidth=2.2,
            markersize=8,
            label=ALL_DISPLAY[model],
        )
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=13)
    ax.set_ylabel("Mean best test accuracy", fontsize=14)
    ax.set_title("Protein-oriented benchmark package", fontsize=16, fontweight="bold")
    # Adaptive y-axis
    margin = (max(all_acc_values) - min(all_acc_values)) * 0.18
    ax.set_ylim(max(0, min(all_acc_values) - margin), max(all_acc_values) + margin)
    style_axis(ax)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22), fontsize=12)
    fig.tight_layout()
    save(fig, "fig5_protein_package_summary")
    plt.close(fig)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate benchmark figures from versioned logs.")
    parser.add_argument("--version", default=DEFAULT_EXPERIMENT_VERSION)
    args = parser.parse_args()

    ensure_version_manifest(ROOT)
    version = normalize_version(args.version)
    active_log_dir = log_dir(ROOT, version)
    global OUTPUT_SUFFIX
    OUTPUT_SUFFIX = version
    ensure_dirs()
    rows = load_rows(active_log_dir)
    summary = summarize(rows)
    plot_full_suite(summary)
    plot_cross_heatmap(summary)
    plot_rank_summary(summary)
    plot_topic_focus(summary)
    plot_protein_package(summary)
    for name in [
        "fig1_full_suite_results.pdf",
        "fig2_cross_advantage_heatmap.pdf",
        "fig3_rank_winner_summary.pdf",
        "fig4_topic_focus_results.pdf",
        "fig5_protein_package_summary.pdf",
    ]:
        print((ROOT / "paper" / "figures" / "exp" / name).as_posix())


if __name__ == "__main__":
    main()
