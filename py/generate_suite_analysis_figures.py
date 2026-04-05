from __future__ import annotations

import glob
import json
import statistics as st
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
FIG_DIRS = [
    ROOT / "figures" / "exp",
    ROOT / "paper" / "figures" / "exp",
]

DATASETS = ["MUTAG", "PROTEINS", "DD", "ENZYMES", "MSRC_9", "AIDS", "Mutagenicity"]
TOPIC_DATASETS = ["PROTEINS", "DD", "ENZYMES"]
MODELS = ["PlainGNN", "NodeResGNN", "NodeCrossGNN", "GraphResGNN", "GraphCrossGNN"]

DISPLAY = {
    "PlainGNN": "Plain",
    "NodeResGNN": "NodeRes",
    "NodeCrossGNN": "NodeCross",
    "GraphResGNN": "GraphRes",
    "GraphCrossGNN": "GraphCross",
}
COLORS = {
    "PlainGNN": "#6b7280",
    "NodeResGNN": "#2563eb",
    "NodeCrossGNN": "#dc2626",
    "GraphResGNN": "#059669",
    "GraphCrossGNN": "#d97706",
}


def latest_matching_log(dataset: str, model: str, fold: int) -> Path:
    pattern = str(LOG_DIR / f"train_{dataset}_{model}_GCNConv_fold{fold}__*.json")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(pattern)
    return Path(matches[-1])


def load_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for dataset in DATASETS:
        for model in MODELS:
            for fold in range(5):
                payload = json.loads(latest_matching_log(dataset, model, fold).read_text(encoding="utf-8"))
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


def ensure_dirs() -> None:
    for path in FIG_DIRS:
        path.mkdir(parents=True, exist_ok=True)


def save(fig: plt.Figure, filename: str) -> None:
    for out_dir in FIG_DIRS:
        fig.savefig(out_dir / f"{filename}.pdf", dpi=300, bbox_inches="tight")
        fig.savefig(out_dir / f"{filename}.png", dpi=300, bbox_inches="tight")


def plot_full_suite(summary: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 5.8))
    x = np.arange(len(DATASETS))
    width = 0.15
    offsets = np.linspace(-2, 2, len(MODELS)) * width

    for offset, model in zip(offsets, MODELS):
        means = [summary[dataset][model]["mean_acc"] for dataset in DATASETS]
        stds = [summary[dataset][model]["std_acc"] for dataset in DATASETS]
        ax.bar(
            x + offset,
            means,
            width=width,
            color=COLORS[model],
            label=DISPLAY[model],
            edgecolor="black",
            linewidth=0.6,
            yerr=stds,
            capsize=3,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS, rotation=20)
    ax.set_ylabel("Mean 5-fold best test accuracy")
    ax.set_title("Full-suite benchmark across seven datasets")
    ax.grid(axis="y", linestyle="--", alpha=0.25)
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 1.18))
    fig.tight_layout()
    save(fig, "fig1_full_suite_results")
    plt.close(fig)


def plot_cross_heatmap(summary: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    matrix = []
    for dataset in DATASETS:
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
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(DATASETS)))
    ax.set_yticklabels(DATASETS)
    ax.set_title("Accuracy deltas that define the cross-residual advantage")
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:+.3f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9)
    cbar.set_label("Accuracy delta")
    fig.tight_layout()
    save(fig, "fig2_cross_advantage_heatmap")
    plt.close(fig)


def plot_rank_summary(summary: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    rank_sum = {model: 0 for model in MODELS}
    win_count = {model: 0 for model in MODELS}
    for dataset in DATASETS:
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

    avg_rank = np.array([rank_sum[model] / len(DATASETS) for model in MODELS])
    wins = np.array([win_count[model] for model in MODELS])

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    axes[0].bar(
        np.arange(len(MODELS)),
        avg_rank,
        color=[COLORS[model] for model in MODELS],
        edgecolor="black",
        linewidth=0.6,
    )
    axes[0].set_xticks(np.arange(len(MODELS)))
    axes[0].set_xticklabels([DISPLAY[model] for model in MODELS], rotation=20)
    axes[0].set_ylabel("Average rank")
    axes[0].set_title("Average rank across datasets")
    axes[0].grid(axis="y", linestyle="--", alpha=0.25)
    axes[0].invert_yaxis()

    axes[1].bar(
        np.arange(len(MODELS)),
        wins,
        color=[COLORS[model] for model in MODELS],
        edgecolor="black",
        linewidth=0.6,
    )
    axes[1].set_xticks(np.arange(len(MODELS)))
    axes[1].set_xticklabels([DISPLAY[model] for model in MODELS], rotation=20)
    axes[1].set_ylabel("Winner count")
    axes[1].set_title("Number of dataset wins")
    axes[1].grid(axis="y", linestyle="--", alpha=0.25)

    fig.tight_layout()
    save(fig, "fig3_rank_winner_summary")
    plt.close(fig)


def plot_topic_focus(summary: Dict[str, Dict[str, Dict[str, float]]]) -> None:
    fig, axes = plt.subplots(1, len(TOPIC_DATASETS), figsize=(11.5, 4.1), sharey=True)
    for ax, dataset in zip(axes, TOPIC_DATASETS):
        means = [summary[dataset][model]["mean_acc"] for model in MODELS]
        ax.bar(
            np.arange(len(MODELS)),
            means,
            color=[COLORS[model] for model in MODELS],
            edgecolor="black",
            linewidth=0.6,
        )
        ax.set_title(dataset)
        ax.set_xticks(np.arange(len(MODELS)))
        ax.set_xticklabels([DISPLAY[model] for model in MODELS], rotation=30, ha="right")
        ax.grid(axis="y", linestyle="--", alpha=0.25)
    axes[0].set_ylabel("Mean 5-fold best test accuracy")
    fig.suptitle("Topic-facing biological datasets", y=1.04)
    fig.tight_layout()
    save(fig, "fig4_topic_focus_results")
    plt.close(fig)


def main() -> None:
    ensure_dirs()
    rows = load_rows()
    summary = summarize(rows)
    plot_full_suite(summary)
    plot_cross_heatmap(summary)
    plot_rank_summary(summary)
    plot_topic_focus(summary)
    for name in [
        "fig1_full_suite_results.pdf",
        "fig2_cross_advantage_heatmap.pdf",
        "fig3_rank_winner_summary.pdf",
        "fig4_topic_focus_results.pdf",
    ]:
        print((ROOT / "paper" / "figures" / "exp" / name).as_posix())


if __name__ == "__main__":
    main()
