"""
Generate figures and LaTeX tables for the paper's experiments section.

This script consolidates the benchmark summary files, validates the
associated record exports, and writes the figures/tables used in the
paper manuscript.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams


ROOT = Path(__file__).resolve().parents[1]
RECORDS_DIR = ROOT / "records"

FIGURE_DIRS = [
    ROOT / "figures" / "exp",
    ROOT / "paper" / "figures" / "exp",
    ROOT / "py" / "figures" / "exp",
]
TABLE_DIRS = [
    ROOT / "md",
    ROOT / "py" / "md",
]

MODEL_ORDER = [
    "BlockGNN",
    "ResBlockGnn",
    "CrossBlockGnn",
    "GraphBlockGnn",
    "ResGraphBlockGnn",
    "CrossGraphBlockGnn",
]
DATASET_ORDER = ["MUTAG", "DD", "MSRC_9", "AIDS"]
DEPTH_ORDER = [1, 2, 3, 4, 5]
DIM_ORDER = [32, 64]

PALETTE = {
    "BlockGNN": "#B03A2E",
    "ResBlockGnn": "#1F618D",
    "CrossBlockGnn": "#2874A6",
    "GraphBlockGnn": "#117864",
    "ResGraphBlockGnn": "#148F77",
    "CrossGraphBlockGnn": "#7D3C98",
}

DISPLAY_NAMES = {
    "BlockGNN": "PlainGNN",
    "ResBlockGnn": "NodeResGNN",
    "CrossBlockGnn": "NodeCrossGNN",
    "GraphBlockGnn": "GraphCondGNN",
    "ResGraphBlockGnn": "GraphResGNN",
    "CrossGraphBlockGnn": "GraphCrossGNN",
}


rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
rcParams["font.size"] = 10
rcParams["figure.dpi"] = 300
rcParams["savefig.dpi"] = 300
rcParams["savefig.bbox"] = "tight"


def ensure_output_dirs() -> None:
    for path in [*FIGURE_DIRS, *TABLE_DIRS]:
        path.mkdir(parents=True, exist_ok=True)


def load_merged_results() -> pd.DataFrame:
    v3 = pd.read_excel(RECORDS_DIR / "v3result.xlsx")
    v4 = pd.read_excel(RECORDS_DIR / "v4result.xlsx")

    base = v3[~((v3["ds"] == "MUTAG") & (v3["model"] == "GCNConv"))].copy()
    merged = pd.concat([base, v4], ignore_index=True)

    merged["fold_range"] = merged[[f"acc{i}" for i in range(5)]].max(axis=1) - merged[
        [f"acc{i}" for i in range(5)]
    ].min(axis=1)
    return merged


def validate_record_exports() -> None:
    checks = {
        "v3result.xlsx": sorted(RECORDS_DIR.glob("graph_classify_v3_*")),
        "v4result.xlsx": sorted(RECORDS_DIR.glob("graph_classify_v4_*")),
    }
    for excel_name, files in checks.items():
        expected = len(pd.read_excel(RECORDS_DIR / excel_name))
        observed = 0
        for path in files:
            with path.open("r", encoding="utf-8") as handle:
                observed += sum(1 for line in handle if line.strip())
        if observed != expected:
            raise ValueError(
                f"Record export mismatch for {excel_name}: "
                f"expected {expected} rows, found {observed} raw lines."
            )


def save_figure(fig: plt.Figure, filename: str) -> None:
    for out_dir in FIGURE_DIRS:
        fig.savefig(out_dir / f"{filename}.pdf")
        fig.savefig(out_dir / f"{filename}.png")


def save_tables(content: str) -> None:
    for out_dir in TABLE_DIRS:
        (out_dir / "exp_tables.tex").write_text(content, encoding="utf-8")


def model_label(model: str) -> str:
    return DISPLAY_NAMES[model]


def build_main_results_table(df: pd.DataFrame) -> str:
    rows = []
    for model in MODEL_ORDER:
        row = []
        for dataset in DATASET_ORDER:
            subset = df[(df["gm"] == model) & (df["ds"] == dataset)]
            best = subset.loc[subset["acc"].idxmax()]
            cell = f"{best['acc']:.3f} $\\pm$ {best['acc_std_dev']:.4f}"
            if abs(best["acc"] - df[df["ds"] == dataset]["acc"].max()) < 1e-9:
                cell = f"\\textbf{{{cell}}}"
            row.append(cell)
        rows.append(f"{model_label(model)} & {' & '.join(row)} \\\\")

    return "\n".join(
        [
            "% Table 1: Main Results",
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\setlength{\tabcolsep}{4pt}",
            r"\caption{Best 5-fold graph classification accuracy (mean $\pm$ std) for each architecture on each dataset under the final benchmark setting.}",
            r"\label{tab:main_results}",
            r"\begin{tabular}{lcccc}",
            r"\toprule",
            r"Model & MUTAG & DD & MSRC\_9 & AIDS \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )


def build_mutag_subset_table(df: pd.DataFrame) -> str:
    subset = df[(df["ds"] == "MUTAG") & (df["model"] == "GCNConv")]
    rows = []
    for model in MODEL_ORDER:
        model_df = subset[subset["gm"] == model]
        best = model_df.loc[model_df["acc"].idxmax()]
        rows.append(
            f"{model_label(model)} & {int(best['dim'])} & {int(best['h'])} & "
            f"{best['acc']:.3f} & {best['acc_std_dev']:.4f} & {best['execution_time']:.1f} \\\\"
        )

    return "\n".join(
        [
            "% Table 2: MUTAG GCN subset summary",
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\caption{Best configuration of each architecture on the refreshed MUTAG + GCNConv subset.}",
            r"\label{tab:ablation}",
            r"\begin{tabular}{lccccc}",
            r"\toprule",
            r"Model & Dim & Depth & Acc. & Std. & Time (s) \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )


def build_efficiency_table(df: pd.DataFrame) -> str:
    subset = df[
        (df["ds"] == "MUTAG")
        & (df["model"] == "GCNConv")
        & (df["dim"] == 32)
        & (df["h"] == 2)
    ].copy()
    baseline = float(subset[subset["gm"] == "BlockGNN"]["execution_time"].iloc[0])

    rows = []
    for model in MODEL_ORDER:
        row = subset[subset["gm"] == model].iloc[0]
        overhead = (float(row["execution_time"]) / baseline - 1.0) * 100.0
        rows.append(
            f"{model_label(model)} & {row['execution_time']:.1f} & {overhead:+.1f}\\% & {row['acc']:.3f} \\\\"
        )

    return "\n".join(
        [
            "% Table 3: Efficiency comparison",
            r"\begin{table}[t]",
            r"\centering",
            r"\small",
            r"\caption{Efficiency comparison on MUTAG with GCNConv, dim=32, and depth=2. Relative overhead is measured against PlainGNN.}",
            r"\label{tab:efficiency}",
            r"\begin{tabular}{lccc}",
            r"\toprule",
            r"Model & Time (s) & Relative Overhead & Accuracy \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )


def generate_tables(df: pd.DataFrame) -> None:
    tables = "\n\n".join(
        [
            build_main_results_table(df),
            build_mutag_subset_table(df),
            build_efficiency_table(df),
        ]
    )
    save_tables(tables)


def generate_average_performance_figure(df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(8.2, 5.2))

    mean_acc = df.groupby("gm")["acc"].mean().reindex(MODEL_ORDER)
    std_acc = df.groupby("gm")["acc"].std().reindex(MODEL_ORDER)

    bars = ax.bar(
        range(len(MODEL_ORDER)),
        mean_acc,
        yerr=std_acc,
        capsize=5,
        alpha=0.9,
        edgecolor="black",
        linewidth=1.0,
        color=[PALETTE[model] for model in MODEL_ORDER],
    )

    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels(
        [f"{model_label(model)}\n(n={len(df[df['gm'] == model])})" for model in MODEL_ORDER],
        fontsize=9,
    )
    ax.set_ylabel("Mean accuracy", fontweight="bold")
    ax.set_xlabel("Architecture", fontweight="bold")
    ax.set_ylim(0.65, 0.82)
    ax.grid(axis="y", alpha=0.3, linestyle="--")
    ax.set_title("Average Performance Across All 720 Configurations", fontweight="bold")

    for bar, mean, std in zip(bars, mean_acc, std_acc):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + std + 0.006,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    fig.tight_layout()
    save_figure(fig, "fig1_average_performance")
    plt.close(fig)


def generate_depth_sensitivity_figure(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

    markers = {
        "BlockGNN": "o",
        "ResBlockGnn": "s",
        "CrossBlockGnn": "^",
        "GraphBlockGnn": "D",
        "ResGraphBlockGnn": "v",
        "CrossGraphBlockGnn": "P",
    }

    for ax, dim in zip(axes, DIM_ORDER):
        subset = df[(df["ds"] == "MUTAG") & (df["model"] == "GCNConv") & (df["dim"] == dim)]
        pivot = subset.pivot(index="gm", columns="h", values="acc").reindex(MODEL_ORDER)

        for model in MODEL_ORDER:
            ax.plot(
                DEPTH_ORDER,
                pivot.loc[model, DEPTH_ORDER].values,
                marker=markers[model],
                color=PALETTE[model],
                linewidth=2,
                markersize=6,
                label=model_label(model),
            )

        ax.set_xticks(DEPTH_ORDER)
        ax.set_xlabel(f"Depth (dim={dim})", fontweight="bold")
        ax.grid(alpha=0.25, linestyle="--")
        ax.set_title(f"Refreshed MUTAG + GCNConv Subset (dim={dim})", fontweight="bold")

    axes[0].set_ylabel("Accuracy", fontweight="bold")
    axes[0].set_ylim(0.55, 0.87)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, "fig2_depth_sensitivity")
    plt.close(fig)


def generate_fold_distribution_figure(df: pd.DataFrame) -> None:
    subset = df[
        (df["ds"] == "MUTAG")
        & (df["model"] == "GCNConv")
        & (df["dim"] == 64)
        & (df["h"] == 2)
    ].copy()
    subset = subset.set_index("gm").reindex(MODEL_ORDER).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5.8))
    box_data = [[row[f"acc{i}"] for i in range(5)] for _, row in subset.iterrows()]
    model_keys = subset["gm"].tolist()
    labels = [model_label(model) for model in model_keys]

    bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True, showmeans=True, meanline=True)
    for patch, model in zip(bp["boxes"], model_keys):
        patch.set_facecolor(PALETTE[model])
        patch.set_alpha(0.75)

    ax.set_ylabel("Fold accuracy", fontweight="bold")
    ax.set_xlabel("Architecture", fontweight="bold")
    ax.set_ylim(0.5, 0.95)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_title("Five-Fold Distribution on MUTAG + GCNConv (dim=64, depth=2)", fontweight="bold")
    plt.xticks(rotation=15, ha="right")

    fig.tight_layout()
    save_figure(fig, "fig3_boxplot_distribution")
    plt.close(fig)


def generate_heatmap_figure(df: pd.DataFrame) -> None:
    heatmap_rows = []
    for model in MODEL_ORDER:
        row = []
        for dataset in DATASET_ORDER:
            subset = df[(df["gm"] == model) & (df["ds"] == dataset)]
            row.append(float(subset["acc"].max()))
        heatmap_rows.append(row)

    heatmap = pd.DataFrame(
        heatmap_rows,
        index=[model_label(model) for model in MODEL_ORDER],
        columns=DATASET_ORDER,
    )

    fig, ax = plt.subplots(figsize=(8.0, 5.8))
    im = ax.imshow(heatmap.values, cmap="YlGnBu", aspect="auto", vmin=0.58, vmax=1.0)

    ax.set_xticks(np.arange(len(DATASET_ORDER)))
    ax.set_yticks(np.arange(len(MODEL_ORDER)))
    ax.set_xticklabels(DATASET_ORDER)
    ax.set_yticklabels([model_label(model) for model in MODEL_ORDER])
    ax.set_xlabel("Dataset", fontweight="bold")
    ax.set_ylabel("Architecture", fontweight="bold")
    ax.set_title("Best Accuracy Heatmap Across Datasets", fontweight="bold")

    for i in range(len(MODEL_ORDER)):
        for j in range(len(DATASET_ORDER)):
            ax.text(
                j,
                i,
                f"{heatmap.values[i, j]:.3f}",
                ha="center",
                va="center",
                fontsize=8.5,
                fontweight="bold",
                color="black",
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Best accuracy", fontweight="bold")
    fig.tight_layout()
    save_figure(fig, "fig4_heatmap_performance")
    plt.close(fig)


def main() -> None:
    ensure_output_dirs()
    validate_record_exports()
    merged = load_merged_results()

    generate_tables(merged)
    generate_average_performance_figure(merged)
    generate_depth_sensitivity_figure(merged)
    generate_fold_distribution_figure(merged)
    generate_heatmap_figure(merged)

    print("Generated experiment figures and tables successfully.")


if __name__ == "__main__":
    main()
