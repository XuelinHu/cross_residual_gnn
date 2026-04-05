from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper" / "figures"


def add_box(ax, xy, width, height, title, body, facecolor, edgecolor="#1f2937") -> None:
    x, y = xy
    patch = FancyBboxPatch(
        (x, y),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.2,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height * 0.76, title, ha="center", va="center", fontsize=12, fontweight="bold")
    ax.text(x + width / 2, y + height * 0.40, body, ha="center", va="center", fontsize=10, linespacing=1.4)


def add_arrow(ax, start, end, text=None, color="#374151", rad=0.0) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=12,
        linewidth=1.3,
        color=color,
        connectionstyle=f"arc3,rad={rad}",
    )
    ax.add_patch(arrow)
    if text:
        mx = (start[0] + end[0]) / 2
        my = (start[1] + end[1]) / 2
        ax.text(mx, my + 0.03, text, ha="center", va="bottom", fontsize=9, color=color)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12.5, 6.8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5,
        0.96,
        "CR-GNN family overview: from plain propagation to residual and cross-residual reuse",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )

    add_box(
        ax,
        (0.05, 0.57),
        0.18,
        0.24,
        "PlainGNN",
        "Single branch\nstacked message passing\nno explicit reuse",
        "#fde68a",
    )
    add_box(
        ax,
        (0.29, 0.57),
        0.18,
        0.24,
        "NodeResGNN",
        "Single branch\nreuse previous node state\nwithin the branch",
        "#bfdbfe",
    )
    add_box(
        ax,
        (0.53, 0.57),
        0.18,
        0.24,
        "NodeCrossGNN",
        "Two node branches\nexchange cached states\nacross branches",
        "#c7d2fe",
    )
    add_box(
        ax,
        (0.17, 0.18),
        0.22,
        0.24,
        "GraphResGNN",
        "Sequential blocks\nreuse pooled graph summary\nacross stages",
        "#bbf7d0",
    )
    add_box(
        ax,
        (0.56, 0.18),
        0.22,
        0.24,
        "GraphCrossGNN",
        "Paired graph blocks\nswap graph summaries\nbetween stages",
        "#fecdd3",
    )

    add_arrow(ax, (0.23, 0.69), (0.29, 0.69), "add reuse")
    add_arrow(ax, (0.47, 0.69), (0.53, 0.69), "cross exchange")
    add_arrow(ax, (0.38, 0.57), (0.30, 0.42), "graph-level extension", rad=0.05)
    add_arrow(ax, (0.62, 0.57), (0.67, 0.42), "graph-level extension", rad=-0.05)

    add_arrow(ax, (0.60, 0.63), (0.64, 0.75), color="#4f46e5", rad=0.25)
    add_arrow(ax, (0.64, 0.75), (0.68, 0.63), color="#4f46e5", rad=0.25)
    ax.text(0.64, 0.79, "node-state exchange", ha="center", va="bottom", fontsize=9, color="#4f46e5")

    add_arrow(ax, (0.62, 0.30), (0.72, 0.30), color="#be123c", rad=0.22)
    add_arrow(ax, (0.72, 0.30), (0.62, 0.30), color="#be123c", rad=-0.22)
    ax.text(0.67, 0.36, "graph-summary swap", ha="center", va="bottom", fontsize=9, color="#be123c")

    ax.text(
        0.5,
        0.08,
        "Main paper comparison: PlainGNN, NodeResGNN, NodeCrossGNN, GraphResGNN, GraphCrossGNN",
        ha="center",
        va="center",
        fontsize=10,
    )

    fig.tight_layout()
    fig.savefig(OUT_DIR / "cr_gnn_family_overview.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "cr_gnn_family_overview.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(OUT_DIR / "cr_gnn_family_overview.pdf")


if __name__ == "__main__":
    main()
