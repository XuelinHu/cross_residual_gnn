from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "paper" / "figures"

COLORS = {
    "plain": "#E69F00",
    "residual": "#0072B2",
    "cross": "#009E73",
    "graph": "#CC79A7",
    "muted": "#4D4D4D",
    "light": "#F7F7F7",
    "panel": "#D9D9D9",
    "edge": "#2F2F2F",
}


def configure_style() -> None:
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    rcParams["axes.titlesize"] = 12
    rcParams["axes.labelsize"] = 11
    rcParams["xtick.labelsize"] = 9
    rcParams["ytick.labelsize"] = 9
    rcParams["legend.fontsize"] = 9
    rcParams["figure.facecolor"] = "white"
    rcParams["axes.facecolor"] = "white"
    rcParams["savefig.dpi"] = 300


def add_round_box(
    ax,
    x: float,
    y: float,
    w: float,
    h: float,
    title: str,
    subtitle: str = "",
    facecolor: str = "white",
    edgecolor: str = COLORS["edge"],
    linewidth: float = 1.2,
    fontsize: int = 9,
    title_weight: str = "bold",
    zorder: int = 2,
) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=zorder,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h * 0.64,
        title,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=title_weight,
        color=COLORS["edge"],
        zorder=zorder + 1,
    )
    if subtitle:
        ax.text(
            x + w / 2,
            y + h * 0.32,
            subtitle,
            ha="center",
            va="center",
            fontsize=fontsize - 1,
            color=COLORS["muted"],
            zorder=zorder + 1,
            linespacing=1.3,
        )


def add_arrow(
    ax,
    start: tuple[float, float],
    end: tuple[float, float],
    color: str = COLORS["edge"],
    lw: float = 1.3,
    linestyle: str = "-",
    rad: float = 0.0,
    label: str | None = None,
    label_dx: float = 0.0,
    label_dy: float = 0.0,
    zorder: int = 3,
) -> None:
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=11,
        linewidth=lw,
        color=color,
        linestyle=linestyle,
        connectionstyle=f"arc3,rad={rad}",
        zorder=zorder,
    )
    ax.add_patch(arrow)
    if label:
        mx = (start[0] + end[0]) / 2 + label_dx
        my = (start[1] + end[1]) / 2 + label_dy
        ax.text(mx, my, label, fontsize=7.5, color=color, ha="center", va="center", zorder=zorder + 1)


def add_panel_frame(ax, x: float, y: float, w: float, h: float, title: str, accent: str) -> None:
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.01,rounding_size=0.02",
        linewidth=1.5,
        edgecolor=accent,
        facecolor="white",
        zorder=0,
    )
    ax.add_patch(patch)
    ax.text(
        x + w / 2,
        y + h - 0.07,
        title,
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        color=COLORS["edge"],
        zorder=1,
    )
    ax.plot([x + 0.015, x + w - 0.015], [y + h - 0.085, y + h - 0.085], color=accent, lw=1.0, zorder=1)


def add_pool_box(ax, x: float, y: float, w: float, h: float, label: str) -> None:
    add_round_box(
        ax,
        x,
        y,
        w,
        h,
        title=label,
        subtitle="Global mean pooling",
        facecolor="#F3F4F6",
        edgecolor="#6B7280",
        linewidth=1.1,
        fontsize=8,
    )


def add_legend(ax) -> None:
    handles = [
        Line2D([0], [0], color=COLORS["edge"], lw=1.4, label="Main forward path"),
        Line2D([0], [0], color=COLORS["residual"], lw=1.6, linestyle="--", label="Residual injection"),
        Line2D([0], [0], color=COLORS["cross"], lw=1.6, linestyle="--", label="Cross-residual injection"),
        Line2D([0], [0], color=COLORS["graph"], lw=1.6, linestyle=":", label="Graph-summary conditioning"),
    ]
    ax.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 0.02))


def draw_node_plain(ax, x0: float, y0: float, w: float, h: float) -> None:
    add_panel_frame(ax, x0, y0, w, h, "Basic Node-Level Stack", COLORS["plain"])
    boxes = [
        ("Input Node\nFeatures", x0 + 0.03),
        ("Input\nProjection", x0 + 0.18),
        ("Hidden Layer 1\nMessage Passing", x0 + 0.36),
        ("Hidden Layer 2\nMessage Passing", x0 + 0.54),
        ("Node\nEmbedding", x0 + 0.72),
    ]
    by = y0 + 0.22
    bw = 0.11
    bh = 0.18
    last_center = None
    for title, bx in boxes:
        add_round_box(ax, bx, by, bw, bh, title=title, facecolor="#FFF6E5", edgecolor=COLORS["plain"], fontsize=8)
        center = (bx + bw / 2, by + bh / 2)
        if last_center is not None:
            add_arrow(ax, (last_center[0] + bw / 2, center[1]), (center[0] - bw / 2, center[1]))
        last_center = center
    ax.text(x0 + w / 2, y0 + 0.12, "No explicit residual reuse", ha="center", fontsize=8, color=COLORS["muted"])


def draw_node_residual(ax, x0: float, y0: float, w: float, h: float) -> None:
    add_panel_frame(ax, x0, y0, w, h, "Residual Node-Level Stack", COLORS["residual"])
    positions = [x0 + 0.03, x0 + 0.18, x0 + 0.36, x0 + 0.54, x0 + 0.72]
    titles = [
        "Input Node\nFeatures",
        "Input\nProjection",
        "Hidden Layer 1\nMessage Passing",
        "Hidden Layer 2\nMessage Passing",
        "Node\nEmbedding",
    ]
    by = y0 + 0.22
    bw = 0.11
    bh = 0.18
    centers = []
    for bx, title in zip(positions, titles):
        add_round_box(ax, bx, by, bw, bh, title=title, facecolor="#EAF4FB", edgecolor=COLORS["residual"], fontsize=8)
        centers.append((bx + bw / 2, by + bh / 2))
    for idx in range(len(centers) - 1):
        add_arrow(ax, (centers[idx][0] + bw / 2, centers[idx][1]), (centers[idx + 1][0] - bw / 2, centers[idx + 1][1]))

    add_round_box(
        ax,
        x0 + 0.19,
        y0 + 0.51,
        0.14,
        0.10,
        title="Cached Previous\nNode State",
        subtitle="gate × residual",
        facecolor="#F4F8FC",
        edgecolor=COLORS["residual"],
        fontsize=8,
    )
    add_arrow(
        ax,
        (centers[1][0], by + bh + 0.01),
        (x0 + 0.26, y0 + 0.51),
        color=COLORS["residual"],
        linestyle="--",
        rad=0.22,
        label="cache",
        label_dy=0.02,
    )
    add_arrow(
        ax,
        (x0 + 0.33, y0 + 0.56),
        (centers[2][0], by + bh + 0.01),
        color=COLORS["residual"],
        linestyle="--",
        rad=-0.18,
        label="residual 1",
        label_dy=0.02,
    )
    add_arrow(
        ax,
        (centers[2][0], by + bh + 0.01),
        (x0 + 0.26, y0 + 0.51),
        color=COLORS["residual"],
        linestyle="--",
        rad=0.18,
        label="update cache",
        label_dy=0.02,
    )
    add_arrow(
        ax,
        (x0 + 0.33, y0 + 0.56),
        (centers[3][0], by + bh + 0.01),
        color=COLORS["residual"],
        linestyle="--",
        rad=-0.06,
        label="residual 2",
        label_dy=0.02,
    )
    ax.text(
        x0 + w / 2,
        y0 + 0.12,
        "Two-step same-branch residual reuse: previous hidden state is injected before each deeper layer",
        ha="center",
        fontsize=8,
        color=COLORS["muted"],
    )


def draw_node_cross(ax, x0: float, y0: float, w: float, h: float) -> None:
    add_panel_frame(ax, x0, y0, w, h, "Cross-Residual Node-Level Stack", COLORS["cross"])
    left_x = [x0 + 0.05, x0 + 0.20, x0 + 0.34, x0 + 0.48]
    right_x = [x0 + 0.56, x0 + 0.70, x0 + 0.84, x0 + 0.98]
    y = y0 + 0.22
    bw = 0.09
    bh = 0.16

    left_titles = ["Input\nProj A", "Hidden 1A", "Hidden 2A", "Embed A"]
    right_titles = ["Input\nProj B", "Hidden 1B", "Hidden 2B", "Embed B"]

    left_centers = []
    right_centers = []
    for bx, title in zip(left_x, left_titles):
        add_round_box(ax, bx, y, bw, bh, title=title, facecolor="#EAF8F3", edgecolor=COLORS["cross"], fontsize=8)
        left_centers.append((bx + bw / 2, y + bh / 2))
    for bx, title in zip(right_x, right_titles):
        add_round_box(ax, bx, y, bw, bh, title=title, facecolor="#EAF8F3", edgecolor=COLORS["cross"], fontsize=8)
        right_centers.append((bx + bw / 2, y + bh / 2))

    for centers in (left_centers, right_centers):
        for idx in range(len(centers) - 1):
            add_arrow(ax, (centers[idx][0] + bw / 2, centers[idx][1]), (centers[idx + 1][0] - bw / 2, centers[idx + 1][1]))

    add_round_box(
        ax,
        x0 + 0.40,
        y0 + 0.51,
        0.22,
        0.10,
        title="Cross Residual Cache",
        subtitle="prev A, prev B with gate × routing",
        facecolor="#F3FBF7",
        edgecolor=COLORS["cross"],
        fontsize=8,
    )

    add_arrow(ax, (left_centers[0][0], y + bh + 0.01), (x0 + 0.46, y0 + 0.51), color=COLORS["cross"], linestyle="--", rad=0.18, label="cache A", label_dy=0.018)
    add_arrow(ax, (right_centers[0][0], y + bh + 0.01), (x0 + 0.56, y0 + 0.51), color=COLORS["cross"], linestyle="--", rad=-0.18, label="cache B", label_dy=0.018)

    add_arrow(ax, (x0 + 0.50, y0 + 0.56), (left_centers[1][0], y + bh + 0.01), color=COLORS["cross"], linestyle="--", rad=-0.22, label="X1: prev B → H1A", label_dy=0.02)
    add_arrow(ax, (x0 + 0.52, y0 + 0.56), (right_centers[1][0], y + bh + 0.01), color=COLORS["cross"], linestyle="--", rad=0.22, label="X2: prev A → H1B", label_dy=0.02)
    add_arrow(ax, (x0 + 0.50, y0 + 0.56), (left_centers[2][0], y + bh + 0.01), color=COLORS["cross"], linestyle="--", rad=-0.08, label="X3: cache B → H2A", label_dy=0.02)
    add_arrow(ax, (x0 + 0.52, y0 + 0.56), (right_centers[2][0], y + bh + 0.01), color=COLORS["cross"], linestyle="--", rad=0.08, label="X4: cache A → H2B", label_dy=0.02)

    add_round_box(
        ax,
        x0 + 0.46,
        y0 + 0.10,
        0.14,
        0.10,
        title="Branch Sum",
        subtitle="Embed A + Embed B",
        facecolor="#F3FBF7",
        edgecolor=COLORS["cross"],
        fontsize=8,
    )
    add_arrow(ax, (left_centers[3][0], y), (x0 + 0.53, y0 + 0.20), color=COLORS["edge"], rad=-0.10)
    add_arrow(ax, (right_centers[3][0], y), (x0 + 0.53, y0 + 0.20), color=COLORS["edge"], rad=0.10)

    ax.text(
        x0 + w / 2,
        y0 + 0.05,
        "Four explicit cross-residual interaction sites are shown across two branches and two hidden layers",
        ha="center",
        fontsize=8,
        color=COLORS["muted"],
    )


def draw_graph_plain(ax, x0: float, y0: float, w: float, h: float) -> None:
    add_panel_frame(ax, x0, y0, w, h, "Basic Graph-Level Stack", COLORS["plain"])
    bx = [x0 + 0.05, x0 + 0.28, x0 + 0.51, x0 + 0.74]
    titles = [
        "Input Graph\n(Node Features + Topology)",
        "Hidden Layer 1\nMessage Passing",
        "Hidden Layer 2\nMessage Passing",
        "Graph\nEmbedding",
    ]
    y = y0 + 0.28
    bw = 0.16
    bh = 0.17
    centers = []
    for x, title in zip(bx, titles):
        add_round_box(ax, x, y, bw, bh, title=title, facecolor="#FFF6E5", edgecolor=COLORS["plain"], fontsize=8)
        centers.append((x + bw / 2, y + bh / 2))
    for idx in range(len(centers) - 1):
        add_arrow(ax, (centers[idx][0] + bw / 2, centers[idx][1]), (centers[idx + 1][0] - bw / 2, centers[idx + 1][1]))
    add_pool_box(ax, x0 + 0.57, y0 + 0.14, 0.18, 0.10, "Readout")
    add_arrow(ax, (centers[2][0], y), (x0 + 0.66, y0 + 0.24), color=COLORS["edge"])
    add_arrow(ax, (x0 + 0.75, y0 + 0.19), (centers[3][0], y), color=COLORS["edge"])


def draw_graph_residual(ax, x0: float, y0: float, w: float, h: float) -> None:
    add_panel_frame(ax, x0, y0, w, h, "Residual Graph-Level Stack", COLORS["graph"])
    block_x = [x0 + 0.05, x0 + 0.36, x0 + 0.67]
    centers = []
    for idx, bx in enumerate(block_x, start=1):
        add_round_box(
            ax,
            bx,
            y0 + 0.24,
            0.22,
            0.24,
            title=f"Graph Block {idx}",
            subtitle="Input proj\nHidden 1 + Hidden 2\nPooling + classifier",
            facecolor="#FBEFF7",
            edgecolor=COLORS["graph"],
            fontsize=8,
        )
        add_pool_box(ax, bx + 0.03, y0 + 0.12, 0.16, 0.08, f"Summary g{idx}")
        add_arrow(ax, (bx + 0.11, y0 + 0.24), (bx + 0.11, y0 + 0.20), color=COLORS["edge"])
        centers.append((bx + 0.11, y0 + 0.36))
    add_arrow(ax, (block_x[0] + 0.22, y0 + 0.36), (block_x[1], y0 + 0.36))
    add_arrow(ax, (block_x[1] + 0.22, y0 + 0.36), (block_x[2], y0 + 0.36))

    add_arrow(
        ax,
        (block_x[0] + 0.11, y0 + 0.12),
        (block_x[1] + 0.11, y0 + 0.49),
        color=COLORS["graph"],
        linestyle="--",
        rad=0.18,
        label="R1: g1 → block 2",
        label_dy=0.02,
    )
    add_arrow(
        ax,
        (block_x[1] + 0.11, y0 + 0.12),
        (block_x[2] + 0.11, y0 + 0.49),
        color=COLORS["graph"],
        linestyle="--",
        rad=0.18,
        label="R2: g2 → block 3",
        label_dy=0.02,
    )
    ax.text(
        x0 + w / 2,
        y0 + 0.05,
        "Each later block reuses the previous graph summary and injects it into deeper node updates before pooling",
        ha="center",
        fontsize=8,
        color=COLORS["muted"],
    )


def draw_graph_cross(ax, x0: float, y0: float, w: float, h: float) -> None:
    add_panel_frame(ax, x0, y0, w, h, "Cross-Residual Graph-Level Stack", COLORS["cross"])

    coords = {
        "b1": (x0 + 0.06, y0 + 0.30),
        "b2": (x0 + 0.33, y0 + 0.30),
        "b3": (x0 + 0.60, y0 + 0.30),
        "b4": (x0 + 0.87, y0 + 0.30),
    }
    for name, (bx, by) in coords.items():
        stage = "Stage 1" if name in {"b1", "b2"} else "Stage 2"
        add_round_box(
            ax,
            bx,
            by,
            0.17,
            0.21,
            title=name.upper().replace("B", "Block "),
            subtitle=f"{stage}\n2 hidden layers + pooling",
            facecolor="#EAF8F3",
            edgecolor=COLORS["cross"],
            fontsize=8,
        )
        add_pool_box(ax, bx + 0.025, y0 + 0.16, 0.12, 0.08, f"g{name[-1]}")
        add_arrow(ax, (bx + 0.085, by), (bx + 0.085, y0 + 0.24), color=COLORS["edge"])

    add_arrow(ax, (coords["b1"][0] + 0.17, y0 + 0.40), (coords["b3"][0], y0 + 0.40))
    add_arrow(ax, (coords["b2"][0] + 0.17, y0 + 0.40), (coords["b4"][0], y0 + 0.40))

    add_arrow(
        ax,
        (coords["b2"][0] + 0.08, y0 + 0.16),
        (coords["b3"][0] + 0.08, y0 + 0.52),
        color=COLORS["cross"],
        linestyle="--",
        rad=0.18,
        label="X1: g2 → block 3",
        label_dy=0.025,
    )
    add_arrow(
        ax,
        (coords["b1"][0] + 0.08, y0 + 0.16),
        (coords["b4"][0] + 0.08, y0 + 0.52),
        color=COLORS["cross"],
        linestyle="--",
        rad=0.28,
        label="X2: g1 → block 4",
        label_dy=0.04,
    )
    add_arrow(
        ax,
        (coords["b3"][0] + 0.08, y0 + 0.52),
        (coords["b3"][0] + 0.04, y0 + 0.44),
        color=COLORS["graph"],
        linestyle=":",
        label="X3: broadcast into block 3 hidden layers",
        label_dx=0.10,
        label_dy=0.03,
    )
    add_arrow(
        ax,
        (coords["b4"][0] + 0.08, y0 + 0.52),
        (coords["b4"][0] + 0.04, y0 + 0.44),
        color=COLORS["graph"],
        linestyle=":",
        label="X4: broadcast into block 4 hidden layers",
        label_dx=0.08,
        label_dy=0.03,
    )

    add_round_box(
        ax,
        x0 + 0.48,
        y0 + 0.05,
        0.16,
        0.09,
        title="Final Fusion",
        subtitle="g3 + g4",
        facecolor="#F3FBF7",
        edgecolor=COLORS["cross"],
        fontsize=8,
    )
    add_arrow(ax, (coords["b3"][0] + 0.085, y0 + 0.16), (x0 + 0.56, y0 + 0.14), color=COLORS["edge"], rad=-0.08)
    add_arrow(ax, (coords["b4"][0] + 0.085, y0 + 0.16), (x0 + 0.56, y0 + 0.14), color=COLORS["edge"], rad=0.08)

    ax.text(
        x0 + w / 2,
        y0 + 0.01,
        "Four graph-level cross-residual interactions are made explicit: two summary swaps and two hidden-layer conditioning routes",
        ha="center",
        fontsize=8,
        color=COLORS["muted"],
    )


def build_node_figure() -> None:
    fig, ax = plt.subplots(figsize=(18, 5.8))
    ax.set_xlim(0, 3.08)
    ax.set_ylim(0, 1.0)
    ax.axis("off")

    ax.text(
        1.54,
        0.95,
        "Node-Level CR-GNN Architecture",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        1.54,
        0.91,
        "Unified abstract view of plain stacking, same-branch residual reuse, and cross-residual node interaction",
        ha="center",
        va="center",
        fontsize=9,
        color=COLORS["muted"],
    )

    draw_node_plain(ax, 0.03, 0.18, 0.95, 0.66)
    draw_node_residual(ax, 1.06, 0.18, 0.95, 0.66)
    draw_node_cross(ax, 2.09, 0.18, 0.96, 0.66)
    add_legend(ax)

    fig.tight_layout()
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT_DIR / f"cr_gnn_node_architecture.{ext}", bbox_inches="tight")
    plt.close(fig)


def build_graph_figure() -> None:
    fig, ax = plt.subplots(figsize=(18, 6.0))
    ax.set_xlim(0, 3.35)
    ax.set_ylim(0, 1.0)
    ax.axis("off")

    ax.text(
        1.675,
        0.95,
        "Graph-Level CR-GNN Architecture",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        1.675,
        0.91,
        "Unified abstract view of graph-level feature extraction, graph-summary residual propagation, and cross-summary exchange",
        ha="center",
        va="center",
        fontsize=9,
        color=COLORS["muted"],
    )

    draw_graph_plain(ax, 0.03, 0.18, 0.98, 0.68)
    draw_graph_residual(ax, 1.13, 0.18, 0.98, 0.68)
    draw_graph_cross(ax, 2.23, 0.18, 1.09, 0.68)
    add_legend(ax)

    fig.tight_layout()
    for ext in ("pdf", "svg", "png"):
        fig.savefig(OUT_DIR / f"cr_gnn_graph_architecture.{ext}", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    configure_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    build_node_figure()
    build_graph_figure()
    print(OUT_DIR / "cr_gnn_node_architecture.pdf")
    print(OUT_DIR / "cr_gnn_graph_architecture.pdf")


if __name__ == "__main__":
    main()
