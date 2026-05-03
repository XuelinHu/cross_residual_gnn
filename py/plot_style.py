from __future__ import annotations

from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
from matplotlib import rcParams


ROOT = Path(__file__).resolve().parents[1]

MODEL_COLORS: Dict[str, str] = {
    "PlainGNN": "#4D4D4D",
    "NodeResGNN": "#0072B2",
    "NodeCrossGNN": "#D55E00",
    "GraphResGNN": "#009E73",
    "GraphCrossGNN": "#CC79A7",
}

MODEL_MARKERS: Dict[str, str] = {
    "PlainGNN": "o",
    "NodeResGNN": "s",
    "NodeCrossGNN": "^",
    "GraphResGNN": "D",
    "GraphCrossGNN": "P",
}

MODEL_LINESTYLES: Dict[str, str] = {
    "PlainGNN": "-",
    "NodeResGNN": "--",
    "NodeCrossGNN": "-.",
    "GraphResGNN": ":",
    "GraphCrossGNN": (0, (5, 1.5)),
}


def apply_paper_style() -> None:
    rcParams["font.family"] = "serif"
    rcParams["font.serif"] = ["Times New Roman", "Times", "DejaVu Serif"]
    rcParams["font.size"] = 13
    rcParams["axes.titlesize"] = 16
    rcParams["axes.labelsize"] = 14
    rcParams["xtick.labelsize"] = 12
    rcParams["ytick.labelsize"] = 12
    rcParams["legend.fontsize"] = 12
    rcParams["figure.titlesize"] = 17
    rcParams["figure.dpi"] = 300
    rcParams["savefig.dpi"] = 300
    rcParams["savefig.facecolor"] = "white"
    rcParams["figure.facecolor"] = "white"
    rcParams["axes.facecolor"] = "white"
    rcParams["axes.edgecolor"] = "black"
    rcParams["axes.linewidth"] = 0.9
    rcParams["grid.color"] = "#D0D0D0"
    rcParams["grid.linewidth"] = 0.6
    rcParams["grid.alpha"] = 0.5
    rcParams["grid.linestyle"] = "--"
    rcParams["legend.frameon"] = False
    rcParams["savefig.bbox"] = "tight"


def style_axis(ax: plt.Axes, with_grid: bool = True) -> None:
    if with_grid:
        ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)

