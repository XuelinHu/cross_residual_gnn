from __future__ import annotations

import glob
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from py.plot_style import MODEL_COLORS, MODEL_MARKERS, apply_paper_style, style_axis

LOG_DIR = ROOT / "logs"
MD_DIR = ROOT / "md"
FIG_DIR = ROOT / "paper" / "figures" / "exp"

DATASETS = ["PROTEINS", "DD", "ENZYMES"]
MODELS = ["NodeCrossGNN", "GraphCrossGNN"]
SWEEPS = ["h_layer", "drop", "lr"]

BASE_CONFIGS: Dict[Tuple[str, str], Dict[str, float]] = {
    ("PROTEINS", "NodeCrossGNN"): {"lr": 0.003, "drop": 0.2, "h_layer": 4},
    ("PROTEINS", "GraphCrossGNN"): {"lr": 0.003, "drop": 0.3, "h_layer": 4},
    ("DD", "NodeCrossGNN"): {"lr": 0.003, "drop": 0.3, "h_layer": 3},
    ("DD", "GraphCrossGNN"): {"lr": 0.002, "drop": 0.2, "h_layer": 4},
    ("ENZYMES", "NodeCrossGNN"): {"lr": 0.003, "drop": 0.3, "h_layer": 4},
    ("ENZYMES", "GraphCrossGNN"): {"lr": 0.003, "drop": 0.3, "h_layer": 4},
}

DISPLAY = {"NodeCrossGNN": "NodeCross", "GraphCrossGNN": "GraphCross"}


def load_rows() -> List[Dict[str, object]]:
    rows = []
    for path in glob.glob(str(LOG_DIR / "train_*_sensitivity_*__*.json")):
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        cfg = payload["config"]
        rows.append(
            {
                "path": path,
                "dataset": cfg["ds"],
                "model": cfg["gname"],
                "fold": cfg["fold"],
                "lr": float(cfg["lr"]),
                "drop": float(cfg["drop"]),
                "h_layer": int(cfg["h_layer"]),
                "best_test_acc": float(payload["best_test_acc"]),
                "best_epoch": int(payload["best_epoch"]) + 1,
                "test_loss": float(payload["test_loss"]),
            }
        )
    return rows


def build_frame(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    if not rows:
        raise RuntimeError("No sensitivity-tagged logs found.")
    rows = sorted(rows, key=lambda row: row["path"])
    dedup: Dict[Tuple[object, ...], Dict[str, object]] = {}
    for row in rows:
        key = (
            row["dataset"],
            row["model"],
            row["fold"],
            row["lr"],
            row["drop"],
            row["h_layer"],
        )
        dedup[key] = row
    return list(dedup.values())


def sensitivity_rows(rows: List[Dict[str, object]], dataset: str, model: str, sweep: str) -> List[Dict[str, object]]:
    base = BASE_CONFIGS[(dataset, model)]
    subset = [
        row for row in rows
        if row["dataset"] == dataset and row["model"] == model and row["fold"] == 0
    ]
    for key, value in base.items():
        if key != sweep:
            subset = [row for row in subset if row[key] == value]
    return sorted(subset, key=lambda row: row[sweep])


def build_markdown(rows: List[Dict[str, object]]) -> str:
    lines = ["# Sensitivity Summary", ""]
    for dataset in DATASETS:
        lines.append(f"## {dataset}")
        for model in MODELS:
            lines.append(f"### {model}")
            for sweep in SWEEPS:
                subset = sensitivity_rows(rows, dataset, model, sweep)
                if not subset:
                    continue
                best = max(subset, key=lambda row: row["best_test_acc"])
                values = ", ".join(
                    f"{row[sweep]} -> {row['best_test_acc']:.5f}"
                    for row in subset
                )
                lines.append(
                    f"- `{sweep}`: best `{best[sweep]}` with acc `{best['best_test_acc']:.5f}`; values: {values}"
                )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def plot_sweep(rows: List[Dict[str, object]], sweep: str, filename: str, ylabel: str = "Fold-0 Best Test Accuracy") -> None:
    apply_paper_style()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(DATASETS), 1, figsize=(8.2, 10.5), sharex=False)
    if len(DATASETS) == 1:
        axes = [axes]
    for ax, dataset in zip(axes, DATASETS):
        for model in MODELS:
            subset = sensitivity_rows(rows, dataset, model, sweep)
            if not subset:
                continue
            ax.plot(
                [row[sweep] for row in subset],
                [row["best_test_acc"] for row in subset],
                marker=MODEL_MARKERS[model],
                linewidth=1.9,
                markersize=5.2,
                color=MODEL_COLORS[model],
                label=DISPLAY[model],
            )
        ax.set_title(dataset)
        ax.set_ylabel(ylabel)
        style_axis(ax)
        ax.legend(frameon=False, ncol=2)
    axes[-1].set_xlabel(sweep)
    fig.tight_layout()
    fig.savefig(FIG_DIR / f"{filename}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(FIG_DIR / f"{filename}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    rows = load_rows()
    rows = build_frame(rows)
    MD_DIR.mkdir(parents=True, exist_ok=True)
    (MD_DIR / "sensitivity_summary.md").write_text(build_markdown(rows), encoding="utf-8")
    plot_sweep(rows, "h_layer", "fig5_depth_sensitivity_v3")
    plot_sweep(rows, "drop", "fig6_dropout_sensitivity_v3")
    plot_sweep(rows, "lr", "fig7_lr_sensitivity_v3")
    print(MD_DIR / "sensitivity_summary.md")
    print(FIG_DIR / "fig5_depth_sensitivity_v3.pdf")
    print(FIG_DIR / "fig6_dropout_sensitivity_v3.pdf")
    print(FIG_DIR / "fig7_lr_sensitivity_v3.pdf")


if __name__ == "__main__":
    main()
