from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"

FOCUSED_MODELS = ["PlainGNN", "NodeResGNN", "NodeCrossGNN", "GraphResGNN", "GraphCrossGNN"]
MAIN_DATASETS = ["MUTAG", "PROTEINS", "DD", "MSRC_9"]
TOPIC_DATASETS = ["PROTEINS", "DD", "ENZYMES"]
EXTENDED_DATASETS = ["AIDS", "Mutagenicity"]


def latest_matching_log(dataset: str, model: str, fold: int) -> Path:
    pattern = str(LOG_DIR / f"train_{dataset}_{model}_GCNConv_fold{fold}__*.json")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(pattern)
    return Path(matches[-1])


def load_result(dataset: str, model: str, fold: int) -> Dict[str, object]:
    path = latest_matching_log(dataset, model, fold)
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_model(dataset: str, model: str) -> Dict[str, object]:
    rows = [load_result(dataset, model, fold) for fold in range(5)]
    accs = [float(row["best_test_acc"]) for row in rows]
    losses = [float(row["test_loss"]) for row in rows]
    epochs = [int(row["best_epoch"]) + 1 for row in rows]
    return {
        "dataset": dataset,
        "model": model,
        "mean_acc": st.mean(accs),
        "std_acc": st.pstdev(accs),
        "mean_loss": st.mean(losses),
        "std_loss": st.pstdev(losses),
        "mean_best_epoch": st.mean(epochs),
        "fold_accs": accs,
    }


def format_table(datasets: Iterable[str], models: Iterable[str]) -> str:
    lines: List[str] = []
    for dataset in datasets:
        lines.append(f"## {dataset}")
        summaries = [summarize_model(dataset, model) for model in models]
        summaries.sort(key=lambda row: (-row["mean_acc"], row["mean_loss"]))
        for index, row in enumerate(summaries, 1):
            fold_text = ",".join(f"{value:.5f}" for value in row["fold_accs"])
            lines.append(
                f"{index}. {row['model']}\tmean_acc={row['mean_acc']:.5f}\t"
                f"std_acc={row['std_acc']:.5f}\tmean_loss={row['mean_loss']:.5f}\t"
                f"mean_best_epoch={row['mean_best_epoch']:.1f}"
            )
            lines.append(f"   folds={fold_text}")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize paper-ready V3 experiments from logs.")
    parser.add_argument(
        "--dataset_group",
        choices=["main", "topic", "extended", "all"],
        default="main",
    )
    parser.add_argument("--models", nargs="+", default=FOCUSED_MODELS)
    args = parser.parse_args()

    if args.dataset_group == "main":
        datasets = MAIN_DATASETS
    elif args.dataset_group == "topic":
        datasets = TOPIC_DATASETS
    elif args.dataset_group == "extended":
        datasets = EXTENDED_DATASETS
    else:
        datasets = MAIN_DATASETS + EXTENDED_DATASETS

    print(format_table(datasets, args.models))


if __name__ == "__main__":
    main()
