from __future__ import annotations

import argparse
import glob
import json
import statistics as st
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geomatric.experiment_catalog import (
    ALL_ACTIVE_DATASETS,
    FOCUSED_MODELS,
    MAIN_BIOLOGICAL_DATASETS,
    SUPPLEMENTARY_DATASETS,
)
from geomatric.experiment_paths import DEFAULT_EXPERIMENT_VERSION, ensure_version_manifest, log_dir, normalize_version

MAIN_DATASETS = MAIN_BIOLOGICAL_DATASETS
TOPIC_DATASETS = MAIN_BIOLOGICAL_DATASETS
EXTENDED_DATASETS = SUPPLEMENTARY_DATASETS
ALL_DATASETS = ALL_ACTIVE_DATASETS


def latest_matching_log(dataset: str, model: str, fold: int, active_log_dir: Path) -> Path:
    pattern = str(active_log_dir / f"train_{dataset}_{model}_GCNConv_fold{fold}__*.json")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(pattern)
    return Path(matches[-1])


def load_result(dataset: str, model: str, fold: int, active_log_dir: Path) -> Dict[str, object]:
    path = latest_matching_log(dataset, model, fold, active_log_dir)
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_model(dataset: str, model: str, active_log_dir: Path) -> Dict[str, object]:
    try:
        rows = [load_result(dataset, model, fold, active_log_dir) for fold in range(5)]
    except FileNotFoundError:
        return {
            "dataset": dataset,
            "model": model,
            "pending": True,
        }
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


def format_table(datasets: Iterable[str], models: Iterable[str], active_log_dir: Path) -> str:
    lines: List[str] = []
    for dataset in datasets:
        lines.append(f"## {dataset}")
        summaries = [summarize_model(dataset, model, active_log_dir) for model in models]
        ready = [row for row in summaries if not row.get("pending")]
        pending = [row for row in summaries if row.get("pending")]
        ready.sort(key=lambda row: (-row["mean_acc"], row["mean_loss"]))
        for index, row in enumerate(ready, 1):
            fold_text = ",".join(f"{value:.5f}" for value in row["fold_accs"])
            lines.append(
                f"{index}. {row['model']}\tmean_acc={row['mean_acc']:.5f}\t"
                f"std_acc={row['std_acc']:.5f}\tmean_loss={row['mean_loss']:.5f}\t"
                f"mean_best_epoch={row['mean_best_epoch']:.1f}"
            )
            lines.append(f"   folds={fold_text}")
        for row in pending:
            lines.append(f"- {row['model']}\tpending=no complete log set found")
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
    parser.add_argument("--version", default=DEFAULT_EXPERIMENT_VERSION)
    args = parser.parse_args()
    ensure_version_manifest(ROOT)
    active_log_dir = log_dir(ROOT, normalize_version(args.version))

    if args.dataset_group == "main":
        datasets = MAIN_DATASETS
    elif args.dataset_group == "topic":
        datasets = TOPIC_DATASETS
    elif args.dataset_group == "extended":
        datasets = EXTENDED_DATASETS
    else:
        datasets = ALL_DATASETS

    print(format_table(datasets, args.models, active_log_dir))


if __name__ == "__main__":
    main()
