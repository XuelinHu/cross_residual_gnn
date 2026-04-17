from __future__ import annotations

import argparse
import glob
import json
import statistics as st
from pathlib import Path
from typing import Dict, List

from geomatric.experiment_paths import DEFAULT_EXPERIMENT_VERSION, ensure_version_manifest, log_dir, normalize_version

ROOT = Path(__file__).resolve().parents[1]
OUT_PATH = ROOT / "md" / "topic_exp_tables.tex"

DATASETS = ["PROTEINS", "DD", "ENZYMES"]
MODELS = ["PlainGNN", "NodeResGNN", "NodeCrossGNN", "GraphResGNN", "GraphCrossGNN"]


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
    rows = [load_result(dataset, model, fold, active_log_dir) for fold in range(5)]
    accs = [float(row["best_test_acc"]) for row in rows]
    losses = [float(row["test_loss"]) for row in rows]
    epochs = [int(row["best_epoch"]) + 1 for row in rows]
    params = int(rows[0]["parameter_stats"]["total_params"])
    return {
        "dataset": dataset,
        "model": model,
        "mean_acc": st.mean(accs),
        "std_acc": st.pstdev(accs),
        "mean_loss": st.mean(losses),
        "mean_best_epoch": st.mean(epochs),
        "params": params,
    }


def collect_summary(active_log_dir: Path) -> Dict[str, Dict[str, Dict[str, object]]]:
    summary: Dict[str, Dict[str, Dict[str, object]]] = {}
    for dataset in DATASETS:
        summary[dataset] = {}
        for model in MODELS:
            summary[dataset][model] = summarize_model(dataset, model, active_log_dir)
    return summary


def format_metric(mean_acc: float, std_acc: float, is_best: bool) -> str:
    text = f"{mean_acc:.4f} $\\pm$ {std_acc:.4f}"
    return f"\\textbf{{{text}}}" if is_best else text


def build_main_table(summary: Dict[str, Dict[str, Dict[str, object]]]) -> str:
    best_by_dataset = {
        dataset: max(rows.values(), key=lambda row: row["mean_acc"])["mean_acc"]
        for dataset, rows in summary.items()
    }
    lines: List[str] = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Topic-facing benchmark results on protein-related graph datasets. Results are reported as mean best test accuracy $\pm$ standard deviation over 5 stratified folds.}",
        r"\label{tab:topic_main_results}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Model & PROTEINS & DD & ENZYMES \\",
        r"\midrule",
    ]
    for model in MODELS:
        cells = []
        for dataset in DATASETS:
            row = summary[dataset][model]
            is_best = abs(row["mean_acc"] - best_by_dataset[dataset]) < 1e-12
            cells.append(format_metric(row["mean_acc"], row["std_acc"], is_best))
        lines.append(f"{model} & {' & '.join(cells)} \\\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def build_aux_table(summary: Dict[str, Dict[str, Dict[str, object]]]) -> str:
    lines: List[str] = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Optimization and complexity summary on the topic-facing datasets. Parameter counts depend on dataset input dimension, so they are reported per dataset.}",
        r"\label{tab:topic_aux_results}",
        r"\begin{tabular}{llccc}",
        r"\toprule",
        r"Dataset & Model & Params & Mean Test Loss & Mean Best Epoch \\",
        r"\midrule",
    ]
    for dataset in DATASETS:
        rows = [summary[dataset][model] for model in MODELS]
        rows.sort(key=lambda row: (-row["mean_acc"], row["mean_loss"]))
        for row in rows:
            lines.append(
                f"{dataset} & {row['model']} & {row['params']} & "
                f"{row['mean_loss']:.4f} & {row['mean_best_epoch']:.1f} \\\\"
            )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.extend([r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate topic-facing LaTeX tables from latest V3 logs.")
    parser.add_argument("--output", default=str(OUT_PATH))
    parser.add_argument("--version", default=DEFAULT_EXPERIMENT_VERSION)
    args = parser.parse_args()

    ensure_version_manifest(ROOT)
    version = normalize_version(args.version)
    output_default = ROOT / "md" / f"topic_exp_tables_{version}.tex"
    summary = collect_summary(log_dir(ROOT, version))
    content = "\n\n".join([build_main_table(summary), build_aux_table(summary)]) + "\n"
    output_path = Path(args.output) if args.output != str(OUT_PATH) else output_default
    output_path.write_text(content, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
