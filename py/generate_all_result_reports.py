from __future__ import annotations

import glob
import json
import statistics as st
import sys
from pathlib import Path
from typing import Dict, List


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geomatric.experiment_catalog import (
    ALL_ACTIVE_DATASETS,
    EXTERNAL_BASELINES,
    FOCUSED_MODELS,
    MAIN_BIOLOGICAL_DATASETS,
    MODEL_DISPLAY,
    SUPPLEMENTARY_DATASETS,
)
from geomatric.experiment_paths import DEFAULT_EXPERIMENT_VERSION, ensure_version_manifest, log_dir, normalize_version

MD_DIR = ROOT / "md"

DATASETS = ALL_ACTIVE_DATASETS
MAIN_DATASETS = MAIN_BIOLOGICAL_DATASETS
SUPP_DATASETS = SUPPLEMENTARY_DATASETS
# CR-GNN family + external baselines
CR_MODELS = list(FOCUSED_MODELS)
EXT_MODELS = [name for name, _ in EXTERNAL_BASELINES]
ALL_MODELS = CR_MODELS + EXT_MODELS

# Operator mapping: CR models all use GCNConv in main benchmark;
# external baselines each use their native operator.
_MODEL_OPERATOR = {m: "GCNConv" for m in CR_MODELS}
_MODEL_OPERATOR.update({name: op for name, op in EXTERNAL_BASELINES})

_BASELINE_DISPLAY = {
    "GraphSAGEBaseline": "GraphSAGE",
    "GINBaseline": "GIN",
    "JKNetBaseline": "JKNet",
    "APPNPBaseline": "APPNP",
}
_ALL_DISPLAY = {**MODEL_DISPLAY, **_BASELINE_DISPLAY}
MODELS = ALL_MODELS

TXT_OUT = MD_DIR / "all_results_summary.txt"
TEX_OUT = MD_DIR / "all_exp_tables.tex"
ANALYSIS_OUT = MD_DIR / "all_ablation_analysis.md"


def tex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def latest_matching_log(dataset: str, model: str, fold: int, active_log_dir: Path) -> Path:
    op = _MODEL_OPERATOR.get(model, "GCNConv")
    pattern = str(active_log_dir / f"train_{dataset}_{model}_{op}_fold{fold}__*.json")
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
    params = int(rows[0]["parameter_stats"]["total_params"])
    return {
        "dataset": dataset,
        "model": model,
        "mean_acc": st.mean(accs),
        "std_acc": st.pstdev(accs),
        "max_acc": max(accs),
        "min_acc": min(accs),
        "mean_loss": st.mean(losses),
        "std_loss": st.pstdev(losses),
        "mean_best_epoch": st.mean(epochs),
        "params": params,
        "fold_accs": accs,
    }


def collect_summary(active_log_dir: Path) -> Dict[str, Dict[str, Dict[str, object]]]:
    summary: Dict[str, Dict[str, Dict[str, object]]] = {}
    for dataset in DATASETS:
        summary[dataset] = {}
        for model in MODELS:
            summary[dataset][model] = summarize_model(dataset, model, active_log_dir)
    return summary


def ranked_rows(summary: Dict[str, Dict[str, Dict[str, object]]], dataset: str) -> List[Dict[str, object]]:
    rows = [summary[dataset][model] for model in MODELS if not summary[dataset][model].get("pending")]
    rows.sort(key=lambda row: (-row["mean_acc"], row["mean_loss"]))
    return rows


def build_text_summary(summary: Dict[str, Dict[str, Dict[str, object]]]) -> str:
    lines: List[str] = []
    for dataset in DATASETS:
        lines.append(f"## {dataset}")
        rows = ranked_rows(summary, dataset)
        for index, row in enumerate(rows, 1):
            fold_text = ",".join(f"{value:.5f}" for value in row["fold_accs"])
            lines.append(
                f"{index}. {row['model']}\tmean_acc={row['mean_acc']:.5f}\t"
                f"std_acc={row['std_acc']:.5f}\tmax_acc={row['max_acc']:.5f}\t"
                f"min_acc={row['min_acc']:.5f}\tmean_loss={row['mean_loss']:.5f}\t"
                f"mean_best_epoch={row['mean_best_epoch']:.1f}\tparams={row['params']}"
            )
            lines.append(f"   folds={fold_text}")
        for model in MODELS:
            if summary[dataset][model].get("pending"):
                lines.append(f"- {model}\tpending=no complete log set found")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def format_metric(row: Dict[str, object], best_acc: float) -> str:
    """Format a cell as: mean ± std [min–max], bold if best mean."""
    text = (
        f"{row['mean_acc']:.4f} $\\pm$ {row['std_acc']:.4f} "
        f"[{row['min_acc']:.3f}, {row['max_acc']:.3f}]"
    )
    if abs(row["mean_acc"] - best_acc) < 1e-12:
        return f"\\textbf{{{text}}}"
    return text


def build_tex_tables(summary: Dict[str, Dict[str, Dict[str, object]]]) -> str:
    best_by_dataset = {
        dataset: max(
            (row for row in summary[dataset].values() if not row.get("pending")),
            key=lambda row: row["mean_acc"],
        )["mean_acc"]
        for dataset in DATASETS
        if ranked_rows(summary, dataset)
    }

    chunks = [MAIN_DATASETS, SUPP_DATASETS]
    tables: List[str] = []
    for table_index, datasets in enumerate(chunks, 1):
        colspec = "l" + ("c" * len(datasets))
        lines: List[str] = [
            r"\begin{table*}[t]",
            r"\centering",
            r"\footnotesize",
            r"\setlength{\tabcolsep}{4pt}",
            (
                r"\caption{Main biological benchmark results. "
                r"Cells show mean $\pm$ std [min, max] over five folds. Bold: best mean per dataset.}"
                if table_index == 1
                else r"\caption{Supplementary robustness benchmark results. "
                r"Cells show mean $\pm$ std [min, max] over five folds. Bold: best mean per dataset.}"
            ),
            rf"\label{{tab:all_results_{table_index}}}",
            rf"\begin{{tabular}}{{{colspec}}}",
            r"\toprule",
            "Model & " + " & ".join(tex_escape(dataset) for dataset in datasets) + r" \\",
            r"\midrule",
        ]
        for model in MODELS:
            cells = []
            for dataset in datasets:
                row = summary[dataset][model]
                if row.get("pending"):
                    cells.append("--")
                else:
                    cells.append(format_metric(row, best_by_dataset[dataset]))
            display = _ALL_DISPLAY.get(model, model)
            lines.append(f"{display} & {' & '.join(cells)} \\\\")
        lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
        tables.append("\n".join(lines))

    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\small",
        r"\caption{Winner summary with max/min fold accuracy and parameter count for each dataset.}",
        r"\label{tab:all_winners}",
        r"\begin{tabular}{l l c c c c c}",
        r"\toprule",
        r"Dataset & Winner & Mean Acc & Max Acc & Min Acc & Best Epoch & Params \\",
        r"\midrule",
    ]
    for dataset in DATASETS:
        rows = ranked_rows(summary, dataset)
        if not rows:
            lines.append(f"{tex_escape(dataset)} & pending & -- & -- & -- & -- & -- \\\\")
        else:
            winner = rows[0]
            winner_display = _ALL_DISPLAY.get(winner['model'], winner['model'])
            lines.append(
                f"{tex_escape(dataset)} & {winner_display} & {winner['mean_acc']:.4f} & "
                f"{winner['max_acc']:.4f} & {winner['min_acc']:.4f} & "
                f"{winner['mean_best_epoch']:.1f} & {winner['params']} \\\\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    tables.append("\n".join(lines))
    return "\n\n".join(tables) + "\n"


def build_analysis(summary: Dict[str, Dict[str, Dict[str, object]]]) -> str:
    win_counts = {model: 0 for model in MODELS}
    rank_sums = {model: 0 for model in MODELS}
    lines: List[str] = [
        "# Full-Suite Ablation Analysis",
        "",
        "## Scope",
        "",
        "This note summarizes the completed full-suite run over all seven available datasets.",
        "",
        "Datasets:",
        "",
    ]
    for dataset in DATASETS:
        lines.append(f"- `{dataset}`")
    lines.extend(["", "Compared models:", ""])
    for model in MODELS:
        lines.append(f"- `{model}`")

    lines.extend(["", "## Dataset Winners", ""])
    for dataset in DATASETS:
        rows = ranked_rows(summary, dataset)
        if not rows:
            lines.append(f"- `{dataset}`: pending, no complete log set found yet.")
            continue
        winner = rows[0]
        win_counts[winner["model"]] += 1
        for rank, row in enumerate(rows, 1):
            rank_sums[row["model"]] += rank
        lines.append(
            f"- `{dataset}`: `{winner['model']}` with `{winner['mean_acc']:.5f} ± {winner['std_acc']:.5f}`"
        )

    avg_ranks = {model: rank_sums[model] / len(DATASETS) for model in MODELS}
    lines.extend(["", "## Overall Ranking Signals", ""])
    for model in sorted(MODELS, key=lambda item: (win_counts[item], -avg_ranks[item]), reverse=True):
        lines.append(
            f"- `{model}`: wins `{win_counts[model]}/7`, average rank `{avg_ranks[model]:.2f}`"
        )

    lines.extend(["", "## Cross vs Plain and Residual", ""])
    cross_wins_over_plain = 0
    cross_wins_over_residual = 0
    node_cross_over_node_res = 0
    graph_cross_over_graph_res = 0

    for dataset in DATASETS:
        if not ranked_rows(summary, dataset):
            lines.append(f"- `{dataset}`: pending, no complete log set found yet.")
            continue
        plain = summary[dataset]["PlainGNN"]["mean_acc"]
        node_res = summary[dataset]["NodeResGNN"]["mean_acc"]
        node_cross = summary[dataset]["NodeCrossGNN"]["mean_acc"]
        graph_res = summary[dataset]["GraphResGNN"]["mean_acc"]
        graph_cross = summary[dataset]["GraphCrossGNN"]["mean_acc"]
        best_cross = max(node_cross, graph_cross)
        best_residual = max(node_res, graph_res)
        if best_cross > plain:
            cross_wins_over_plain += 1
        if best_cross > best_residual:
            cross_wins_over_residual += 1
        if node_cross > node_res:
            node_cross_over_node_res += 1
        if graph_cross > graph_res:
            graph_cross_over_graph_res += 1
        lines.append(
            f"- `{dataset}`: best cross vs plain `{best_cross - plain:+.5f}`, "
            f"best cross vs best residual `{best_cross - best_residual:+.5f}`, "
            f"`NodeCrossGNN - NodeResGNN = {node_cross - node_res:+.5f}`, "
            f"`GraphCrossGNN - GraphResGNN = {graph_cross - graph_res:+.5f}`"
        )

    lines.extend(
        [
            "",
            "## Aggregated Conclusions",
            "",
            f"- Best cross model beats `PlainGNN` on `{cross_wins_over_plain}` completed datasets.",
            f"- Best cross model beats the best residual baseline on `{cross_wins_over_residual}` completed datasets.",
            f"- `NodeCrossGNN` beats `NodeResGNN` on `{node_cross_over_node_res}` completed datasets.",
            f"- `GraphCrossGNN` beats `GraphResGNN` on `{graph_cross_over_graph_res}` completed datasets.",
            "",
            "## Interpretation",
            "",
        ]
    )

    best_cross_datasets = [
        dataset
        for dataset in DATASETS
        if ranked_rows(summary, dataset)
        if max(
            summary[dataset]["NodeCrossGNN"]["mean_acc"],
            summary[dataset]["GraphCrossGNN"]["mean_acc"],
        )
        > max(
            summary[dataset]["NodeResGNN"]["mean_acc"],
            summary[dataset]["GraphResGNN"]["mean_acc"],
        )
    ]
    lines.append(
        "- `Cross` is not the strongest default family in the full suite. "
        f"It only beats the best residual baseline on {len(best_cross_datasets)} completed datasets: "
        + ", ".join(f"`{dataset}`" for dataset in best_cross_datasets)
        + "."
    )
    lines.append(
        "- `Residual` remains the strongest default family across the active benchmark package, "
        "especially on the topic-facing protein-oriented datasets."
    )
    lines.append(
        "- `Cross` still has selective value. It wins outright on selected datasets such as `MUTAG` and `Mutagenicity`, "
        "which means the idea is useful, but not universally dominant."
    )
    lines.append(
        "- `PlainGNN` never wins the full-suite benchmark. It stays competitive on `PROTEINS` and `DD`, "
        "but the stronger information-flow variants dominate the top ranks."
    )
    lines.append(
        "- The most defensible final claim is that cross-residual design is a meaningful alternative information-flow mechanism "
        "whose gains are dataset-dependent, while residual reuse remains the stronger default baseline."
    )
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Generate aggregate result reports from versioned logs.")
    parser.add_argument("--version", default=DEFAULT_EXPERIMENT_VERSION)
    args = parser.parse_args()

    ensure_version_manifest(ROOT)
    version = normalize_version(args.version)
    active_log_dir = log_dir(ROOT, version)
    txt_out = MD_DIR / f"all_results_summary_{version}.txt"
    tex_out = MD_DIR / f"all_exp_tables_{version}.tex"
    analysis_out = MD_DIR / f"all_ablation_analysis_{version}.md"

    summary = collect_summary(active_log_dir)
    txt_out.write_text(build_text_summary(summary), encoding="utf-8")
    tex_out.write_text(build_tex_tables(summary), encoding="utf-8")
    analysis_out.write_text(build_analysis(summary), encoding="utf-8")
    print(txt_out)
    print(tex_out)
    print(analysis_out)


if __name__ == "__main__":
    main()
