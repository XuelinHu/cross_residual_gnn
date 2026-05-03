"""
Statistical significance tests for key CR-GNN comparisons.
Reads fold-level results from logs/LATEST and computes paired t-tests
and Wilcoxon signed-rank tests for the main benchmark comparisons.

Note: With n=5 folds, the Wilcoxon signed-rank test has a hard floor
at p=0.0625 (all 5 differences in the same direction). The t-test is
used as the primary significance indicator in the generated table.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geomatric.experiment_catalog import (
    ALL_ACTIVE_DATASETS,
    FOCUSED_MODELS,
    ACTIVE_OPERATORS,
)

LOG_DIR = ROOT / "logs" / "LATEST"
OUTPUT_DIR = ROOT / "md"


def load_fold_accs(dataset: str, model: str, operator: str) -> List[float]:
    accs = []
    for fold in range(5):
        pattern = str(
            LOG_DIR / f"train_{dataset}_{model}_{operator}_fold{fold}__*.json"
        )
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise FileNotFoundError(
                f"No log for {dataset}/{model}/{operator}/fold{fold}"
            )
        data = json.loads(Path(matches[-1]).read_text(encoding="utf-8"))
        accs.append(float(data["best_test_acc"]))
    return accs


def paired_tests(accs_a: List[float], accs_b: List[float], alpha: float = 0.05) -> Dict:
    a = np.array(accs_a)
    b = np.array(accs_b)
    diff = a - b

    t_stat, t_p = stats.ttest_rel(a, b)

    try:
        w_stat, w_p = stats.wilcoxon(a, b, alternative="two-sided")
    except ValueError:
        w_stat, w_p = np.nan, 1.0

    d = np.mean(diff) / (np.std(diff, ddof=1) + 1e-10)

    return {
        "mean_a": float(np.mean(a)),
        "std_a": float(np.std(a, ddof=1)),
        "mean_b": float(np.mean(b)),
        "std_b": float(np.std(b, ddof=1)),
        "mean_diff": float(np.mean(diff)),
        "t_stat": float(t_stat),
        "t_p_value": float(t_p),
        "t_significant": t_p < alpha,
        "w_stat": float(w_stat) if not np.isnan(w_stat) else None,
        "w_p_value": float(w_p),
        "w_significant": w_p < alpha,
        "cohens_d": float(d),
    }


COMPARISONS = [
    ("NodeResGNN", "PlainGNN", "residual vs plain"),
    ("NodeCrossGNN", "PlainGNN", "node-cross vs plain"),
    ("GraphResGNN", "PlainGNN", "graph-res vs plain"),
    ("GraphCrossGNN", "PlainGNN", "graph-cross vs plain"),
    ("NodeCrossGNN", "NodeResGNN", "node-cross vs node-res"),
    ("GraphCrossGNN", "GraphResGNN", "graph-cross vs graph-res"),
    ("GraphResGNN", "NodeResGNN", "graph-res vs node-res"),
    ("NodeCrossGNN", "GraphCrossGNN", "node-cross vs graph-cross"),
]


def format_result(result: Dict) -> str:
    sig_marker = ""
    if result["t_significant"] and result["w_significant"]:
        sig_marker = " **"
    elif result["t_significant"] or result["w_significant"]:
        sig_marker = " *"

    d = result["cohens_d"]
    if abs(d) < 0.2:
        effect = "negligible"
    elif abs(d) < 0.5:
        effect = "small"
    elif abs(d) < 0.8:
        effect = "medium"
    else:
        effect = "large"

    r = result
    return (
        f"Delta={r['mean_diff']:+.4f}, "
        f"t={r['t_stat']:+.3f} (p={r['t_p_value']:.4f}), "
        f"W p={r['w_p_value']:.4f}, "
        f"d={r['cohens_d']:+.3f} ({effect})"
        f"{sig_marker}"
    )


COMPARISON_SHORT = {
    "residual vs plain": "NodeRes $-$ Plain",
    "node-cross vs plain": "NodeCross $-$ Plain",
    "graph-res vs plain": "GraphRes $-$ Plain",
    "graph-cross vs plain": "GraphCross $-$ Plain",
    "node-cross vs node-res": "NodeCross $-$ NodeRes",
    "graph-cross vs graph-res": "GraphCross $-$ GraphRes",
    "graph-res vs node-res": "GraphRes $-$ NodeRes",
    "node-cross vs graph-cross": "NodeCross $-$ GraphCross",
}


def generate_latex_table(all_results: Dict) -> str:
    """Generate a transposed LaTeX table for statistical tests.

    Rows: dataset × comparison (48 rows), grouped by dataset.
    Columns: operator (4 cols).  Each cell shows Δ (d, p).
    Bold: p < 0.05.

    This transposed format avoids the 10-column overflow of the
    original layout and removes the need for \\resizebox.
    """
    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{3.5pt}",
        "\\caption{Paired $t$-tests for architecture comparisons (transposed). "
        "Rows: dataset $\\times$ comparison. Columns: operator. "
        "Cells: $\\Delta$ (Cohen's $d$, $p$). "
        "\\textbf{Bold}: $p<0.05$ ($n{=}5$ folds).}",
        "\\label{tab:statistical_tests}",
        "\\begin{tabular}{llcccc}",
        "\\toprule",
        "Dataset & Comparison & GCNConv & GATConv & SAGEConv & GINConv \\\\",
        "\\midrule",
    ]

    datasets = ALL_ACTIVE_DATASETS
    operators = ACTIVE_OPERATORS

    for ds_idx, ds in enumerate(datasets):
        for comp_idx, (model_a, model_b, label) in enumerate(COMPARISONS):
            short_label = COMPARISON_SHORT.get(label, label)
            if comp_idx == 0:
                row = f"{ds} & {short_label}"
            else:
                row = f" & {short_label}"

            for op in operators:
                key = f"{model_a}_vs_{model_b}"
                if key in all_results.get(ds, {}).get(op, {}):
                    r = all_results[ds][op][key]
                    md = r["mean_diff"]
                    d_val = r["cohens_d"]
                    p_val = r["t_p_value"]
                    cell = f"{md:+.3f} ({d_val:+.2f}, {p_val:.3f})"
                    if r["t_significant"]:
                        cell = "\\textbf{" + cell + "}"
                else:
                    cell = "N/A"
                row += f" & {cell}"
            row += " \\\\"
            lines.append(row)
        if ds_idx < len(datasets) - 1:
            lines.append("\\midrule")

    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")

    return "\n".join(lines)


def main():
    all_results: Dict[str, Dict[str, Dict]] = {}

    for ds in ALL_ACTIVE_DATASETS:
        all_results[ds] = {}
        for op in ACTIVE_OPERATORS:
            all_results[ds][op] = {}
            model_accs = {}
            for model in FOCUSED_MODELS:
                try:
                    model_accs[model] = load_fold_accs(ds, model, op)
                except FileNotFoundError:
                    print(f"Missing: {ds}/{model}/{op}, skipping comparisons")
                    continue

            for model_a, model_b, label in COMPARISONS:
                if model_a in model_accs and model_b in model_accs:
                    key = f"{model_a}_vs_{model_b}"
                    all_results[ds][op][key] = paired_tests(
                        model_accs[model_a], model_accs[model_b]
                    )

    latex = generate_latex_table(all_results)
    (OUTPUT_DIR / "statistical_tests_table.tex").write_text(
        latex, encoding="utf-8"
    )

    # Print summary
    print("=== Statistical Test Summary ===")
    for ds in ALL_ACTIVE_DATASETS:
        print(f"\n## {ds}")
        for op in ACTIVE_OPERATORS:
            print(f"  Operator: {op}")
            for model_a, model_b, label in COMPARISONS:
                key = f"{model_a}_vs_{model_b}"
                if key in all_results.get(ds, {}).get(op, {}):
                    print(f"    {label}: {format_result(all_results[ds][op][key])}")
    print("\nDone. Outputs written to md/")


if __name__ == "__main__":
    main()
