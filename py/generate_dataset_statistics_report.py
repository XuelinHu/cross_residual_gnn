from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from geomatric.experiment_catalog import DATASET_METADATA, MAIN_BIOLOGICAL_DATASETS, SUPPLEMENTARY_DATASETS
from geomatric.graph_classify_v3 import dataset_statistics, load_dataset

MD_DIR = ROOT / "md"

SUMMARY_OUT = MD_DIR / "dataset_statistics_summary.md"
TABLES_OUT = MD_DIR / "dataset_statistics_tables.tex"
JSON_OUT = MD_DIR / "dataset_statistics_summary.json"


def tu_stats(dataset_name: str) -> Dict[str, object]:
    dataset = load_dataset(dataset_name)
    return dataset_statistics(dataset, dataset_name)


def collect_rows(datasets: List[str]) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for dataset_name in datasets:
        metadata = DATASET_METADATA[dataset_name]
        stats = tu_stats(dataset_name)
        rows.append(
            {
                "dataset": dataset_name,
                "source": metadata["source"],
                "task_type": metadata["task_type"],
                "split_protocol": metadata["split_protocol"],
                "role": metadata["role"],
                "graphs": stats["graphs"],
                "classes": stats["classes"],
                "num_features": stats["num_features"],
                "avg_nodes": stats["avg_nodes"],
                "avg_edges": stats["avg_edges"],
                "statistics_note": stats.get("statistics_note", "Computed from local dataset export"),
            }
        )
    return rows


def build_summary(main_rows: List[Dict[str, object]], supp_rows: List[Dict[str, object]]) -> str:
    lines = [
        "# Dataset Statistics Summary",
        "",
        "This note consolidates the dataset facts used in the revised paper narrative.",
        "",
        "## Main Biological Package",
        "",
    ]
    for row in main_rows:
        lines.extend(
            [
                f"### {row['dataset']}",
                f"- source: {row['source']}",
                f"- task_type: {row['task_type']}",
                f"- split_protocol: {row['split_protocol']}",
                f"- role: {row['role']}",
                f"- graphs: {row['graphs']}",
                f"- classes: {row['classes']}",
                f"- num_features: {row['num_features']}",
                f"- avg_nodes: {row['avg_nodes']:.2f}",
                f"- avg_edges: {row['avg_edges']:.2f}",
                f"- note: {row['statistics_note']}",
                "",
            ]
        )
    lines.extend(["## Supplementary Robustness Package", ""])
    for row in supp_rows:
        lines.extend(
            [
                f"### {row['dataset']}",
                f"- source: {row['source']}",
                f"- task_type: {row['task_type']}",
                f"- split_protocol: {row['split_protocol']}",
                f"- role: {row['role']}",
                f"- graphs: {row['graphs']}",
                f"- classes: {row['classes']}",
                f"- num_features: {row['num_features']}",
                f"- avg_nodes: {row['avg_nodes']:.2f}",
                f"- avg_edges: {row['avg_edges']:.2f}",
                f"- note: {row['statistics_note']}",
                "",
            ]
        )
    return "\n".join(lines).rstrip() + "\n"


def tex_escape(text: str) -> str:
    return text.replace("_", r"\_")


def build_table(rows: List[Dict[str, object]], caption: str, label: str) -> str:
    def short_source(text: str) -> str:
        return text.replace("PyG TUDataset", "TU/PyG")

    def short_task(text: str) -> str:
        return text.replace("graph classification", "classification")

    def short_split(text: str) -> str:
        return text.replace("stratified 5-fold CV + inner validation split", "5-fold CV + val")

    def short_role(text: str) -> str:
        return text.replace("main biological benchmark", "main").replace("supplementary robustness dataset", "supp.")

    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{l l l l c c c c l}",
        r"\toprule",
        r"Dataset & Source & Task & Split & \#Graphs & \#Cls. & Feat. & Avg. N / E & Role \\",
        r"\midrule",
    ]
    for row in rows:
        avg_text = f"{row['avg_nodes']:.1f} / {row['avg_edges']:.1f}"
        lines.append(
            f"{tex_escape(row['dataset'])} & {short_source(row['source'])} & {short_task(row['task_type'])} & "
            f"{tex_escape(short_split(row['split_protocol']))} & {row['graphs']} & {row['classes']} & "
            f"{row['num_features']} & {avg_text} & {short_role(row['role'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    return "\n".join(lines)


def main() -> None:
    MD_DIR.mkdir(parents=True, exist_ok=True)
    main_rows = collect_rows(MAIN_BIOLOGICAL_DATASETS)
    supp_rows = collect_rows(SUPPLEMENTARY_DATASETS)

    SUMMARY_OUT.write_text(build_summary(main_rows, supp_rows), encoding="utf-8")
    TABLES_OUT.write_text(
        "\n\n".join(
            [
                build_table(main_rows, "Unified summary of the main biological benchmark package.", "tab:dataset_main_package"),
                build_table(supp_rows, "Supplementary robustness datasets retained outside the core biological narrative.", "tab:dataset_supp_package"),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    JSON_OUT.write_text(json.dumps({"main": main_rows, "supplementary": supp_rows}, indent=2), encoding="utf-8")
    print(SUMMARY_OUT)
    print(TABLES_OUT)
    print(JSON_OUT)


if __name__ == "__main__":
    main()
