from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple


ROOT = Path(__file__).resolve().parents[1]
CORPUS_PATH = ROOT / "paper" / "paper_corpus_merged.json"
OUT_PATH = ROOT / "md" / "reference_logic_map.md"

GROUP_ORDER = [
    "GNN Foundations",
    "Deep GNN Stability",
    "Residual And Cross-Layer Architectures",
    "Graph Classification And Pooling",
    "Biomolecular And Protein Graph Learning",
    "Graph Datasets And Biological Benchmarks",
    "Plant-Oriented Motivation",
    "Additional Related Work",
]


def load_corpus() -> List[Dict[str, object]]:
    payload = json.loads(CORPUS_PATH.read_text(encoding="utf-8"))
    return payload["papers"]


def infer_group(paper: Dict[str, object]) -> str:
    title = ((paper.get("titles") or {}).get("en") or "").lower()
    relevance = ((paper.get("reference_context") or {}).get("relevance_to_cross_residual_gnn") or "").lower()
    category = ((paper.get("reference_context") or {}).get("category") or "").lower()

    if any(token in title for token in ["tudataset", "benchmark datasets for learning with graphs", "graph property prediction benchmark"]):
        return "Graph Datasets And Biological Benchmarks"
    if any(token in title for token in ["plant ", "arabidopsis", "planteome", "crop", "abiotic stress", "mirna in plants"]):
        return "Plant-Oriented Motivation"
    if any(token in title for token in ["protein", "biomolecular", "drug-protein", "ppi", "enzyme"]):
        return "Biomolecular And Protein Graph Learning"
    if any(token in title for token in ["jumping knowledge", "residual", "densegnn", "graph u-net", "appnp", "mixhop"]):
        return "Residual And Cross-Layer Architectures"
    if any(token in title for token in ["oversmoothing", "over-smoothing", "over-squashing", "bottleneck", "dropedge", "pairnorm", "graphnorm", "1000-layer"]):
        return "Deep GNN Stability"
    if any(token in title for token in ["pool", "graph classification", "molecular fingerprint", "graph representation learning"]):
        return "Graph Classification And Pooling"
    if category in {"foundational", "directly_relevant"} or any(
        token in title for token in ["graph neural", "message passing", "graph convolution", "graph attention", "graphsage", "how powerful are graph neural networks"]
    ):
        return "GNN Foundations"
    return "Additional Related Work"


def compact_entry(paper: Dict[str, object]) -> Tuple[int, str]:
    year = int(((paper.get("publication") or {}).get("year")) or 0)
    title = ((paper.get("titles") or {}).get("en") or "").strip()
    paper_id = paper["paper_id"]
    return year, f"- `{paper_id}` ({year}): {title}"


def build_markdown(papers: List[Dict[str, object]]) -> str:
    groups: Dict[str, List[Tuple[int, str]]] = {group: [] for group in GROUP_ORDER}
    for paper in papers:
        group = infer_group(paper)
        groups.setdefault(group, []).append(compact_entry(paper))

    lines: List[str] = [
        "# Reference Logic Map",
        "",
        "This note reorganizes the merged paper corpus into the final citation flow for the revised manuscript.",
        "",
        "Recommended section order:",
        "",
        "1. Introduction: foundations, deep GNN degradation, residual information flow, biomolecular motivation, plant extension logic",
        "2. Related Work: graph classification, pooling, hybrid architectures, biological graph learning, benchmark dataset context",
        "3. Datasets: TU biological datasets and supplementary robustness datasets",
        "4. Conclusion: plant-oriented future integration and omics extension",
        "",
    ]

    for group in GROUP_ORDER:
        entries = sorted(groups.get(group, []), key=lambda item: (item[0], item[1].lower()))
        lines.append(f"## {group}")
        lines.append("")
        if not entries:
            lines.append("- No papers assigned.")
            lines.append("")
            continue
        lines.extend(entry for _, entry in entries)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    papers = load_corpus()
    OUT_PATH.write_text(build_markdown(papers), encoding="utf-8")
    print(OUT_PATH)


if __name__ == "__main__":
    main()
